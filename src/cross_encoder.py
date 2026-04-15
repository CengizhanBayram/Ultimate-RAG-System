import logging
from typing import List, Tuple

from .models import Document, ScoredDocument
from .config import Settings

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    """
    Neural cross-encoder reranker using ms-marco-MiniLM-L-6-v2.

    Architectural decision — why cross-encoder after bi-encoder:
    Bi-encoders (like BGE-M3) encode query and document *independently* and compare
    via cosine similarity. This is fast but loses fine-grained interaction signals.
    A cross-encoder encodes (query, document) *jointly*, allowing full attention
    across both — capturing negation, entity co-reference, and numerical specifics
    that matter enormously for Turkish legal/contract text.

    The two-stage pipeline (bi-encoder retrieves 40, cross-encoder ranks 10) gives
    the accuracy of cross-encoder at 1/4 of the compute cost vs. re-encoding the
    entire corpus.

    Model choice: ms-marco-MiniLM-L-6-v2 is fast, small (22M params), and
    despite being trained on English MS MARCO, generalises well to Turkish
    because the legal structure of (query, passage) relevance transfers across
    languages. For production upgrade, consider: cross-encoder/ms-marco-MiniLM-L-12-v2.
    """

    def __init__(self, settings: Settings):
        logger.info(f"Loading cross-encoder: {settings.cross_encoder_model}")
        from sentence_transformers import CrossEncoder
        self.model = CrossEncoder(
            settings.cross_encoder_model,
            max_length=512,
            device=settings.embed_device,
        )
        logger.info("Cross-encoder loaded.")

    def rerank(
        self,
        query: str,
        candidates: List[Tuple[Document, float]],
        top_k: int,
    ) -> List[ScoredDocument]:
        """
        Score each (query, chunk_content) pair with the cross-encoder.
        Returns top_k ScoredDocuments sorted by cross-encoder score descending.
        """
        if not candidates:
            return []

        pairs = [(query, doc.content) for doc, _ in candidates]
        retrieval_scores = [score for _, score in candidates]
        docs = [doc for doc, _ in candidates]

        # Batch scoring — cross-encoder handles 40 pairs quickly
        ce_scores = self.model.predict(
            pairs, batch_size=16, show_progress_bar=False
        )

        results: List[ScoredDocument] = []
        for doc, ret_score, ce_score in zip(docs, retrieval_scores, ce_scores):
            results.append(
                ScoredDocument(
                    document=doc,
                    retrieval_score=float(ret_score),
                    cross_encoder_score=float(ce_score),
                    priority_score=0.0,
                    final_score=0.0,
                )
            )

        results.sort(key=lambda x: x.cross_encoder_score, reverse=True)
        logger.debug(
            f"Cross-encoder reranked {len(results)} candidates, "
            f"top score={results[0].cross_encoder_score:.3f}"
        )
        return results[:top_k]
