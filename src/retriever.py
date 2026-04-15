import logging
import pickle
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import numpy as np
from rank_bm25 import BM25Okapi

from .models import Document
from .config import Settings

logger = logging.getLogger(__name__)


class HybridRetriever:
    """
    Hybrid BM25 (sparse) + BGE-M3 dense retrieval with RRF fusion.

    Architectural decision — why hybrid retrieval:
    - BM25 excels at keyword-exact matches (contract article numbers, package names,
      specific Turkish legal terms that embeddings may conflate).
    - Dense retrieval excels at semantic similarity (paraphrased questions,
      synonym-rich queries).
    - RRF (Reciprocal Rank Fusion) combines both rank lists without requiring
      score normalisation, which is notoriously difficult across different spaces.
    - Running retrieval for ALL query variants (original + expanded) in parallel
      maximises recall at the cost of compute, which is acceptable given the
      small corpus size.

    RRF formula: score(d) = Σ 1 / (k + rank(d))  where k=60 dampens top ranks.
    """

    def __init__(self, vector_store, embedder, settings: Settings):
        self.vector_store = vector_store
        self.embedder = embedder
        self.bm25: Optional[BM25Okapi] = None
        self.bm25_docs: List[Document] = []
        self.bm25_weight = settings.bm25_weight
        self.dense_weight = settings.dense_weight
        self.rrf_k = settings.rrf_k

    def build_bm25(self, documents: List[Document]) -> None:
        """Build BM25 index from document list."""
        tokenized = [doc.content.lower().split() for doc in documents]
        self.bm25 = BM25Okapi(tokenized)
        self.bm25_docs = documents
        logger.info(f"BM25 index built: {len(documents)} documents")

    def retrieve(
        self,
        queries: List[str],
        top_k: int,
        query_embedding: Optional[np.ndarray] = None,
    ) -> List[Tuple[Document, float]]:
        """
        Run retrieval for ALL query variants in parallel, merge via multi-query RRF.
        The first query in `queries` uses the pre-computed embedding if provided.
        """
        if self.bm25 is None:
            raise RuntimeError("BM25 index not built. Call build_bm25() first.")

        def dense_search(q: str, emb: Optional[np.ndarray]) -> List[Tuple[Document, float]]:
            vec = emb if emb is not None else self.embedder.embed_query(q)
            return self.vector_store.search(vec, top_k=top_k * 2)

        def bm25_search(q: str) -> List[Tuple[Document, float]]:
            tokens = q.lower().split()
            scores = self.bm25.get_scores(tokens)
            top_indices = np.argsort(scores)[::-1][: top_k * 2]
            return [(self.bm25_docs[i], float(scores[i])) for i in top_indices]

        max_workers = min(len(queries) * 2, 8)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            dense_futures = [
                executor.submit(
                    dense_search,
                    q,
                    query_embedding if i == 0 else None,
                )
                for i, q in enumerate(queries)
            ]
            bm25_futures = [
                executor.submit(bm25_search, q) for q in queries
            ]
            all_dense = [f.result() for f in dense_futures]
            all_bm25 = [f.result() for f in bm25_futures]

        return self._multi_query_rrf(all_dense, all_bm25, top_k)

    def _multi_query_rrf(
        self,
        dense_per_query: List[List[Tuple[Document, float]]],
        bm25_per_query: List[List[Tuple[Document, float]]],
        top_k: int,
    ) -> List[Tuple[Document, float]]:
        """Merge multiple ranked lists via weighted RRF."""
        scores: Dict[str, float] = defaultdict(float)
        id_to_doc: Dict[str, Document] = {}

        for results in dense_per_query:
            for rank, (doc, _) in enumerate(results):
                cid = doc.metadata["chunk_id"]
                scores[cid] += self.dense_weight * (1.0 / (self.rrf_k + rank + 1))
                id_to_doc[cid] = doc

        for results in bm25_per_query:
            for rank, (doc, _) in enumerate(results):
                cid = doc.metadata["chunk_id"]
                scores[cid] += self.bm25_weight * (1.0 / (self.rrf_k + rank + 1))
                id_to_doc[cid] = doc

        sorted_ids = sorted(scores, key=lambda x: scores[x], reverse=True)[:top_k]
        return [(id_to_doc[cid], scores[cid]) for cid in sorted_ids]

    def save_bm25(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump((self.bm25, self.bm25_docs), f)

    def load_bm25(self, path: Path) -> None:
        with open(path, "rb") as f:
            self.bm25, self.bm25_docs = pickle.load(f)
        logger.info(f"BM25 index loaded from {path}")
