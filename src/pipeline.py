import hashlib
import logging
import pickle
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, Optional

from .cache import QueryCache
from .chunker import HierarchicalChunker
from .config import Settings, settings
from .context_builder import ContextBuilder
from .cross_encoder import CrossEncoderReranker
from .embedder import EmbeddingEngine
from .evaluator import RAGEvaluator
from .generator import RAGGenerator
from .guardrail import HallucinationGuardrail
from .loaders import DocumentLoader
from .models import Document, GeneratorResponse, QueryExpansion
from .query_expander import QueryExpander
from .reranker import PriorityReranker
from .retriever import HybridRetriever
from .vector_store import VectorStore

logger = logging.getLogger(__name__)


class RAGPipeline:
    """
    Master orchestrator for the 6-stage RAG pipeline.

    Stage 1 — Query Expansion (Claude, async)
    Stage 2 — Parallel Hybrid Retrieval (BM25 + BGE-M3 dense + RRF)
    Stage 3 — Cross-Encoder Reranking (ms-marco-MiniLM)
    Stage 4 — Priority Scoring + Conflict Resolution
    Stage 5 — Context Assembly with Parent Promotion
    Stage 6 — Generation (Claude) → Evaluation → Guardrail
    """

    def __init__(self, s: Optional[Settings] = None):
        self._settings = s or settings
        self.loader = DocumentLoader()
        self.chunker = HierarchicalChunker(self._settings)
        self.embedder = EmbeddingEngine(self._settings)
        self.vector_store = VectorStore(self._settings.index_dir)
        self.retriever = HybridRetriever(
            self.vector_store, self.embedder, self._settings
        )
        self.query_expander = QueryExpander(self._settings)
        self.cross_encoder = CrossEncoderReranker(self._settings)
        self.priority_reranker = PriorityReranker()
        self.generator = RAGGenerator(self._settings)
        self.evaluator = RAGEvaluator(self._settings)
        self.guardrail = HallucinationGuardrail()
        self.cache = QueryCache(
            self._settings.cache_dir, self._settings.cache_ttl_seconds
        )
        self.parent_map: Dict[str, Document] = {}

    # ──────────────────────────────────────────────────────────────────────
    # QUERY
    # ──────────────────────────────────────────────────────────────────────

    def query(self, question: str) -> GeneratorResponse:
        t_start = time.perf_counter()

        # Stage 0: Cache check
        cached = self.cache.get(question)
        if cached:
            cached.was_cache_hit = True
            return cached

        # Stage 1 + 2A: Query expansion & query embedding — PARALLEL
        with ThreadPoolExecutor(max_workers=2) as executor:
            if self._settings.query_expansion_enabled:
                expansion_future = executor.submit(
                    self.query_expander.expand,
                    question,
                    self._settings.query_expansion_variants,
                )
            else:
                expansion_future = None
            embed_future = executor.submit(self.embedder.embed_query, question)

        query_embedding = embed_future.result()
        if expansion_future is not None:
            expansion = expansion_future.result()
        else:
            expansion = QueryExpansion(
                original=question, variants=[], all_queries=[question]
            )

        # Stage 2B: Hybrid retrieval (BM25 + dense, all variants, parallel)
        raw_candidates = self.retriever.retrieve(
            queries=expansion.all_queries,
            top_k=self._settings.top_k_retrieval,
            query_embedding=query_embedding,
        )

        if not raw_candidates:
            logger.warning("No candidates retrieved — returning empty response.")
            return self._empty_response(question, expansion, t_start)

        # Stage 3: Cross-encoder reranking
        cross_ranked = self.cross_encoder.rerank(
            query=question,
            candidates=raw_candidates,
            top_k=self._settings.top_k_after_crossencoder,
        )

        # Stage 4: Priority scoring + conflict detection
        priority_ranked = self.priority_reranker.apply_priority(cross_ranked)
        resolved = self.priority_reranker.detect_and_resolve_conflicts(priority_ranked)
        final_chunks = resolved[: self._settings.top_k_final]

        # Stage 5: Context assembly with parent promotion
        context_builder = ContextBuilder(self.parent_map, self._settings)
        context = context_builder.build(final_chunks)

        # Stage 6: Generation
        raw_answer = self.generator.generate(
            question, context, expansion.all_queries
        )

        # Stage 7: Faithfulness evaluation + guardrail
        evaluation = self.evaluator.evaluate(question, raw_answer, final_chunks)
        final_answer, was_blocked = self.guardrail.check(raw_answer, evaluation)

        # Stage 8: Assemble response
        latency_ms = (time.perf_counter() - t_start) * 1000
        response = GeneratorResponse(
            answer=final_answer,
            sources_used=list(
                {sd.document.metadata["source"] for sd in final_chunks}
            ),
            conflicts_resolved=[
                sd.conflict_note
                for sd in final_chunks
                if sd.conflict_note
            ],
            evaluation=evaluation,
            latency_ms=latency_ms,
            chunk_count=len(final_chunks),
            queries_used=expansion.all_queries,
            was_cache_hit=False,
        )

        if not was_blocked:
            self.cache.set(question, response)

        return response

    # ──────────────────────────────────────────────────────────────────────
    # INDEX BUILD
    # ──────────────────────────────────────────────────────────────────────

    def build_index(self, force_rebuild: bool = False) -> None:
        if not force_rebuild and self._index_is_fresh():
            logger.info("Index is fresh — loading from disk cache.")
            self.vector_store.load()
            self._load_bm25_cache()
            self._load_parent_map()
            return

        logger.info("Building index from scratch...")
        t0 = time.perf_counter()

        documents = self.loader.load_all_parallel(self._settings.data_dir)
        children, parent_map = self.chunker.chunk(documents)
        self.parent_map = parent_map

        embeddings = self.embedder.embed_documents(children)

        with ThreadPoolExecutor(max_workers=2) as executor:
            faiss_f = executor.submit(self.vector_store.build, embeddings, children)
            bm25_f = executor.submit(self.retriever.build_bm25, children)
        faiss_f.result()
        bm25_f.result()

        self._save_checksum()
        self._save_bm25_cache()
        self._save_parent_map()

        elapsed = time.perf_counter() - t0
        logger.info(
            f"Index built in {elapsed:.2f}s | "
            f"{len(children)} children | {len(parent_map)} parents"
        )

    # ──────────────────────────────────────────────────────────────────────
    # HELPERS
    # ──────────────────────────────────────────────────────────────────────

    def _index_is_fresh(self) -> bool:
        checksum_path = Path(self._settings.index_dir) / "checksum.txt"
        if not checksum_path.exists():
            return False
        return checksum_path.read_text().strip() == self._compute_checksum()

    def _compute_checksum(self) -> str:
        h = hashlib.md5()
        for fname in ["sozlesme.txt", "paket_fiyatlari.csv", "guncellemeler.json"]:
            fp = Path(self._settings.data_dir) / fname
            if fp.exists():
                h.update(fp.read_bytes())
        return h.hexdigest()

    def _save_checksum(self) -> None:
        path = Path(self._settings.index_dir) / "checksum.txt"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self._compute_checksum())

    def _save_bm25_cache(self) -> None:
        self.retriever.save_bm25(
            Path(self._settings.index_dir) / "bm25.pkl"
        )

    def _load_bm25_cache(self) -> None:
        bm25_path = Path(self._settings.index_dir) / "bm25.pkl"
        if bm25_path.exists():
            self.retriever.load_bm25(bm25_path)
        else:
            logger.warning("BM25 cache not found — rebuilding BM25 only.")
            self.retriever.build_bm25(self.vector_store.documents)

    def _save_parent_map(self) -> None:
        path = Path(self._settings.index_dir) / "parent_map.pkl"
        with open(path, "wb") as f:
            pickle.dump(self.parent_map, f)

    def _load_parent_map(self) -> None:
        path = Path(self._settings.index_dir) / "parent_map.pkl"
        if path.exists():
            with open(path, "rb") as f:
                self.parent_map = pickle.load(f)
        else:
            logger.warning("Parent map not found — context builder will use child content.")

    def _empty_response(
        self, question: str, expansion: QueryExpansion, t_start: float
    ) -> GeneratorResponse:
        from .models import EvaluationResult
        return GeneratorResponse(
            answer="Bu konuda bilgi tabanımda güncel veri bulunmuyor. Lütfen destek ekibimize başvurun.",
            sources_used=[],
            conflicts_resolved=[],
            evaluation=EvaluationResult(
                faithfulness_score=1.0,
                relevance_score=0.0,
                passes_guardrail=True,
                unfaithful_claims=[],
                explanation="No candidates retrieved.",
            ),
            latency_ms=(time.perf_counter() - t_start) * 1000,
            chunk_count=0,
            queries_used=expansion.all_queries,
        )
