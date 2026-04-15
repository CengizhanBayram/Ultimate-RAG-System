"""Integration tests for RAGPipeline (mocked LLM and ML models)."""
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock
import numpy as np

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


def _make_settings(tmp_path):
    s = MagicMock()
    s.google_api_key = "test-key"
    s.gemini_model = "gemini-2.5-flash-preview-04-17"
    s.data_dir = Path(__file__).parent.parent / "data"
    s.index_dir = tmp_path / "index"
    s.cache_dir = tmp_path / "cache"
    s.embed_model = "BAAI/bge-m3"
    s.embed_batch_size = 8
    s.embed_workers = 1
    s.embed_device = "cpu"
    s.cross_encoder_model = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    s.cross_encoder_workers = 1
    s.top_k_retrieval = 10
    s.top_k_after_crossencoder = 5
    s.top_k_final = 3
    s.bm25_weight = 0.35
    s.dense_weight = 0.65
    s.rrf_k = 60
    s.query_expansion_variants = 2
    s.query_expansion_enabled = False  # disable for speed
    s.child_chunk_size = 100
    s.parent_chunk_size = 300
    s.chunk_overlap = 20
    s.max_context_tokens = 2000
    s.context_overflow_strategy = "truncate"
    s.faithfulness_threshold = 0.70
    s.relevance_threshold = 0.50
    s.cache_ttl_seconds = 60
    s.max_tokens = 512
    s.log_level = "WARNING"
    return s


class TestPipelineBuildAndQuery:
    """Integration tests that build index and run a query with mocked ML models."""

    @pytest.fixture
    def pipeline(self, tmp_path):
        """Build a pipeline with mocked ML components."""
        from src.pipeline import RAGPipeline

        settings = _make_settings(tmp_path)
        (tmp_path / "index").mkdir()
        (tmp_path / "cache").mkdir()

        with patch("src.embedder.EmbeddingEngine._get_model") as mock_model:
            # Mock embedding model
            fake_model = MagicMock()
            fake_model.encode.return_value = {"dense_vecs": np.random.rand(1, 768).astype(np.float32)}
            mock_model.return_value = fake_model

            with patch("src.cross_encoder.CrossEncoderReranker.__init__", return_value=None):
                with patch("src.query_expander.QueryExpander.__init__", return_value=None):
                    p = RAGPipeline(s=settings)
                    p.cross_encoder.model = MagicMock()
                    p.cross_encoder.model.predict.return_value = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1])
                    p.query_expander.enabled = False
                    p.query_expander.expand = lambda q, n=4: MagicMock(
                        original=q, variants=[], all_queries=[q]
                    )

                    # Build index with fake embeddings
                    with patch.object(p.embedder, "embed_documents",
                                      return_value=np.random.rand(20, 768).astype(np.float32)):
                        with patch.object(p.embedder, "embed_query",
                                          return_value=np.random.rand(768).astype(np.float32)):
                            p.build_index(force_rebuild=True)

        return p, settings

    def test_index_builds_without_error(self, pipeline):
        p, settings = pipeline
        assert p.vector_store.index is not None
        assert p.retriever.bm25 is not None

    def test_query_returns_response(self, pipeline):
        p, settings = pipeline

        # Mock generation and evaluation
        gen_response = MagicMock()
        gen_response.content = [MagicMock(text="Test yanıt [Kaynak: sozlesme.txt | 4.1 | 2023-01-15]")]
        eval_response = MagicMock()
        eval_response.content = [MagicMock(text=json.dumps({
            "faithfulness_score": 0.9,
            "relevance_score": 0.8,
            "unfaithful_claims": [],
            "explanation": "ok"
        }))]

        with patch.object(p.generator.client.messages, "create", return_value=gen_response):
            with patch.object(p.evaluator.client.messages, "create", return_value=eval_response):
                with patch.object(p.embedder, "embed_query",
                                  return_value=np.random.rand(768).astype(np.float32)):
                    result = p.query("Pro paket fiyatı nedir?")

        assert result is not None
        assert result.answer != ""
        assert result.chunk_count >= 0

    def test_cache_hit_on_second_query(self, pipeline):
        p, settings = pipeline

        gen_response = MagicMock()
        gen_response.content = [MagicMock(text="Yanıt")]
        eval_response = MagicMock()
        eval_response.content = [MagicMock(text=json.dumps({
            "faithfulness_score": 0.9, "relevance_score": 0.8,
            "unfaithful_claims": [], "explanation": "ok"
        }))]

        q = "Basic paket nedir?"
        with patch.object(p.generator.client.messages, "create", return_value=gen_response):
            with patch.object(p.evaluator.client.messages, "create", return_value=eval_response):
                with patch.object(p.embedder, "embed_query",
                                  return_value=np.random.rand(768).astype(np.float32)):
                    r1 = p.query(q)

        r2 = p.query(q)
        assert r2.was_cache_hit is True
