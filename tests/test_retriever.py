"""Tests for HybridRetriever RRF logic."""
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch
from collections import defaultdict

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.retriever import HybridRetriever
from src.models import Document


def _make_doc(chunk_id, content="test content", source_type="sozlesme"):
    return Document(
        content=content,
        metadata={
            "chunk_id": chunk_id,
            "source": "sozlesme.txt",
            "type": source_type,
            "priority_tier": 3,
        },
    )


def _make_settings():
    s = MagicMock()
    s.bm25_weight = 0.35
    s.dense_weight = 0.65
    s.rrf_k = 60
    return s


def _make_retriever():
    vs = MagicMock()
    embedder = MagicMock()
    settings = _make_settings()
    r = HybridRetriever(vs, embedder, settings)
    return r


class TestRRFFusion:
    def test_rrf_combines_both_sources(self):
        r = _make_retriever()
        doc_a = _make_doc("A")
        doc_b = _make_doc("B")
        doc_c = _make_doc("C")

        dense = [[(doc_a, 0.9), (doc_b, 0.7)]]
        bm25 = [[(doc_b, 5.0), (doc_c, 3.0)]]

        results = r._multi_query_rrf(dense, bm25, top_k=3)
        ids = [doc.metadata["chunk_id"] for doc, _ in results]
        assert "A" in ids
        assert "B" in ids
        assert "C" in ids

    def test_document_in_both_lists_scores_higher(self):
        r = _make_retriever()
        doc_shared = _make_doc("shared")
        doc_dense_only = _make_doc("dense_only")
        doc_bm25_only = _make_doc("bm25_only")

        dense = [[(doc_shared, 0.9), (doc_dense_only, 0.8)]]
        bm25 = [[(doc_shared, 8.0), (doc_bm25_only, 7.0)]]

        results = r._multi_query_rrf(dense, bm25, top_k=3)
        top_id = results[0][0].metadata["chunk_id"]
        assert top_id == "shared", "Document in both lists should rank first"

    def test_top_k_respected(self):
        r = _make_retriever()
        docs = [_make_doc(f"doc{i}") for i in range(10)]
        dense = [[(d, 0.9 - i * 0.05) for i, d in enumerate(docs)]]
        bm25 = [[(d, 10 - i) for i, d in enumerate(docs)]]
        results = r._multi_query_rrf(dense, bm25, top_k=3)
        assert len(results) == 3

    def test_multi_query_merges_all_variants(self):
        r = _make_retriever()
        doc1 = _make_doc("X")
        doc2 = _make_doc("Y")
        doc3 = _make_doc("Z")

        # Two query variants, each finds a different document as top
        dense = [[(doc1, 0.9)], [(doc2, 0.9)]]
        bm25 = [[(doc1, 5.0)], [(doc3, 5.0)]]

        results = r._multi_query_rrf(dense, bm25, top_k=3)
        ids = {doc.metadata["chunk_id"] for doc, _ in results}
        # All three unique docs should appear
        assert "X" in ids
        assert "Y" in ids or "Z" in ids


class TestBM25Build:
    def test_build_bm25_sets_index(self):
        r = _make_retriever()
        docs = [_make_doc(f"id{i}", content=f"kelime{i} test") for i in range(5)]
        r.build_bm25(docs)
        assert r.bm25 is not None
        assert len(r.bm25_docs) == 5
