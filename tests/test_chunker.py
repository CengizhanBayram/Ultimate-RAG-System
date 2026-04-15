"""Tests for HierarchicalChunker."""
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.chunker import HierarchicalChunker
from src.models import Document


def _make_settings(child=200, parent=600, overlap=40):
    s = MagicMock()
    s.child_chunk_size = child
    s.parent_chunk_size = parent
    s.chunk_overlap = overlap
    return s


def _sozlesme_doc(content, madde="4.1"):
    return Document(
        content=content,
        metadata={
            "source": "sozlesme.txt",
            "type": "sozlesme",
            "madde": madde,
            "belge_tarihi": "2023-01-15",
            "chunk_id": "test-id",
            "priority_tier": 3,
        },
    )


def _csv_doc():
    return Document(
        content="Paket: Pro | Aylık: ₺599",
        metadata={
            "source": "paket_fiyatlari.csv",
            "type": "fiyat_tablosu",
            "paket": "Pro",
            "chunk_id": "csv-id",
            "priority_tier": 2,
        },
    )


def _json_doc():
    return Document(
        content="[UPD-001 | 2024-01-10] Değişiklik",
        metadata={
            "source": "guncellemeler.json",
            "type": "guncelleme_logu",
            "guncelleme_id": "UPD-001",
            "chunk_id": "json-id",
            "priority_tier": 1,
        },
    )


class TestHierarchicalChunker:
    def test_atomic_csv_not_split(self):
        chunker = HierarchicalChunker(_make_settings())
        doc = _csv_doc()
        children, parent_map = chunker.chunk([doc])
        assert len(children) == 1
        assert children[0].content == doc.content

    def test_atomic_json_not_split(self):
        chunker = HierarchicalChunker(_make_settings())
        doc = _json_doc()
        children, parent_map = chunker.chunk([doc])
        assert len(children) == 1

    def test_parent_map_populated(self):
        chunker = HierarchicalChunker(_make_settings())
        docs = [_csv_doc(), _json_doc()]
        children, parent_map = chunker.chunk(docs)
        assert len(parent_map) >= len(children)

    def test_child_has_parent_id(self):
        chunker = HierarchicalChunker(_make_settings())
        docs = [_csv_doc()]
        children, parent_map = chunker.chunk(docs)
        for child in children:
            assert child.parent_id is not None
            assert child.parent_id in parent_map

    def test_long_sozlesme_creates_multiple_children(self):
        chunker = HierarchicalChunker(_make_settings(child=50, parent=150, overlap=10))
        long_text = "Madde 4.1: " + " ".join(["kelime"] * 300)
        doc = _sozlesme_doc(long_text)
        children, parent_map = chunker.chunk([doc])
        assert len(children) > 1, "Long text should be split into multiple children"

    def test_children_preserve_madde_metadata(self):
        chunker = HierarchicalChunker(_make_settings(child=50, parent=150, overlap=10))
        long_text = "Madde 5: " + " ".join(["metin"] * 200)
        doc = _sozlesme_doc(long_text, madde="5")
        children, _ = chunker.chunk([doc])
        for child in children:
            assert child.metadata.get("madde") == "5"

    def test_parent_content_available(self):
        chunker = HierarchicalChunker(_make_settings())
        doc = _csv_doc()
        children, parent_map = chunker.chunk([doc])
        child = children[0]
        assert child.parent_id in parent_map
        parent = parent_map[child.parent_id]
        assert parent.content == doc.content
