"""Tests for DocumentLoader."""
import json
import os
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.loaders import DocumentLoader


@pytest.fixture
def loader():
    return DocumentLoader()


@pytest.fixture
def data_dir():
    return Path(__file__).parent.parent / "data"


class TestTxtLoader:
    def test_loads_sozlesme(self, loader, data_dir):
        docs = loader.load_txt(data_dir / "sozlesme.txt")
        assert len(docs) > 0, "Should load at least one document"

    def test_madde_metadata(self, loader, data_dir):
        docs = loader.load_txt(data_dir / "sozlesme.txt")
        maddes = [d.metadata.get("madde") for d in docs]
        assert "4.1" in maddes, "Madde 4.1 must be present"
        assert "7.3" in maddes, "Madde 7.3 must be present"

    def test_source_metadata(self, loader, data_dir):
        docs = loader.load_txt(data_dir / "sozlesme.txt")
        for doc in docs:
            assert doc.metadata["source"] == "sozlesme.txt"
            assert doc.metadata["type"] == "sozlesme"
            assert doc.metadata["priority_tier"] == 3
            assert "chunk_id" in doc.metadata

    def test_belge_tarihi(self, loader, data_dir):
        docs = loader.load_txt(data_dir / "sozlesme.txt")
        assert docs[0].metadata.get("belge_tarihi") == "2023-01-15"

    def test_no_empty_content(self, loader, data_dir):
        docs = loader.load_txt(data_dir / "sozlesme.txt")
        for doc in docs:
            assert doc.content.strip(), "No document should have empty content"


class TestCsvLoader:
    def test_loads_three_rows(self, loader, data_dir):
        docs = loader.load_csv(data_dir / "paket_fiyatlari.csv")
        assert len(docs) == 3, "Should load exactly 3 package rows"

    def test_package_names(self, loader, data_dir):
        docs = loader.load_csv(data_dir / "paket_fiyatlari.csv")
        names = {d.metadata["paket"] for d in docs}
        assert names == {"Basic", "Pro", "Enterprise"}

    def test_row_not_split(self, loader, data_dir):
        docs = loader.load_csv(data_dir / "paket_fiyatlari.csv")
        for doc in docs:
            assert "Paket:" in doc.content
            assert "Aylık:" in doc.content
            assert "SLA:" in doc.content

    def test_priority_tier(self, loader, data_dir):
        docs = loader.load_csv(data_dir / "paket_fiyatlari.csv")
        for doc in docs:
            assert doc.metadata["priority_tier"] == 2

    def test_no_comment_rows(self, loader, data_dir):
        docs = loader.load_csv(data_dir / "paket_fiyatlari.csv")
        # None of the documents should contain the comment
        for doc in docs:
            assert "Bu dosya" not in doc.content


class TestJsonLoader:
    def test_loads_eight_records(self, loader, data_dir):
        docs = loader.load_json(data_dir / "guncellemeler.json")
        assert len(docs) == 8, "Should load exactly 8 update records"

    def test_first_record(self, loader, data_dir):
        docs = loader.load_json(data_dir / "guncellemeler.json")
        upd001 = next(d for d in docs if d.metadata["guncelleme_id"] == "UPD-001")
        assert upd001.metadata["etkilenen_madde"] == "4.1"
        assert upd001.metadata["yeni_deger"] == "21 gün"

    def test_priority_tier(self, loader, data_dir):
        docs = loader.load_json(data_dir / "guncellemeler.json")
        for doc in docs:
            assert doc.metadata["priority_tier"] == 1

    def test_tarih_date_parsed(self, loader, data_dir):
        from datetime import date
        docs = loader.load_json(data_dir / "guncellemeler.json")
        for doc in docs:
            assert isinstance(doc.metadata["tarih_date"], date)

    def test_content_format(self, loader, data_dir):
        docs = loader.load_json(data_dir / "guncellemeler.json")
        for doc in docs:
            assert "Değişim:" in doc.content
            assert "Onaylayan:" in doc.content


class TestParallelLoader:
    def test_loads_all_sources(self, loader, data_dir):
        docs = loader.load_all_parallel(data_dir)
        types = {d.metadata["type"] for d in docs}
        assert "sozlesme" in types
        assert "fiyat_tablosu" in types
        assert "guncelleme_logu" in types

    def test_total_count(self, loader, data_dir):
        docs = loader.load_all_parallel(data_dir)
        assert len(docs) >= 12, "Should have at least 12 total documents"
