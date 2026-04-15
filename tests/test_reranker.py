"""Tests for PriorityReranker conflict detection."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.reranker import PriorityReranker
from src.models import Document, ScoredDocument


def _sd(doc_type, madde=None, paket=None, tarih="2024-01-01",
        gid="UPD-001", onceki="14 gün", yeni="21 gün", ce_score=1.0):
    meta = {
        "source": f"{doc_type}.txt",
        "type": doc_type,
        "chunk_id": f"chunk-{doc_type}-{gid}",
        "priority_tier": 1 if doc_type == "guncelleme_logu" else 3,
    }
    if madde:
        meta["madde"] = madde
        meta["etkilenen_madde"] = madde
    if paket:
        meta["paket"] = paket
        meta["etkilenen_paket"] = paket
    if doc_type == "guncelleme_logu":
        meta["guncelleme_id"] = gid
        meta["tarih"] = tarih
        meta["onceki_deger"] = onceki
        meta["yeni_deger"] = yeni

    return ScoredDocument(
        document=Document(content="test", metadata=meta),
        retrieval_score=0.5,
        cross_encoder_score=ce_score,
    )


class TestPriorityMultipliers:
    def test_update_log_highest_score(self):
        r = PriorityReranker()
        docs = [
            _sd("guncelleme_logu", madde="4.1", ce_score=1.0),
            _sd("fiyat_tablosu", ce_score=1.0),
            _sd("sozlesme", madde="4.1", ce_score=1.0),
        ]
        ranked = r.apply_priority(docs)
        assert ranked[0].document.metadata["type"] == "guncelleme_logu"

    def test_price_table_beats_contract(self):
        r = PriorityReranker()
        docs = [
            _sd("fiyat_tablosu", ce_score=1.0),
            _sd("sozlesme", ce_score=1.0),
        ]
        ranked = r.apply_priority(docs)
        assert ranked[0].document.metadata["type"] == "fiyat_tablosu"

    def test_final_score_computed(self):
        r = PriorityReranker()
        docs = [_sd("guncelleme_logu", ce_score=2.0)]
        ranked = r.apply_priority(docs)
        assert abs(ranked[0].final_score - 2.0 * 1.8) < 0.01


class TestConflictDetection:
    def test_sozlesme_superseded_by_update(self):
        r = PriorityReranker()
        update = _sd("guncelleme_logu", madde="4.1", paket=None,
                     tarih="2024-06-01", gid="UPD-002", yeni="30 gün")
        contract = _sd("sozlesme", madde="4.1")
        docs = r.apply_priority([update, contract])
        resolved = r.detect_and_resolve_conflicts(docs)
        contract_doc = next(
            d for d in resolved if d.document.metadata["type"] == "sozlesme"
        )
        assert contract_doc.is_superseded
        assert contract_doc.superseded_by == "UPD-002"

    def test_newer_update_supersedes_older(self):
        r = PriorityReranker()
        old_upd = _sd("guncelleme_logu", madde="4.1", paket="Basic",
                      tarih="2024-01-10", gid="UPD-001", yeni="21 gün")
        new_upd = _sd("guncelleme_logu", madde="4.1", paket="Basic",
                      tarih="2024-08-01", gid="UPD-007", yeni="7 gün")
        docs = r.apply_priority([old_upd, new_upd])
        resolved = r.detect_and_resolve_conflicts(docs)
        old_doc = next(
            d for d in resolved if d.document.metadata.get("guncelleme_id") == "UPD-001"
        )
        assert old_doc.is_superseded
        assert old_doc.superseded_by == "UPD-007"

    def test_package_specific_update_doesnt_affect_other_packages(self):
        r = PriorityReranker()
        pro_upd = _sd("guncelleme_logu", madde="4.1", paket="Pro",
                      tarih="2024-06-01", gid="UPD-002", yeni="30 gün")
        # Enterprise contract for same madde — should NOT be superseded by Pro-only update
        ent_contract = _sd("sozlesme", madde="4.1", paket="Enterprise")
        docs = r.apply_priority([pro_upd, ent_contract])
        resolved = r.detect_and_resolve_conflicts(docs)
        ent_doc = next(
            d for d in resolved if d.document.metadata.get("paket") == "Enterprise"
        )
        # Enterprise not superseded by Pro-specific update
        assert not ent_doc.is_superseded

    def test_conflict_note_populated(self):
        r = PriorityReranker()
        update = _sd("guncelleme_logu", madde="4.1", tarih="2024-06-01",
                     gid="UPD-002", yeni="30 gün")
        contract = _sd("sozlesme", madde="4.1")
        docs = r.apply_priority([update, contract])
        resolved = r.detect_and_resolve_conflicts(docs)
        contract_doc = next(
            d for d in resolved if d.document.metadata["type"] == "sozlesme"
        )
        assert contract_doc.conflict_note is not None
        assert "UPD-002" in contract_doc.conflict_note
