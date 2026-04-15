import logging
from collections import defaultdict
from datetime import date
from typing import List, Dict, Tuple, Optional

from .models import ScoredDocument, Document

logger = logging.getLogger(__name__)

MULTIPLIERS = {
    "guncelleme_logu": 1.8,    # Most recent — always overrides
    "fiyat_tablosu": 1.3,       # Structured, authoritative pricing
    "sozlesme": 1.0,            # Baseline — valid unless overridden
}


class PriorityReranker:
    """
    Applies source-priority multipliers and resolves temporal conflicts.

    Architectural decision — priority multiplier rationale:
    In a multi-source RAG system with overlapping facts, we need a tie-breaking
    rule beyond semantic similarity. We encode domain knowledge into multipliers:
    - Updates (1.8x): JSON change logs are the ground truth for the current state.
      If a chunk from guncellemeler.json says "iade 30 gün", it overrides any
      contract text saying "14 gün".
    - Price table (1.3x): Structured CSV data is more precise than prose descriptions
      in the contract.
    - Contract (1.0x): The baseline; valid only when no update supersedes it.

    The conflict detection algorithm:
    1. Groups update records by (etkilenen_madde, etkilenen_paket) key.
    2. For each group, the latest-dated update wins.
    3. Contract/price-table chunks for the same (madde, paket) are marked [SUPERSEDED].
    4. The LLM system prompt instructs it to never use superseded content as primary.
    """

    def apply_priority(
        self, docs: List[ScoredDocument]
    ) -> List[ScoredDocument]:
        """Apply source-priority multipliers to final_score."""
        for sd in docs:
            source_type = sd.document.metadata.get("type", "sozlesme")
            multiplier = MULTIPLIERS.get(source_type, 1.0)
            sd.priority_score = multiplier
            sd.final_score = sd.cross_encoder_score * multiplier
        docs.sort(key=lambda x: x.final_score, reverse=True)
        return docs

    def detect_and_resolve_conflicts(
        self, docs: List[ScoredDocument]
    ) -> List[ScoredDocument]:
        """
        Group documents by (etkilenen_madde, etkilened_paket).
        For each group, mark older/lower-priority entries as superseded.
        The latest-dated JSON update always wins.
        """
        # Step 1: Collect all JSON update records from retrieved set
        updates_by_key: Dict[Tuple, List[Document]] = defaultdict(list)
        for sd in docs:
            meta = sd.document.metadata
            if meta.get("type") == "guncelleme_logu":
                madde = meta.get("etkilenen_madde")
                # Handle typo variant in data ("etkilened_paket" vs "etkilenen_paket")
                paket = meta.get("etkilenen_paket") or meta.get("etkilened_paket")
                key = (madde, paket)
                updates_by_key[key].append(sd.document)
                # Also register under (madde, None) for global updates
                if paket is not None:
                    updates_by_key[(madde, None)].append(sd.document)

        # Step 2: Sort each update group by date DESC → first = winner
        winning_updates: Dict[Tuple, Document] = {}
        for key, update_docs in updates_by_key.items():
            sorted_updates = sorted(
                update_docs,
                key=lambda d: _parse_date(d.metadata.get("tarih", "1970-01-01")),
                reverse=True,
            )
            winning_updates[key] = sorted_updates[0]

        # Step 3: Mark superseded documents
        for sd in docs:
            meta = sd.document.metadata
            doc_type = meta.get("type")

            if doc_type in ("sozlesme", "fiyat_tablosu"):
                # Determine what madde/paket this chunk corresponds to
                madde = meta.get("madde") or meta.get("etkilenen_madde")
                paket = meta.get("paket")

                # Check package-specific update first, then global
                for key in [(madde, paket), (madde, None)]:
                    if key in winning_updates:
                        winner = winning_updates[key]
                        winner_date = winner.metadata.get("tarih", "unknown")
                        winner_id = winner.metadata.get("guncelleme_id", "unknown")
                        sd.is_superseded = True
                        sd.superseded_by = winner_id
                        sd.conflict_note = (
                            f"'{meta.get('source')}' içindeki bu bilgi, "
                            f"{winner_date} tarihli {winner_id} güncellemesiyle "
                            f"geçersiz kılınmıştır. "
                            f"Yeni değer: {winner.metadata.get('yeni_deger', 'N/A')}"
                        )
                        break

            elif doc_type == "guncelleme_logu":
                # Check if this update is itself superseded by a newer one
                madde = meta.get("etkilenen_madde")
                paket = meta.get("etkilenen_paket") or meta.get("etkilened_paket")
                key = (madde, paket)
                if key in winning_updates:
                    winner = winning_updates[key]
                    if winner.metadata.get("guncelleme_id") != meta.get("guncelleme_id"):
                        winner_date = winner.metadata.get("tarih", "unknown")
                        winner_id = winner.metadata.get("guncelleme_id", "unknown")
                        sd.is_superseded = True
                        sd.superseded_by = winner_id
                        sd.conflict_note = (
                            f"Bu güncelleme kaydı ({meta.get('guncelleme_id')}), "
                            f"{winner_date} tarihli {winner_id} ile geçersiz kılınmıştır."
                        )

        return docs


def _parse_date(date_str: str) -> date:
    try:
        return date.fromisoformat(date_str)
    except (ValueError, AttributeError):
        return date(1970, 1, 1)
