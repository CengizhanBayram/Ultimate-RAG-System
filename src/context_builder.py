import logging
from typing import List, Dict

from .models import ScoredDocument, Document
from .config import Settings

logger = logging.getLogger(__name__)


def _approx_tokens(text: str) -> float:
    return len(text.split()) * 1.3


LABEL_MAP = {
    "guncelleme_logu": "UPDATE_LOG",
    "fiyat_tablosu": "PRICE_TABLE",
    "sozlesme": "CONTRACT",
}


class ContextBuilder:
    """
    Token-aware context assembler with parent-chunk promotion.

    Architectural decision — parent chunk promotion:
    During retrieval, small child chunks (200 tokens) are matched because they're
    semantically tight. But at generation time, the LLM needs more context to
    avoid confusion from partial sentences or missing co-references. We swap each
    child for its parent (~600 tokens) before assembling the final prompt.

    Section ordering: UPDATE_LOG → PRICE_TABLE → CONTRACT
    This mirrors source priority and trains the LLM's attention to read the most
    authoritative sources first. The system prompt reinforces this ordering.

    Overflow strategy:
    - UPDATE_LOG sections are NEVER truncated (they are small and critical).
    - PRICE_TABLE rows are NEVER truncated (structured, compact).
    - CONTRACT sections are truncated last to fit remaining token budget.
    """

    def __init__(self, parent_map: Dict[str, Document], settings: Settings):
        self.parent_map = parent_map
        self.max_tokens = settings.max_context_tokens
        self.overflow_strategy = settings.context_overflow_strategy

    def build(self, ranked_docs: List[ScoredDocument]) -> str:
        """Build the final context string for the LLM prompt."""
        sections: Dict[str, List[str]] = {
            "guncelleme_logu": [],
            "fiyat_tablosu": [],
            "sozlesme": [],
        }

        for sd in ranked_docs:
            content = self._get_parent_content(sd.document)
            meta = sd.document.metadata
            source_type = meta.get("type", "sozlesme")
            label = LABEL_MAP.get(source_type, "UNKNOWN")

            # Superseded warning annotation
            superseded_note = ""
            if sd.is_superseded:
                superseded_note = (
                    f"\n  ⚠️ [SUPERSEDED by {sd.superseded_by}]: {sd.conflict_note}"
                )

            # Override note (for update logs that override something)
            override_note = ""
            if sd.conflict_note and not sd.is_superseded:
                override_note = f"\n  ℹ️ OVERRIDES: {sd.conflict_note}"

            ref = (
                meta.get("guncelleme_id")
                or meta.get("madde")
                or meta.get("paket", "")
            )
            date_val = meta.get("tarih") or meta.get("belge_tarihi", "N/A")

            chunk_text = (
                f"[{label}] {meta.get('source')} "
                f"| ref={ref} "
                f"| date={date_val}"
                f"{superseded_note}{override_note}\n"
                f"{content}"
            )
            bucket = source_type if source_type in sections else "sozlesme"
            sections[bucket].append(chunk_text)

        # Order: updates first (highest priority), then price, then contract
        ordered = (
            sections["guncelleme_logu"]
            + sections["fiyat_tablosu"]
            + sections["sozlesme"]
        )

        full_context = "\n\n---\n\n".join(ordered)
        estimated_tokens = _approx_tokens(full_context)

        if estimated_tokens > self.max_tokens:
            full_context = self._handle_overflow(ordered)
            logger.warning(
                f"Context overflow: {estimated_tokens:.0f} est. tokens → truncated."
            )

        return full_context

    def _get_parent_content(self, doc: Document) -> str:
        """Return parent chunk content if available, else child content."""
        if doc.parent_id and doc.parent_id in self.parent_map:
            return self.parent_map[doc.parent_id].content
        return doc.content

    def _handle_overflow(self, sections: List[str]) -> str:
        """
        Fit context within token budget:
        - Keep UPDATE_LOG and PRICE_TABLE sections intact.
        - Truncate CONTRACT sections to fit remaining budget.
        """
        truncated: List[str] = []
        budget_remaining = self.max_tokens

        for section in sections:
            tokens = _approx_tokens(section)
            is_update = section.startswith("[UPDATE_LOG]")
            is_price = section.startswith("[PRICE_TABLE]")

            if is_update or is_price:
                # Always keep these intact
                truncated.append(section)
                budget_remaining -= tokens
            elif tokens <= budget_remaining:
                truncated.append(section)
                budget_remaining -= tokens
            elif budget_remaining > 100:
                # Truncate to available budget
                words = section.split()
                keep = max(1, int(budget_remaining / 1.3))
                truncated.append(" ".join(words[:keep]) + " [...truncated]")
                budget_remaining = 0
            # else: skip section entirely

        return "\n\n---\n\n".join(truncated)
