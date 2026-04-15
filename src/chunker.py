import re
import logging
from uuid import uuid4
from dataclasses import dataclass
from typing import List, Dict, Tuple

from .models import Document
from .config import Settings

logger = logging.getLogger(__name__)


def _approx_tokens(text: str) -> float:
    return len(text.split()) * 1.3


@dataclass
class ChunkPair:
    child: Document      # what gets embedded and retrieved
    parent: Document     # what gets sent to LLM after retrieval


class HierarchicalChunker:
    """
    Parent-child indexing strategy:
    - Child chunks (~200 tokens) → indexed in FAISS for retrieval precision
    - Parent chunks (~600 tokens) → stored in metadata, sent to LLM for richer context

    Architectural decision:
    The parent-child split improves the precision/recall tradeoff.
    Small children are more likely to match specific queries (high precision),
    while the parent provides enough surrounding context for the LLM to reason
    correctly without losing important co-references or sentence boundaries.
    """

    def __init__(self, settings: Settings):
        self.child_size = settings.child_chunk_size
        self.parent_size = settings.parent_chunk_size
        self.overlap = settings.chunk_overlap

    def chunk(
        self, documents: List[Document]
    ) -> Tuple[List[Document], Dict[str, Document]]:
        """
        Returns:
            children: list of child Documents (used for embedding + FAISS)
            parent_map: dict mapping parent_id -> parent Document
        """
        children: List[Document] = []
        parent_map: Dict[str, Document] = {}

        for doc in documents:
            source_type = doc.metadata.get("type", "sozlesme")

            if source_type == "sozlesme":
                child_list, parent_list = self._chunk_sozlesme(doc)
            elif source_type in ("fiyat_tablosu", "guncelleme_logu"):
                # Never split: child == parent == the whole record
                child_list, parent_list = self._chunk_atomic(doc)
            else:
                child_list, parent_list = self._chunk_atomic(doc)

            for child, parent in zip(child_list, parent_list):
                parent_map[parent.metadata["chunk_id"]] = parent
                child.parent_id = parent.metadata["chunk_id"]
                child.parent_content = parent.content
                children.append(child)

        logger.info(
            f"Chunking complete: {len(children)} children, {len(parent_map)} parents"
        )
        return children, parent_map

    def _chunk_atomic(self, doc: Document) -> Tuple[List[Document], List[Document]]:
        """For CSV rows and JSON records: child == parent, never split."""
        parent_id = str(uuid4())
        parent = Document(
            content=doc.content,
            metadata={**doc.metadata, "chunk_id": parent_id, "is_parent": True},
        )
        child_id = str(uuid4())
        child = Document(
            content=doc.content,
            metadata={
                **doc.metadata,
                "chunk_id": child_id,
                "parent_id": parent_id,
                "chunk_index": 0,
                "total_chunks": 1,
                "is_child": True,
            },
        )
        return [child], [parent]

    def _chunk_sozlesme(
        self, doc: Document
    ) -> Tuple[List[Document], List[Document]]:
        """
        For sozlesme articles: create parent chunks (~parent_size tokens) and
        child chunks (~child_size tokens) derived from parents.
        """
        text = doc.content
        base_meta = {k: v for k, v in doc.metadata.items() if k != "chunk_id"}

        # Step 1: Build parent chunks
        parent_texts = self._sliding_window(text, self.parent_size, self.overlap)

        children_out: List[Document] = []
        parents_out: List[Document] = []

        for pi, parent_text in enumerate(parent_texts):
            parent_id = str(uuid4())
            parent = Document(
                content=parent_text,
                metadata={
                    **base_meta,
                    "chunk_id": parent_id,
                    "chunk_index": pi,
                    "total_chunks": len(parent_texts),
                    "is_parent": True,
                },
            )
            parents_out.append(parent)

            # Step 2: Build child chunks from parent text
            child_texts = self._sliding_window(parent_text, self.child_size, self.overlap)
            for ci, child_text in enumerate(child_texts):
                child_id = str(uuid4())
                child = Document(
                    content=child_text,
                    metadata={
                        **base_meta,
                        "chunk_id": child_id,
                        "parent_id": parent_id,
                        "chunk_index": ci,
                        "total_chunks": len(child_texts),
                        "is_child": True,
                    },
                )
                children_out.append(child)
                parents_out.append(parent)  # each child maps to its parent

        # parents_out has duplicates (one entry per child for zip alignment)
        # Reconstruct as aligned pairs
        pairs: List[Tuple[Document, Document]] = []
        pidx = 0
        for pi, parent_text in enumerate(parent_texts):
            parent = [p for p in parents_out if p.metadata.get("chunk_index") == pi
                      and p.metadata.get("is_parent")][0]
            child_texts = self._sliding_window(parent_text, self.child_size, self.overlap)
            for ci in range(len(child_texts)):
                pairs.append((children_out[pidx], parent))
                pidx += 1
            if pidx >= len(children_out):
                break

        if not pairs:
            return self._chunk_atomic(doc)

        c_list = [p[0] for p in pairs]
        p_list = [p[1] for p in pairs]
        return c_list, p_list

    def _sliding_window(self, text: str, max_tokens: int, overlap: int) -> List[str]:
        """Sentence-boundary sliding window chunker with token overlap."""
        # Split into sentences
        sentence_pattern = re.compile(r"(?<=[.!?;])\s+")
        sentences = sentence_pattern.split(text)
        sentences = [s.strip() for s in sentences if s.strip()]

        if not sentences:
            return [text]

        chunks: List[str] = []
        current: List[str] = []
        current_tokens = 0.0
        overlap_sentences = max(1, overlap // 10)  # approx sentences for overlap

        for sent in sentences:
            tok = _approx_tokens(sent)
            if current_tokens + tok > max_tokens and current:
                chunks.append(" ".join(current))
                tail = current[-overlap_sentences:]
                current = list(tail)
                current_tokens = sum(_approx_tokens(s) for s in current)
            current.append(sent)
            current_tokens += tok

        if current:
            chunks.append(" ".join(current))

        return chunks if chunks else [text]
