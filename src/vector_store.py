import logging
import pickle
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import faiss
import numpy as np

from .models import Document

logger = logging.getLogger(__name__)


class VectorStore:
    """
    FAISS-based vector store using IndexFlatIP (inner product = cosine on L2-normalised vecs).

    Architectural decision — why IndexFlatIP over IVF/HNSW:
    Our corpus is small (<5k chunks). IndexFlatIP gives exact nearest-neighbour
    search with zero approximation error. IVF/HNSW are worthwhile only when
    the corpus exceeds ~50k vectors where the exact search latency becomes
    noticeable. We keep it simple and correct for now.
    """

    def __init__(self, index_dir: Path):
        self.index_dir = Path(index_dir)
        self.index: Optional[faiss.Index] = None
        self.documents: List[Document] = []

    def build(self, embeddings: np.ndarray, documents: List[Document]) -> None:
        """Build FAISS index from embeddings and associate documents."""
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings.astype(np.float32))
        self.documents = documents
        self._save()
        logger.info(f"FAISS index built: {self.index.ntotal} vectors, dim={dim}")

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int,
        filters: Optional[Dict] = None,
    ) -> List[Tuple[Document, float]]:
        """Search index, optionally filtering by metadata key-value pairs."""
        if self.index is None:
            raise RuntimeError("VectorStore not initialised. Call build() or load() first.")

        search_k = top_k * 5 if filters else top_k
        search_k = min(search_k, len(self.documents))

        scores, indices = self.index.search(
            query_embedding.reshape(1, -1).astype(np.float32),
            search_k,
        )

        results: List[Tuple[Document, float]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            doc = self.documents[idx]
            if filters and not all(
                doc.metadata.get(k) == v for k, v in filters.items()
            ):
                continue
            results.append((doc, float(score)))
            if len(results) >= top_k:
                break

        return results

    def _save(self) -> None:
        self.index_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(self.index_dir / "faiss.index"))
        with open(self.index_dir / "documents.pkl", "wb") as f:
            pickle.dump(self.documents, f)
        logger.info(f"VectorStore saved to {self.index_dir}")

    def load(self) -> None:
        index_path = self.index_dir / "faiss.index"
        docs_path = self.index_dir / "documents.pkl"
        if not index_path.exists() or not docs_path.exists():
            raise FileNotFoundError(
                f"Index files not found in {self.index_dir}. Run build_index() first."
            )
        self.index = faiss.read_index(str(index_path))
        with open(docs_path, "rb") as f:
            self.documents = pickle.load(f)
        logger.info(
            f"VectorStore loaded: {self.index.ntotal} vectors, "
            f"{len(self.documents)} documents"
        )
