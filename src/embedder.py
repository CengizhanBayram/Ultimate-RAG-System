import logging
import pickle
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional

import numpy as np
from tqdm import tqdm

from .models import Document
from .config import Settings

logger = logging.getLogger(__name__)


class EmbeddingEngine:
    """
    BGE-M3 batch embedder with disk cache.

    Architectural decision — why BGE-M3:
    BGE-M3 (BAAI/bge-m3) is currently the best multilingual embedding model
    for Turkish text because:
    1. It was trained on 100+ languages including Turkish.
    2. It natively supports dense, sparse, and multi-vector retrieval.
    3. It uses asymmetric retrieval (query prefix vs. passage prefix) which
       dramatically improves recall for question-to-document matching.
    4. It outperforms OpenAI text-embedding-3-small on MIRACL Turkish benchmark.

    We use dense-only mode here because BM25 already covers the sparse axis
    (preventing redundancy), keeping embeddings lean at ~1024 dims.
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self.batch_size = settings.embed_batch_size
        self.workers = settings.embed_workers
        self._cache_path = Path(settings.index_dir) / "embeddings.npy"
        self._model = None  # lazy-loaded

    def _get_model(self):
        if self._model is None:
            logger.info(f"Loading embedding model: {self.settings.embed_model}")
            try:
                from FlagEmbedding import BGEM3FlagModel
                self._model = BGEM3FlagModel(
                    self.settings.embed_model,
                    use_fp16=True,
                    device=self.settings.embed_device,
                )
                logger.info("BGE-M3 model loaded successfully.")
            except ImportError:
                logger.warning(
                    "FlagEmbedding not found, falling back to sentence-transformers."
                )
                from sentence_transformers import SentenceTransformer
                self._model = _SentenceTransformerWrapper(
                    self.settings.embed_model, self.settings.embed_device
                )
        return self._model

    def embed_documents(self, documents: List[Document]) -> np.ndarray:
        """Parallel batch embedding with progress tracking."""
        model = self._get_model()
        texts = [doc.content for doc in documents]
        batches = [
            texts[i: i + self.batch_size]
            for i in range(0, len(texts), self.batch_size)
        ]

        all_embeddings = []

        def _encode_batch(batch):
            return model.encode(
                batch,
                batch_size=self.batch_size,
                max_length=512,
                return_dense=True,
                return_sparse=False,
            )

        with ThreadPoolExecutor(max_workers=self.workers) as executor:
            futures = [executor.submit(_encode_batch, batch) for batch in batches]
            for future in tqdm(
                as_completed(futures), total=len(futures), desc="Embedding batches"
            ):
                result = future.result()
                if isinstance(result, dict):
                    all_embeddings.append(result["dense_vecs"])
                else:
                    all_embeddings.append(result)

        embeddings = np.vstack(all_embeddings).astype(np.float32)
        Path(self.settings.index_dir).mkdir(parents=True, exist_ok=True)
        np.save(self._cache_path, embeddings)
        logger.info(f"Embeddings saved: shape={embeddings.shape}")
        return embeddings

    def embed_query(self, query: str) -> np.ndarray:
        """Encode a single query using BGE-M3's query prefix for asymmetric retrieval."""
        model = self._get_model()
        result = model.encode(
            [f"query: {query}"],
            return_dense=True,
        )
        if isinstance(result, dict):
            return result["dense_vecs"][0].astype(np.float32)
        return np.array(result[0], dtype=np.float32)

    def load_cached_embeddings(self) -> Optional[np.ndarray]:
        """Load embeddings from disk cache if available."""
        if self._cache_path.exists():
            logger.info("Loading cached embeddings from disk.")
            return np.load(self._cache_path)
        return None


class _SentenceTransformerWrapper:
    """
    Fallback wrapper around SentenceTransformer to match BGE-M3 interface.
    Used when FlagEmbedding is not installed.
    """

    def __init__(self, model_name: str, device: str):
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer(model_name, device=device)

    def encode(self, texts, batch_size=32, max_length=512,
               return_dense=True, return_sparse=False, normalize_embeddings=True):
        vecs = self._model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=normalize_embeddings,
            show_progress_bar=False,
        )
        return {"dense_vecs": vecs}
