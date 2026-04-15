import hashlib
import logging
from pathlib import Path
from typing import Optional, TYPE_CHECKING

import diskcache

if TYPE_CHECKING:
    from .models import GeneratorResponse

logger = logging.getLogger(__name__)


class QueryCache:
    """
    Persistent disk-based query cache using diskcache.

    Architectural decision:
    Disk cache (not in-memory) survives process restarts, which is important for
    a Streamlit app that re-imports the module on every interaction. diskcache
    uses SQLite under the hood, supports TTL natively, and is thread-safe.

    Cache key: SHA-256 of lowercased, stripped query — identical questions
    differing only in whitespace or capitalisation share one cache entry.
    """

    def __init__(self, cache_dir: Path, ttl: int):
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache = diskcache.Cache(str(cache_dir))
        self.ttl = ttl

    def _key(self, query: str) -> str:
        return hashlib.sha256(query.strip().lower().encode()).hexdigest()

    def get(self, query: str) -> Optional["GeneratorResponse"]:
        result = self.cache.get(self._key(query))
        if result is not None:
            logger.info("Cache HIT")
        return result

    def set(self, query: str, response: "GeneratorResponse") -> None:
        self.cache.set(self._key(query), response, expire=self.ttl)
        logger.debug(f"Cached response (ttl={self.ttl}s)")

    def invalidate_all(self) -> None:
        self.cache.clear()
        logger.info("Query cache cleared.")

    def size(self) -> int:
        return len(self.cache)
