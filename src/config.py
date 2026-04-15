from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path


class Settings(BaseSettings):
    # API
    google_api_key: str

    # Paths
    data_dir: Path = Path("./data")
    index_dir: Path = Path("./index_cache")
    cache_dir: Path = Path("./query_cache")

    # Embedding — BGE-M3 for best Turkish multilingual quality
    embed_model: str = "BAAI/bge-m3"
    embed_batch_size: int = 32           # BGE-M3 is heavier, reduce batch size
    embed_workers: int = 4
    embed_device: str = "cpu"            # or "cuda" if available

    # Cross-Encoder reranker
    cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    cross_encoder_workers: int = 4

    # Retrieval
    # 15 aday yeterli: 31 belgeli korpüste cross-encoder 40→12s, 15→5.6s, 10→3s
    top_k_retrieval: int = 15
    top_k_after_crossencoder: int = 8
    top_k_final: int = 6
    bm25_weight: float = 0.35
    dense_weight: float = 0.65
    rrf_k: int = 60

    # Query expansion
    query_expansion_variants: int = 4
    query_expansion_enabled: bool = True

    # Hierarchical chunking
    child_chunk_size: int = 200          # tokens — retrieved
    parent_chunk_size: int = 600         # tokens — sent to LLM
    chunk_overlap: int = 40

    # Context window
    max_context_tokens: int = 6000
    context_overflow_strategy: str = "summarize"  # or "truncate"

    # Guardrail thresholds
    faithfulness_threshold: float = 0.70
    relevance_threshold: float = 0.50

    # Cache
    cache_ttl_seconds: int = 3600

    # Generation
    gemini_model: str = "models/gemini-2.5-flash"
    max_tokens: int = 2048
    log_level: str = "INFO"

    model_config = SettingsConfigDict(env_file=".env")


settings = Settings()
