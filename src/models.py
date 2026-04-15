from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from datetime import date


@dataclass
class Document:
    content: str                           # human-readable text
    metadata: Dict[str, Any]               # source, type, chunk_id, priority_tier, etc.
    parent_id: Optional[str] = None        # for hierarchical chunks
    parent_content: Optional[str] = None


@dataclass
class ScoredDocument:
    document: Document
    retrieval_score: float                 # RRF fusion score
    cross_encoder_score: float = 0.0
    priority_score: float = 0.0
    final_score: float = 0.0
    is_superseded: bool = False
    superseded_by: Optional[str] = None
    conflict_note: Optional[str] = None


@dataclass
class QueryExpansion:
    original: str
    variants: List[str]                    # LLM-generated alternatives
    all_queries: List[str]                 # original + variants


@dataclass
class EvaluationResult:
    faithfulness_score: float              # 0.0–1.0: are all claims grounded in context?
    relevance_score: float                 # 0.0–1.0: are retrieved chunks relevant?
    passes_guardrail: bool
    unfaithful_claims: List[str]           # claims not supported by any chunk
    explanation: str


@dataclass
class GeneratorResponse:
    answer: str
    sources_used: List[str]
    conflicts_resolved: List[str]
    evaluation: EvaluationResult
    latency_ms: float
    chunk_count: int
    queries_used: List[str]               # original + expanded
    was_cache_hit: bool = False
