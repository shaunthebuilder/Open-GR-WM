from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class Fact:
    subject: str
    predicate: str
    object: str
    value: str = ""
    unit: str = ""
    timeframe: str = ""
    source_ref: str = ""
    confidence: float = 0.0
    source_id: str = ""
    chunk_id: str = ""


@dataclass
class ChunkRecord:
    source_id: str
    page_or_doc_idx: int
    chunk_id: str
    section_title: str
    char_start: int
    char_end: int
    is_vision: bool
    created_at: str
    text: str


@dataclass
class RetrievalEvidence:
    chunk_ids: List[int] = field(default_factory=list)
    chunk_lines: List[str] = field(default_factory=list)
    graph_lines: List[str] = field(default_factory=list)
    confidence: float = 0.0
    entity_hits: List[str] = field(default_factory=list)
    metric_hits: List[str] = field(default_factory=list)
    allowed_citations: List[str] = field(default_factory=list)
    context_citation_ids: List[str] = field(default_factory=list)
    answerability_signals: Dict[str, float] = field(default_factory=dict)
    debug: Dict[str, float] = field(default_factory=dict)


@dataclass
class QueryPlan:
    intents: List[str]
    timeframe: str
    entities: List[str]
    metrics: List[str]
    comparison_anchors: List[str] = field(default_factory=list)
    requires_derivation: bool = False
    derivation_type: str = ""


JsonDict = Dict[str, object]
