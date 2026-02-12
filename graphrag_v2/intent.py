from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Dict, List


@dataclass
class QueryIntentSpec:
    normalized_question: str
    intent_labels: List[str]
    comparison_anchors: List[str]
    requires_derivation: bool
    derivation_type: str
    timeframe: str
    should_visualize: bool
    canonical_retrieval_query: str

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


_STOP_PREFIX_RE = re.compile(
    r"^(?:the\s+)?(?:adventures?\s+of\s+|timeline\s+of\s+|events?\s+of\s+|story\s+of\s+)",
    flags=re.IGNORECASE,
)
_TEMPORAL_DERIVATION_RE = re.compile(
    r"\b(?:time\s+difference|difference|timeline|years?\s+apart|far\s+apart|before|after|apart)\b",
    flags=re.IGNORECASE,
)
_BETWEEN_ANCHORS_RE = re.compile(
    r"\bbetween\s+(.+?)\s*,?\s+and\s+(.+?)(?:\?|$)",
    flags=re.IGNORECASE,
)


def _dedup_ci(values: List[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for raw in values:
        val = str(raw or "").strip()
        if not val:
            continue
        key = val.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(val)
    return out


def _normalize_anchor(value: str) -> str:
    text = str(value or "").strip(" ,.;:")
    text = _STOP_PREFIX_RE.sub("", text)
    text = re.sub(r"^(?:the|a)\s+", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip(" ,.;:")
    return text


def _extract_anchors(question: str) -> List[str]:
    src = str(question or "").strip()
    if not src:
        return []
    m = _BETWEEN_ANCHORS_RE.search(src)
    if not m:
        return []
    a = _normalize_anchor(m.group(1))
    b = _normalize_anchor(m.group(2))
    return _dedup_ci([a, b])[:2]


def _infer_timeframe(question: str) -> str:
    years = sorted(set(re.findall(r"\b(?:19|20)\d{2}\b", str(question or ""))))
    if years:
        if len(years) > 1:
            return f"{years[0]}-{years[-1]}"
        return years[0]
    quarters = [q.upper() for q in re.findall(r"\bq[1-4]\b", str(question or ""), flags=re.IGNORECASE)]
    if quarters:
        return quarters[0]
    return "unspecified"


def parse_intent_spec(question: str) -> QueryIntentSpec:
    q = str(question or "").strip()
    ql = q.lower()
    normalized_question = re.sub(r"\s+", " ", q).strip()
    comparison_anchors = _extract_anchors(normalized_question)

    has_temporal_words = bool(_TEMPORAL_DERIVATION_RE.search(ql)) or ("years" in ql or "year" in ql)
    has_between = "between" in ql and " and " in ql
    requires_derivation = bool(has_temporal_words and has_between and len(comparison_anchors) >= 2)
    derivation_type = "year_difference" if requires_derivation else ""

    intent_labels: List[str] = []
    if any(k in ql for k in ["summary", "summarize", "overview", "about this", "high level"]):
        intent_labels.append("summarize_document")
    if any(k in ql for k in ["ebit", "ebitda", "revenue", "profit", "margin", "volume", "container", "how much"]):
        intent_labels.append("extract_key_metrics")
    if requires_derivation or any(k in ql for k in ["compare", "vs", "versus", "difference", "higher", "lower", "increase", "decrease"]):
        intent_labels.append("compare_values")
    if has_temporal_words or any(k in ql for k in ["trend", "over time", "from", "to", "quarter", "q1", "q2", "q3", "q4"]):
        intent_labels.append("trend_check")
    if any(k in ql for k in ["why", "driver", "reason", "cause"]):
        intent_labels.append("driver_analysis")
    if not intent_labels:
        intent_labels.append("direct_lookup")
    intent_labels = _dedup_ci(intent_labels)[:3]

    timeframe = _infer_timeframe(normalized_question)
    should_visualize = ("trend_check" in intent_labels) or ("compare_values" in intent_labels)

    if requires_derivation and len(comparison_anchors) >= 2:
        canonical_retrieval_query = (
            f"timeline year difference between {comparison_anchors[0]} and {comparison_anchors[1]} before after years"
        )
    else:
        canonical_retrieval_query = normalized_question

    return QueryIntentSpec(
        normalized_question=normalized_question,
        intent_labels=intent_labels,
        comparison_anchors=comparison_anchors,
        requires_derivation=requires_derivation,
        derivation_type=derivation_type,
        timeframe=timeframe,
        should_visualize=should_visualize,
        canonical_retrieval_query=canonical_retrieval_query,
    )
