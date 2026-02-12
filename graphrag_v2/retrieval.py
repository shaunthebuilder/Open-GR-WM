from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np

from .indexing import dense_scores, mmr_select, sparse_scores
from .intent import parse_intent_spec
from .types import ChunkRecord, QueryPlan, RetrievalEvidence
from .utils import dedup_keep_order, normalize_ws

try:
    from rapidfuzz import fuzz
except Exception:  # pragma: no cover
    fuzz = None


STOP = {"what", "which", "show", "give", "about", "from", "with", "this", "that", "the", "and", "for", "how", "much"}
TEMPORAL_PATTERNS = [
    re.compile(r"\b\d+\s+years?\s+(?:before|after)\b", flags=re.IGNORECASE),
    re.compile(r"\btakes place\b.{0,80}\b\d+\s+years?\b", flags=re.IGNORECASE),
    re.compile(r"\bset\b.{0,80}\b\d+\s+years?\b", flags=re.IGNORECASE),
]


def _tokens(text: str) -> List[str]:
    return [t for t in re.findall(r"[a-z0-9]{2,}", str(text or "").lower()) if t not in STOP]


def _extract_between_anchors(question: str) -> List[str]:
    q = str(question or "").strip()
    if not q:
        return []
    anchors: List[str] = []
    m = re.search(r"\bbetween\s+(.+?)\s+and\s+(.+?)(?:\?|$)", q, flags=re.IGNORECASE)
    if m:
        anchors = [normalize_ws(m.group(1)), normalize_ws(m.group(2))]
    elif re.search(r"\b(?:difference|years?)\s+between\b", q, flags=re.IGNORECASE):
        m2 = re.search(r"\b(?:difference|years?)\s+between\s+(.+?)\s+and\s+(.+?)(?:\?|$)", q, flags=re.IGNORECASE)
        if m2:
            anchors = [normalize_ws(m2.group(1)), normalize_ws(m2.group(2))]

    cleaned: List[str] = []
    for anchor in anchors:
        value = normalize_ws(anchor)
        value = re.sub(r"^(?:the\s+)?(?:adventures?\s+of\s+)", "", value, flags=re.IGNORECASE)
        value = re.sub(r"^(?:the\s+)", "", value, flags=re.IGNORECASE)
        value = value.strip(" ,.;:")
        if value:
            cleaned.append(value)
    return dedup_keep_order(cleaned)[:2]


def _rec_field(rec: object, name: str, default: str = "") -> str:
    if isinstance(rec, dict):
        return str(rec.get(name, default))
    if isinstance(rec, ChunkRecord):
        return str(getattr(rec, name, default))
    return str(getattr(rec, name, default))


def _anchor_chunk_score(anchor: str, text: str) -> float:
    a = normalize_ws(anchor).lower()
    t = str(text or "").lower()
    if not a or not t:
        return 0.0
    score = 0.0
    if a in t:
        score += 1.5
    a_toks = [tok for tok in re.findall(r"[a-z0-9]{2,}", a) if tok not in STOP]
    if a_toks:
        overlap = sum(1 for tok in a_toks if tok in t)
        score += min(1.0, overlap / float(len(a_toks)))
    if fuzz is not None:
        try:
            score += min(1.0, float(fuzz.partial_ratio(a, t)) / 100.0) * 0.5
        except Exception:
            pass
    return score


def _top_anchor_candidates(anchor: str, chunk_texts: List[str], top_n: int = 6) -> List[int]:
    scored: List[Tuple[float, int]] = []
    for idx, txt in enumerate(chunk_texts):
        s = _anchor_chunk_score(anchor, txt)
        if s > 0.30:
            scored.append((s, idx))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [idx for _score, idx in scored[:top_n]]


def _has_temporal_signal(text: str) -> bool:
    src = str(text or "")
    return any(p.search(src) for p in TEMPORAL_PATTERNS)


def _temporal_signal_score(text: str) -> float:
    src = str(text or "")
    score = 0.0
    for pattern in TEMPORAL_PATTERNS:
        if pattern.search(src):
            score += 0.34
    return min(1.0, score)


def _extract_chunk_ids_from_lines(lines: List[str]) -> List[str]:
    ids: List[str] = []
    for line in lines:
        for match in re.findall(r"\[\s*chunk\s*([^\]]+?)\s*\]", str(line), flags=re.IGNORECASE):
            val = normalize_ws(str(match))
            if val:
                ids.append(val)
    return dedup_keep_order(ids)


def build_query_plan(
    question: str,
    graph_nodes: List[str],
    intent_hint: Optional[Dict[str, object]] = None,
) -> QueryPlan:
    q = str(question or "")
    if isinstance(intent_hint, dict):
        hint = dict(intent_hint)
    else:
        hint = parse_intent_spec(str(question or "")).to_dict()
    normalized_q = str(hint.get("normalized_question", q) or q)
    normalized_ql = normalized_q.lower()
    intents: List[str] = []
    if any(k in normalized_ql for k in ["trend", "over time", "vs", "versus", "compared", "change"]):
        intents.append("trend_or_compare")
    if any(k in normalized_ql for k in ["guidance", "forecast", "outlook"]):
        intents.append("guidance_lookup")
    if any(k in normalized_ql for k in ["why", "driver", "reason", "cause"]):
        intents.append("driver_analysis")
    if not intents:
        intents.append("direct_lookup")
    if isinstance(hint.get("intent_labels"), list):
        hint_intents = [normalize_ws(str(x)) for x in hint.get("intent_labels", []) if str(x).strip()]
        if hint_intents:
            intents = dedup_keep_order(hint_intents)[:3]

    timeframe = "unspecified"
    ym = re.findall(r"(?:19|20)\d{2}", normalized_q)
    qm = re.findall(r"q[1-4]", normalized_ql)
    fy = re.findall(r"fy\s*\d{2}", normalized_ql)
    if ym:
        timeframe = ",".join(dedup_keep_order(ym))
    elif fy:
        timeframe = ",".join(dedup_keep_order([x.replace(" ", "") for x in fy]))
    elif qm:
        timeframe = ",".join(dedup_keep_order([x.upper() for x in qm]))

    query_toks = _tokens(normalized_q)
    entity_hits: List[str] = []
    for n in graph_nodes:
        nl = str(n).lower()
        score = 0
        if nl in normalized_ql or normalized_ql in nl:
            score += 5
        overlap = sum(1 for t in query_toks if t in nl)
        score += overlap
        if score > 1:
            entity_hits.append((score, str(n)))
    entity_hits = [x[1] for x in sorted(entity_hits, key=lambda z: z[0], reverse=True)[:8]]

    metric_terms = [
        "ebit",
        "ebitda",
        "revenue",
        "margin",
        "cash",
        "fcf",
        "capex",
        "volume",
        "guidance",
        "profit",
    ]
    metrics = [m for m in metric_terms if m in normalized_ql]

    comparison_anchors = _extract_between_anchors(normalized_q)
    if isinstance(hint.get("comparison_anchors"), list):
        hint_anchors = [normalize_ws(str(x)) for x in hint.get("comparison_anchors", []) if str(x).strip()]
        if hint_anchors:
            comparison_anchors = dedup_keep_order(hint_anchors)[:2]

    temporal_request = bool(
        re.search(r"\b(?:year|years|before|after|difference|time difference|timeline|apart)\b", normalized_ql)
    )
    requires_derivation = bool(temporal_request and len(comparison_anchors) >= 2)
    derivation_type = "year_difference" if requires_derivation else ""
    if "requires_derivation" in hint:
        requires_derivation = bool(hint.get("requires_derivation", requires_derivation))
    if "derivation_type" in hint and str(hint.get("derivation_type", "")).strip():
        derivation_type = str(hint.get("derivation_type", "")).strip()
    if requires_derivation and not derivation_type:
        derivation_type = "year_difference"

    for anchor in comparison_anchors:
        if anchor.lower() not in {e.lower() for e in entity_hits}:
            entity_hits.append(anchor)
    entity_hits = entity_hits[:8]

    return QueryPlan(
        intents=intents,
        timeframe=timeframe,
        entities=entity_hits,
        metrics=metrics,
        comparison_anchors=comparison_anchors,
        requires_derivation=requires_derivation,
        derivation_type=derivation_type,
    )


def _expand_graph_context(graph: nx.MultiDiGraph, seeds: List[str], max_edges: int = 120) -> List[str]:
    if not seeds:
        return []
    lines: List[str] = []
    seen = set()
    for seed in seeds[:12]:
        for u, v, data in graph.edges(seed, data=True):
            p = str(data.get("label", "")).strip()
            key = (u, p, v)
            if key in seen:
                continue
            seen.add(key)
            src = str(data.get("source_ref", "")).strip()
            line = f"{u} -[{p}]-> {v} {src}".strip()
            lines.append(line)
            if len(lines) >= max_edges:
                return lines
        for u, v, data in graph.in_edges(seed, data=True):
            p = str(data.get("label", "")).strip()
            key = (u, p, v)
            if key in seen:
                continue
            seen.add(key)
            src = str(data.get("source_ref", "")).strip()
            line = f"{u} -[{p}]-> {v} {src}".strip()
            lines.append(line)
            if len(lines) >= max_edges:
                return lines
    return lines


def _scores_to_rank(scores: np.ndarray, top_n: int) -> List[int]:
    if scores is None or getattr(scores, "size", 0) == 0:
        return []
    idx = np.argsort(scores)[-top_n:][::-1]
    return [int(i) for i in idx]


def _selected_has_anchor(selected: List[int], anchor: str, chunk_texts: List[str]) -> bool:
    return any(_anchor_chunk_score(anchor, chunk_texts[idx]) >= 0.65 for idx in selected)


def retrieve_evidence_bundle(
    question: str,
    graph: nx.MultiDiGraph,
    embedder,
    chunk_texts: List[str],
    embeddings: np.ndarray,
    bm25_model,
    bm25_tokens: List[List[str]],
    chunk_records: List[object],
    metric_map: Optional[Dict[str, List[int]]] = None,
    entity_map: Optional[Dict[str, List[int]]] = None,
    top_k: int = 10,
    intent_hint: Optional[Dict[str, object]] = None,
) -> RetrievalEvidence:
    if not question or not chunk_texts:
        return RetrievalEvidence()

    nodes = [str(n) for n in graph.nodes]
    has_external_hint = isinstance(intent_hint, dict)
    if has_external_hint:
        hint = dict(intent_hint)
    else:
        hint = parse_intent_spec(str(question or "")).to_dict()
    qp = build_query_plan(question, nodes, intent_hint=hint if hint else None)
    query_text = str(hint.get("canonical_retrieval_query", "")).strip() or str(question)
    intent_source = "hint" if has_external_hint else "inferred"

    d_scores = dense_scores(query_text, embeddings, embedder)
    s_scores = sparse_scores(query_text, bm25_model, bm25_tokens)

    d_ids = _scores_to_rank(d_scores, top_n=max(16, top_k * 2))
    s_ids = _scores_to_rank(s_scores, top_n=max(16, top_k * 2))
    prior_ids: List[int] = []
    metric_map = metric_map if isinstance(metric_map, dict) else {}
    entity_map = entity_map if isinstance(entity_map, dict) else {}
    for metric in qp.metrics:
        prior_ids.extend(metric_map.get(metric.lower(), []))
    for ent in qp.entities:
        prior_ids.extend(entity_map.get(ent.lower(), []))

    anchor_candidates: Dict[str, List[int]] = {}
    for anchor in qp.comparison_anchors[:2]:
        ids = _top_anchor_candidates(anchor, chunk_texts, top_n=6)
        anchor_candidates[anchor] = ids
        prior_ids.extend(ids)

    candidate_ids = dedup_keep_order([str(x) for x in d_ids + s_ids + prior_ids])
    candidate_ints = [int(x) for x in candidate_ids]

    blend: Dict[int, float] = {}
    if d_scores is not None and getattr(d_scores, "size", 0) > 0:
        dmax = float(np.max(d_scores)) if len(d_scores) else 1.0
        dmin = float(np.min(d_scores)) if len(d_scores) else 0.0
        span = (dmax - dmin) + 1e-8
        for i in candidate_ints:
            blend[i] = blend.get(i, 0.0) + 0.56 * float((d_scores[i] - dmin) / span)
    if s_scores is not None and getattr(s_scores, "size", 0) > 0:
        smax = float(np.max(s_scores)) if len(s_scores) else 1.0
        smin = float(np.min(s_scores)) if len(s_scores) else 0.0
        span = (smax - smin) + 1e-8
        for i in candidate_ints:
            blend[i] = blend.get(i, 0.0) + 0.34 * float((s_scores[i] - smin) / span)

    ent_tokens = [e.lower() for e in qp.entities]
    for i in candidate_ints:
        txt = chunk_texts[i].lower()
        bonus = 0.0
        for ent in ent_tokens:
            if ent and ent in txt:
                bonus += 0.06
        for met in qp.metrics:
            if met in txt:
                bonus += 0.05
        if i in prior_ids:
            bonus += 0.08
        for anchor, ids in anchor_candidates.items():
            if i in ids:
                bonus += 0.10
            elif _anchor_chunk_score(anchor, txt) >= 0.75:
                bonus += 0.05
        if _has_temporal_signal(txt):
            bonus += min(0.18, _temporal_signal_score(txt) * 0.18)
        blend[i] = blend.get(i, 0.0) + min(0.35, bonus)

    selected = mmr_select(candidate_ints, blend, embeddings, top_k=top_k, lambda_mult=0.72)
    if not selected:
        selected = candidate_ints[:top_k]

    must_keep = set()
    for anchor, ids in anchor_candidates.items():
        if not ids:
            continue
        if _selected_has_anchor(selected, anchor, chunk_texts):
            continue
        best = ids[0]
        if best not in selected:
            selected.append(best)
            must_keep.add(best)

    selected = [int(x) for x in dedup_keep_order([str(x) for x in selected])]
    while len(selected) > top_k:
        removable = [x for x in selected if x not in must_keep]
        if not removable:
            removable = list(selected)
        worst = min(removable, key=lambda cid: blend.get(cid, 0.0))
        selected = [x for x in selected if x != worst]

    chunk_lines: List[str] = []
    citations: List[str] = []
    for i in selected:
        rec = chunk_records[i] if i < len(chunk_records) else {}
        chunk_id = _rec_field(rec, "chunk_id", f"{i + 1}")
        source_id = _rec_field(rec, "source_id", "source")
        text = re.sub(r"\s+", " ", chunk_texts[i]).strip()[:760]
        line = f"[Chunk {chunk_id}] ({source_id}) {text}"
        chunk_lines.append(line)
        citations.append(f"[Chunk {chunk_id}]")

    graph_lines = _expand_graph_context(graph, qp.entities, max_edges=110)
    if graph_lines:
        citations.insert(0, "[Graph]")

    anchor_coverage_score = 1.0
    temporal_anchor_score = min(
        1.0,
        float(np.mean([_temporal_signal_score(chunk_texts[i]) for i in selected])) if selected else 0.0,
    )
    if qp.comparison_anchors:
        covered = sum(1 for anchor in qp.comparison_anchors if _selected_has_anchor(selected, anchor, chunk_texts))
        anchor_coverage_score = float(covered / max(1, len(qp.comparison_anchors)))
        temporal_covered = 0
        for anchor in qp.comparison_anchors:
            anchor_idxs = [idx for idx in selected if _anchor_chunk_score(anchor, chunk_texts[idx]) >= 0.65]
            if any(_has_temporal_signal(chunk_texts[idx]) for idx in anchor_idxs):
                temporal_covered += 1
        temporal_anchor_score = float(temporal_covered / max(1, len(qp.comparison_anchors)))

    derivation_ready_score = min(anchor_coverage_score, temporal_anchor_score)
    base_relevance = float(np.mean([min(1.0, blend.get(i, 0.0)) for i in selected])) if selected else 0.0
    citation_diversity = min(1.0, len(dedup_keep_order(citations)) / float(max(2, top_k)))
    confidence = (
        0.45 * base_relevance
        + 0.20 * citation_diversity
        + 0.20 * anchor_coverage_score
        + 0.15 * temporal_anchor_score
    )

    confidence_capped = 0.0
    if qp.requires_derivation and anchor_coverage_score < 1.0:
        confidence = min(confidence, 0.35)
        confidence_capped = 1.0
    if qp.requires_derivation and temporal_anchor_score < 0.30:
        confidence = min(confidence, 0.45)
        confidence_capped = 1.0
    confidence = max(0.0, min(1.0, float(confidence)))

    context_citation_ids = _extract_chunk_ids_from_lines(chunk_lines + graph_lines)

    return RetrievalEvidence(
        chunk_ids=selected,
        chunk_lines=chunk_lines,
        graph_lines=graph_lines,
        confidence=confidence,
        entity_hits=qp.entities,
        metric_hits=qp.metrics,
        allowed_citations=dedup_keep_order(citations)[:20],
        context_citation_ids=context_citation_ids,
        answerability_signals={
            "anchor_coverage_score": anchor_coverage_score,
            "temporal_anchor_score": temporal_anchor_score,
            "derivation_ready_score": derivation_ready_score,
        },
        debug={
            "dense_candidates": float(len(d_ids)),
            "sparse_candidates": float(len(s_ids)),
            "selected": float(len(selected)),
            "confidence": confidence,
            "anchor_coverage_score": anchor_coverage_score,
            "temporal_anchor_score": temporal_anchor_score,
            "confidence_capped": confidence_capped,
            "intent_source": intent_source,
            "canonical_query_used": query_text[:240],
            "anchor_count": float(len(qp.comparison_anchors)),
        },
    )
