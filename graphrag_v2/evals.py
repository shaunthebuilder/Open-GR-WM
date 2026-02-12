from __future__ import annotations

import re
from typing import Dict, List

import networkx as nx

from .indexing import RetrievalIndex
from .types import Fact, RetrievalEvidence


NULL_LIKE = {"", "none", "null", "n/a", "na", "unknown", "-", "--"}


def _has_numeric(text: str) -> bool:
    return bool(re.search(r"-?\d[\d,]*(?:\.\d+)?", str(text or "")))


def _chunk_ids_from_citations(citations: List[str]) -> List[str]:
    out: List[str] = []
    for cit in citations:
        for match in re.findall(r"\[\s*chunk\s*([^\]]+?)\s*\]", str(cit), flags=re.IGNORECASE):
            value = str(match).strip()
            if value and value.lower() not in {x.lower() for x in out}:
                out.append(value)
    return out


def run_compositional_retrieval_eval(question: str, evidence: RetrievalEvidence) -> Dict[str, float]:
    _ = str(question or "").strip()
    signals = evidence.answerability_signals if isinstance(evidence.answerability_signals, dict) else {}
    anchor_coverage_rate = float(signals.get("anchor_coverage_score", 0.0) or 0.0)
    temporal_anchor_score = float(signals.get("temporal_anchor_score", 0.0) or 0.0)
    derivation_success_rate = 1.0 if (anchor_coverage_rate >= 1.0 and temporal_anchor_score >= 0.5) else 0.0

    allowed_ids = _chunk_ids_from_citations(list(evidence.allowed_citations or []))
    context_ids = [str(x).strip() for x in list(evidence.context_citation_ids or []) if str(x).strip()]
    context_set = {x.lower() for x in context_ids}
    valid_hits = [cid for cid in allowed_ids if cid.lower() in context_set]
    citation_context_validity_rate = float(len(valid_hits) / max(1, len(allowed_ids))) if allowed_ids else 1.0

    return {
        "anchor_coverage_rate": round(anchor_coverage_rate, 4),
        "derivation_success_rate": round(derivation_success_rate, 4),
        "citation_context_validity_rate": round(citation_context_validity_rate, 4),
    }


def run_build_evals(
    facts: List[Fact],
    graph: nx.MultiDiGraph,
    retrieval_index: RetrievalIndex,
    total_pages: int,
    timings: Dict[str, float],
    compositional_eval: Dict[str, float] = None,
) -> Dict[str, object]:
    total_pages = max(1, int(total_pages or 1))
    nodes = list(graph.nodes())
    edges = int(graph.number_of_edges())
    facts_n = int(len(facts))

    null_nodes = [n for n in nodes if str(n).strip().lower() in NULL_LIKE]
    none_node_rate = float(len(null_nodes) / max(1, len(nodes)))

    facts_with_citation = [f for f in facts if str(f.source_ref or "").strip()]
    citation_validity_rate = float(len(facts_with_citation) / max(1, facts_n))

    numeric_supported = [f for f in facts if _has_numeric(f.value) or _has_numeric(f.object)]
    numeric_support_rate = float(len(numeric_supported) / max(1, facts_n))

    edges_per_page = float(edges / total_pages)
    retrieval_chunks = int(len(retrieval_index.chunk_texts))
    retrieval_coverage = float(
        len([k for k, v in retrieval_index.metric_map.items() if isinstance(v, list) and v]) / max(1, len(retrieval_index.metric_map))
    ) if retrieval_index.metric_map else 0.0
    entity_coverage = float(
        len([k for k, v in retrieval_index.entity_map.items() if isinstance(v, list) and v]) / max(1, len(retrieval_index.entity_map))
    ) if retrieval_index.entity_map else 0.0
    retrieval_recall_proxy = min(1.0, 0.55 * retrieval_coverage + 0.30 * entity_coverage + 0.15 * min(1.0, edges_per_page / 8.0))

    build_total_sec = float((timings or {}).get("build_total_sec", 0.0) or 0.0)
    extract_sec = float((timings or {}).get("extract_sec", 0.0) or 0.0)
    extracted_per_sec = float(facts_n / extract_sec) if extract_sec > 0 else 0.0

    checks = {
        "edges_per_page>=6": edges_per_page >= 6.0,
        "none_node_rate==0": none_node_rate == 0.0,
        "citation_validity>=0.95": citation_validity_rate >= 0.95,
        "retrieval_recall@8_proxy>=0.85": retrieval_recall_proxy >= 0.85,
    }

    comp = compositional_eval if isinstance(compositional_eval, dict) else {}
    if comp:
        checks["anchor_coverage_rate>=0.9"] = float(comp.get("anchor_coverage_rate", 0.0) or 0.0) >= 0.9
        checks["citation_context_validity_rate>=0.95"] = (
            float(comp.get("citation_context_validity_rate", 0.0) or 0.0) >= 0.95
        )

    return {
        "summary": {
            "nodes": len(nodes),
            "edges": edges,
            "facts": facts_n,
            "retrieval_chunks": retrieval_chunks,
            "pages": total_pages,
        },
        "quality_metrics": {
            "edges_per_page": round(edges_per_page, 4),
            "none_node_rate": round(none_node_rate, 4),
            "citation_validity_rate": round(citation_validity_rate, 4),
            "numeric_support_rate": round(numeric_support_rate, 4),
            "retrieval_recall_at_8_proxy": round(retrieval_recall_proxy, 4),
            "anchor_coverage_rate": round(float(comp.get("anchor_coverage_rate", 0.0) or 0.0), 4),
            "derivation_success_rate": round(float(comp.get("derivation_success_rate", 0.0) or 0.0), 4),
            "citation_context_validity_rate": round(
                float(comp.get("citation_context_validity_rate", 0.0) or 0.0), 4
            ),
        },
        "performance_metrics": {
            "build_total_sec": round(build_total_sec, 3),
            "extract_sec": round(extract_sec, 3),
            "facts_per_sec": round(extracted_per_sec, 4),
        },
        "checks": checks,
        "pass": all(checks.values()),
    }
