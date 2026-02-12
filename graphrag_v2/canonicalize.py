from __future__ import annotations

import re
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Tuple

import networkx as nx

from .types import Fact
from .utils import dedup_keep_order, normalize_ws

try:
    from rapidfuzz import fuzz
except Exception:  # pragma: no cover
    fuzz = None


NULL_LIKE = {"", "none", "null", "n/a", "na", "unknown", "-", "--"}


def _clean_entity(value: str) -> str:
    text = normalize_ws(value)
    if not text:
        return ""
    if text.lower() in NULL_LIKE:
        return ""
    text = re.sub(r"\s+", " ", text).strip(" ,;:")
    return text


def _entity_key(value: str) -> str:
    text = _clean_entity(value).lower()
    text = re.sub(r"[^a-z0-9%$€£\-\.\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _merge_aliases(entities: List[str], threshold: int = 92) -> Dict[str, str]:
    canonical: Dict[str, str] = {}
    ordered = dedup_keep_order(entities)
    reps: List[str] = []

    for ent in ordered:
        key = _entity_key(ent)
        if not key:
            continue
        best = None
        if fuzz is not None:
            best_score = -1
            for rep in reps:
                score = int(fuzz.token_sort_ratio(key, _entity_key(rep)))
                if score > best_score:
                    best_score = score
                    best = rep
            if best is not None and best_score >= threshold:
                canonical[ent] = best
                continue
        # fallback exact-ish match
        for rep in reps:
            if key == _entity_key(rep):
                best = rep
                break
        if best is not None:
            canonical[ent] = best
            continue
        reps.append(ent)
        canonical[ent] = ent

    return canonical


def canonicalize_facts(facts: Iterable[Fact]) -> List[Fact]:
    items = [f for f in facts if isinstance(f, Fact)]
    ents: List[str] = []
    for f in items:
        ents.append(_clean_entity(f.subject))
        ents.append(_clean_entity(f.object))

    alias = _merge_aliases([e for e in ents if e])

    dedup: Dict[Tuple[str, str, str, str, str, str], Fact] = {}
    provenance: Dict[Tuple[str, str, str, str, str, str], List[str]] = defaultdict(list)
    conf_acc: Dict[Tuple[str, str, str, str, str, str], List[float]] = defaultdict(list)

    for f in items:
        s0 = _clean_entity(f.subject)
        p0 = normalize_ws(f.predicate)
        o0 = _clean_entity(f.object)
        if not (s0 and p0 and o0):
            continue
        if s0.lower() in NULL_LIKE or o0.lower() in NULL_LIKE:
            continue

        s = alias.get(s0, s0)
        o = alias.get(o0, o0)
        p = p0.strip(" ,;:")
        timeframe = normalize_ws(f.timeframe)
        value = normalize_ws(f.value)
        unit = normalize_ws(f.unit)

        key = (
            _entity_key(s),
            p.lower(),
            _entity_key(o),
            timeframe.lower(),
            value.lower(),
            unit.lower(),
        )
        if key not in dedup:
            dedup[key] = Fact(
                subject=s,
                predicate=p,
                object=o,
                value=value,
                unit=unit,
                timeframe=timeframe,
                source_ref=f.source_ref,
                confidence=f.confidence,
                source_id=f.source_id,
                chunk_id=f.chunk_id,
            )
        if f.source_ref:
            provenance[key].append(f.source_ref)
        conf_acc[key].append(float(f.confidence or 0.0))

    out: List[Fact] = []
    for key, fact in dedup.items():
        refs = dedup_keep_order(provenance.get(key, []))
        if refs:
            fact.source_ref = ", ".join(refs[:8])
        vals = conf_acc.get(key, [])
        if vals:
            fact.confidence = sum(vals) / len(vals)
        out.append(fact)

    return out


def facts_to_graph(facts: Iterable[Fact]) -> Tuple[nx.MultiDiGraph, List[Tuple[str, str, str]]]:
    graph = nx.MultiDiGraph()
    triples: List[Tuple[str, str, str]] = []
    for f in facts:
        s = _clean_entity(f.subject)
        p = normalize_ws(f.predicate)
        o = _clean_entity(f.object)
        if not (s and p and o):
            continue
        graph.add_node(s)
        graph.add_node(o)
        graph.add_edge(
            s,
            o,
            label=p,
            value=f.value,
            unit=f.unit,
            timeframe=f.timeframe,
            source_ref=f.source_ref,
            confidence=float(f.confidence or 0.0),
        )
        triples.append((s, p, o))
    return graph, triples


def build_entities_table(facts: Iterable[Fact]) -> List[Dict[str, object]]:
    counter = Counter()
    for f in facts:
        s = _clean_entity(f.subject)
        o = _clean_entity(f.object)
        if s:
            counter[s] += 1
        if o:
            counter[o] += 1
    out = []
    for ent, freq in counter.most_common():
        out.append({"entity": ent, "frequency": int(freq)})
    return out

