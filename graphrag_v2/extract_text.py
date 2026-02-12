from __future__ import annotations

import hashlib
import re
from typing import Callable, Dict, List

from .types import ChunkRecord, Fact
from .utils import dedup_keep_order, normalize_ws, parse_json_any


def _fact_from_row(row: Dict[str, object], source_ref: str, source_id: str, chunk_id: str, default_conf: float = 0.74) -> Fact:
    subject = normalize_ws(str(row.get("subject", "")))
    predicate = normalize_ws(str(row.get("predicate", "")))
    obj = normalize_ws(str(row.get("object", "")))
    if not (subject and predicate and obj):
        return Fact("", "", "")
    confidence = row.get("confidence", default_conf)
    try:
        confidence_f = float(confidence)
    except Exception:
        confidence_f = default_conf
    confidence_f = max(0.0, min(1.0, confidence_f))
    return Fact(
        subject=subject,
        predicate=predicate,
        object=obj,
        value=normalize_ws(str(row.get("value", ""))),
        unit=normalize_ws(str(row.get("unit", ""))),
        timeframe=normalize_ws(str(row.get("timeframe", ""))),
        source_ref=source_ref,
        confidence=confidence_f,
        source_id=source_id,
        chunk_id=chunk_id,
    )


def _extract_json_facts(raw: str, source_ref: str, source_id: str, chunk_id: str, default_conf: float = 0.74) -> List[Fact]:
    parsed = parse_json_any(raw)
    rows: List[Dict[str, object]] = []
    if isinstance(parsed, dict):
        if isinstance(parsed.get("facts"), list):
            rows = [x for x in parsed.get("facts", []) if isinstance(x, dict)]
        elif all(k in parsed for k in ["subject", "predicate", "object"]):
            rows = [parsed]
    elif isinstance(parsed, list):
        rows = [x for x in parsed if isinstance(x, dict)]

    out: List[Fact] = []
    for row in rows:
        fact = _fact_from_row(row, source_ref, source_id, chunk_id, default_conf=default_conf)
        if fact.subject:
            out.append(fact)
    return out


def _extract_entities(raw: str) -> List[str]:
    parsed = parse_json_any(raw)
    entities: List[str] = []
    if isinstance(parsed, dict):
        for key in ["entities", "key_entities"]:
            if isinstance(parsed.get(key), list):
                entities.extend([normalize_ws(str(x)) for x in parsed.get(key, [])])
                break
    entities = [e for e in entities if e]
    return dedup_keep_order(entities)[:24]


def _pass_a_prompt(chunk: ChunkRecord) -> str:
    return (
        "Extract dense factual records from the chunk. Return strict JSON only.\n"
        "JSON schema:\n"
        "{\"facts\":[{\"subject\":str,\"predicate\":str,\"object\":str,\"value\":str,\"unit\":str,\"timeframe\":str,\"confidence\":0..1}],\"entities\":[str]}\n"
        "Rules:\n"
        "- Capture financial facts, operational facts, guidance, comparisons, and constraints.\n"
        "- Always capture explicit timeline relations: 'X years before Y', 'X years after Y', and 'takes place/set ... years ...'.\n"
        "- No nulls, no placeholders, no prose outside JSON.\n"
        "- Ignore citation formatting in extraction output.\n\n"
        f"Section: {chunk.section_title}\n"
        f"Text:\n{chunk.text}\n"
    )


def _pass_b_prompt(chunk: ChunkRecord, entities: List[str], pass_a_facts: List[Fact]) -> str:
    seed = "\n".join([f"- {f.subject} | {f.predicate} | {f.object}" for f in pass_a_facts[:24]])
    ents = ", ".join(entities[:20])
    return (
        "Densify relationships between the extracted entities and facts. Return strict JSON only.\n"
        "JSON schema:\n"
        "{\"facts\":[{\"subject\":str,\"predicate\":str,\"object\":str,\"value\":str,\"unit\":str,\"timeframe\":str,\"confidence\":0..1}]}\n"
        "Rules:\n"
        "- Add missing, direct, factual relations implied by the chunk.\n"
        "- Prioritize timeline links and temporal derivations across entities when explicit in text.\n"
        "- No speculative facts.\n"
        "- No duplicates of existing relations.\n"
        f"Entities: {ents}\n"
        f"Existing facts:\n{seed}\n\n"
        f"Text:\n{chunk.text}\n"
    )


NUMERIC_SENTENCE_RE = re.compile(r"([^.!?\n]{0,240}\d[^.!?\n]{0,240})")
VALUE_RE = re.compile(
    r"(?P<value>-?\d[\d,]*(?:\.\d+)?)\s*(?P<unit>%|percent|bn|billion|million|m|k|usd|eur|dkk|gbp|teu|tons?)?",
    flags=re.IGNORECASE,
)
METRIC_RE = re.compile(
    r"(revenue|ebitda|ebit|margin|cash flow|fcf|capex|volume|cost|profit|guidance|growth)",
    flags=re.IGNORECASE,
)
YEAR_RE = re.compile(r"\b((?:19|20)\d{2}|Q[1-4]|FY\s?\d{2})\b", flags=re.IGNORECASE)
TEMPORAL_REL_RE = re.compile(
    r"(?:(?:takes place|set)\s+(?:about|around|approximately|nearly)?\s*)?(?P<value>\d{1,4})\s+years?\s+"
    r"(?P<rel>before|after)\s+(?P<object>[A-Z][^.;:\n]{2,120})",
    flags=re.IGNORECASE,
)


def _numeric_boost_facts(chunk: ChunkRecord, seed_subject: str, source_ref: str) -> List[Fact]:
    out: List[Fact] = []
    text = str(chunk.text or "")
    if len(text) < 40:
        return out

    for sentence_match in NUMERIC_SENTENCE_RE.finditer(text):
        sentence = normalize_ws(sentence_match.group(1))
        if len(sentence) < 16:
            continue
        metric_match = METRIC_RE.search(sentence)
        if not metric_match:
            continue
        value_match = VALUE_RE.search(sentence)
        if not value_match:
            continue
        metric = normalize_ws(metric_match.group(1))
        value = normalize_ws(value_match.group("value"))
        unit = normalize_ws(value_match.group("unit") or "")
        timeframe_match = YEAR_RE.search(sentence)
        timeframe = normalize_ws(timeframe_match.group(1) if timeframe_match else "")
        obj = value if not unit else f"{value} {unit}"
        out.append(
            Fact(
                subject=seed_subject,
                predicate=f"reported_{metric.lower().replace(' ', '_')}",
                object=obj,
                value=value,
                unit=unit,
                timeframe=timeframe,
                source_ref=source_ref,
                confidence=0.66,
                source_id=chunk.source_id,
                chunk_id=chunk.chunk_id,
            )
        )
        if len(out) >= 6:
            break
    return out


def _clean_timeline_object(raw: str) -> str:
    text = normalize_ws(raw)
    if not text:
        return ""
    text = re.split(r"\s+(?:in|from|for|where|which)\s+", text, maxsplit=1, flags=re.IGNORECASE)[0]
    text = text.strip(" ,.;:")
    return text


def _timeline_boost_facts(chunk: ChunkRecord, seed_subject: str, source_ref: str) -> List[Fact]:
    out: List[Fact] = []
    text = str(chunk.text or "")
    if len(text) < 20:
        return out

    for m in TEMPORAL_REL_RE.finditer(text):
        value = normalize_ws(m.group("value"))
        rel = normalize_ws(m.group("rel")).lower()
        obj = _clean_timeline_object(m.group("object"))
        if not (value and rel and obj):
            continue
        predicate = "occurs_years_before" if rel == "before" else "occurs_years_after"
        out.append(
            Fact(
                subject=seed_subject,
                predicate=predicate,
                object=obj,
                value=value,
                unit="years",
                timeframe="",
                source_ref=source_ref,
                confidence=0.68,
                source_id=chunk.source_id,
                chunk_id=chunk.chunk_id,
            )
        )
        if len(out) >= 8:
            break
    return out


def extract_facts_from_text_chunk(
    chunk: ChunkRecord,
    llm_generate: Callable[..., str],
    model: str,
    profile: str,
    cache_lookup: Callable[[str], str],
    cache_store: Callable[[str, str], None],
) -> List[Fact]:
    if not chunk.text.strip():
        return []

    mode = str(profile or "balanced").lower()
    if mode == "fast":
        predict_a = 180
        predict_b = 100
    elif mode == "quality":
        predict_a = 260
        predict_b = 150
    else:
        predict_a = 220
        predict_b = 120

    prompt_a = _pass_a_prompt(chunk)
    key_a = hashlib.md5((model + "|v2.1-passA|" + prompt_a).encode("utf-8")).hexdigest()
    raw_a = cache_lookup(key_a)
    if not raw_a:
        raw_a = llm_generate(
            prompt=prompt_a,
            system="You are a strict JSON information extraction engine.",
            model=model,
            timeout=80,
            options={"temperature": 0, "num_predict": predict_a, "num_ctx": 4096},
            response_format="json",
        )
        cache_store(key_a, raw_a)

    source_ref = f"[Chunk {chunk.chunk_id}]"
    pass_a_facts = _extract_json_facts(raw_a, source_ref, chunk.source_id, chunk.chunk_id, default_conf=0.76)
    entities = _extract_entities(raw_a)
    if len(entities) < 2 and pass_a_facts:
        entities = dedup_keep_order([f.subject for f in pass_a_facts] + [f.object for f in pass_a_facts])[:20]

    pass_b_facts: List[Fact] = []
    if len(entities) >= 2:
        prompt_b = _pass_b_prompt(chunk, entities, pass_a_facts)
        key_b = hashlib.md5((model + "|v2.1-passB|" + prompt_b).encode("utf-8")).hexdigest()
        raw_b = cache_lookup(key_b)
        if not raw_b:
            raw_b = llm_generate(
                prompt=prompt_b,
                system="You are a strict JSON relation densification engine.",
                model=model,
                timeout=70,
                options={"temperature": 0, "num_predict": predict_b, "num_ctx": 4096},
                response_format="json",
            )
            cache_store(key_b, raw_b)
        pass_b_facts = _extract_json_facts(raw_b, source_ref, chunk.source_id, chunk.chunk_id, default_conf=0.70)

    # Rule-based numeric boost to improve density for financial documents.
    seed_subject = (
        pass_a_facts[0].subject
        if pass_a_facts
        else normalize_ws(chunk.section_title) or f"source:{chunk.source_id}"
    )
    boost = _numeric_boost_facts(chunk, seed_subject, source_ref)
    timeline_boost = _timeline_boost_facts(chunk, seed_subject, source_ref)

    merged = pass_a_facts + pass_b_facts + boost + timeline_boost

    # Dedup local chunk facts.
    dedup = {}
    for f in merged:
        key = (
            normalize_ws(f.subject).lower(),
            normalize_ws(f.predicate).lower(),
            normalize_ws(f.object).lower(),
            normalize_ws(f.timeframe).lower(),
            normalize_ws(f.value).lower(),
            normalize_ws(f.unit).lower(),
        )
        if not all(key[:3]):
            continue
        if key not in dedup or f.confidence > dedup[key].confidence:
            dedup[key] = f
    return list(dedup.values())
