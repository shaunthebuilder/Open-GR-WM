from __future__ import annotations

import hashlib
from typing import Callable, Dict, List

from .types import Fact
from .utils import normalize_ws, parse_json_any


def _fact_from_row(row: Dict[str, object], source_ref: str, source_id: str, chunk_id: str, default_conf: float = 0.70) -> Fact:
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


def _extract_json_facts(raw: str, source_ref: str, source_id: str, chunk_id: str) -> List[Fact]:
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
        f = _fact_from_row(row, source_ref, source_id, chunk_id)
        if f.subject:
            out.append(f)
    return out


def extract_facts_from_vision_artifact(
    artifact_b64: str,
    source_id: str,
    chunk_id: str,
    label_hint: str,
    llm_generate: Callable[..., str],
    model: str,
    profile: str,
    cache_lookup: Callable[[str], str],
    cache_store: Callable[[str, str], None],
) -> List[Fact]:
    mode = str(profile or "balanced").lower()
    if mode == "fast":
        num_predict = 220
    elif mode == "quality":
        num_predict = 320
    else:
        num_predict = 270

    prompt = (
        "Extract chart/table facts from this image. Return strict JSON only.\n"
        "JSON schema:\n"
        "{\"facts\":[{\"subject\":str,\"predicate\":str,\"object\":str,\"value\":str,\"unit\":str,\"timeframe\":str,\"confidence\":0..1}]}\n"
        "Rules:\n"
        "- Prioritize numeric metrics, trend points, guidance values, comparisons.\n"
        "- No speculative interpretation, no prose.\n"
        f"Hint: {label_hint}\n"
    )
    image_sig = hashlib.md5(str(artifact_b64 or "").encode("utf-8")).hexdigest()[:16]
    key = hashlib.md5((model + "|v2.1-vision|" + prompt + "|" + image_sig).encode("utf-8")).hexdigest()
    raw = cache_lookup(key)
    if not raw:
        raw = llm_generate(
            prompt=prompt,
            system="You are a strict JSON chart extraction engine.",
            model=model,
            images=[artifact_b64],
            timeout=120,
            options={"temperature": 0, "num_predict": num_predict, "num_ctx": 4096},
            response_format="json",
        )
        cache_store(key, raw)

    source_ref = f"[Vision {chunk_id}]"
    return _extract_json_facts(raw, source_ref, source_id, chunk_id)
