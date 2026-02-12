from __future__ import annotations

import ast
import json
import re
from typing import Any, Dict, List, Optional, Tuple


def now_iso() -> str:
    from datetime import datetime

    return datetime.utcnow().isoformat() + "Z"


def normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def approx_tokens(text: str) -> int:
    # Lightweight approximation; good enough for local batching.
    words = len(re.findall(r"\S+", str(text or "")))
    return max(1, int(words * 1.33))


def split_sections(text: str) -> List[Tuple[str, str, int, int]]:
    """Return list of (title, body, start_idx, end_idx)."""
    src = str(text or "")
    if not src.strip():
        return []

    lines = src.splitlines()
    offsets: List[int] = []
    cursor = 0
    for ln in lines:
        offsets.append(cursor)
        cursor += len(ln) + 1

    def looks_like_heading(line: str) -> bool:
        raw = line.strip()
        if not raw:
            return False
        if len(raw) > 110:
            return False
        if re.match(r"^(\d+(?:\.\d+)*|Q[1-4]|FY\d{2})\b", raw, flags=re.IGNORECASE):
            return True
        if raw.endswith(":") and len(raw.split()) <= 12:
            return True
        if raw.isupper() and 2 <= len(raw.split()) <= 12:
            return True
        if re.match(r"^[A-Z][A-Za-z0-9 &/\-]{2,60}$", raw) and len(raw.split()) <= 10:
            return True
        return False

    sections: List[Tuple[str, str, int, int]] = []
    current_title = "Section"
    current_start = 0
    current_lines: List[str] = []

    for idx, ln in enumerate(lines):
        if looks_like_heading(ln):
            if current_lines:
                body = "\n".join(current_lines).strip()
                if body:
                    sections.append((current_title, body, current_start, offsets[idx]))
            current_title = normalize_ws(ln)
            current_start = offsets[idx]
            current_lines = []
        else:
            current_lines.append(ln)

    if current_lines:
        body = "\n".join(current_lines).strip()
        if body:
            sections.append((current_title, body, current_start, len(src)))

    if not sections:
        return [("Section", src, 0, len(src))]
    return sections


def extract_balanced_block(text: str, start_at: int, opening: str, closing: str) -> Tuple[str, int, int]:
    if not text or start_at >= len(text):
        return "", -1, -1
    i = start_at
    while i < len(text) and text[i].isspace():
        i += 1
    if i >= len(text) or text[i] != opening:
        return "", -1, -1
    depth = 0
    quote = ""
    escaped = False
    for j in range(i, len(text)):
        ch = text[j]
        if quote:
            if escaped:
                escaped = False
                continue
            if ch == "\\":
                escaped = True
                continue
            if ch == quote:
                quote = ""
            continue
        if ch in {'"', "'"}:
            quote = ch
            continue
        if ch == opening:
            depth += 1
            continue
        if ch == closing:
            depth -= 1
            if depth == 0:
                return text[i : j + 1], i, j + 1
    return "", -1, -1


def extract_first_json_value(text: str, start_at: int = 0) -> Tuple[str, int, int]:
    if not text:
        return "", -1, -1
    cursor = max(0, int(start_at))
    while cursor < len(text):
        obj_pos = text.find("{", cursor)
        arr_pos = text.find("[", cursor)
        candidates = [p for p in [obj_pos, arr_pos] if p != -1]
        if not candidates:
            return "", -1, -1
        pos = min(candidates)
        ch = text[pos]
        if ch == "{":
            block, s, e = extract_balanced_block(text, pos, "{", "}")
        else:
            block, s, e = extract_balanced_block(text, pos, "[", "]")
        if block:
            return block, s, e
        cursor = pos + 1
    return "", -1, -1


def normalize_jsonish(raw: str) -> str:
    text = str(raw or "")
    if not text:
        return text
    text = text.replace("\u201c", '"').replace("\u201d", '"').replace("\u2018", "'").replace("\u2019", "'")
    text = re.sub(
        r'("source_ref"\s*:\s*)\[\s*([^\]\n]+)\s*\]',
        lambda m: f'{m.group(1)}"[{m.group(2).strip()}]"',
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r",\s*([\]}])", r"\1", text)
    return text


def parse_json_any(text: str) -> Any:
    src = str(text or "").strip()
    if not src:
        return None
    block, s, e = extract_first_json_value(src, 0)
    candidate = block if block else src
    for trial in [candidate, normalize_jsonish(candidate)]:
        try:
            return json.loads(trial)
        except Exception:
            try:
                return ast.literal_eval(trial)
            except Exception:
                continue
    return None


def to_float(val: Any) -> Optional[float]:
    if isinstance(val, (int, float)):
        return float(val)
    text = normalize_ws(str(val or ""))
    if not text:
        return None
    m = re.search(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
    if not m:
        return None
    try:
        return float(m.group(0))
    except Exception:
        return None


def dedup_keep_order(items: List[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for item in items:
        v = str(item).strip()
        if not v:
            continue
        k = v.lower()
        if k in seen:
            continue
        seen.add(k)
        out.append(v)
    return out

