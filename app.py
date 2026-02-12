import ast
import base64
import copy
import html
import hashlib
import io
import json
import os
import re
import shutil
import tempfile
import time
import uuid
import zipfile
from datetime import datetime
from typing import Callable, Dict, List, Optional, Tuple
from urllib.parse import urljoin, urlparse, urlunparse

import networkx as nx
import numpy as np
import plotly.graph_objects as go
import requests
import streamlit as st
from PyPDF2 import PdfReader
from pyvis.network import Network
from sentence_transformers import SentenceTransformer

from graphrag_v2.canonicalize import canonicalize_facts, facts_to_graph
from graphrag_v2.chunking import build_chunk_tracks_for_pages
from graphrag_v2.evals import run_build_evals, run_compositional_retrieval_eval
from graphrag_v2.extract_text import extract_facts_from_text_chunk
from graphrag_v2.extract_vision import extract_facts_from_vision_artifact
from graphrag_v2.indexing import build_retrieval_index
from graphrag_v2.intent import parse_intent_spec
from graphrag_v2.retrieval import retrieve_evidence_bundle
from graphrag_v2.storage import (
    cache_lookup as v2_cache_lookup,
    cache_store as v2_cache_store,
    clear_build_checkpoint as v2_clear_checkpoint,
    ensure_v2_dir,
    load_v2_artifacts,
    save_build_checkpoint as v2_save_checkpoint,
    save_v2_artifacts,
)

try:
    from bs4 import BeautifulSoup
except Exception:
    BeautifulSoup = None


OLLAMA_URL = "http://localhost:11434/api/generate"
SCANNER_MODEL = "llama3.2:latest"
VISION_MODEL = "llama3.2-vision:latest"
BRAIN_MODEL = "deepseek-r1:14b"
EMBED_MODEL = "all-MiniLM-L6-v2"
RAG_STORE_DIR = os.path.join(os.getcwd(), "rag_store")
SETTINGS_PATH = os.path.join(RAG_STORE_DIR, "settings.json")
CHAT_SESSIONS_PATH = os.path.join(RAG_STORE_DIR, "chat_sessions.json")
EXTERNAL_STYLE_PATH = "/Users/shantanurastogi/Downloads/style.css"
LOCAL_STYLE_PATH = os.path.join(os.getcwd(), "style.css")


def strip_think(text: str) -> str:
    if not text:
        return ""
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    cleaned = cleaned.replace("</think>", "").replace("<think>", "")
    return cleaned.strip()


def split_think_and_answer(text: str) -> Tuple[str, str]:
    if not text:
        return "", ""
    lower = text.lower()
    i = 0
    in_think = False
    think_chars: List[str] = []
    answer_chars: List[str] = []
    while i < len(text):
        if lower.startswith("<think>", i):
            in_think = True
            i += len("<think>")
            continue
        if lower.startswith("</think>", i):
            in_think = False
            i += len("</think>")
            continue
        if in_think:
            think_chars.append(text[i])
        else:
            answer_chars.append(text[i])
        i += 1
    return "".join(think_chars).strip(), "".join(answer_chars).strip()


def is_ollama_error(text: str) -> bool:
    return str(text).startswith("[ERROR]")


def call_ollama(
    prompt: str,
    system: str = "",
    model: str = BRAIN_MODEL,
    images: List[str] = None,
    timeout: int = 120,
    options: Dict[str, object] = None,
    response_format: str = "",
) -> str:
    payload = {
        "model": model,
        "prompt": prompt,
        "system": system,
        "stream": False,
    }
    if images:
        payload["images"] = images
    if options:
        payload["options"] = options
    if response_format:
        payload["format"] = response_format
    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        return strip_think(data.get("response", ""))
    except Exception as exc:
        return f"[ERROR] Ollama request failed: {exc}"


def call_ollama_stream(
    prompt: str,
    system: str = "",
    model: str = BRAIN_MODEL,
    images: List[str] = None,
    timeout: int = 300,
    options: Dict[str, object] = None,
    on_update: Optional[Callable[[str, str], None]] = None,
) -> Tuple[str, str, str]:
    payload = {
        "model": model,
        "prompt": prompt,
        "system": system,
        "stream": True,
    }
    if images:
        payload["images"] = images
    if options:
        payload["options"] = options
    try:
        resp = requests.post(
            OLLAMA_URL,
            json=payload,
            stream=True,
            timeout=(10, timeout),
        )
        resp.raise_for_status()
        raw_full = ""
        done_reason = ""
        for line in resp.iter_lines(decode_unicode=True):
            if not line:
                continue
            try:
                data = json.loads(line)
            except Exception:
                continue
            if data.get("error"):
                return f"[ERROR] Ollama stream error: {data.get('error')}", "", "error"
            token = data.get("response", "")
            if token:
                raw_full += token
                think_text, answer_text = split_think_and_answer(raw_full)
                if on_update:
                    on_update(think_text, answer_text)
            if data.get("done"):
                done_reason = str(data.get("done_reason", "")).strip().lower()
        think_text, answer_text = split_think_and_answer(raw_full)
        return answer_text.strip(), think_text.strip(), done_reason
    except Exception as exc:
        return f"[ERROR] Ollama request failed: {exc}", "", "error"


def should_continue_answer(answer_text: str, done_reason: str = "") -> bool:
    txt = str(answer_text or "").strip()
    reason = str(done_reason or "").strip().lower()
    if reason == "length":
        return True
    if not txt:
        return False
    if txt.endswith((":", "-", "•", ",")):
        return True
    if txt.lower().endswith("hero figures:"):
        return True
    last_line = txt.splitlines()[-1].strip()
    if not last_line:
        return False
    terminal_ok = bool(re.search(r"[.!?\"'\]\)}]$", last_line))
    tail_word = re.findall(r"[A-Za-z]+", last_line.lower())
    tail_word = tail_word[-1] if tail_word else ""
    if not terminal_ok and tail_word in {"approximately", "around", "about", "due", "with", "to", "from", "and", "or"}:
        return True
    if re.match(r"^[-*]\s+[A-Za-z0-9$%]{1,24}$", last_line):
        return True
    if len(last_line.split()) <= 3 and not re.search(r"[.!?\"'\]\)}]$", last_line):
        return True
    return False


def read_pdf_pages(pdf_bytes: bytes) -> List[str]:
    reader = PdfReader(io.BytesIO(pdf_bytes))
    pages = []
    for page in reader.pages:
        text = page.extract_text() or ""
        pages.append(text)
    return pages


def read_pdf_text(pdf_bytes: bytes) -> str:
    return "\n".join(read_pdf_pages(pdf_bytes))


def normalize_source_url(url: str) -> str:
    value = str(url or "").strip()
    if not value:
        return ""
    if not re.match(r"^https?://", value, flags=re.IGNORECASE):
        value = "https://" + value
    parsed = urlparse(value)
    if not parsed.netloc:
        return ""
    parsed = parsed._replace(fragment="")
    return urlunparse(parsed)


def url_to_graph_name(url: str) -> str:
    normalized = normalize_source_url(url)
    if not normalized:
        return ""
    parsed = urlparse(normalized)
    host = (parsed.netloc or "").replace("www.", "").strip()
    tail = parsed.path.strip("/").split("/")[-1] if parsed.path.strip("/") else ""
    tail = re.sub(r"[-_]+", " ", tail).strip()
    if tail:
        return normalize_graph_name(f"{host} {tail}", fallback=host or "web graph")
    return normalize_graph_name(host, fallback="web graph")


def parse_web_url_inputs(primary_url: str, batch_text: str) -> List[str]:
    candidates: List[str] = []
    for raw in [primary_url] + str(batch_text or "").splitlines():
        value = normalize_source_url(raw)
        if value:
            candidates.append(value)
    deduped: List[str] = []
    seen = set()
    for url in candidates:
        key = url.lower().strip()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(url)
    return deduped


def make_web_text_segments(blocks: List[str], max_chars: int = 2500, min_chars: int = 70) -> List[str]:
    cleaned_blocks: List[str] = []
    seen = set()
    for block in blocks:
        text = re.sub(r"\s+", " ", str(block or "")).strip()
        if len(text) < min_chars and not text.lower().startswith(("title:", "summary:")):
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        cleaned_blocks.append(text)
    if not cleaned_blocks:
        return []

    segments: List[str] = []
    bucket: List[str] = []
    size = 0
    for blk in cleaned_blocks:
        add = len(blk) + 1
        if bucket and size + add > max_chars:
            segments.append("\n".join(bucket))
            bucket = [blk]
            size = len(blk)
        else:
            bucket.append(blk)
            size += add
    if bucket:
        segments.append("\n".join(bucket))
    return segments


def score_web_image_candidate(url: str, alt_text: str, width: int, height: int) -> float:
    combined = f"{url} {alt_text}".lower()
    score = 0.0
    if any(x in combined for x in ["chart", "graph", "table", "figure", "trend", "ebit", "ebitda", "revenue", "profit", "margin", "volume", "financial"]):
        score += 4.0
    if any(x in combined for x in ["logo", "icon", "avatar", "sprite", "favicon", "social", "emoji"]):
        score -= 5.0
    if width >= 380 or height >= 250:
        score += 1.4
    if width and height and width < 90 and height < 90:
        score -= 2.2
    if url.lower().endswith(".svg"):
        score -= 1.5
    return score


def _to_int(value) -> int:
    try:
        return int(float(str(value).strip()))
    except Exception:
        return 0


def extract_web_content(
    url: str,
    include_images: bool = True,
    max_images: int = 20,
) -> Dict[str, object]:
    normalized_url = normalize_source_url(url)
    if not normalized_url:
        raise ValueError("Enter a valid URL (example: https://example.com).")

    headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X) OpenGR-WM/1.0"}
    resp = requests.get(normalized_url, headers=headers, timeout=30, allow_redirects=True)
    resp.raise_for_status()
    final_url = normalize_source_url(resp.url) or normalized_url
    content_type = str(resp.headers.get("content-type", "")).lower()
    if "text/html" not in content_type and "application/xhtml+xml" not in content_type:
        raise ValueError(f"Unsupported URL content-type for graph ingest: {content_type or 'unknown'}")

    html_text = resp.text or ""
    text_blocks: List[str] = []
    title = ""
    image_candidates: List[Dict[str, object]] = []

    if BeautifulSoup is not None:
        soup = BeautifulSoup(html_text, "html.parser")
        for tag in soup(["script", "style", "noscript", "header", "footer", "nav", "aside", "form"]):
            tag.decompose()

        if soup.title:
            title = re.sub(r"\s+", " ", soup.title.get_text(" ", strip=True)).strip()
        meta_desc = soup.find("meta", attrs={"name": re.compile(r"^description$", re.IGNORECASE)})
        if title:
            text_blocks.append(f"Title: {title}")
        if meta_desc and meta_desc.get("content"):
            text_blocks.append(f"Summary: {str(meta_desc.get('content')).strip()}")

        main = soup.find("main") or soup.body or soup
        for elem in main.find_all(["h1", "h2", "h3", "p", "li", "td", "th"], limit=3500):
            txt = re.sub(r"\s+", " ", elem.get_text(" ", strip=True)).strip()
            if txt:
                text_blocks.append(txt)

        if include_images:
            seen_urls = set()
            for img in soup.find_all("img", limit=600):
                src = img.get("src") or img.get("data-src") or img.get("data-original") or ""
                src = str(src).strip()
                if not src or src.startswith("data:"):
                    continue
                abs_url = urljoin(final_url, src)
                if abs_url in seen_urls:
                    continue
                seen_urls.add(abs_url)
                alt = re.sub(r"\s+", " ", str(img.get("alt", "")).strip())
                width = _to_int(img.get("width"))
                height = _to_int(img.get("height"))
                score = score_web_image_candidate(abs_url, alt, width, height)
                image_candidates.append(
                    {
                        "url": abs_url,
                        "alt": alt,
                        "width": width,
                        "height": height,
                        "score": score,
                    }
                )
    else:
        # Fallback if bs4 is unavailable.
        stripped = re.sub(r"<(script|style).*?>.*?</\1>", " ", html_text, flags=re.IGNORECASE | re.DOTALL)
        stripped = re.sub(r"<[^>]+>", " ", stripped)
        stripped = re.sub(r"\s+", " ", stripped).strip()
        if stripped:
            text_blocks = [stripped]

    segments = make_web_text_segments(text_blocks, max_chars=2500, min_chars=70)
    if not segments and text_blocks:
        segments = make_web_text_segments([" ".join(text_blocks)], max_chars=2500, min_chars=10)

    vision_payloads: List[Tuple[int, str, str]] = []
    if include_images and image_candidates:
        ranked = sorted(image_candidates, key=lambda x: float(x.get("score", 0.0)), reverse=True)
        # If heuristics are weak, still keep a few largest/non-icon images as fallback.
        shortlisted = [x for x in ranked if float(x.get("score", 0.0)) > 0][: max_images * 3]
        if not shortlisted:
            shortlisted = ranked[: max_images * 2]
        for idx, cand in enumerate(shortlisted[: max_images]):
            img_url = str(cand.get("url", "")).strip()
            if not img_url:
                continue
            try:
                img_resp = requests.get(img_url, headers=headers, timeout=25)
                img_resp.raise_for_status()
                ct = str(img_resp.headers.get("content-type", "")).lower()
                if not ct.startswith("image/"):
                    continue
                data = img_resp.content or b""
                if not data or len(data) > 2_500_000:
                    continue
                img_b64 = base64.b64encode(data).decode("utf-8")
                label_hint = str(cand.get("alt", "")).strip() or os.path.basename(urlparse(img_url).path) or f"image {idx + 1}"
                vision_payloads.append((idx, img_b64, label_hint[:96]))
            except Exception:
                continue

    return {
        "url": final_url,
        "title": title,
        "segments": segments,
        "vision_images": vision_payloads,
        "image_candidate_count": len(image_candidates),
    }


def page_has_image(page) -> bool:
    try:
        resources = page.get("/Resources")
        if not resources:
            return False
        xobject = resources.get("/XObject")
        if not xobject:
            return False
        for obj in xobject.values():
            obj = obj.get_object()
            if obj.get("/Subtype") == "/Image":
                return True
    except Exception:
        return False
    return False


def analyze_pdf_pages(pdf_bytes: bytes) -> Tuple[List[int], List[int], int]:
    reader = PdfReader(io.BytesIO(pdf_bytes))
    image_pages = []
    text_lengths = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        text_lengths.append(len(text))
        if page_has_image(page):
            image_pages.append(i)
    return image_pages, text_lengths, len(reader.pages)


def prepare_vision_images(pdf_bytes: bytes, page_indices: List[int], max_pages: int = 5, scale: float = 1.6) -> List[Tuple[int, str]]:
    try:
        import pypdfium2 as pdfium
    except Exception:
        return []
    pdf = pdfium.PdfDocument(pdf_bytes)
    images_b64: List[Tuple[int, str]] = []
    for idx in page_indices[:max_pages]:
        try:
            page = pdf[idx]
            pil_img = page.render(scale=scale).to_pil()
            buf = io.BytesIO()
            pil_img.save(buf, format="PNG")
            images_b64.append((idx, base64.b64encode(buf.getvalue()).decode("utf-8")))
        except Exception:
            continue
    return images_b64


def select_vision_pages(
    image_pages: List[int],
    text_lengths: List[int],
    text_limit: int,
    max_pages: int,
) -> List[int]:
    total = len(text_lengths)
    image_set = set(image_pages)
    candidates = [i for i in range(total) if (i in image_set) or (text_lengths[i] <= text_limit)]
    candidates.sort(key=lambda i: (0 if i in image_set else 1, text_lengths[i], i))
    return candidates[:max_pages]


def build_text_batches(
    page_texts: List[str],
    max_chars: int = 2800,
    max_pages_per_batch: int = 3,
    min_chars: int = 80,
) -> List[Dict[str, object]]:
    tasks: List[Dict[str, object]] = []
    total_pages = len(page_texts)
    i = 0
    while i < total_pages:
        page_text = (page_texts[i] or "").strip()
        if len(page_text) < min_chars:
            i += 1
            continue
        batch_pages = [i]
        batch_parts = [f"[Page {i + 1}]\n{page_text}"]
        used = len(page_text)
        j = i + 1
        while j < total_pages and len(batch_pages) < max_pages_per_batch:
            nxt = (page_texts[j] or "").strip()
            if len(nxt) < min_chars:
                j += 1
                continue
            if used + len(nxt) > max_chars:
                break
            batch_pages.append(j)
            batch_parts.append(f"[Page {j + 1}]\n{nxt}")
            used += len(nxt)
            j += 1
        tasks.append(
            {
                "type": "text",
                "data": "\n\n".join(batch_parts),
                "label": f"text batch pages {batch_pages[0] + 1}-{batch_pages[-1] + 1}/{total_pages}",
            }
        )
        i = max(j, i + 1)
    return tasks


def chunk_text(text: str, size: int = 1200, overlap: int = 200) -> List[str]:
    clean = re.sub(r"\s+", " ", text).strip()
    if not clean:
        return []
    chunks = []
    start = 0
    while start < len(clean):
        end = min(len(clean), start + size)
        chunks.append(clean[start:end])
        if end >= len(clean):
            break
        start = end - overlap
        if start < 0:
            start = 0
    return chunks


def build_retrieval_chunks_from_pages(
    page_texts: List[str],
    chunk_size: int = 900,
    overlap: int = 160,
    min_chars: int = 80,
    max_chunks_per_page: int = 14,
) -> List[str]:
    chunks: List[str] = []
    for page_idx, page_text in enumerate(page_texts):
        clean = re.sub(r"\s+", " ", str(page_text or "")).strip()
        if len(clean) < min_chars:
            continue
        parts = chunk_text(clean, size=chunk_size, overlap=overlap)
        for part_idx, part in enumerate(parts[:max_chunks_per_page]):
            part_clean = str(part).strip()
            if len(part_clean) < min_chars:
                continue
            chunks.append(f"[Page {page_idx + 1} | Part {part_idx + 1}] {part_clean}")
    return chunks


def extract_balanced_block(text: str, start_at: int, opening: str, closing: str) -> Tuple[str, int, int]:
    if not text or start_at >= len(text):
        return "", -1, -1
    i = start_at
    while i < len(text) and text[i].isspace():
        i += 1
    if i >= len(text) or text[i] != opening:
        return "", -1, -1
    depth = 0
    quote_char = ""
    escaped = False
    for j in range(i, len(text)):
        ch = text[j]
        if quote_char:
            if escaped:
                escaped = False
                continue
            if ch == "\\":
                escaped = True
                continue
            if ch == quote_char:
                quote_char = ""
            continue
        if ch in {"'", '"'}:
            quote_char = ch
            continue
        if ch == opening:
            depth += 1
            continue
        if ch == closing:
            depth -= 1
            if depth == 0:
                return text[i : j + 1], i, j + 1
    return "", -1, -1


def extract_json_block(text: str) -> str:
    if not text:
        return ""
    cleaned = strip_think(text)
    # Keep fenced content; strip only markdown fence markers.
    cleaned = re.sub(r"```(?:json|javascript|js|text)?", "", cleaned, flags=re.IGNORECASE)
    cleaned = cleaned.replace("```", "").strip()
    if not cleaned:
        return ""

    if cleaned[0] == "{":
        block, _, _ = extract_balanced_block(cleaned, 0, "{", "}")
        if block:
            return block
    if cleaned[0] == "[":
        block, _, _ = extract_balanced_block(cleaned, 0, "[", "]")
        if block:
            return block

    obj_pos = cleaned.find("{")
    arr_pos = cleaned.find("[")
    candidates = []
    if obj_pos != -1:
        candidates.append((obj_pos, "{", "}"))
    if arr_pos != -1:
        candidates.append((arr_pos, "[", "]"))
    candidates.sort(key=lambda x: x[0])

    for pos, opening, closing in candidates:
        block, _, _ = extract_balanced_block(cleaned, pos, opening, closing)
        if block:
            return block
    return cleaned.strip()


def parse_triples(text: str) -> List[Dict[str, str]]:
    data = parse_json_any(text)
    if data is None:
        return []
    if isinstance(data, dict):
        if isinstance(data.get("triples"), list):
            data = data.get("triples")
        else:
            data = [data]
    triples = []
    if isinstance(data, list):
        for item in data:
            if not isinstance(item, dict):
                continue
            subj = str(item.get("subject", "")).strip()
            pred = str(item.get("predicate", "")).strip()
            obj = str(item.get("object", "")).strip()
            if subj and pred and obj:
                triples.append({"subject": subj, "predicate": pred, "object": obj})
    return triples


def parse_json_object(text: str) -> Dict[str, object]:
    data = parse_json_any(text)
    return data if isinstance(data, dict) else {}


def normalize_jsonish(block: str) -> str:
    text = str(block or "")
    if not text:
        return text
    text = text.replace("\u201c", '"').replace("\u201d", '"').replace("\u2018", "'").replace("\u2019", "'")
    # Common model artifact: source: [Chunk 9] (invalid JSON/Python literal)
    text = re.sub(
        r'("source"\s*:\s*)\[\s*([^\]\n]+)\s*\]',
        lambda m: f'{m.group(1)}"[{m.group(2).strip()}]"',
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"('source'\s*:\s*)\[\s*([^\]\n]+)\s*\]",
        lambda m: f'{m.group(1)}"[{m.group(2).strip()}]"',
        text,
        flags=re.IGNORECASE,
    )
    # Strip trailing commas before container close.
    text = re.sub(r",\s*([\]}])", r"\1", text)
    return text


def parse_json_any(text: str):
    block = extract_json_block(text)
    if not block:
        return None
    try:
        return json.loads(block)
    except Exception:
        try:
            return ast.literal_eval(block)
        except Exception:
            normalized = normalize_jsonish(block)
            if normalized != block:
                try:
                    return json.loads(normalized)
                except Exception:
                    try:
                        return ast.literal_eval(normalized)
                    except Exception:
                        return None
            return None


def extract_balanced_json(text: str, start_at: int = 0) -> Tuple[str, int, int]:
    return extract_balanced_block(text, start_at, "{", "}")


def extract_first_json_object(text: str, start_at: int = 0) -> Tuple[str, int, int]:
    if not text:
        return "", -1, -1
    pos = text.find("{", max(start_at, 0))
    while pos != -1:
        block, s, e = extract_balanced_json(text, pos)
        if block:
            return block, s, e
        pos = text.find("{", pos + 1)
    return "", -1, -1


def extract_first_json_value(text: str, start_at: int = 0) -> Tuple[str, int, int]:
    if not text:
        return "", -1, -1
    cursor = max(start_at, 0)
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


def extend_json_remove_end(text: str, end_idx: int) -> int:
    i = min(max(end_idx, 0), len(text))
    while i < len(text) and text[i] in " \t\r\n":
        i += 1
    if text.startswith("```", i):
        i += 3
        while i < len(text) and text[i] not in "\r\n":
            i += 1
        while i < len(text) and text[i] in "\r\n":
            i += 1
    return i


def splice_text(prefix: str, suffix: str) -> str:
    left = str(prefix or "")
    right = str(suffix or "")
    if not left:
        return right.strip()
    if not right:
        return left.strip()
    joiner = ""
    if left[-1].isalnum() and right[0].isalnum():
        joiner = "\n\n"
    elif left.endswith("\n") or right.startswith("\n"):
        joiner = ""
    else:
        joiner = "\n"
    return f"{left}{joiner}{right}".strip()


def build_graph_from_text(chunks: List[str]) -> Tuple[nx.MultiDiGraph, List[Tuple[str, str, str]]]:
    graph = nx.MultiDiGraph()
    all_triples = []
    system = (
        "You are an information extraction engine. "
        "Extract factual relationships from the text."
    )
    for chunk in chunks:
        prompt = (
            "Extract entities and relationships from the text below.\n"
            "Return ONLY valid JSON as an array of objects with keys: "
            "subject, predicate, object.\n"
            "No prose, no commentary.\n\n"
            f"Text:\n{chunk}\n"
        )
        raw = call_ollama(
            prompt,
            system=system,
            model=SCANNER_MODEL,
            timeout=90,
            options={"temperature": 0, "num_predict": 500, "num_ctx": 4096},
        )
        triples = parse_triples(raw)
        for t in triples:
            s = t["subject"]
            p = t["predicate"]
            o = t["object"]
            graph.add_node(s)
            graph.add_node(o)
            graph.add_edge(s, o, label=p)
            all_triples.append((s, p, o))
    return graph, all_triples


def build_graph_from_images(images_b64: List[str]) -> Tuple[nx.MultiDiGraph, List[Tuple[str, str, str]]]:
    graph = nx.MultiDiGraph()
    all_triples = []
    system = (
        "You are an information extraction engine for charts and tables. "
        "Extract factual relationships and numeric values from the image."
    )
    for img_b64 in images_b64:
        prompt = (
            "Extract key facts from the chart/table in the image.\n"
            "Return ONLY valid JSON as an array of objects with keys: "
            "subject, predicate, object.\n"
            "No prose, no commentary."
        )
        raw = call_ollama(
            prompt,
            system=system,
            model=VISION_MODEL,
            images=[img_b64],
            timeout=120,
            options={"temperature": 0, "num_predict": 450, "num_ctx": 4096},
        )
        triples = parse_triples(raw)
        for t in triples:
            s = t["subject"]
            p = t["predicate"]
            o = t["object"]
            graph.add_node(s)
            graph.add_node(o)
            graph.add_edge(s, o, label=p)
            all_triples.append((s, p, o))
    return graph, all_triples


def compact_label(text: str, max_len: int = 34) -> str:
    value = str(text)
    if len(value) <= max_len:
        return value
    return value[: max_len - 3] + "..."


def graph_to_pyvis(graph: nx.MultiDiGraph) -> str:
    net = Network(height="640px", width="100%", bgcolor="#00000000", font_color="#dbe7f5", directed=True)
    degrees = dict(graph.degree())
    max_degree = max(degrees.values()) if degrees else 1
    for node in graph.nodes:
        degree = int(degrees.get(node, 0))
        if degree >= 10:
            node_color = {"background": "#00F0FF", "border": "#8EF9FF", "highlight": {"background": "#2CF4FF", "border": "#A8FCFF"}}
        elif degree >= 4:
            node_color = {"background": "#3FA3FF", "border": "#7EC4FF", "highlight": {"background": "#56B2FF", "border": "#9CD2FF"}}
        else:
            node_color = {"background": "#2E6DFF", "border": "#7B9DFF", "highlight": {"background": "#4E83FF", "border": "#9EB6FF"}}
        scaled = degree / max_degree if max_degree else 0
        node_size = 14 + int(26 * scaled)
        net.add_node(
            node,
            label=compact_label(node),
            title=f"{node}\nDegree: {degree}",
            size=node_size,
            color=node_color,
            borderWidth=1.2,
            borderWidthSelected=2.0,
            shape="dot",
        )
    for u, v, data in graph.edges(data=True):
        label = data.get("label", "")
        net.add_edge(
            u,
            v,
            label=compact_label(label, 28),
            title=label,
            color={"color": "rgba(212, 220, 235, 0.45)", "highlight": "rgba(240, 245, 255, 0.90)"},
            width=1.2,
            smooth={"type": "dynamic", "roundness": 0.15},
        )
    net.set_options(
        """
        var options = {
            "nodes": {
            "font": {"size": 13, "face": "Inter", "color": "#EAF4FF", "strokeWidth": 0},
            "shadow": {"enabled": true, "color": "rgba(0,255,255,0.35)", "size": 12}
            },
          "interaction": {
            "hover": true,
            "navigationButtons": true,
            "keyboard": {"enabled": true, "bindToWindow": false},
            "multiselect": true
          },
          "physics": {
            "solver": "forceAtlas2Based",
            "forceAtlas2Based": {
              "gravitationalConstant": -52,
              "centralGravity": 0.012,
              "springLength": 150,
              "springConstant": 0.055,
              "damping": 0.5,
              "avoidOverlap": 0.8
            },
            "stabilization": {"enabled": true, "iterations": 240, "updateInterval": 25, "fit": true},
            "minVelocity": 0.75
          },
          "edges": {
            "arrows": {"to": {"enabled": true, "scaleFactor": 0.55}},
            "font": {"size": 10, "color": "#a8b0c2", "strokeWidth": 0},
            "selectionWidth": 2
          }
        }
        """
    )
    with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp:
        try:
            net.write_html(tmp.name, open_browser=False, notebook=False)
            with open(tmp.name, "r", encoding="utf-8") as f:
                html = f.read()
            freeze_script = """
            <script>
            (function() {
              function lockPhysics() {
                if (window.network && typeof window.network.once === "function") {
                  window.network.once("stabilizationIterationsDone", function () {
                    window.network.setOptions({physics: false});
                  });
                  return true;
                }
                return false;
              }
              if (!lockPhysics()) {
                let tries = 0;
                const t = setInterval(function() {
                  tries += 1;
                  if (lockPhysics() || tries > 40) {
                    clearInterval(t);
                  }
                }, 120);
              }
            })();
            </script>
            """
            if "</body>" in html:
                html = html.replace("</body>", freeze_script + "\n</body>")
                with open(tmp.name, "w", encoding="utf-8") as f:
                    f.write(html)
        except Exception as exc:
            fallback = (
                "<html><body style='background:#0f1117;color:#e6e6e6;"
                "font-family:monospace;padding:16px;'>"
                "<h3>Graph Render Fallback</h3>"
                f"<p>PyVis could not render interactive graph: {str(exc)}</p>"
                f"<p>Nodes: {graph.number_of_nodes()} | Edges: {graph.number_of_edges()}</p>"
                "</body></html>"
            )
            tmp.write(fallback.encode("utf-8"))
        return tmp.name


def keyword_nodes(graph: nx.MultiDiGraph, query: str) -> List[str]:
    q = str(query or "").lower().strip()
    if not q:
        return []
    tokens = [t for t in re.findall(r"[a-z0-9]{2,}", q) if t not in {"what", "which", "show", "give", "about", "from", "with", "this", "that", "the"}]
    scored: List[Tuple[float, str]] = []
    for node in graph.nodes:
        name = str(node)
        lower = name.lower()
        score = 0.0
        if q in lower:
            score += 4.0
        overlap = 0
        for tok in tokens:
            if tok in lower:
                overlap += 1
        if overlap:
            score += min(3.0, 0.9 * overlap)
        if score > 0:
            score += min(1.2, float(graph.degree(node)) * 0.03)
            scored.append((score, name))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [name for _score, name in scored[:18]]


def neighbors_context(graph: nx.MultiDiGraph, nodes: List[str], max_edges: int = 90) -> str:
    edges = []
    for node in nodes:
        for u, v, data in graph.edges(node, data=True):
            edges.append((u, data.get("label", ""), v))
        for u, v, data in graph.in_edges(node, data=True):
            edges.append((u, data.get("label", ""), v))
    dedup = []
    seen = set()
    for u, p, v in edges:
        key = (u, p, v)
        if key in seen:
            continue
        seen.add(key)
        dedup.append(key)
        if len(dedup) >= max_edges:
            break
    lines = [f"{u} -[{p}]-> {v}" for u, p, v in dedup]
    return "\n".join(lines)


def embed_chunks(embedder: SentenceTransformer, chunks: List[str]) -> np.ndarray:
    if not chunks:
        return np.array([])
    vectors = embedder.encode(chunks, show_progress_bar=False)
    return np.array(vectors)


def merge_embeddings(existing: np.ndarray, new_values: np.ndarray) -> np.ndarray:
    if existing is None or getattr(existing, "size", 0) == 0:
        return np.array(new_values) if new_values is not None else np.array([])
    if new_values is None or getattr(new_values, "size", 0) == 0:
        return np.array(existing)
    a = np.array(existing)
    b = np.array(new_values)
    if a.ndim == 1:
        a = a.reshape(1, -1)
    if b.ndim == 1:
        b = b.reshape(1, -1)
    try:
        return np.vstack([a, b])
    except Exception:
        return np.array(existing)


def top_k_chunks(question: str, chunks: List[str], embeddings: np.ndarray, embedder: SentenceTransformer, k: int = 4) -> List[str]:
    if not chunks or embeddings.size == 0:
        return []
    q_vec = embedder.encode([question], show_progress_bar=False)[0]
    denom = np.linalg.norm(embeddings, axis=1) * (np.linalg.norm(q_vec) + 1e-8)
    scores = (embeddings @ q_vec) / (denom + 1e-8)
    idxs = np.argsort(scores)[-k:][::-1]
    return [chunks[i] for i in idxs]


def top_k_chunks_with_ids(
    question: str,
    chunks: List[str],
    embeddings: np.ndarray,
    embedder: SentenceTransformer,
    k: int = 4,
) -> List[Tuple[int, str]]:
    if not chunks or embeddings.size == 0:
        return []
    q_vec = embedder.encode([question], show_progress_bar=False)[0]
    denom = np.linalg.norm(embeddings, axis=1) * (np.linalg.norm(q_vec) + 1e-8)
    scores = (embeddings @ q_vec) / (denom + 1e-8)
    idxs = np.argsort(scores)[-k:][::-1]
    return [(int(i), chunks[int(i)]) for i in idxs]


def lexical_top_k_chunks_with_ids(question: str, chunks: List[str], k: int = 8) -> List[Tuple[int, str, float]]:
    if not question or not chunks:
        return []
    q = question.lower()
    q_tokens = set(re.findall(r"[a-z0-9]{2,}", q))
    q_tokens = {t for t in q_tokens if t not in {"what", "which", "show", "give", "about", "from", "with", "this", "that", "the", "and"}}
    year_tokens = set(re.findall(r"(?:19|20)\\d{2}", q))
    quarter_tokens = set(re.findall(r"q[1-4]", q))
    scored: List[Tuple[float, int, str]] = []
    for i, text in enumerate(chunks):
        lower = str(text).lower()
        c_tokens = set(re.findall(r"[a-z0-9]{2,}", lower))
        overlap = len(q_tokens & c_tokens)
        phrase_bonus = 2.0 if len(q) > 10 and q in lower else 0.0
        year_bonus = 0.0
        if year_tokens and any(y in lower for y in year_tokens):
            year_bonus += 1.0
        if quarter_tokens and any(qt in lower for qt in quarter_tokens):
            year_bonus += 0.7
        score = float(overlap) + phrase_bonus + year_bonus
        if score > 0:
            scored.append((score, i, text))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [(i, text, score) for score, i, text in scored[:k]]


def hybrid_top_k_chunks_with_ids(
    question: str,
    chunks: List[str],
    embeddings: np.ndarray,
    embedder: SentenceTransformer,
    k: int = 8,
) -> List[Tuple[int, str, float]]:
    if not chunks:
        return []
    dense_n = max(10, k * 2)
    lexical_n = max(10, k * 2)
    dense_raw = top_k_chunks_with_ids(question, chunks, embeddings, embedder, k=dense_n)
    lex_raw = lexical_top_k_chunks_with_ids(question, chunks, k=lexical_n)
    score_map: Dict[int, float] = {}
    for rank, (idx, _text) in enumerate(dense_raw):
        score_map[idx] = score_map.get(idx, 0.0) + (2.3 * (dense_n - rank) / dense_n)
    for rank, (idx, _text, lex_score) in enumerate(lex_raw):
        score_map[idx] = score_map.get(idx, 0.0) + (1.6 * (lexical_n - rank) / lexical_n) + min(1.0, lex_score * 0.08)
    ranked = sorted(score_map.items(), key=lambda x: x[1], reverse=True)[:k]
    return [(idx, chunks[idx], float(score)) for idx, score in ranked]


def extract_chunk_citations(answer: str) -> List[str]:
    matches = re.findall(r"\[\s*chunk\s*([^\]]+?)\s*\]", str(answer or ""), flags=re.IGNORECASE)
    out: List[str] = []
    seen = set()
    for m in matches:
        val = str(m).strip()
        if not val:
            continue
        key = val.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(val)
    return out


def dedup_ci(values: List[object]) -> List[str]:
    out: List[str] = []
    seen = set()
    for item in values:
        val = str(item).strip()
        if not val:
            continue
        key = val.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(val)
    return out


def quick_grounding_check(
    answer: str,
    valid_chunk_ids: List[object],
    has_graph_context: bool,
) -> Dict[str, str]:
    text = str(answer or "").strip()
    if not text:
        return {"verdict": "retry", "issues": "empty_answer"}
    lower = text.lower()
    if "couldn't find" in lower or "not found" in lower:
        return {"verdict": "not_found", "issues": "model_not_found"}
    chunk_cites = extract_chunk_citations(text)
    has_graph_cite = bool(re.search(r"\[\s*graph\s*\]", text, flags=re.IGNORECASE))
    if not chunk_cites and not (has_graph_context and has_graph_cite):
        return {"verdict": "retry", "issues": "missing_citations"}
    valid_set = {str(x).strip().lower() for x in valid_chunk_ids}
    invalid = [c for c in chunk_cites if str(c).strip().lower() not in valid_set]
    if invalid:
        return {"verdict": "retry", "issues": "invalid_chunk_citations"}
    has_numbers = bool(re.search(r"(?<!\w)[\$€£]?[0-9][0-9,\.]*\s*(?:%|million|billion|m|bn)?", text, flags=re.IGNORECASE))
    if has_numbers and not chunk_cites and not has_graph_cite:
        return {"verdict": "retry", "issues": "numbers_without_citation"}
    return {"verdict": "pass", "issues": "rule_pass"}


def extract_visualization(text: str) -> Tuple[str, Dict[str, object]]:
    def parse_rows_from_raw(raw: str) -> List[Dict[str, object]]:
        rows: List[Dict[str, object]] = []
        pattern = re.compile(
            r'\{[^{}]*"label"\s*:\s*"([^"]+)"[^{}]*"value"\s*:\s*(?:"([^"]+)"|(-?\d+(?:\.\d+)?))[^{}]*\}',
            flags=re.DOTALL,
        )
        for m in pattern.finditer(str(raw or "")):
            label = str(m.group(1) or "").strip()
            if not label:
                continue
            if m.group(3) is not None:
                try:
                    value = float(m.group(3))
                except Exception:
                    value = m.group(3)
            else:
                value = str(m.group(2) or "").strip()
            rows.append({"label": label, "value": value})
        return rows

    def fallback_viz_from_raw(raw: str) -> Dict[str, object]:
        raw_text = str(raw or "")
        chart_type = "bar"
        title = "Visualization"
        mt = re.search(r'"type"\s*:\s*"([^"]+)"', raw_text, flags=re.IGNORECASE)
        if mt:
            chart_type = str(mt.group(1)).strip().lower() or "bar"
        mh = re.search(r'"title"\s*:\s*"([^"]+)"', raw_text, flags=re.IGNORECASE)
        if mh:
            title = str(mh.group(1)).strip() or "Visualization"
        rows = parse_rows_from_raw(raw_text)
        if not rows:
            return {}
        return {"type": chart_type, "title": title, "data": rows}

    def coerce_viz_payload(parsed_obj) -> Dict[str, object]:
        if isinstance(parsed_obj, dict):
            if isinstance(parsed_obj.get("data"), list):
                return parsed_obj
            if isinstance(parsed_obj.get("data"), dict):
                out = dict(parsed_obj)
                out["data"] = [parsed_obj.get("data")]
                if "type" not in out:
                    out["type"] = "bar"
                return out
            if isinstance(parsed_obj.get("values"), list):
                out = dict(parsed_obj)
                out["data"] = out.pop("values")
                if "type" not in out:
                    out["type"] = "bar"
                return out
            if parsed_obj.get("label") and parsed_obj.get("value"):
                return {"type": "bar", "title": "Visualization", "data": [parsed_obj]}
            return {}
        if isinstance(parsed_obj, list):
            rows = [x for x in parsed_obj if isinstance(x, dict)]
            if rows:
                return {"type": "bar", "title": "Visualization", "data": rows}
        return {}

    if not text:
        return "", {}
    match = re.search(r"<visualization>(.*?)</visualization>", text, flags=re.DOTALL | re.IGNORECASE)
    if match:
        raw = match.group(1)
        parsed = coerce_viz_payload(parse_json_any(raw))
        if not parsed:
            parsed = fallback_viz_from_raw(raw)
        cleaned = splice_text(text[: match.start()], text[match.end() :])
        return cleaned, parsed if isinstance(parsed, dict) else {}
    marker = re.search(r"<visualization>", text, flags=re.IGNORECASE)
    if not marker:
        label_match = re.search(
            r"(?:^|\n)\s*(?:\*\*)?\s*visuali[sz]ation\s*:?\s*(?:\*\*)?\s*",
            text,
            flags=re.IGNORECASE,
        )
        if not label_match:
            label_match = re.search(
                r"(?:\*\*)?\s*\bvisuali[sz]ation\b\s*:?\s*(?:\*\*)?\s*",
                text,
                flags=re.IGNORECASE,
            )
        if not label_match:
            return text, {}
        raw, _, raw_end = extract_first_json_value(text, label_match.end())
        if not raw:
            return text, {}
        parsed = coerce_viz_payload(parse_json_any(raw))
        if not parsed:
            parsed = fallback_viz_from_raw(raw)
        remove_end = extend_json_remove_end(text, raw_end)
        cleaned = splice_text(text[: label_match.start()], text[remove_end:])
        return cleaned, parsed if isinstance(parsed, dict) else {}
    raw, _, raw_end = extract_first_json_value(text, marker.end())
    if not raw:
        cleaned = splice_text(text[: marker.start()], text[marker.end() :])
        return cleaned, {}
    tail = text[raw_end:]
    close = re.match(r"\s*</visualization>", tail, flags=re.IGNORECASE)
    end = raw_end + (close.end() if close else 0)
    cleaned = splice_text(text[: marker.start()], text[end:])
    parsed = coerce_viz_payload(parse_json_any(raw))
    if not parsed:
        parsed = fallback_viz_from_raw(raw)
    return cleaned, parsed if isinstance(parsed, dict) else {}


def extract_hero_figures(text: str) -> Tuple[str, List[Dict[str, str]]]:
    def parse_figure_rows_from_raw(raw: str) -> List[Dict[str, str]]:
        figures_local: List[Dict[str, str]] = []
        pattern = re.compile(
            r'\{[^{}]*"label"\s*:\s*"([^"]+)"[^{}]*"value"\s*:\s*(?:"([^"]+)"|(-?\d+(?:\.\d+)?))'
            r'(?:[^{}]*"delta"\s*:\s*"([^"]*)")?'
            r'(?:[^{}]*"source"\s*:\s*(?:"([^"]*)"|\[\s*([^\]]+)\s*\]))?[^{}]*\}',
            flags=re.DOTALL,
        )
        for m in pattern.finditer(str(raw or "")):
            label = str(m.group(1) or "").strip()
            value = str((m.group(2) if m.group(2) is not None else m.group(3)) or "").strip()
            delta = str(m.group(4) or "").strip()
            source = str(m.group(5) or "").strip()
            if not source:
                src_bracket = str(m.group(6) or "").strip()
                if src_bracket:
                    source = f"[{src_bracket}]"
            if label and value:
                figures_local.append({"label": label, "value": value, "delta": delta, "source": source})
        return figures_local

    def parse_figure_rows(parsed_obj) -> List[Dict[str, str]]:
        rows = []
        if isinstance(parsed_obj, dict):
            if isinstance(parsed_obj.get("hero_figures"), list):
                rows = parsed_obj.get("hero_figures")
            elif isinstance(parsed_obj.get("figures"), list):
                rows = parsed_obj.get("figures")
            elif isinstance(parsed_obj.get("data"), list):
                rows = parsed_obj.get("data")
            elif parsed_obj.get("label") and parsed_obj.get("value"):
                rows = [parsed_obj]
            else:
                for k, v in parsed_obj.items():
                    if str(k).strip().lower() in {"hero_figures", "figures"} and isinstance(v, list):
                        rows = v
                        break
        elif isinstance(parsed_obj, list):
            rows = parsed_obj
        figures_local: List[Dict[str, str]] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            label = str(row.get("label", "")).strip()
            value = str(row.get("value", "")).strip()
            delta = str(row.get("delta", "")).strip()
            source_val = row.get("source", "")
            if isinstance(source_val, list):
                source = ", ".join([str(x).strip() for x in source_val if str(x).strip()])
            else:
                source = str(source_val).strip()
            if label and value:
                figures_local.append({"label": label, "value": value, "delta": delta, "source": source})
        return figures_local

    if not text:
        return "", []
    match = re.search(r"<hero_figures>(.*?)</hero_figures>", text, flags=re.DOTALL | re.IGNORECASE)
    if match:
        raw = match.group(1)
        parsed = parse_json_any(raw)
        figures = parse_figure_rows(parsed)
        if not figures:
            figures = parse_figure_rows_from_raw(raw)
        cleaned = splice_text(text[: match.start()], text[match.end() :])
        return cleaned, figures

    marker = re.search(r"<hero_figures>", text, flags=re.IGNORECASE)
    if marker:
        raw, _, raw_end = extract_first_json_value(text, marker.end())
        if raw:
            parsed = parse_json_any(raw)
            figures = parse_figure_rows(parsed)
            if not figures:
                figures = parse_figure_rows_from_raw(raw)
            tail = text[raw_end:]
            close = re.match(r"\s*</hero_figures>", tail, flags=re.IGNORECASE)
            end = raw_end + (close.end() if close else 0)
            cleaned = splice_text(text[: marker.start()], text[end:])
            return cleaned, figures

    label_match = re.search(
        r"(?:^|\n)\s*(?:\*\*)?\s*hero\s*figures\s*:?\s*(?:\*\*)?\s*",
        text,
        flags=re.IGNORECASE,
    )
    if not label_match:
        label_match = re.search(
            r"(?:\*\*)?\s*\bhero\s*figures\b\s*:?\s*(?:\*\*)?\s*",
            text,
            flags=re.IGNORECASE,
        )
    if not label_match:
        label_match = re.search(
            r"(?:\*\*)?\s*\bhero\s*figure\b\s*:?\s*(?:\*\*)?\s*",
            text,
            flags=re.IGNORECASE,
        )
    if label_match:
        raw, _, raw_end = extract_first_json_value(text, label_match.end())
        if raw:
            parsed = parse_json_any(raw)
            figures = parse_figure_rows(parsed)
            if not figures:
                figures = parse_figure_rows_from_raw(raw)
            remove_end = extend_json_remove_end(text, raw_end)
            cleaned = splice_text(text[: label_match.start()], text[remove_end:])
            return cleaned, figures

    return text, []


def extract_answer_artifacts(text: str) -> Tuple[str, Dict[str, object], List[Dict[str, str]]]:
    remaining = text or ""
    remaining, viz = extract_visualization(remaining)
    remaining, figures = extract_hero_figures(remaining)
    return remaining, viz, figures


def is_artifact_json_payload(parsed_obj) -> bool:
    if isinstance(parsed_obj, dict):
        keys = {str(k).strip().lower() for k in parsed_obj.keys()}
        if {"type", "title", "data"} & keys:
            return True
        if {"hero_figures", "figures"} & keys:
            return True
        if {"label", "value"} <= keys:
            return True
        return False
    if isinstance(parsed_obj, list):
        if not parsed_obj:
            return False
        dict_rows = [x for x in parsed_obj if isinstance(x, dict)]
        if dict_rows and len(dict_rows) == len(parsed_obj):
            for row in dict_rows[:4]:
                row_keys = {str(k).strip().lower() for k in row.keys()}
                if {"label", "value"} <= row_keys:
                    return True
                if {"source", "delta"} & row_keys:
                    return True
        return False
    return False


def strip_residual_json_artifacts(text: str) -> str:
    cleaned = str(text or "")
    if not cleaned:
        return ""

    # Remove artifact-like fenced JSON/JS blocks.
    def _strip_fenced(match: re.Match) -> str:
        body = str(match.group(1) or "").strip()
        parsed = parse_json_any(body)
        return "" if is_artifact_json_payload(parsed) else match.group(0)

    cleaned = re.sub(
        r"```(?:json|javascript|js|text)?\s*(.*?)```",
        _strip_fenced,
        cleaned,
        flags=re.DOTALL | re.IGNORECASE,
    )

    # Remove unfenced JSON artifacts that still leak into body text.
    cursor = 0
    scan_limit = 0
    while scan_limit < 12:
        raw, start, end = extract_first_json_value(cleaned, cursor)
        if not raw or start < 0 or end <= start:
            break
        parsed = parse_json_any(raw)
        if is_artifact_json_payload(parsed):
            cleaned = splice_text(cleaned[:start], cleaned[end:])
            cursor = max(0, start - 2)
        else:
            cursor = end
        scan_limit += 1

    return re.sub(r"\n{3,}", "\n\n", cleaned).strip()


def sanitize_assistant_payload(
    content: str,
    viz_existing: Dict[str, object],
    hero_existing: List[Dict[str, str]],
) -> Tuple[str, Dict[str, object], List[Dict[str, str]]]:
    cleaned = strip_think(str(content or "")).strip()
    viz = viz_existing if isinstance(viz_existing, dict) else {}
    figures = hero_existing if isinstance(hero_existing, list) else []

    # Run multiple passes to remove artifacts even when sections are repeated.
    for _ in range(3):
        next_clean, viz_found = extract_visualization(cleaned)
        cleaned = next_clean
        if not viz and viz_found:
            viz = viz_found
        next_clean, fig_found = extract_hero_figures(cleaned)
        cleaned = next_clean
        if not figures and fig_found:
            figures = fig_found

    # Remove dangling section headers if the model emitted labels without valid payload.
    cleaned = re.sub(
        r"(?:^|\n)\s*(?:\*\*)?\s*visuali[sz]ation\s*:?\s*(?:\*\*)?\s*$",
        "",
        cleaned,
        flags=re.IGNORECASE | re.MULTILINE,
    )
    cleaned = re.sub(
        r"(?:^|\n)\s*(?:\*\*)?\s*hero\s*figures?\s*:?\s*(?:\*\*)?\s*$",
        "",
        cleaned,
        flags=re.IGNORECASE | re.MULTILINE,
    )
    cleaned = strip_residual_json_artifacts(cleaned)
    # Defensive cleanup for partially emitted blocks that could not be parsed.
    cleaned = re.sub(
        r"(?:^|\n)\s*(?:\*\*)?\s*visuali[sz]ation\s*:?\s*(?:\*\*)?\s*```.*?(?:```|$)",
        "",
        cleaned,
        flags=re.IGNORECASE | re.DOTALL,
    )
    cleaned = re.sub(
        r"(?:^|\n)\s*(?:\*\*)?\s*hero\s*figures?\s*:?\s*(?:\*\*)?\s*```.*?(?:```|$)",
        "",
        cleaned,
        flags=re.IGNORECASE | re.DOTALL,
    )
    cleaned = re.sub(
        r"(?:^|\n)\s*(?:\*\*)?\s*visuali[sz]ation\s*:?\s*(?:\*\*)?\s*[\{\[].*?(?=\n\s*(?:\*\*)?\s*(?:hero\s*figures?|sources?|key\s*insights?)\b|$)",
        "",
        cleaned,
        flags=re.IGNORECASE | re.DOTALL,
    )
    cleaned = re.sub(
        r"(?:^|\n)\s*(?:\*\*)?\s*hero\s*figures?\s*:?\s*(?:\*\*)?\s*[\{\[].*?(?=\n\s*(?:\*\*)?\s*(?:sources?|notes?|disclaimer)\b|$)",
        "",
        cleaned,
        flags=re.IGNORECASE | re.DOTALL,
    )
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    return cleaned, viz, figures


def render_visualization(viz: Dict[str, object]) -> None:
    if not viz:
        return
    data = viz.get("data", [])
    if not isinstance(data, list) or not data:
        return
    rows = []
    numeric_rows = []
    for row in data:
        if not isinstance(row, dict):
            continue
        label = str(row.get("label", "")).strip()
        if not label:
            continue
        value_raw = row.get("value", "")
        value_text = str(value_raw).strip()
        number = None
        if isinstance(value_raw, (int, float)):
            number = float(value_raw)
        else:
            is_phone_like = bool(re.match(r"^\+?\d[\d\s\-]{7,}$", value_text))
            cleaned = re.sub(r"[^0-9.\-]", "", value_text.replace(",", ""))
            if (not is_phone_like) and cleaned and re.search(r"\d", cleaned):
                try:
                    number = float(cleaned)
                except Exception:
                    number = None
        rows.append({"label": label, "value_raw": value_text, "value": number})
        if number is not None:
            numeric_rows.append({"label": label, "value": number})
    if not rows:
        return
    chart_type = str(viz.get("type", "bar")).lower()
    title = str(viz.get("title", "")).strip()
    numeric_ratio = len(numeric_rows) / max(len(rows), 1)
    if len(numeric_rows) >= 2 and numeric_ratio >= 0.6:
        mark = "line" if chart_type == "line" else "bar"
        spec = {
            "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
            "data": {"values": numeric_rows},
            "mark": {"type": mark, "tooltip": True},
            "encoding": {
                "x": {"field": "label", "type": "nominal", "axis": {"labelColor": "#e9eefb"}},
                "y": {"field": "value", "type": "quantitative", "axis": {"labelColor": "#e9eefb"}},
            },
        }
        if title:
            spec["title"] = {"text": title, "color": "#e9eefb"}
        st.vega_lite_chart(spec, use_container_width=True)
    else:
        if title:
            st.markdown(f"**{title}**")
        st.table([{"Label": x["label"], "Value": x["value_raw"]} for x in rows])


def render_hero_figures(figures: List[Dict[str, str]]) -> None:
    if not figures:
        return
    top = figures[:4]
    st.markdown("**Hero figures**")
    cols = st.columns(len(top))
    for col, fig in zip(cols, top):
        label = fig.get("label", "")
        value = fig.get("value", "")
        delta = fig.get("delta", "")
        if delta:
            col.metric(label, value, delta)
        else:
            col.metric(label, value)
    sources = [f"{fig['label']}: {fig['source']}" for fig in top if fig.get("source")]
    if sources:
        st.caption("Sources: " + " | ".join(sources))


def format_enrichment_summary(enrichment: Dict[str, object]) -> str:
    if not enrichment:
        return ""
    intents = enrichment.get("insight_intents", [])
    entities = enrichment.get("key_entities", [])
    timeframe = str(enrichment.get("timeframe", "unspecified")).strip()
    viz_hint = str(enrichment.get("visualization_hint", "none")).strip()
    intent_txt = ", ".join([str(x) for x in intents[:2]]) if isinstance(intents, list) else ""
    entity_txt = ", ".join([str(x) for x in entities[:2]]) if isinstance(entities, list) else ""
    parts = []
    if intent_txt:
        parts.append(f"Intent: {intent_txt}")
    if entity_txt:
        parts.append(f"Focus: {entity_txt}")
    if timeframe and timeframe.lower() != "unspecified":
        parts.append(f"Timeframe: {timeframe}")
    if viz_hint and viz_hint.lower() != "none":
        parts.append(f"Viz: {viz_hint}")
    return " | ".join(parts)


def slim_context(context: str, max_chars: int = 1800, max_lines: int = 18) -> str:
    if not context:
        return ""
    lines = [ln.strip() for ln in context.splitlines() if ln.strip()]
    if len(lines) > max_lines:
        lines = lines[:max_lines]
    compact = "\n".join(lines)
    if len(compact) > max_chars:
        compact = compact[:max_chars]
    return compact


def approx_token_count(text: str) -> int:
    words = len(re.findall(r"\S+", str(text or "")))
    return max(1, int(words * 1.33))


def parse_comparison_anchors(question: str) -> List[str]:
    spec = parse_intent_spec(str(question or ""))
    return dedup_ci(list(spec.comparison_anchors or []))[:2]


def pack_answer_context(
    question: str,
    chunk_lines: List[str],
    graph_lines: List[str],
    max_tokens: int = 5200,
    anchors: Optional[List[str]] = None,
) -> str:
    resolved = [str(x).strip().lower() for x in (anchors or parse_comparison_anchors(question)) if str(x).strip()]
    temporal_re = re.compile(
        r"\b\d+\s+years?\s+(?:before|after)\b|\btakes place\b|\bset\b",
        flags=re.IGNORECASE,
    )

    scored_chunks: List[Tuple[float, str]] = []
    for line in chunk_lines:
        score = 0.0
        lower = str(line).lower()
        for anchor in resolved:
            if anchor and anchor in lower:
                score += 2.0
        if temporal_re.search(line):
            score += 1.5
        scored_chunks.append((score, line))

    scored_chunks.sort(key=lambda x: x[0], reverse=True)
    prioritized_chunks = [line for _score, line in scored_chunks]

    header_graph = "Graph edges (cite as [Graph]):"
    header_chunk = "Relevant chunks (cite as [Chunk <id>]):"
    lines: List[str] = [header_chunk]
    used_tokens = approx_token_count(header_chunk)

    for line in prioritized_chunks:
        t = approx_token_count(line)
        if used_tokens + t > max_tokens:
            break
        lines.append(line)
        used_tokens += t

    if used_tokens + approx_token_count(header_graph) <= max_tokens:
        lines.append("")
        lines.append(header_graph)
        used_tokens += approx_token_count(header_graph)

    for line in graph_lines:
        t = approx_token_count(line)
        if used_tokens + t > max_tokens:
            break
        lines.append(line)
        used_tokens += t

    return "\n".join(lines).strip()


def build_judge_context(
    answer_text: str,
    chunk_lines: List[str],
    graph_lines: List[str],
    fallback_context: str,
) -> str:
    cited_chunk_ids = {x.lower() for x in extract_chunk_citations(answer_text)}
    has_graph_cite = bool(re.search(r"\[\s*graph\s*\]", str(answer_text), flags=re.IGNORECASE))

    picked_chunks: List[str] = []
    if cited_chunk_ids:
        for line in chunk_lines:
            ids = [x.lower() for x in extract_chunk_citations(line)]
            if any(cid in cited_chunk_ids for cid in ids):
                picked_chunks.append(line)

    picked_graph = list(graph_lines[:28]) if has_graph_cite else []

    if not picked_chunks and not picked_graph:
        return slim_context(fallback_context, max_chars=3200, max_lines=30)

    block = []
    if picked_graph:
        block.append("Graph edges (cite as [Graph]):")
        block.extend(picked_graph)
    if picked_chunks:
        if block:
            block.append("")
        block.append("Relevant chunks (cite as [Chunk <id>]):")
        block.extend(picked_chunks[:24])
    return slim_context("\n".join(block), max_chars=3200, max_lines=40)


def citations_supported_in_context(answer_text: str, context_text: str) -> bool:
    cited = [x.lower() for x in extract_chunk_citations(answer_text)]
    if not cited:
        return False
    ctx_ids = {x.lower() for x in extract_chunk_citations(context_text)}
    return all(cid in ctx_ids for cid in cited)


def normalize_small_list(values, limit: int = 3) -> List[str]:
    out: List[str] = []
    if not isinstance(values, list):
        return out
    for item in values:
        val = str(item).strip()
        if not val:
            continue
        if val.lower() in {x.lower() for x in out}:
            continue
        out.append(val)
        if len(out) >= limit:
            break
    return out


def infer_timeframe(question: str) -> str:
    years = sorted(set(re.findall(r"\b(?:19|20)\d{2}\b", question)))
    quarters = [q.upper() for q in re.findall(r"\bq[1-4]\b", question, flags=re.IGNORECASE)]
    halves = [h.upper() for h in re.findall(r"\bh[12]\b", question, flags=re.IGNORECASE)]
    if years and len(years) > 1:
        return f"{years[0]}-{years[-1]}"
    if years and quarters:
        return f"{quarters[0]} {years[0]}"
    if years:
        return years[0]
    if quarters:
        return quarters[0]
    if halves:
        return halves[0]
    return "unspecified"


def infer_intents(question: str) -> List[str]:
    q = question.lower()
    intents: List[str] = []
    if any(k in q for k in ["summary", "summarize", "overview", "about this", "high level"]):
        intents.append("summarize_document")
    if any(k in q for k in ["ebit", "ebitda", "revenue", "profit", "margin", "volume", "carry", "container", "number", "how much"]):
        intents.append("extract_key_metrics")
    if any(k in q for k in ["compare", "vs", "versus", "difference", "higher", "lower", "increase", "decrease"]):
        intents.append("compare_values")
    if any(k in q for k in ["trend", "over time", "from", "to", "year", "quarter", "q1", "q2", "q3", "q4"]):
        intents.append("trend_check")
    if any(k in q for k in ["why", "driver", "reason", "cause"]):
        intents.append("driver_analysis")
    if not intents:
        intents.append("direct_lookup")
    return intents[:3]


def infer_entities(question: str, graph_hits: List[str]) -> List[str]:
    entities: List[str] = []
    for hit in graph_hits[:6]:
        val = str(hit).strip()
        if val and val.lower() not in {x.lower() for x in entities}:
            entities.append(val)
        if len(entities) >= 3:
            return entities
    tokens = re.findall(r"\b[A-Z][A-Za-z0-9&\.\-]{2,}\b", question)
    for tok in tokens:
        if tok.lower() in {"what", "which", "how", "full", "year"}:
            continue
        if tok.lower() not in {x.lower() for x in entities}:
            entities.append(tok)
        if len(entities) >= 3:
            break
    return entities[:3]


def build_deterministic_enrichment(
    question: str,
    graph_hits: List[str],
    intent_hint: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    hint = intent_hint if isinstance(intent_hint, dict) else {}
    intents = normalize_small_list(hint.get("intent_labels"), limit=3) if hint else infer_intents(question)
    if not intents:
        intents = infer_intents(question)
    timeframe = str(hint.get("timeframe", "")).strip() if hint else ""
    if not timeframe:
        timeframe = infer_timeframe(question)
    entities = infer_entities(question, graph_hits)
    anchors = normalize_small_list(hint.get("comparison_anchors"), limit=2) if hint else []
    for anchor in anchors:
        if anchor.lower() not in {x.lower() for x in entities}:
            entities.append(anchor)
    entities = entities[:4]

    should_visualize = bool(hint.get("should_visualize", ("trend_check" in intents) or ("compare_values" in intents)))
    viz_hint = "line" if "trend_check" in intents else ("bar" if should_visualize else "none")
    return {
        "enriched_question": str(hint.get("normalized_question", question)).strip(),
        "key_entities": entities,
        "insight_intents": intents,
        "timeframe": timeframe,
        "should_visualize": should_visualize,
        "visualization_hint": viz_hint,
        "comparison_anchors": anchors,
        "requires_derivation": bool(hint.get("requires_derivation", False)),
        "derivation_type": str(hint.get("derivation_type", "")).strip(),
        "canonical_retrieval_query": str(hint.get("canonical_retrieval_query", "")).strip(),
        "enrichment_mode": "deterministic_fast",
    }


def enrich_question(
    question: str,
    context: str,
    graph_hits: List[str],
    intent_hint: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    # Deterministic enrichment avoids model-latency spikes and always returns immediately.
    _ = slim_context(context, max_chars=400, max_lines=6)
    return build_deterministic_enrichment(question, graph_hits, intent_hint=intent_hint)


def should_use_enrichment(enrichment: Dict[str, object]) -> bool:
    if not enrichment:
        return False
    intents = enrichment.get("insight_intents", [])
    entities = enrichment.get("key_entities", [])
    timeframe = str(enrichment.get("timeframe", "unspecified")).lower()
    viz_hint = str(enrichment.get("visualization_hint", "none")).lower()
    if isinstance(intents, list) and any(i in {"compare_values", "trend_check", "driver_analysis", "extract_key_metrics", "summarize_document"} for i in intents):
        return True
    if isinstance(entities, list) and len(entities) > 0:
        return True
    if timeframe != "unspecified":
        return True
    if viz_hint in {"bar", "line"}:
        return True
    return False


def judge_answer(question: str, answer: str, context: str) -> Dict[str, str]:
    system = "Fast fact checker. Output strict JSON only."
    context = slim_context(context, max_chars=3200, max_lines=30)
    answer = answer[:1400]
    prompt = (
        "Judge the answer using only CONTEXT.\n"
        "Return exactly one JSON object:\n"
        "{\"verdict\":\"pass|retry|not_found\",\"issues\":\"short reason\"}\n"
        "Rules:\n"
        "- pass: answer fully supported by context.\n"
        "- retry: unsupported claim, contradiction, or missing citation.\n"
        "- not_found: context lacks required facts.\n"
        "- issues max 20 words.\n\n"
        f"QUESTION:\n{question}\n\n"
        f"ANSWER:\n{answer}\n\n"
        f"CONTEXT:\n{context}\n"
    )
    raw = call_ollama(
        prompt,
        system=system,
        model=SCANNER_MODEL,
        timeout=35,
        options={"temperature": 0, "num_predict": 90, "num_ctx": 3072},
    )
    if is_ollama_error(raw):
        return {"verdict": "retry", "issues": "judge_timeout"}
    parsed = parse_json_object(raw)
    if not parsed:
        lower = raw.lower()
        if "not_found" in lower or "not found" in lower:
            return {"verdict": "not_found", "issues": "judge_text_parse"}
        if "retry" in lower:
            return {"verdict": "retry", "issues": "judge_text_parse"}
        if "pass" in lower:
            return {"verdict": "pass", "issues": "judge_text_parse"}
        return {"verdict": "pass", "issues": "judge_parse_fallback"}
    verdict = str(parsed.get("verdict", "pass")).lower().strip()
    if verdict not in {"pass", "retry", "not_found"}:
        verdict = "pass"
    issues = str(parsed.get("issues", parsed.get("issue", ""))).strip()
    return {"verdict": verdict, "issues": issues}


def do_rerun() -> None:
    rerun_fn = getattr(st, "rerun", None)
    if rerun_fn is None:
        rerun_fn = getattr(st, "experimental_rerun")
    rerun_fn()


def set_status_bubble(message: str, kind: str = "processing") -> None:
    st.session_state.status_bubble = {
        "message": message,
        "kind": kind,
        "ts": time.time(),
    }


def render_status_bubble() -> None:
    bubble = st.session_state.get("status_bubble")
    if not bubble:
        return
    if time.time() - float(bubble.get("ts", 0)) > 2.2:
        st.session_state.status_bubble = None
        return
    kind = str(bubble.get("kind", "processing")).lower()
    css_kind = kind if kind in {"success", "processing", "warning"} else "processing"
    message = str(bubble.get("message", ""))
    if not message:
        return
    st.markdown(
        f'<div class="bubble {css_kind} auto-hide">{message}</div>',
        unsafe_allow_html=True,
    )


def inject_theme() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=Sora:wght@600;700&display=swap');
        :root {
          --bg: #0b1020;
          --panel: rgba(16, 22, 38, 0.85);
          --panel-strong: rgba(20, 28, 48, 0.95);
          --text: #e9eefb;
          --muted: #a8b0c2;
          --mint: #9fe7d3;
          --peach: #f7c6a0;
          --sky: #9ed2ff;
          --rose: #f6b4d5;
          --glow: rgba(159, 231, 211, 0.25);
        }
        html, body, [class*="css"] {
          font-family: 'Space Grotesk', sans-serif;
          color: var(--text);
        }
        .stApp {
          background:
            radial-gradient(1200px 800px at 10% 10%, rgba(158, 210, 255, 0.08), transparent 60%),
            radial-gradient(900px 700px at 90% 0%, rgba(247, 198, 160, 0.08), transparent 55%),
            radial-gradient(900px 900px at 80% 90%, rgba(246, 180, 213, 0.07), transparent 55%),
            var(--bg);
        }
        div[data-testid="stSidebar"] {
          background: linear-gradient(180deg, rgba(12, 17, 32, 0.98), rgba(10, 14, 26, 0.98));
          border-right: 1px solid rgba(160, 170, 200, 0.12);
        }
        .brand-dock {
          position: fixed;
          top: 10px;
          left: 16px;
          z-index: 9998;
          display: flex;
          align-items: center;
          gap: 10px;
          padding: 8px 12px 8px 8px;
          border-radius: 14px;
          backdrop-filter: blur(10px);
          background: rgba(12, 18, 34, 0.72);
          border: 1px solid rgba(160, 170, 200, 0.22);
          box-shadow: 0 14px 34px rgba(0, 0, 0, 0.35), 0 0 20px rgba(158, 210, 255, 0.2);
        }
        .brand-title {
          font-family: "Sora", sans-serif;
          font-size: 14px;
          letter-spacing: 0.3px;
          color: #e9eefb;
          font-weight: 700;
          line-height: 1.05;
        }
        .brand-sub {
          color: #a8b0c2;
          font-size: 11px;
          margin-top: 2px;
          letter-spacing: 0.35px;
        }
        .hero {
          padding: 20px 24px;
          margin-top: 26px;
          border-radius: 18px;
          background: linear-gradient(135deg, rgba(18, 26, 46, 0.9), rgba(12, 16, 30, 0.9));
          border: 1px solid rgba(160, 170, 200, 0.18);
          box-shadow: 0 20px 50px rgba(0, 0, 0, 0.35), 0 0 40px var(--glow);
          margin-bottom: 18px;
        }
        .hero h1 {
          font-family: "Sora", sans-serif;
          font-size: 30px;
          margin: 0 0 6px 0;
          letter-spacing: 0.2px;
        }
        .hero p {
          color: var(--muted);
          margin: 0;
        }
        .chip {
          display: inline-flex;
          align-items: center;
          gap: 8px;
          padding: 6px 12px;
          border-radius: 999px;
          font-size: 12px;
          background: rgba(20, 28, 48, 0.9);
          border: 1px solid rgba(160, 170, 200, 0.2);
          color: var(--muted);
          transition: all 0.3s ease;
        }
        .chip.active {
          color: #0e1a18;
          background: linear-gradient(135deg, var(--mint), var(--sky));
          border-color: transparent;
          box-shadow: 0 0 18px rgba(159, 231, 211, 0.45);
          animation: pulse 2.4s infinite;
        }
        .chip.ready {
          color: #20160d;
          background: linear-gradient(135deg, var(--peach), var(--rose));
          border-color: transparent;
          box-shadow: 0 0 18px rgba(247, 198, 160, 0.35);
        }
        .section-title {
          font-weight: 600;
          letter-spacing: 0.2px;
          color: var(--text);
          margin: 6px 0 10px 0;
        }
        .workspace-title {
          font-family: "Sora", sans-serif;
          font-size: 22px;
          font-weight: 700;
          margin: 2px 0 6px 0;
          letter-spacing: 0.1px;
        }
        .panel-card {
          background: linear-gradient(165deg, rgba(20, 28, 48, 0.84), rgba(12, 17, 30, 0.86));
          border: 1px solid rgba(160, 170, 200, 0.18);
          border-radius: 16px;
          padding: 14px 16px;
          box-shadow: 0 10px 26px rgba(0, 0, 0, 0.25);
        }
        .subtle {
          color: #9aa6bd;
          font-size: 12px;
          letter-spacing: 0.2px;
        }
        .active-note {
          color: #d5f9ef;
          text-shadow: 0 0 14px rgba(159, 231, 211, 0.35);
        }
        .muted {
          color: var(--muted);
          font-size: 13px;
        }
        [data-testid="stTabs"] [data-baseweb="tab-list"] {
          gap: 10px;
          background: rgba(12, 17, 32, 0.68);
          border: 1px solid rgba(160, 170, 200, 0.2);
          border-radius: 14px;
          padding: 6px;
          width: fit-content;
          box-shadow: 0 8px 20px rgba(0, 0, 0, 0.25);
        }
        [data-testid="stTabs"] [data-baseweb="tab"] {
          border-radius: 10px;
          color: #8f9ab0;
          font-weight: 600;
          padding: 8px 16px;
          transition: all 0.2s ease;
        }
        [data-testid="stTabs"] [aria-selected="true"] {
          background: linear-gradient(135deg, rgba(159, 231, 211, 0.18), rgba(158, 210, 255, 0.2));
          color: #e9eefb !important;
          border: 1px solid rgba(159, 231, 211, 0.3);
          box-shadow: 0 0 16px rgba(158, 210, 255, 0.25);
        }
        div[data-testid="stChatMessage"] {
          background: rgba(14, 20, 36, 0.8);
          border: 1px solid rgba(160, 170, 200, 0.16);
          border-radius: 14px;
          padding: 8px 12px;
          margin-bottom: 8px;
        }
        div[data-testid="stChatMessage"]:hover {
          border-color: rgba(159, 231, 211, 0.35);
        }
        .chat-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          gap: 12px;
          margin-bottom: 10px;
        }
        .float-badge {
          position: fixed;
          right: 26px;
          bottom: 26px;
          padding: 10px 14px;
          border-radius: 999px;
          background: linear-gradient(135deg, rgba(159, 231, 211, 0.18), rgba(158, 210, 255, 0.18));
          border: 1px solid rgba(159, 231, 211, 0.5);
          color: var(--text);
          box-shadow: 0 0 22px rgba(158, 210, 255, 0.4);
          z-index: 9999;
          animation: pulse 2.4s infinite;
        }
        .particles {
          position: fixed;
          inset: 0;
          overflow: hidden;
          z-index: 0;
          pointer-events: none;
        }
        .particle {
          position: absolute;
          width: 6px;
          height: 6px;
          background: rgba(158, 210, 255, 0.15);
          border-radius: 50%;
          filter: blur(0.5px);
          animation: drift 20s linear infinite;
        }
        .particle.p2 { width: 8px; height: 8px; background: rgba(159, 231, 211, 0.18); animation-duration: 24s; }
        .particle.p3 { width: 5px; height: 5px; background: rgba(246, 180, 213, 0.14); animation-duration: 18s; }
        @keyframes drift {
          0% { transform: translateY(0) translateX(0); opacity: 0.2; }
          50% { opacity: 0.6; }
          100% { transform: translateY(-120vh) translateX(30vw); opacity: 0.0; }
        }
        .glass-card {
          background: var(--panel);
          border: 1px solid rgba(160, 170, 200, 0.18);
          border-radius: 16px;
          padding: 14px 16px;
          box-shadow: 0 12px 30px rgba(0, 0, 0, 0.25);
        }
        .bubble {
          padding: 10px 14px;
          border-radius: 14px;
          background: rgba(20, 28, 48, 0.9);
          border: 1px solid rgba(160, 170, 200, 0.2);
          display: inline-block;
          animation: float-in 0.6s ease;
        }
        .bubble.auto-hide {
          animation: float-in 0.4s ease, fade-out 0.8s ease 1.4s forwards;
        }
        .bubble.success {
          background: linear-gradient(135deg, rgba(159, 231, 211, 0.2), rgba(158, 210, 255, 0.2));
          border-color: rgba(159, 231, 211, 0.5);
          box-shadow: 0 0 18px rgba(159, 231, 211, 0.2);
        }
        .bubble.processing {
          background: linear-gradient(135deg, rgba(247, 198, 160, 0.15), rgba(246, 180, 213, 0.18));
          border-color: rgba(247, 198, 160, 0.5);
          box-shadow: 0 0 18px rgba(247, 198, 160, 0.25);
        }
        .bubble.warning {
          background: linear-gradient(135deg, rgba(255, 211, 128, 0.2), rgba(247, 198, 160, 0.18));
          border-color: rgba(255, 211, 128, 0.5);
          box-shadow: 0 0 16px rgba(255, 211, 128, 0.2);
        }
        @keyframes pulse {
          0% { box-shadow: 0 0 18px rgba(159, 231, 211, 0.35); }
          50% { box-shadow: 0 0 30px rgba(159, 231, 211, 0.6); }
          100% { box-shadow: 0 0 18px rgba(159, 231, 211, 0.35); }
        }
        @keyframes float-in {
          from { transform: translateY(6px); opacity: 0.0; }
          to { transform: translateY(0); opacity: 1.0; }
        }
        @keyframes fade-out {
          to { opacity: 0.0; transform: translateY(-4px); height: 0; margin: 0; padding: 0; border: 0; }
        }
        .stButton > button {
          background: linear-gradient(135deg, rgba(158, 210, 255, 0.25), rgba(159, 231, 211, 0.2));
          border: 1px solid rgba(160, 170, 200, 0.35);
          color: var(--text);
          border-radius: 12px;
          transition: transform 0.15s ease, box-shadow 0.2s ease;
        }
        .stButton > button:hover {
          transform: translateY(-1px);
          box-shadow: 0 10px 24px rgba(0, 0, 0, 0.25);
        }
        .stTextInput input, .stTextArea textarea {
          background: rgba(10, 14, 26, 0.7);
          border: 1px solid rgba(160, 170, 200, 0.25);
          border-radius: 12px;
          color: var(--text);
        }
        div[data-testid="stProgress"] > div > div > div {
          background: linear-gradient(90deg, var(--mint), var(--sky), var(--rose));
          box-shadow: 0 0 18px rgba(158, 210, 255, 0.5);
        }
        code, pre {
          background: rgba(10, 14, 26, 0.85) !important;
          border: 1px solid rgba(160, 170, 200, 0.2) !important;
          border-radius: 12px !important;
        }
        @media (max-width: 900px) {
          .brand-dock {
            left: 8px;
            top: 8px;
            transform: scale(0.9);
            transform-origin: left top;
          }
          .hero {
            margin-top: 54px;
          }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def ensure_rag_store() -> str:
    os.makedirs(RAG_STORE_DIR, exist_ok=True)
    return RAG_STORE_DIR


def format_disk_size(num_bytes: int) -> str:
    value = float(num_bytes or 0)
    for unit in ["B", "KB", "MB", "GB"]:
        if value < 1024 or unit == "GB":
            return f"{value:.2f} {unit}"
        value /= 1024
    return f"{value:.2f} GB"


def normalize_graph_name(name: str, fallback: str = "Untitled Graph") -> str:
    clean = re.sub(r"\s+", " ", str(name or "")).strip()
    return clean[:120] if clean else fallback


def ellipsize(text: str, max_chars: int = 44) -> str:
    value = re.sub(r"\s+", " ", str(text or "")).strip()
    if len(value) <= max_chars:
        return value
    if max_chars <= 3:
        return value[:max_chars]
    return value[: max_chars - 3].rstrip() + "..."


def graph_name_to_slug(name: str) -> str:
    value = normalize_graph_name(name, fallback="graph").lower()
    value = re.sub(r"[^a-z0-9]+", "-", value).strip("-")
    return value[:72] if value else "graph"


def choose_new_rag_id(graph_name: str) -> str:
    base = graph_name_to_slug(graph_name)
    existing = {str(m.get("rag_id", "")) for m in list_saved_rags() if m.get("rag_id")}
    if base not in existing:
        return base
    i = 2
    while f"{base}-{i}" in existing:
        i += 1
    return f"{base}-{i}"


def choose_unique_rag_id(base_id: str) -> str:
    clean = graph_name_to_slug(base_id)
    if not clean:
        clean = "graph"
    existing = {str(m.get("rag_id", "")) for m in list_saved_rags() if m.get("rag_id")}
    if clean not in existing:
        return clean
    i = 2
    while f"{clean}-{i}" in existing:
        i += 1
    return f"{clean}-{i}"


def load_settings() -> Dict[str, object]:
    ensure_rag_store()
    if not os.path.exists(SETTINGS_PATH):
        return {}
    try:
        with open(SETTINGS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def save_settings(settings: Dict[str, object]) -> None:
    ensure_rag_store()
    with open(SETTINGS_PATH, "w", encoding="utf-8") as f:
        json.dump(settings, f, indent=2)


def save_rag_to_disk(
    rag_id: str,
    pdf_name: str,
    graph: nx.MultiDiGraph,
    triples: List[Tuple[str, str, str]],
    chunks: List[str],
    embeddings: np.ndarray,
    meta_extra: Dict[str, str],
    previous_meta: Optional[Dict[str, object]] = None,
    source_file: str = "",
    source_files: Optional[List[str]] = None,
    facts: Optional[List[object]] = None,
    extraction_chunks: Optional[List[object]] = None,
    retrieval_chunks_records: Optional[List[object]] = None,
    retrieval_index: Optional[object] = None,
    build_profile: str = "balanced",
    eval_report: Optional[Dict[str, object]] = None,
    timings: Optional[Dict[str, float]] = None,
) -> Dict[str, object]:
    ensure_rag_store()
    rag_dir = os.path.join(RAG_STORE_DIR, rag_id)
    os.makedirs(rag_dir, exist_ok=True)

    now = datetime.utcnow().isoformat() + "Z"
    prev = previous_meta if isinstance(previous_meta, dict) else {}
    existing_sources = prev.get("source_files", [])
    if not isinstance(existing_sources, list):
        existing_sources = []
    if not existing_sources and prev.get("pdf_name"):
        existing_sources = [str(prev.get("pdf_name"))]
    incoming_sources: List[str] = []
    if isinstance(source_files, list):
        incoming_sources.extend([str(x).strip() for x in source_files if str(x).strip()])
    source_candidate = source_file.strip() if isinstance(source_file, str) else ""
    if source_candidate:
        incoming_sources.append(source_candidate)
    if not incoming_sources:
        incoming_sources.append(pdf_name)
    all_sources = list(existing_sources) + incoming_sources
    dedup_sources: List[str] = []
    seen_sources = set()
    for item in all_sources:
        val = str(item).strip()
        if not val:
            continue
        key = val.lower()
        if key in seen_sources:
            continue
        seen_sources.add(key)
        dedup_sources.append(val)

    # v2-first persistence path.
    manifest = {}
    if facts is not None and retrieval_index is not None and extraction_chunks is not None and retrieval_chunks_records is not None:
        manifest = save_v2_artifacts(
            rag_id=rag_id,
            graph_name=str(prev.get("pdf_name", pdf_name)),
            graph=graph,
            facts=facts,
            extraction_chunks=extraction_chunks,
            retrieval_chunks=retrieval_chunks_records,
            retrieval_index=retrieval_index,
            build_profile=str(build_profile or "balanced").lower(),
            models={
                "scanner": SCANNER_MODEL,
                "vision": VISION_MODEL,
                "brain": BRAIN_MODEL,
                "embed": EMBED_MODEL,
            },
            source_files=dedup_sources,
            timings=timings or {},
            quality_metrics=eval_report or {},
        )

    # Keep a lightweight root metadata file for quick listing/export/import UX.
    meta = {
        "rag_id": rag_id,
        "pdf_name": str(prev.get("pdf_name", pdf_name)),
        "created_at": str(prev.get("created_at", now)),
        "updated_at": str((manifest or {}).get("updated_at", now)),
        "scanner_model": SCANNER_MODEL,
        "vision_model": VISION_MODEL,
        "brain_model": BRAIN_MODEL,
        "embed_model": EMBED_MODEL,
        "source_files": dedup_sources,
        "source_file_count": len(dedup_sources),
        "format_version": "2.0",
        "v2_dir": os.path.relpath(ensure_v2_dir(rag_id), rag_dir),
        "edges": int(graph.number_of_edges()),
        "nodes": int(graph.number_of_nodes()),
    }
    if manifest:
        meta["quality_metrics"] = manifest.get("quality_metrics", {})
        meta["counts"] = manifest.get("counts", {})
    meta.update(meta_extra)
    with open(os.path.join(rag_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    return meta


def _rag_required_files() -> List[str]:
    return [
        "meta.json",
        "v2/manifest.json",
        "v2/graph.json",
        "v2/facts.jsonl",
        "v2/chunks.jsonl",
        "v2/embeddings.npy",
    ]


def export_rag_archive(rag_id: str) -> Tuple[str, bytes]:
    ensure_rag_store()
    rag_dir = os.path.join(RAG_STORE_DIR, rag_id)
    if not os.path.isdir(rag_dir):
        raise FileNotFoundError(f"Graph '{rag_id}' was not found on disk.")
    missing = [fn for fn in _rag_required_files() if not os.path.exists(os.path.join(rag_dir, fn))]
    if missing:
        raise ValueError(f"Cannot export graph. Missing files: {', '.join(missing)}")

    with open(os.path.join(rag_dir, "meta.json"), "r", encoding="utf-8") as f:
        meta = json.load(f)
    pdf_name = normalize_graph_name(str(meta.get("pdf_name", rag_id)), fallback=rag_id)
    now_tag = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    filename = f"{graph_name_to_slug(pdf_name) or rag_id}-{now_tag}.opengrwm.zip"

    archive_bytes = io.BytesIO()
    with zipfile.ZipFile(archive_bytes, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for root, _dirs, files in os.walk(rag_dir):
            for fn in files:
                src = os.path.join(root, fn)
                arcname = os.path.relpath(src, rag_dir)
                if str(arcname).lower().endswith(".zip"):
                    continue
                zf.write(src, arcname=arcname)
    return filename, archive_bytes.getvalue()


def import_rag_archive(archive_bytes: bytes, archive_name: str = "") -> Tuple[str, Dict[str, object]]:
    ensure_rag_store()
    if not archive_bytes:
        raise ValueError("Imported archive is empty.")

    with tempfile.TemporaryDirectory() as tmpdir:
        archive_path = os.path.join(tmpdir, "graph_import.zip")
        with open(archive_path, "wb") as f:
            f.write(archive_bytes)

        with zipfile.ZipFile(archive_path, "r") as zf:
            for member in zf.infolist():
                name = member.filename
                if os.path.isabs(name) or ".." in name.replace("\\", "/").split("/"):
                    raise ValueError("Archive contains unsafe paths.")
            zf.extractall(tmpdir)

        candidates = [tmpdir]
        for name in os.listdir(tmpdir):
            d = os.path.join(tmpdir, name)
            if os.path.isdir(d):
                candidates.append(d)

        source_dir = ""
        required = _rag_required_files()
        for candidate in candidates:
            if all(os.path.exists(os.path.join(candidate, fn)) for fn in required):
                source_dir = candidate
                break
        if not source_dir:
            raise ValueError("Invalid graph archive format. Required files are missing.")

        with open(os.path.join(source_dir, "meta.json"), "r", encoding="utf-8") as f:
            meta = json.load(f)
        preferred_rag_id = str(meta.get("rag_id", "")).strip()
        if not preferred_rag_id:
            base_from_name = os.path.splitext(os.path.basename(str(archive_name or "imported-graph")))[0]
            preferred_rag_id = graph_name_to_slug(base_from_name) or "imported-graph"
        rag_id = choose_unique_rag_id(preferred_rag_id)

        target_dir = os.path.join(RAG_STORE_DIR, rag_id)
        os.makedirs(target_dir, exist_ok=True)
        for root, _dirs, files in os.walk(source_dir):
            rel_root = os.path.relpath(root, source_dir)
            dst_root = target_dir if rel_root == "." else os.path.join(target_dir, rel_root)
            os.makedirs(dst_root, exist_ok=True)
            for fn in files:
                src = os.path.join(root, fn)
                if os.path.basename(src) == os.path.basename(archive_path) or str(fn).lower().endswith(".zip"):
                    continue
                shutil.copy2(src, os.path.join(dst_root, fn))

        with open(os.path.join(target_dir, "meta.json"), "r", encoding="utf-8") as f:
            saved_meta = json.load(f)
        now = datetime.utcnow().isoformat() + "Z"
        saved_meta["rag_id"] = rag_id
        saved_meta["updated_at"] = now
        if not saved_meta.get("created_at"):
            saved_meta["created_at"] = now
        if not saved_meta.get("pdf_name"):
            saved_meta["pdf_name"] = normalize_graph_name(rag_id, fallback=rag_id)
        with open(os.path.join(target_dir, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(saved_meta, f, indent=2)

    return rag_id, saved_meta


def list_saved_rags() -> List[Dict[str, str]]:
    ensure_rag_store()
    items = []
    for name in os.listdir(RAG_STORE_DIR):
        rag_dir = os.path.join(RAG_STORE_DIR, name)
        meta_path = os.path.join(rag_dir, "meta.json")
        v2_manifest = os.path.join(rag_dir, "v2", "manifest.json")
        if not os.path.isdir(rag_dir) or not os.path.exists(meta_path) or not os.path.exists(v2_manifest):
            continue
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            total_size = 0
            for root, _dirs, files in os.walk(rag_dir):
                for fn in files:
                    try:
                        total_size += os.path.getsize(os.path.join(root, fn))
                    except Exception:
                        continue
            meta["disk_bytes"] = total_size
            items.append(meta)
        except Exception:
            continue
    items.sort(key=lambda x: x.get("updated_at", x.get("created_at", "")), reverse=True)
    return items


def load_rag_from_disk(rag_id: str) -> Dict[str, object]:
    rag_dir = os.path.join(RAG_STORE_DIR, rag_id)
    root_meta = {}
    meta_path = os.path.join(rag_dir, "meta.json")
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            try:
                root_meta = json.load(f)
            except Exception:
                root_meta = {}

    data = load_v2_artifacts(rag_id)
    manifest = data.get("manifest", {}) if isinstance(data.get("manifest", {}), dict) else {}
    meta = dict(root_meta or {})
    if not meta:
        meta = {
            "rag_id": rag_id,
            "pdf_name": manifest.get("graph_name", rag_id),
            "created_at": manifest.get("created_at", ""),
            "updated_at": manifest.get("updated_at", ""),
            "format_version": "2.0",
        }
    if not meta.get("pdf_name"):
        meta["pdf_name"] = manifest.get("graph_name", rag_id)
    meta["quality_metrics"] = manifest.get("quality_metrics", {})
    meta["counts"] = manifest.get("counts", {})
    retrieval_index = data["retrieval_index"]
    return {
        "graph": data["graph"],
        "triples": data["triples"],
        "chunks": list(retrieval_index.chunk_texts),
        "embeddings": retrieval_index.embeddings,
        "meta": meta,
        "facts": data["facts"],
        "retrieval_chunks": data["retrieval_chunks"],
        "extraction_chunks": data["extraction_chunks"],
        "retrieval_index": retrieval_index,
        "eval_report": data.get("eval_report", {}),
    }


def delete_rag_from_disk(rag_id: str) -> None:
    rag_dir = os.path.join(RAG_STORE_DIR, rag_id)
    if os.path.isdir(rag_dir):
        shutil.rmtree(rag_dir)


def load_rag_into_session(rag_id: str, data: Dict[str, object]) -> None:
    st.session_state.graph = data["graph"]
    st.session_state.triples = data["triples"]
    st.session_state.chunks = data["chunks"]
    st.session_state.embeddings = data["embeddings"]
    st.session_state.v2_facts = list(data.get("facts", []) or [])
    st.session_state.v2_retrieval_chunks = list(data.get("retrieval_chunks", []) or [])
    st.session_state.v2_extraction_chunks = list(data.get("extraction_chunks", []) or [])
    st.session_state.v2_retrieval_index = data.get("retrieval_index")
    st.session_state.v2_eval_report = dict(data.get("eval_report", {}) or {})
    if "embedder" not in st.session_state:
        st.session_state.embedder = SentenceTransformer(EMBED_MODEL)
    st.session_state.graph_ready = len(st.session_state.graph.nodes) > 0
    st.session_state.building = False
    st.session_state.current_rag_id = rag_id
    st.session_state.current_meta = data.get("meta", {})
    st.session_state.current_pdf_name = data["meta"].get("pdf_name", rag_id)


def clear_session_data() -> None:
    keys = [
        "graph",
        "triples",
        "chunks",
        "embeddings",
        "build_tasks",
        "build_index",
        "build_logs",
        "build_total",
        "graph_ready",
        "building",
        "stop_build",
        "current_rag_id",
        "current_meta",
        "current_pdf_name",
        "save_pending",
        "build_target_rag_id",
        "build_graph_name",
        "build_source_file",
        "build_source_files",
        "build_source_type",
        "build_source_url",
        "build_operation",
        "build_previous_meta",
        "build_mode_used",
        "build_use_vision",
        "build_vision_text_limit",
        "build_text_batch_pages",
        "build_max_vision_pages",
        "page_texts",
        "pdf_hash",
        "image_pages",
        "text_lengths",
        "total_pages",
        "vision_pages",
        "web_preview",
        "chat_history",
        "chat_focus",
        "pending_query",
        "pending_enrichment",
        "v2_facts",
        "v2_retrieval_chunks",
        "v2_extraction_chunks",
        "v2_retrieval_index",
        "v2_eval_report",
        "v2_build_facts",
        "v2_build_extraction_chunks",
        "v2_build_retrieval_chunks",
        "v2_build_rag_id",
        "v2_build_profile",
        "v2_build_started_at",
        "v2_build_extract_sec",
    ]
    for k in keys:
        if k in st.session_state:
            del st.session_state[k]


def inject_external_styles() -> None:
    css_blob = ""
    for candidate in [LOCAL_STYLE_PATH, EXTERNAL_STYLE_PATH]:
        if not candidate:
            continue
        try:
            with open(candidate, "r", encoding="utf-8") as f:
                css_blob = f.read()
            if css_blob.strip():
                break
        except Exception:
            css_blob = ""
    if css_blob:
        st.markdown("<style>" + css_blob + "</style>", unsafe_allow_html=True)
    extra_css = """
    <style>
    .module-card {
      background: rgba(14,17,23,0.78);
      border: 1px solid rgba(255,255,255,0.08);
      border-radius: 10px;
      padding: 12px;
      margin-bottom: 10px;
    }
    .module-title {
      color: #00ffff;
      font-size: 0.80rem;
      letter-spacing: 0.12rem;
      font-weight: 700;
      margin-bottom: 8px;
      text-transform: uppercase;
    }
    .deck-title {
      color: #d7e3f6;
      font-size: 1.35rem;
      font-weight: 700;
      margin: 0 0 10px 0;
    }
    .data-pills {
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      margin-top: 8px;
    }
    .data-pill {
      border: 1px solid #4d4dff;
      color: #dfefff;
      border-radius: 999px;
      padding: 4px 10px;
      font-size: 0.78rem;
      box-shadow: 0 0 10px rgba(0,255,255,0.22);
      background: rgba(10,12,18,0.85);
    }
    .terminal-output pre {
      margin: 0;
      white-space: pre-wrap;
      line-height: 1.35;
      animation: termflicker 2.6s ease-in-out infinite;
    }
    @keyframes termflicker {
      0%, 100% { opacity: 0.95; }
      50% { opacity: 1.0; }
    }
    .user-row {
      display: flex;
      justify-content: flex-end;
      margin-bottom: 12px;
      width: 100%;
    }
    .sources-list {
      font-family: 'JetBrains Mono', monospace;
      color: #a8d8ff;
      font-size: 0.9rem;
      line-height: 1.45;
    }
    section[data-testid="stSidebar"] .sessions-divider {
      border-top: 1px solid rgba(140, 151, 181, 0.28);
      margin: 7px 0 4px 0;
      height: 0;
    }
    section[data-testid="stSidebar"] div[data-testid="element-container"] {
      margin: 0 0 0.10rem 0 !important;
      padding: 0 !important;
    }
    section[data-testid="stSidebar"] div[data-testid="stHorizontalBlock"] {
      gap: 0.28rem !important;
      align-items: center !important;
      margin: 0 !important;
      padding: 0 !important;
    }
    section[data-testid="stSidebar"] div[data-testid="column"] {
      padding: 0 !important;
      margin: 0 !important;
    }
    section[data-testid="stSidebar"] .stButton {
      margin: 0 !important;
      padding: 0 !important;
    }
    section[data-testid="stSidebar"] .stButton > button {
      min-height: 30px !important;
      height: 30px !important;
      border-radius: 9px !important;
      padding: 0 9px !important;
      font-size: 0.77rem !important;
      line-height: 1 !important;
      font-weight: 600 !important;
      white-space: nowrap !important;
      overflow: hidden !important;
      text-overflow: ellipsis !important;
    }
    section[data-testid="stSidebar"] .stButton > button p,
    section[data-testid="stSidebar"] .stButton > button span {
      white-space: nowrap !important;
      overflow: hidden !important;
      text-overflow: ellipsis !important;
      line-height: 1 !important;
    }
    section[data-testid="stSidebar"] .stButton > button[kind="secondary"] {
      background: rgba(12, 18, 30, 0.72) !important;
      border: 1px solid rgba(88, 103, 153, 0.35) !important;
      color: #c8d8f2 !important;
      justify-content: flex-start !important;
      text-align: left !important;
    }
    section[data-testid="stSidebar"] .stButton > button[kind="secondary"]:hover {
      border-color: rgba(94, 234, 255, 0.45) !important;
      color: #f1f8ff !important;
      background: rgba(13, 24, 39, 0.88) !important;
    }
    section[data-testid="stSidebar"] .stButton > button[kind="primary"] {
      background: linear-gradient(90deg, rgba(22, 58, 113, 0.95), rgba(20, 95, 122, 0.85)) !important;
      border: 1px solid rgba(76, 200, 255, 0.75) !important;
      color: #dff8ff !important;
      box-shadow: 0 0 16px rgba(0, 234, 255, 0.18) !important;
      justify-content: flex-start !important;
      text-align: left !important;
    }
    section[data-testid="stSidebar"] .stButton > button[aria-label="✕"] {
      min-width: 30px !important;
      width: 30px !important;
      max-width: 30px !important;
      justify-content: center !important;
      text-align: center !important;
      padding: 0 !important;
      opacity: 0 !important;
      pointer-events: none !important;
      transition: opacity 140ms ease !important;
      background: rgba(13, 19, 33, 0.78) !important;
      border: 1px solid rgba(88, 103, 153, 0.35) !important;
      color: #9bb9d8 !important;
    }
    section[data-testid="stSidebar"] div[data-testid="stHorizontalBlock"]:hover button[aria-label="✕"],
    section[data-testid="stSidebar"] button[aria-label="✕"]:focus {
      opacity: 1 !important;
      pointer-events: auto !important;
    }
    section[data-testid="stSidebar"] .stButton > button[aria-label="✕"]:hover {
      border-color: rgba(255, 118, 118, 0.7) !important;
      background: rgba(45, 17, 25, 0.82) !important;
      color: #ffffff !important;
    }
    </style>
    """
    st.markdown(extra_css, unsafe_allow_html=True)


def load_chat_sessions() -> List[Dict[str, object]]:
    ensure_rag_store()
    if not os.path.exists(CHAT_SESSIONS_PATH):
        return []
    try:
        with open(CHAT_SESSIONS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        sessions = data if isinstance(data, list) else []
    except Exception:
        return []
    cleaned: List[Dict[str, object]] = []
    for item in sessions:
        if not isinstance(item, dict):
            continue
        sid = str(item.get("id", "")).strip()
        if not sid:
            continue
        messages = item.get("messages", [])
        if not isinstance(messages, list):
            messages = []
        cleaned.append(
            {
                "id": sid,
                "title": str(item.get("title", "New Session")).strip() or "New Session",
                "created_at": str(item.get("created_at", datetime.utcnow().isoformat() + "Z")),
                "updated_at": str(item.get("updated_at", item.get("created_at", datetime.utcnow().isoformat() + "Z"))),
                "messages": messages,
            }
        )
    cleaned.sort(key=lambda x: x.get("updated_at", ""), reverse=True)
    return cleaned


def save_chat_sessions(sessions: List[Dict[str, object]]) -> None:
    ensure_rag_store()
    payload = sessions if isinstance(sessions, list) else []
    with open(CHAT_SESSIONS_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def create_chat_session(title: str = "New Session") -> Dict[str, object]:
    now = datetime.utcnow().isoformat() + "Z"
    return {
        "id": str(uuid.uuid4()),
        "title": normalize_graph_name(title, fallback="New Session"),
        "created_at": now,
        "updated_at": now,
        "messages": [],
    }


def ensure_chat_state() -> None:
    if "chat_sessions" not in st.session_state:
        st.session_state.chat_sessions = load_chat_sessions()
    if "active_chat_id" not in st.session_state:
        if st.session_state.chat_sessions:
            st.session_state.active_chat_id = st.session_state.chat_sessions[0]["id"]
        else:
            new_session = create_chat_session()
            st.session_state.chat_sessions = [new_session]
            st.session_state.active_chat_id = new_session["id"]
            save_chat_sessions(st.session_state.chat_sessions)
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    active = get_active_chat_session()
    if active and st.session_state.get("_chat_loaded_for") != active["id"]:
        st.session_state.chat_history = copy.deepcopy(active.get("messages", []))
        st.session_state._chat_loaded_for = active["id"]


def get_active_chat_session() -> Optional[Dict[str, object]]:
    sid = st.session_state.get("active_chat_id", "")
    sessions = st.session_state.get("chat_sessions", [])
    for sess in sessions:
        if sess.get("id") == sid:
            return sess
    return sessions[0] if sessions else None


def persist_active_chat_history() -> None:
    sessions = st.session_state.get("chat_sessions", [])
    sid = st.session_state.get("active_chat_id", "")
    if not sessions or not sid:
        return
    now = datetime.utcnow().isoformat() + "Z"
    for sess in sessions:
        if sess.get("id") != sid:
            continue
        messages = copy.deepcopy(st.session_state.get("chat_history", []))
        sess["messages"] = messages
        sess["updated_at"] = now
        if sess.get("title", "New Session") == "New Session":
            for msg in messages:
                if msg.get("role") == "user" and str(msg.get("content", "")).strip():
                    sess["title"] = normalize_graph_name(str(msg.get("content"))[:64], fallback="New Session")
                    break
        break
    sessions.sort(key=lambda x: x.get("updated_at", ""), reverse=True)
    st.session_state.chat_sessions = sessions
    save_chat_sessions(sessions)


def extract_insight_bullets(text: str) -> List[str]:
    lines = [ln.strip() for ln in str(text or "").splitlines() if ln.strip()]
    bullets: List[str] = []
    for ln in lines:
        if ln.startswith("- ") or ln.startswith("* ") or ln.startswith("•"):
            bullets.append(ln.lstrip("-*• ").strip())
    return bullets[:8]


def extract_main_answer(text: str) -> str:
    body = str(text or "").strip()
    if not body:
        return ""
    cut = re.split(r"\n\s*(?:key insights?|insights?)\s*:\s*", body, flags=re.IGNORECASE)
    return cut[0].strip() if cut else body


def extract_sources_from_message(msg: Dict[str, object]) -> List[str]:
    raw = str(msg.get("content", ""))
    cites = re.findall(r"\[(?:Chunk\s*[^\]]+|Graph)\]", raw, flags=re.IGNORECASE)
    for fig in msg.get("hero_figures", []) or []:
        src = str(fig.get("source", "")).strip()
        if src:
            cites.append(src)
    out: List[str] = []
    seen = set()
    for c in cites:
        key = c.lower().strip()
        if key in seen:
            continue
        seen.add(key)
        out.append(c.strip())
    return out


def render_plotly_visualization(viz: Dict[str, object]) -> None:
    if not viz:
        st.info("No visualization data returned.")
        return
    data = viz.get("data", [])
    if not isinstance(data, list) or not data:
        st.info("No visualization data returned.")
        return
    labels: List[str] = []
    values: List[float] = []
    for row in data:
        if not isinstance(row, dict):
            continue
        label = str(row.get("label", "")).strip()
        val_raw = row.get("value", "")
        if not label:
            continue
        num = None
        if isinstance(val_raw, (int, float)):
            num = float(val_raw)
        else:
            txt = str(val_raw).replace(",", "")
            m = re.search(r"-?\d+(?:\.\d+)?", txt)
            if m:
                try:
                    num = float(m.group(0))
                except Exception:
                    num = None
        if num is None:
            continue
        labels.append(label)
        values.append(num)
    if len(values) < 2:
        render_visualization(viz)
        return
    chart_type = str(viz.get("type", "bar")).lower()
    if chart_type == "line":
        fig = go.Figure(
            data=[
                go.Scatter(
                    x=labels,
                    y=values,
                    mode="lines+markers",
                    line={"color": "#00ffff", "width": 3},
                    marker={"color": "#4d4dff", "size": 8},
                )
            ]
        )
    else:
        fig = go.Figure(
            data=[
                go.Bar(
                    x=labels,
                    y=values,
                    marker={"color": "#00c2ff"},
                )
            ]
        )
    fig.update_layout(
        title={"text": str(viz.get("title", "Visualization")), "font": {"color": "#e0e0e0"}},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#d7deef"},
        margin={"l": 20, "r": 20, "t": 55, "b": 25},
        xaxis={"gridcolor": "rgba(255,255,255,0.08)"},
        yaxis={"gridcolor": "rgba(255,255,255,0.10)"},
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

st.set_page_config(page_title="Open GR --WM", layout="wide")
inject_external_styles()
st.markdown("## Open GR --WM")
st.caption("Graph-native local RAG workspace")

for key, default in [
    ("graph_ready", False),
    ("building", False),
    ("stop_build", False),
    ("chat_history", []),
    ("pending_query", ""),
    ("pending_enrichment", {}),
    ("v2_facts", []),
    ("v2_retrieval_chunks", []),
    ("v2_extraction_chunks", []),
    ("v2_eval_report", {}),
]:
    if key not in st.session_state:
        st.session_state[key] = default
if "v2_retrieval_index" not in st.session_state:
    st.session_state.v2_retrieval_index = None
if "embedder" not in st.session_state:
    st.session_state.embedder = SentenceTransformer(EMBED_MODEL)

if "auto_loaded" not in st.session_state:
    st.session_state.auto_loaded = False
if not st.session_state.auto_loaded:
    settings = load_settings()
    if settings.get("auto_load_last") and settings.get("last_loaded_rag_id"):
        rag_id = settings.get("last_loaded_rag_id")
        try:
            data = load_rag_from_disk(rag_id)
            load_rag_into_session(rag_id, data)
        except Exception:
            pass
    st.session_state.auto_loaded = True

ensure_chat_state()

with st.sidebar:
    st.markdown("### PAST SESSIONS")
    if st.button("＋ New Session", use_container_width=True, key="new_chat_session_btn", type="secondary"):
        new_session = create_chat_session()
        st.session_state.chat_sessions.insert(0, new_session)
        st.session_state.active_chat_id = new_session["id"]
        st.session_state._chat_loaded_for = None
        save_chat_sessions(st.session_state.chat_sessions)
        do_rerun()
    st.markdown('<div class="sessions-divider"></div>', unsafe_allow_html=True)
    sessions = st.session_state.get("chat_sessions", [])
    for sess in sessions:
        sid = str(sess.get("id", ""))
        if not sid:
            continue
        full_title = str(sess.get("title", "New Session")).strip() or "New Session"
        title_short = ellipsize(full_title, max_chars=28)
        updated = str(sess.get("updated_at", "")).replace("T", " ")[:19]
        active = sid == st.session_state.get("active_chat_id")
        row = st.columns([8.5, 1.3], gap="small")
        with row[0]:
            open_label = title_short
            if st.button(
                open_label,
                key=f"open_sess_{sid}",
                use_container_width=True,
                help=f"{full_title}\n{updated}" if updated else full_title,
                type="primary" if active else "secondary",
            ):
                st.session_state.active_chat_id = sid
                st.session_state._chat_loaded_for = None
                do_rerun()
        with row[1]:
            if st.button(
                "✕",
                key=f"del_sess_{sid}",
                use_container_width=True,
                help=f"Delete: {full_title}",
                type="secondary",
            ):
                st.session_state.chat_sessions = [s for s in sessions if s.get("id") != sid]
                if not st.session_state.chat_sessions:
                    created = create_chat_session()
                    st.session_state.chat_sessions = [created]
                    st.session_state.active_chat_id = created["id"]
                elif st.session_state.get("active_chat_id") == sid:
                    st.session_state.active_chat_id = st.session_state.chat_sessions[0]["id"]
                st.session_state._chat_loaded_for = None
                save_chat_sessions(st.session_state.chat_sessions)
                do_rerun()

saved_rags = list_saved_rags()
rag_meta_by_id = {m.get("rag_id"): m for m in saved_rags if m.get("rag_id")}
rag_options = {
    f"{m.get('pdf_name', 'Unknown')} ({m.get('updated_at', m.get('created_at', ''))})": m.get("rag_id")
    for m in saved_rags
}
tab_graph, tab_chat = st.tabs(["🕸️ Graph Nexus", "💬 Neural Chat"])

with tab_graph:
    left_col, right_col = st.columns([1, 2.5], gap="large")
    with left_col:
        st.markdown('<div class="control-deck">', unsafe_allow_html=True)
        st.markdown('<div class="module-card"><div class="module-title">SOURCE DATA</div></div>', unsafe_allow_html=True)
        source_mode = st.radio("Source Type", ["PDF", "Web URL"], horizontal=True, key="source_mode_selector")
        uploaded = None
        pdf_bytes = b""
        page_texts: List[str] = []
        web_url_input = ""
        web_urls_batch_text = ""
        web_urls_for_build: List[str] = []
        if source_mode == "PDF":
            uploaded = st.file_uploader("Upload PDF source", type=["pdf"], key="source_pdf_uploader")
            if uploaded:
                try:
                    pdf_bytes = uploaded.getvalue()
                    page_texts = read_pdf_pages(pdf_bytes)
                    pdf_hash = hashlib.md5(pdf_bytes).hexdigest()
                    if st.session_state.get("pdf_hash") != pdf_hash:
                        image_pages, text_lengths, total_pages = analyze_pdf_pages(pdf_bytes)
                        st.session_state.pdf_hash = pdf_hash
                        st.session_state.image_pages = image_pages
                        st.session_state.text_lengths = text_lengths
                        st.session_state.total_pages = total_pages
                        st.session_state.page_texts = page_texts
                except Exception as exc:
                    st.error(f"Failed to read PDF: {exc}")
                    pdf_bytes = b""
                    page_texts = []
        else:
            web_url_input = st.text_input(
                "Primary URL",
                placeholder="https://example.com/report",
                key="source_web_url_input",
            )
            web_urls_batch_text = st.text_area(
                "Additional URLs (one per line)",
                placeholder="https://example.com/page-2\nhttps://example.com/page-3",
                key="source_web_urls_batch",
                height=100,
            )
            web_urls_for_build = parse_web_url_inputs(web_url_input, web_urls_batch_text)
            st.caption(f"URLs ready: {len(web_urls_for_build)}")
            if BeautifulSoup is None:
                st.caption("Install `beautifulsoup4` for richer HTML extraction and image routing.")
            preview_cols = st.columns([1, 1])
            if preview_cols[0].button("Preview URL", use_container_width=True, key="preview_web_url_btn"):
                if not web_urls_for_build:
                    st.warning("Enter a URL first.")
                else:
                    try:
                        with st.spinner("Fetching URL preview..."):
                            preview = extract_web_content(web_urls_for_build[0], include_images=True, max_images=0)
                        st.session_state.web_preview = {
                            "url": str(preview.get("url", "")),
                            "title": str(preview.get("title", "")),
                            "segments": len(preview.get("segments", []) or []),
                            "images": int(preview.get("image_candidate_count", 0) or 0),
                        }
                    except Exception as exc:
                        st.error(f"URL preview failed: {exc}")
                        st.session_state.web_preview = {}
            if preview_cols[1].button("Clear Preview", use_container_width=True, key="clear_web_url_preview_btn"):
                st.session_state.web_preview = {}
            web_preview = st.session_state.get("web_preview", {})
            if isinstance(web_preview, dict) and web_preview.get("url"):
                st.caption(
                    f"Preview: {web_preview.get('title') or 'Untitled page'} | "
                    f"segments {web_preview.get('segments', 0)} | "
                    f"image candidates {web_preview.get('images', 0)}"
                )
                st.caption(str(web_preview.get("url", "")))

        st.markdown('<div class="module-card"><div class="module-title">GRAPH OPERATIONS</div></div>', unsafe_allow_html=True)
        selected_rag_label = st.selectbox("Saved graph", [""] + list(rag_options.keys()), key="library_selected_graph")
        selected_rag_id = rag_options.get(selected_rag_label, "")
        op_cols = st.columns(3)
        load_btn = op_cols[0].button("Load", use_container_width=True)
        unload_btn = op_cols[1].button("Unload RAM", use_container_width=True)
        delete_btn = op_cols[2].button("Delete Disk", use_container_width=True)

        if load_btn and selected_rag_id:
            try:
                data = load_rag_from_disk(selected_rag_id)
                load_rag_into_session(selected_rag_id, data)
                settings = load_settings()
                settings["last_loaded_rag_id"] = selected_rag_id
                save_settings(settings)
                set_status_bubble("Saved graph loaded into RAM.", "success")
                do_rerun()
            except Exception as exc:
                st.error(f"Could not load this graph. Rebuild/export it in v2 format. Details: {exc}")
        if unload_btn:
            clear_session_data()
            ensure_chat_state()
            set_status_bubble("Graph unloaded from RAM.", "warning")
            do_rerun()
        if delete_btn and selected_rag_id:
            delete_rag_from_disk(selected_rag_id)
            settings = load_settings()
            if settings.get("last_loaded_rag_id") == selected_rag_id:
                settings["last_loaded_rag_id"] = ""
                save_settings(settings)
            if st.session_state.get("current_rag_id") == selected_rag_id:
                clear_session_data()
                ensure_chat_state()
            set_status_bubble("Saved graph deleted from disk.", "warning")
            do_rerun()

        archive_cols = st.columns(2)
        if selected_rag_id:
            try:
                export_name, export_blob = export_rag_archive(selected_rag_id)
                archive_cols[0].download_button(
                    "Export Archive",
                    data=export_blob,
                    file_name=export_name,
                    mime="application/zip",
                    use_container_width=True,
                    key=f"export_archive_{selected_rag_id}",
                )
            except Exception as exc:
                archive_cols[0].button("Export Archive", disabled=True, use_container_width=True, key="export_archive_disabled_btn")
                st.caption(f"Export unavailable: {exc}")
        else:
            archive_cols[0].button("Export Archive", disabled=True, use_container_width=True, key="export_archive_disabled_btn_empty")
        auto_load_import = archive_cols[1].checkbox("Auto-load after import", value=True, key="auto_load_imported_graph")

        import_archive = st.file_uploader(
            "Import graph archive (.zip)",
            type=["zip"],
            key="import_graph_archive_uploader",
        )
        import_btn = st.button("Import Archive", use_container_width=True, key="import_graph_archive_btn")
        if import_btn:
            if not import_archive:
                st.warning("Choose a .zip archive first.")
            else:
                try:
                    with st.spinner("Importing graph archive..."):
                        imported_rag_id, imported_meta = import_rag_archive(import_archive.getvalue(), import_archive.name)
                    if auto_load_import:
                        data = load_rag_from_disk(imported_rag_id)
                        load_rag_into_session(imported_rag_id, data)
                        settings = load_settings()
                        settings["last_loaded_rag_id"] = imported_rag_id
                        save_settings(settings)
                    imported_name = str(imported_meta.get("pdf_name", imported_rag_id))
                    set_status_bubble(f"Imported graph: {imported_name}", "success")
                    do_rerun()
                except Exception as exc:
                    st.error(f"Import failed: {exc}")

        with st.expander("⚙️ ADVANCED CONFIGURATION"):
            build_target_mode = st.radio(
                "Build Target",
                ["Create new graph", "Augment loaded graph"],
                horizontal=True,
                key="build_target_mode",
            )
            augment_mode = build_target_mode == "Augment loaded graph"
            source_mode_for_name = st.session_state.get("source_mode_selector", "PDF")
            if source_mode_for_name == "PDF":
                source_default_name = os.path.splitext(uploaded.name)[0] if uploaded and uploaded.name else ""
            else:
                web_name_urls = parse_web_url_inputs(
                    st.session_state.get("source_web_url_input", ""),
                    st.session_state.get("source_web_urls_batch", ""),
                )
                if len(web_name_urls) >= 2:
                    source_default_name = normalize_graph_name(
                        f"{url_to_graph_name(web_name_urls[0])} bundle",
                        fallback="web bundle",
                    )
                else:
                    source_default_name = url_to_graph_name(web_name_urls[0] if web_name_urls else "")
            graph_name_input = st.text_input(
                "Graph Name",
                value=st.session_state.get("current_pdf_name", "") if augment_mode and st.session_state.get("current_rag_id") else source_default_name,
                disabled=augment_mode,
                key="graph_name_input",
            )
            build_mode = st.selectbox("Build Mode", ["Fast", "Balanced", "Thorough"], index=1, key="build_mode_select")
            use_vision = st.checkbox("Use Vision Routing", value=True, key="use_vision_toggle")
            vision_text_limit = st.slider("Vision text limit (chars)", 50, 2000, 400, 50, key="vision_text_limit_slider")
            max_vision_pages = st.slider("Max vision artifacts per source", 1, 120, 24, 1, key="max_vision_pages_slider")
            if build_mode == "Fast":
                text_batch_pages = 4
                text_batch_chars = 2600
                min_text_chars = 120
                vision_scale_default = 1.3
            elif build_mode == "Thorough":
                text_batch_pages = 2
                text_batch_chars = 3600
                min_text_chars = 60
                vision_scale_default = 1.8
            else:
                text_batch_pages = 3
                text_batch_chars = 3000
                min_text_chars = 80
                vision_scale_default = 1.6
            vision_scale = st.slider("Vision detail level", 1.0, 2.5, float(vision_scale_default), 0.1, key="vision_scale_slider")
            if augment_mode and not st.session_state.get("current_rag_id"):
                st.caption("Load a graph first to enable augmentation.")

            settings = load_settings()
            if "auto_load_last" not in st.session_state:
                st.session_state.auto_load_last = bool(settings.get("auto_load_last", False))
            auto_load = st.checkbox("Load most recent graph on startup", key="auto_load_last")
            if auto_load != bool(settings.get("auto_load_last", False)):
                settings["auto_load_last"] = bool(auto_load)
                save_settings(settings)

        st.markdown('<div class="module-card"><div class="module-title">EXECUTION</div></div>', unsafe_allow_html=True)
        exec_cols = st.columns(3)
        build_btn = exec_cols[0].button("Build" if not st.session_state.get("build_target_mode") == "Augment loaded graph" else "Augment", use_container_width=True)
        stop_btn = exec_cols[1].button("Stop", use_container_width=True)
        clear_btn = exec_cols[2].button("Clear RAM", use_container_width=True)

        if clear_btn:
            clear_session_data()
            ensure_chat_state()
            set_status_bubble("Workspace RAM cleared.", "warning")
            do_rerun()
        if stop_btn:
            st.session_state.stop_build = True
            set_status_bubble("Stop requested. Finishing current task...", "warning")

        if build_btn:
            augment_mode = st.session_state.get("build_target_mode") == "Augment loaded graph"
            source_mode_active = st.session_state.get("source_mode_selector", "PDF")
            if augment_mode and not st.session_state.get("current_rag_id"):
                st.warning("Load a graph first, then run augmentation.")
            else:
                page_texts = []
                vision_images: List[Tuple[int, str, str]] = []
                web_text_tasks: List[Dict[str, object]] = []
                source_label = ""
                source_url = ""
                source_type = "pdf"
                build_title_hint = ""
                source_valid = True
                total_pages = 0
                source_files_for_save: List[str] = []
                build_log_prelude: List[str] = []
                build_mode = st.session_state.get("build_mode_select", "Balanced")
                use_vision = bool(st.session_state.get("use_vision_toggle", True))
                vision_text_limit = int(st.session_state.get("vision_text_limit_slider", 400))
                max_vision_pages = int(st.session_state.get("max_vision_pages_slider", 24))
                vision_scale = float(st.session_state.get("vision_scale_slider", 1.6))
                if build_mode == "Fast":
                    text_batch_pages = 4
                    text_batch_chars = 2600
                    min_text_chars = 120
                elif build_mode == "Thorough":
                    text_batch_pages = 2
                    text_batch_chars = 3600
                    min_text_chars = 60
                else:
                    text_batch_pages = 3
                    text_batch_chars = 3000
                    min_text_chars = 80
                if source_mode_active == "PDF":
                    if not uploaded:
                        st.warning("Upload a PDF first.")
                        source_valid = False
                    total_pages = int(st.session_state.get("total_pages", 0) or 0)
                    if source_valid and total_pages == 0:
                        st.warning("Could not read any pages from the PDF.")
                        source_valid = False
                    page_texts = list(st.session_state.get("page_texts", []) or []) if source_valid else []
                    source_label = str(uploaded.name) if (source_valid and uploaded) else ""
                    source_type = "pdf"
                    source_url = ""
                    build_title_hint = os.path.splitext(uploaded.name)[0] if (source_valid and uploaded and uploaded.name) else ""
                    if source_label:
                        source_files_for_save = [source_label]
                    if source_valid and use_vision and pdf_bytes:
                        with st.spinner("Preparing vision pages..."):
                            image_pages = st.session_state.get("image_pages", [])
                            text_lengths = st.session_state.get("text_lengths", [])
                            page_indices = select_vision_pages(
                                image_pages=image_pages,
                                text_lengths=text_lengths,
                                text_limit=vision_text_limit,
                                max_pages=max_vision_pages,
                            )
                            if page_indices:
                                pdf_vision = prepare_vision_images(
                                    pdf_bytes,
                                    page_indices,
                                    max_pages=max_vision_pages,
                                    scale=vision_scale,
                                )
                                for page_idx, img_b64 in pdf_vision:
                                    vision_images.append(
                                        (
                                            page_idx,
                                            img_b64,
                                            f"vision page {len(vision_images) + 1}/{len(pdf_vision)} (pdf page {page_idx + 1}/{total_pages})",
                                        )
                                    )
                else:
                    web_urls = parse_web_url_inputs(
                        st.session_state.get("source_web_url_input", ""),
                        st.session_state.get("source_web_urls_batch", ""),
                    )
                    if not web_urls:
                        st.warning("Enter a Web URL first.")
                        source_valid = False
                    if source_valid:
                        successful_urls = 0
                        first_title = ""
                        first_url = ""
                        for url_idx, web_url in enumerate(web_urls, start=1):
                            try:
                                with st.spinner(f"Fetching URL {url_idx}/{len(web_urls)}..."):
                                    web_payload = extract_web_content(
                                        web_url,
                                        include_images=use_vision,
                                        max_images=max_vision_pages,
                                    )
                            except Exception as exc:
                                build_log_prelude.append(f"Skipped URL {url_idx}/{len(web_urls)}: {web_url} | error: {exc}")
                                continue
                            payload_url = str(web_payload.get("url", "")) or web_url
                            payload_title = str(web_payload.get("title", "")).strip()
                            segments_raw = list(web_payload.get("segments", []) or [])
                            host = urlparse(payload_url).netloc or "web"
                            segments_prefixed = [f"[Source URL {url_idx}/{len(web_urls)}: {payload_url}]\n{seg}" for seg in segments_raw]
                            if segments_prefixed:
                                url_tasks = build_text_batches(
                                    page_texts=segments_prefixed,
                                    max_chars=text_batch_chars,
                                    max_pages_per_batch=text_batch_pages,
                                    min_chars=min_text_chars,
                                )
                                for t_i, task in enumerate(url_tasks, start=1):
                                    task["label"] = f"text from URL {url_idx}/{len(web_urls)} ({host}) batch {t_i}/{len(url_tasks)}"
                                web_text_tasks.extend(url_tasks)
                            page_texts.extend(segments_prefixed)
                            web_vision = list(web_payload.get("vision_images", []) or [])
                            for idx_img, img_b64, label_hint in web_vision:
                                vision_images.append(
                                    (
                                        int(idx_img),
                                        str(img_b64),
                                        f"vision from URL {url_idx}/{len(web_urls)} ({host}) artifact {idx_img + 1}/{len(web_vision)} ({label_hint})",
                                    )
                                )
                            build_log_prelude.append(
                                f"Queued URL {url_idx}/{len(web_urls)}: {payload_url} | text segments: {len(segments_prefixed)} | vision artifacts: {len(web_vision)}"
                            )
                            source_files_for_save.append(payload_url)
                            successful_urls += 1
                            if not first_url:
                                first_url = payload_url
                                first_title = payload_title
                            st.session_state.web_preview = {
                                "url": payload_url,
                                "title": payload_title,
                                "segments": len(segments_prefixed),
                                "images": int(web_payload.get("image_candidate_count", 0) or 0),
                            }
                        if successful_urls == 0:
                            st.warning("None of the provided URLs could be processed.")
                            source_valid = False
                        else:
                            source_url = first_url
                            source_label = first_url if successful_urls == 1 else f"{successful_urls} URLs"
                            source_type = "url"
                            if successful_urls >= 2:
                                build_title_hint = normalize_graph_name(
                                    f"{url_to_graph_name(first_url)} bundle",
                                    fallback="web bundle",
                                )
                            else:
                                build_title_hint = first_title or url_to_graph_name(first_url)
                            st.session_state.total_pages = len(page_texts)
                            st.session_state.page_texts = page_texts
                            total_pages = len(page_texts)

                if (not source_valid) or total_pages == 0 or not page_texts:
                    st.warning("No readable text was extracted from the selected source.")
                else:
                    st.session_state.stop_build = False
                    profile_key = "balanced"
                    if build_mode == "Fast":
                        profile_key = "fast"
                    elif build_mode == "Thorough":
                        profile_key = "quality"

                    if augment_mode:
                        base_facts = list(st.session_state.get("v2_facts", []))
                        base_retrieval_chunks = list(st.session_state.get("v2_retrieval_chunks", []))
                        base_extraction_chunks = list(st.session_state.get("v2_extraction_chunks", []))
                        target_rag_id = st.session_state.get("current_rag_id", st.session_state.get("pdf_hash", "unknown"))
                        graph_name = st.session_state.get("current_pdf_name", target_rag_id)
                        prev_meta = st.session_state.get("current_meta", {})
                    else:
                        preferred_name = normalize_graph_name(
                            st.session_state.get("graph_name_input", "") or build_title_hint or "Untitled Graph",
                            fallback="Untitled Graph",
                        )
                        base_facts = []
                        base_retrieval_chunks = []
                        base_extraction_chunks = []
                        target_rag_id = choose_new_rag_id(preferred_name)
                        graph_name = preferred_name
                        prev_meta = {}

                    source_id = f"{source_type}:{target_rag_id}:{int(time.time())}"
                    base_graph_obj, base_triples = facts_to_graph(canonicalize_facts(base_facts))
                    st.session_state.graph = base_graph_obj
                    st.session_state.triples = base_triples
                    st.session_state.chunks = [c.text for c in base_retrieval_chunks]
                    existing_index = st.session_state.get("v2_retrieval_index")
                    if augment_mode and existing_index is not None:
                        st.session_state.embeddings = np.array(existing_index.embeddings)
                    else:
                        st.session_state.embeddings = np.array([])
                    ex_chunks_new, re_chunks_new = build_chunk_tracks_for_pages(
                        source_id=source_id,
                        page_texts=page_texts,
                        profile=profile_key,
                    )
                    if not ex_chunks_new and not vision_images:
                        st.warning("No extraction tasks were generated from this source.")
                        source_valid = False

                    tasks: List[Dict[str, object]] = []
                    for idx_chunk, chunk in enumerate(ex_chunks_new, start=1):
                        tasks.append(
                            {
                                "type": "text_chunk",
                                "chunk": chunk,
                                "label": f"text chunk {idx_chunk}/{len(ex_chunks_new)} (page {chunk.page_or_doc_idx + 1})",
                            }
                        )
                    st.session_state.vision_pages = len(vision_images)
                    for v_idx, (_idx_img, img_b64, vision_label) in enumerate(vision_images, start=1):
                        tasks.append(
                            {
                                "type": "vision_artifact",
                                "data": str(img_b64),
                                "chunk_id": f"vision-{source_id}-{v_idx}",
                                "source_id": source_id,
                                "label": vision_label,
                            }
                        )
                    st.session_state.build_tasks = tasks
                    st.session_state.build_index = 0
                    st.session_state.build_logs = [f"Mode: {'augment' if augment_mode else 'new'} | Source: {source_label}"] + build_log_prelude
                    st.session_state.build_total = len(tasks)
                    st.session_state.building = True
                    st.session_state.graph_ready = False
                    st.session_state.save_pending = True
                    st.session_state.build_target_rag_id = target_rag_id
                    st.session_state.build_graph_name = graph_name
                    st.session_state.build_source_file = source_label
                    st.session_state.build_source_files = source_files_for_save if source_files_for_save else ([source_label] if source_label else [])
                    st.session_state.build_source_type = source_type
                    st.session_state.build_source_url = source_url
                    st.session_state.build_operation = "augment" if augment_mode else "new"
                    st.session_state.build_previous_meta = prev_meta
                    st.session_state.build_mode_used = build_mode
                    st.session_state.v2_build_profile = profile_key
                    st.session_state.build_use_vision = use_vision
                    st.session_state.build_vision_text_limit = vision_text_limit
                    st.session_state.build_text_batch_pages = text_batch_pages
                    st.session_state.build_max_vision_pages = max_vision_pages
                    st.session_state.v2_build_rag_id = target_rag_id
                    st.session_state.v2_build_facts = base_facts
                    st.session_state.v2_build_retrieval_chunks = base_retrieval_chunks + re_chunks_new
                    st.session_state.v2_build_extraction_chunks = base_extraction_chunks + ex_chunks_new
                    st.session_state.v2_build_started_at = time.time()
                    st.session_state.v2_build_extract_sec = 0.0
                    v2_save_checkpoint(
                        target_rag_id,
                        {
                            "build_index": 0,
                            "build_total": len(tasks),
                            "facts_count": len(base_facts),
                            "profile": profile_key,
                        },
                    )
                    set_status_bubble("Build started.", "processing")
                    do_rerun()

        st.markdown("</div>", unsafe_allow_html=True)

    with right_col:
        st.markdown('<div class="deck-title">SYSTEM ACTIVITY</div>', unsafe_allow_html=True)
        logs = st.session_state.get("build_logs", [])
        total = int(st.session_state.get("build_total", 0) or 0)
        idx = int(st.session_state.get("build_index", 0) or 0)
        if st.session_state.get("building"):
            st.progress(idx / total if total else 0.0)
            elapsed = max(0.001, time.time() - float(st.session_state.get("v2_build_started_at", time.time())))
            rate = idx / elapsed
            eta = (total - idx) / max(rate, 1e-6) if total > idx else 0.0
            st.caption(f"Throughput: {rate:.2f} tasks/s | ETA: {eta:.1f}s")
        safe_logs = html.escape("\n".join(logs[-18:]) if logs else "Awaiting operation...")
        st.markdown(f'<div class="terminal-output"><pre>{safe_logs}</pre></div>', unsafe_allow_html=True)
        render_status_bubble()

        if st.session_state.get("building"):
            if total == 0:
                logs.append("No extraction tasks were generated.")
                st.session_state.building = False
                st.session_state.graph_ready = len(st.session_state.graph.nodes) > 0
                st.session_state.save_pending = False
                set_status_bubble("No extraction tasks were generated.", "warning")
                do_rerun()
            elif st.session_state.get("stop_build"):
                logs.append("Stop requested. Halting build.")
                st.session_state.building = False
                st.session_state.graph_ready = len(st.session_state.graph.nodes) > 0
                set_status_bubble("Build paused.", "warning")
            elif idx < total:
                profile_key = str(st.session_state.get("v2_build_profile", "balanced")).lower()
                if profile_key == "fast":
                    batch_n = 6
                elif profile_key == "quality":
                    batch_n = 3
                else:
                    batch_n = 4

                rag_id = st.session_state.get("v2_build_rag_id", st.session_state.get("build_target_rag_id", "graph"))
                extract_start = time.time()
                processed = 0
                while processed < batch_n and st.session_state.build_index < total and not st.session_state.get("stop_build"):
                    local_idx = int(st.session_state.build_index)
                    task = st.session_state.build_tasks[local_idx]
                    logs.append(f"Processing {task['label']}")
                    facts_new = []
                    if task.get("type") == "text_chunk":
                        chunk = task.get("chunk")
                        if chunk is not None:
                            facts_new = extract_facts_from_text_chunk(
                                chunk=chunk,
                                llm_generate=call_ollama,
                                model=SCANNER_MODEL,
                                profile=profile_key,
                                cache_lookup=lambda key, _rid=rag_id: v2_cache_lookup(_rid, key),
                                cache_store=lambda key, value, _rid=rag_id: v2_cache_store(_rid, key, value),
                            )
                    else:
                        facts_new = extract_facts_from_vision_artifact(
                            artifact_b64=str(task.get("data", "")),
                            source_id=str(task.get("source_id", rag_id)),
                            chunk_id=str(task.get("chunk_id", f"vision-{local_idx + 1}")),
                            label_hint=str(task.get("label", "")),
                            llm_generate=call_ollama,
                            model=VISION_MODEL,
                            profile=profile_key,
                            cache_lookup=lambda key, _rid=rag_id: v2_cache_lookup(_rid, key),
                            cache_store=lambda key, value, _rid=rag_id: v2_cache_store(_rid, key, value),
                        )
                    st.session_state.v2_build_facts.extend(facts_new)
                    logs.append(f"Extracted {len(facts_new)} facts from {task['label']}")
                    st.session_state.build_index = local_idx + 1
                    processed += 1
                    if st.session_state.build_index % 20 == 0 or st.session_state.build_index >= total:
                        v2_save_checkpoint(
                            rag_id,
                            {
                                "build_index": int(st.session_state.build_index),
                                "build_total": int(total),
                                "facts_count": len(st.session_state.get("v2_build_facts", [])),
                                "profile": profile_key,
                            },
                        )

                st.session_state.v2_build_extract_sec = float(st.session_state.get("v2_build_extract_sec", 0.0)) + (
                    time.time() - extract_start
                )
                if st.session_state.build_index >= total:
                    logs.append("Canonicalizing and deduplicating facts...")
                    canonical_facts = canonicalize_facts(st.session_state.get("v2_build_facts", []))
                    graph_obj, triples = facts_to_graph(canonical_facts)
                    retrieval_chunks_all = list(st.session_state.get("v2_build_retrieval_chunks", []) or [])
                    extraction_chunks_all = list(st.session_state.get("v2_build_extraction_chunks", []) or [])
                    retrieval_index = build_retrieval_index(
                        embedder=st.session_state.embedder,
                        retrieval_chunks=retrieval_chunks_all,
                    )
                    compositional_eval = {}
                    try:
                        regression_question = (
                            "What is the time difference in years between the adventures of Dunk & Egg, and Dance of the Dragons?"
                        )
                        regression_intent = parse_intent_spec(regression_question).to_dict()
                        regression_evidence = retrieve_evidence_bundle(
                            question=regression_question,
                            graph=graph_obj,
                            embedder=st.session_state.embedder,
                            chunk_texts=list(retrieval_index.chunk_texts),
                            embeddings=retrieval_index.embeddings,
                            bm25_model=retrieval_index.bm25_model,
                            bm25_tokens=retrieval_index.bm25_tokens,
                            chunk_records=list(retrieval_index.chunk_records),
                            metric_map=retrieval_index.metric_map,
                            entity_map=retrieval_index.entity_map,
                            top_k=8,
                            intent_hint=regression_intent,
                        )
                        compositional_eval = run_compositional_retrieval_eval(regression_question, regression_evidence)
                    except Exception as exc:
                        logs.append(f"Compositional eval skipped: {exc}")
                    timings = {
                        "build_total_sec": float(time.time() - float(st.session_state.get("v2_build_started_at", time.time()))),
                        "extract_sec": float(st.session_state.get("v2_build_extract_sec", 0.0)),
                    }
                    eval_report = run_build_evals(
                        facts=canonical_facts,
                        graph=graph_obj,
                        retrieval_index=retrieval_index,
                        total_pages=int(st.session_state.get("total_pages", 0) or 1),
                        timings=timings,
                        compositional_eval=compositional_eval,
                    )
                    st.session_state.graph = graph_obj
                    st.session_state.triples = triples
                    st.session_state.chunks = list(retrieval_index.chunk_texts)
                    st.session_state.embeddings = retrieval_index.embeddings
                    st.session_state.v2_facts = canonical_facts
                    st.session_state.v2_retrieval_chunks = retrieval_chunks_all
                    st.session_state.v2_extraction_chunks = extraction_chunks_all
                    st.session_state.v2_retrieval_index = retrieval_index
                    st.session_state.v2_eval_report = eval_report
                    logs.append("Build complete.")
                    st.session_state.building = False
                    st.session_state.graph_ready = len(st.session_state.graph.nodes) > 0
                    rag_id = st.session_state.get("build_target_rag_id", st.session_state.get("pdf_hash", "unknown"))
                    if st.session_state.get("save_pending"):
                        graph_name = st.session_state.get("build_graph_name", rag_id)
                        meta_extra = {
                            "total_pages": str(st.session_state.get("total_pages", 0)),
                            "vision_text_limit": str(st.session_state.get("build_vision_text_limit", 400)),
                            "use_vision": str(st.session_state.get("build_use_vision", True)),
                            "vision_pages": str(st.session_state.get("vision_pages", 0)),
                            "build_mode": str(st.session_state.get("build_mode_used", "Balanced")),
                            "text_batch_pages": str(st.session_state.get("build_text_batch_pages", 3)),
                            "max_vision_pages": str(st.session_state.get("build_max_vision_pages", 24)),
                            "last_operation": str(st.session_state.get("build_operation", "new")),
                            "last_source_file": str(st.session_state.get("build_source_file", "")),
                            "last_source_type": str(st.session_state.get("build_source_type", "pdf")),
                            "last_source_url": str(st.session_state.get("build_source_url", "")),
                            "last_source_count": str(len(st.session_state.get("build_source_files", []) or [])),
                        }
                        saved_meta = save_rag_to_disk(
                            rag_id=rag_id,
                            pdf_name=graph_name,
                            graph=st.session_state.graph,
                            triples=st.session_state.triples,
                            chunks=st.session_state.chunks,
                            embeddings=st.session_state.embeddings,
                            meta_extra=meta_extra,
                            previous_meta=st.session_state.get("build_previous_meta", {}),
                            source_file=st.session_state.get("build_source_file", ""),
                            source_files=st.session_state.get("build_source_files", []),
                            facts=st.session_state.get("v2_facts", []),
                            extraction_chunks=st.session_state.get("v2_extraction_chunks", []),
                            retrieval_chunks_records=st.session_state.get("v2_retrieval_chunks", []),
                            retrieval_index=st.session_state.get("v2_retrieval_index"),
                            build_profile=st.session_state.get("v2_build_profile", "balanced"),
                            eval_report=st.session_state.get("v2_eval_report", {}),
                            timings=timings,
                        )
                        st.session_state.current_rag_id = rag_id
                        st.session_state.current_pdf_name = saved_meta.get("pdf_name", graph_name)
                        st.session_state.current_meta = saved_meta
                        settings = load_settings()
                        settings["last_loaded_rag_id"] = rag_id
                        save_settings(settings)
                        st.session_state.save_pending = False
                    v2_clear_checkpoint(rag_id)
                    quality = st.session_state.get("v2_eval_report", {}).get("quality_metrics", {})
                    if isinstance(quality, dict) and quality:
                        logs.append(
                            "Quality: edges/page="
                            + str(quality.get("edges_per_page", "n/a"))
                            + " | citation_validity="
                            + str(quality.get("citation_validity_rate", "n/a"))
                        )
                    set_status_bubble("Graph build complete and saved.", "success")
                    do_rerun()
                else:
                    do_rerun()
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="deck-title">ACTIVE KNOWLEDGE LATTICE</div>', unsafe_allow_html=True)
        if st.session_state.get("graph_ready"):
            graph_obj = st.session_state.graph
            g_nodes = graph_obj.number_of_nodes()
            g_edges = graph_obj.number_of_edges()
            try:
                html_path = graph_to_pyvis(graph_obj)
                with open(html_path, "r", encoding="utf-8") as f:
                    html_block = f.read()
                st.components.v1.html(html_block, height=700, scrolling=True)
            except Exception as exc:
                st.warning(f"Graph visualization failed: {exc}")
            finally:
                try:
                    if "html_path" in locals() and os.path.exists(html_path):
                        os.remove(html_path)
                except Exception:
                    pass
            st.markdown(
                f'<div class="data-pills"><span class="data-pill">Entities: {g_nodes}</span>'
                f'<span class="data-pill">Relations: {g_edges}</span>'
                f'<span class="data-pill">Saved Graphs: {len(saved_rags)}</span></div>',
                unsafe_allow_html=True,
            )
            eval_report = st.session_state.get("v2_eval_report", {})
            if isinstance(eval_report, dict) and eval_report:
                with st.expander("Quality Report", expanded=False):
                    st.json(eval_report)
        else:
            st.info("No active graph in memory.")

with tab_chat:
    active_name = st.session_state.get("current_pdf_name", "None")
    st.markdown(f"### Neural Chat — `{active_name}`")
    clear_chat = st.button("Clear Current Session", key="chat_clear_btn")
    if clear_chat:
        st.session_state.chat_history = []
        persist_active_chat_history()
        do_rerun()

    history_mutated = False
    for idx, msg in enumerate(st.session_state.chat_history):
        role = str(msg.get("role", "assistant"))
        content = str(msg.get("content", ""))
        if role == "user":
            st.markdown(
                f'<div class="user-row"><div class="user-bubble">{html.escape(content)}</div></div>',
                unsafe_allow_html=True,
            )
            continue
        clean_content, clean_viz, clean_figures = sanitize_assistant_payload(
            content=content,
            viz_existing=msg.get("viz", {}) if isinstance(msg.get("viz", {}), dict) else {},
            hero_existing=msg.get("hero_figures", []) if isinstance(msg.get("hero_figures", []), list) else [],
        )
        if clean_content != content or clean_viz != msg.get("viz", {}) or clean_figures != msg.get("hero_figures", []):
            st.session_state.chat_history[idx]["content"] = clean_content
            st.session_state.chat_history[idx]["viz"] = clean_viz
            st.session_state.chat_history[idx]["hero_figures"] = clean_figures
            history_mutated = True
        with st.container():
            st.markdown('<div class="ai-container">', unsafe_allow_html=True)
            summary = extract_main_answer(clean_content)
            if summary:
                st.write(summary)
            subtabs = st.tabs(["📊 Visuals", "🧠 Key Insights & Figures", "📜 Sources"])
            with subtabs[0]:
                render_plotly_visualization(clean_viz)
            with subtabs[1]:
                left_metrics, right_insights = st.columns(2)
                with left_metrics:
                    figures = clean_figures or []
                    if figures:
                        for figure in figures[:4]:
                            label = str(figure.get("label", "")).strip()
                            value = str(figure.get("value", "")).strip()
                            delta = str(figure.get("delta", "")).strip()
                            if label and value:
                                st.metric(label=label, value=value, delta=delta if delta else None)
                    else:
                        st.caption("No hero figures returned.")
                with right_insights:
                    bullets = extract_insight_bullets(clean_content)
                    if bullets:
                        st.markdown("\n".join([f"- {b}" for b in bullets]))
                    else:
                        st.caption("No structured insights returned.")
            with subtabs[2]:
                sources = extract_sources_from_message(
                    {"content": clean_content, "hero_figures": clean_figures}
                )
                if sources:
                    st.markdown('<div class="sources-list">' + "<br/>".join([html.escape(s) for s in sources]) + "</div>", unsafe_allow_html=True)
                else:
                    st.caption("No sources detected.")
            if msg.get("thinking"):
                with st.expander("Reasoning trace"):
                    st.markdown(str(msg.get("thinking", ""))[-12000:])
            st.markdown('</div>', unsafe_allow_html=True)
    if history_mutated:
        persist_active_chat_history()

    user_prompt = st.chat_input("Ask your graph a question")
    if user_prompt:
        if not st.session_state.get("graph_ready"):
            st.warning("Load or build a graph first.")
        else:
            st.session_state.chat_history.append({"role": "user", "content": user_prompt})
            persist_active_chat_history()
            st.session_state.pending_query = user_prompt
            st.session_state.pending_enrichment = {}
            set_status_bubble("Question received. Starting analysis...", "processing")
            do_rerun()

    pending_query = st.session_state.get("pending_query", "")
    if pending_query:
        stage_line = st.empty()
        thinking_panel = st.empty()
        stage_line.markdown("`Analyzing query...`")
        intent_spec = parse_intent_spec(pending_query)
        intent_hint = intent_spec.to_dict()
        intent_label = intent_spec.derivation_type or (intent_spec.intent_labels[0] if intent_spec.intent_labels else "direct_lookup")
        anchor_preview = ", ".join(intent_spec.comparison_anchors[:2]) if intent_spec.comparison_anchors else "none"
        stage_line.markdown(f"`Intent:` {intent_label} | anchors: {anchor_preview}")

        graph = st.session_state.graph
        retrieval_index = st.session_state.get("v2_retrieval_index")
        if retrieval_index is None and st.session_state.get("v2_retrieval_chunks"):
            retrieval_index = build_retrieval_index(
                embedder=st.session_state.embedder,
                retrieval_chunks=list(st.session_state.get("v2_retrieval_chunks", [])),
            )
            st.session_state.v2_retrieval_index = retrieval_index

        if retrieval_index is not None:
            evidence = retrieve_evidence_bundle(
                question=pending_query,
                graph=graph,
                embedder=st.session_state.embedder,
                chunk_texts=list(retrieval_index.chunk_texts),
                embeddings=retrieval_index.embeddings,
                bm25_model=retrieval_index.bm25_model,
                bm25_tokens=retrieval_index.bm25_tokens,
                chunk_records=list(retrieval_index.chunk_records),
                metric_map=retrieval_index.metric_map,
                entity_map=retrieval_index.entity_map,
                top_k=8,
                intent_hint=intent_hint,
            )
            hits = list(evidence.entity_hits)
            graph_lines_list = list(evidence.graph_lines or [])
            chunk_lines_list = list(evidence.chunk_lines or [])
            graph_context = "\n".join(graph_lines_list) if graph_lines_list else "No direct node matches."
            chunk_context = "\n".join(chunk_lines_list) if chunk_lines_list else "None."
            has_graph_context = bool(graph_lines_list)
            allowed_citation_rows = list(evidence.allowed_citations or [])
            retrieval_confidence = float(evidence.confidence or 0.0)
            evidence_context_ids = list(evidence.context_citation_ids or [])
            if evidence_context_ids:
                allowed_citation_rows.extend([f"[Chunk {cid}]" for cid in evidence_context_ids])
                allowed_citation_rows = dedup_ci(allowed_citation_rows)
            retrieval_debug_line = (
                f"Retrieval: {str(evidence.debug.get('intent_source', 'inferred'))} | "
                f"anchors: {int(float(evidence.debug.get('anchor_count', 0.0)))} | "
                f"conf: {retrieval_confidence:.2f}"
            )
        else:
            hits = keyword_nodes(graph, pending_query)
            fallback_graph_context = neighbors_context(graph, hits) or "No direct node matches."
            graph_lines_list = [ln for ln in fallback_graph_context.splitlines() if ln.strip()] if fallback_graph_context != "No direct node matches." else []
            chunk_lines_list = []
            graph_context = "\n".join(graph_lines_list) if graph_lines_list else "No direct node matches."
            chunk_context = "None."
            has_graph_context = graph_context != "No direct node matches."
            allowed_citation_rows = ["[Graph]"] if has_graph_context else []
            retrieval_confidence = 0.0
            evidence_context_ids = []
            retrieval_debug_line = "Retrieval: keyword fallback"

        retrieved_chunk_ids = []
        for cit in allowed_citation_rows:
            m = re.search(r"\[\s*chunk\s*([^\]]+?)\s*\]", str(cit), flags=re.IGNORECASE)
            if m:
                retrieved_chunk_ids.append(str(m.group(1)).strip())
        final_context = pack_answer_context(
            question=pending_query,
            chunk_lines=chunk_lines_list,
            graph_lines=graph_lines_list,
            max_tokens=5200,
            anchors=intent_spec.comparison_anchors,
        )
        context_chunk_ids = extract_chunk_citations(final_context)
        context_chunk_set = {x.lower() for x in context_chunk_ids}
        context_extra_ids = [cid for cid in evidence_context_ids if str(cid).strip().lower() in context_chunk_set]
        valid_chunk_ids = dedup_ci(retrieved_chunk_ids + context_chunk_ids + context_extra_ids)
        allowed_citation_rows = dedup_ci(allowed_citation_rows + [f"[Chunk {cid}]" for cid in context_chunk_ids])
        allowed_citations = ", ".join(allowed_citation_rows[:16])

        enriched = enrich_question(pending_query, final_context, hits, intent_hint=intent_hint)
        st.session_state.pending_enrichment = enriched if enriched else {}
        enrichment_summary = format_enrichment_summary(enriched)
        stage_line.markdown(
            (
                f"`Intent:` {intent_label} | anchors: {anchor_preview}\n\n`{retrieval_debug_line}`\n\n`Query plan:` {enrichment_summary}"
                if enrichment_summary
                else f"`Intent:` {intent_label} | anchors: {anchor_preview}\n\n`{retrieval_debug_line}`\n\n`Query plan: direct retrieval`"
            )
        )

        enriched_question = (enriched.get("enriched_question") or enriched.get("q") or pending_query) if enriched else pending_query
        insight_intents = (enriched.get("insight_intents") or enriched.get("intents") or []) if enriched else []
        intent_text = ", ".join([str(i) for i in insight_intents]) if isinstance(insight_intents, list) else ""
        viz_hint = (enriched.get("visualization_hint") or enriched.get("viz") or "none") if enriched else "none"
        should_visualize = bool(enriched.get("should_visualize", str(viz_hint).lower() in {"bar", "line"})) if enriched else False
        use_enrichment = should_use_enrichment(enriched)
        answer_context = final_context
        viz_line = (
            "A visualization would be helpful. Include a <visualization>{...}</visualization> JSON block."
            if should_visualize
            else "Include a <visualization>{...}</visualization> JSON block only if it adds clear value."
        )
        answer_prompt = (
            "You are a precise analyst. Use ONLY the provided context. "
            "If the context does not contain the answer, say so clearly.\n\n"
            f"Enriched question: {enriched_question if use_enrichment else pending_query}\n"
            f"Insight intents: {intent_text if use_enrichment else 'direct_lookup'}\n"
            f"Visualization hint: {viz_hint if use_enrichment else 'none'}\n"
            f"Retrieval confidence score (higher is better): {retrieval_confidence:.2f}\n"
            f"Allowed citations: {allowed_citations if allowed_citations else '[Chunk N]'}\n"
            "Answer requirements:\n"
            "- Provide a concise answer followed by 3-6 bullet insights.\n"
            "- Every bullet must include at least one citation from Allowed citations.\n"
            "- If you include numbers, they must be cited.\n"
            f"- {viz_line} Use keys type (bar|line), title, data: [{{label, value}}].\n"
            "- Include a <hero_figures>[...]</hero_figures> JSON block with up to 4 entries: "
            "[{label, value, delta, source}] for the most important metrics.\n\n"
            "- Do not output raw JSON outside the visualization and hero_figures tags.\n\n"
            f"Context:\n{answer_context}\n\n"
            f"Question: {pending_query}\nAnswer:"
        )
        stage_line.markdown("`Thinking and drafting answer...`")
        last_think_draw = [0.0]

        def on_thinking_update(think_text: str, answer_preview: str) -> None:
            now = time.time()
            if now - last_think_draw[0] < 0.25:
                return
            last_think_draw[0] = now
            with thinking_panel.container():
                with st.expander("Live reasoning trace", expanded=True):
                    st.markdown((think_text[-7000:].strip() if think_text else "_Thinking..._"))
                    if answer_preview:
                        st.markdown("---")
                        st.markdown(answer_preview[-1500:].strip())

        anchor_signal = float(evidence.answerability_signals.get("anchor_coverage_score", 0.0)) if retrieval_index is not None else 0.0
        derivation_active = bool(intent_hint.get("requires_derivation", False))
        low_confidence_gate = retrieval_confidence < 0.15 and len(valid_chunk_ids) == 0 and not has_graph_context
        if derivation_active and anchor_signal > 0:
            low_confidence_gate = False
        if low_confidence_gate:
            answer_raw = ""
            thinking_trace = ""
            stream_done_reason = ""
        else:
            answer_raw, thinking_trace, stream_done_reason = call_ollama_stream(
                answer_prompt,
                model=BRAIN_MODEL,
                timeout=720,
                options={"temperature": 0.1, "num_predict": 1600, "num_ctx": 8192},
                on_update=on_thinking_update,
            )
        if not thinking_trace:
            thinking_panel.empty()

        verdict = "pass"
        issues = ""
        if low_confidence_gate:
            verdict = "not_found"
            answer_text = "I couldn't find enough grounded evidence in the graph for this question."
            viz = {}
            hero_figures = []
        elif is_ollama_error(answer_raw):
            answer_text = answer_raw
            viz = {}
            hero_figures = []
        else:
            combined_answer_raw = answer_raw
            if should_continue_answer(answer_raw, stream_done_reason):
                tail = call_ollama(
                    "Continue the answer from where it stopped. Keep the same citations and structure. "
                    "Do not repeat completed lines.\n\n"
                    f"Partial answer:\n{answer_raw[-2200:]}\n\nContinuation:",
                    model=BRAIN_MODEL,
                    timeout=220,
                    options={"temperature": 0.0, "num_predict": 600, "num_ctx": 8192},
                )
                if not is_ollama_error(tail):
                    combined_answer_raw = (answer_raw.rstrip() + "\n" + tail.lstrip()).strip()
            answer_text, viz, hero_figures = extract_answer_artifacts(combined_answer_raw)
            answer_text, viz, hero_figures = sanitize_assistant_payload(answer_text, viz, hero_figures)
            rule_eval = quick_grounding_check(answer_text, valid_chunk_ids=valid_chunk_ids, has_graph_context=has_graph_context)
            verdict = str(rule_eval.get("verdict", "pass")).lower()
            issues = str(rule_eval.get("issues", "")).strip()
            if verdict == "pass":
                judge_context = build_judge_context(
                    answer_text=answer_text,
                    chunk_lines=chunk_lines_list,
                    graph_lines=graph_lines_list,
                    fallback_context=answer_context,
                )
                judge = judge_answer(pending_query, answer_text, judge_context)
                judge_verdict = str(judge.get("verdict", "pass")).lower()
                judge_issues = str(judge.get("issues", judge.get("issue", ""))).strip()
                if (
                    judge_verdict == "not_found"
                    and citations_supported_in_context(answer_text, judge_context)
                    and not re.search(r"contradict|inconsisten|conflict", judge_issues, flags=re.IGNORECASE)
                ):
                    verdict = "pass"
                    issues = "judge_override_cited_context"
                else:
                    verdict = judge_verdict
                    issues = judge_issues
            if verdict == "retry":
                retry_prompt = answer_prompt + "\n\nThe previous answer had issues: " + issues + "\nFix using ONLY allowed citations. If unsupported, answer: not found."
                retry_raw = call_ollama(
                    retry_prompt,
                    model=BRAIN_MODEL,
                    timeout=420,
                    options={"temperature": 0, "num_predict": 1000, "num_ctx": 8192},
                )
                if not is_ollama_error(retry_raw):
                    answer_text, viz, hero_figures = extract_answer_artifacts(retry_raw)
                    answer_text, viz, hero_figures = sanitize_assistant_payload(answer_text, viz, hero_figures)
                    rule_eval = quick_grounding_check(answer_text, valid_chunk_ids=valid_chunk_ids, has_graph_context=has_graph_context)
                    verdict = str(rule_eval.get("verdict", "pass")).lower()
                else:
                    verdict = "not_found"
                    answer_text = "I couldn't find that information in the graph or retrieved context."
                    viz = {}
                    hero_figures = []
            if verdict == "retry":
                verdict = "not_found"
                answer_text = "I couldn't find that information in the graph or retrieved context."
                viz = {}
                hero_figures = []

        if verdict == "not_found":
            answer_text = "I couldn't find that information in the graph or retrieved context."
            viz = {}
            hero_figures = []
            set_status_bubble("Answer not found in current graph context.", "warning")
        elif not is_ollama_error(answer_text):
            answer_text, viz, hero_figures = sanitize_assistant_payload(answer_text, viz, hero_figures)
            set_status_bubble("Answer ready.", "success")

        st.session_state.chat_history.append(
            {
                "role": "assistant",
                "content": answer_text,
                "viz": viz,
                "hero_figures": hero_figures,
                "thinking": thinking_trace[-12000:] if thinking_trace else "",
                "enrichment": st.session_state.get("pending_enrichment", {}) if use_enrichment else {},
                "enrichment_summary": enrichment_summary if use_enrichment else "",
            }
        )
        persist_active_chat_history()
        st.session_state.pending_query = ""
        st.session_state.pending_enrichment = {}
        stage_line.empty()
        do_rerun()
