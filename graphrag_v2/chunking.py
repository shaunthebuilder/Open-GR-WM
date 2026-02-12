from __future__ import annotations

import hashlib
import re
from typing import Dict, List, Tuple

from .types import ChunkRecord
from .utils import approx_tokens, now_iso, split_sections


def _chunk_by_tokens(text: str, target_tokens: int, overlap_tokens: int) -> List[Tuple[str, int, int]]:
    src = str(text or "")
    if not src.strip():
        return []

    # Build word spans to map chunks back to character ranges.
    words = list(re.finditer(r"\S+", src))
    if not words:
        return []

    stride = max(1, target_tokens - overlap_tokens)
    chunks: List[Tuple[str, int, int]] = []
    i = 0
    while i < len(words):
        j = min(len(words), i + target_tokens)
        start = words[i].start()
        end = words[j - 1].end()
        part = src[start:end].strip()
        if part:
            chunks.append((part, start, end))
        if j >= len(words):
            break
        i += stride
    return chunks


def _make_chunk_id(source_id: str, idx: int, page_idx: int, text: str, track: str) -> str:
    digest = hashlib.md5(f"{source_id}|{idx}|{page_idx}|{track}|{text[:140]}".encode("utf-8")).hexdigest()[:10]
    return f"{track}-{page_idx + 1}-{idx + 1}-{digest}"


def build_dual_chunks(
    source_id: str,
    page_or_doc_idx: int,
    text: str,
    extraction_tokens: int,
    extraction_overlap: int,
    retrieval_tokens: int,
    retrieval_overlap: int,
    is_vision: bool = False,
) -> Tuple[List[ChunkRecord], List[ChunkRecord]]:
    sections = split_sections(text)
    created_at = now_iso()

    extraction_chunks: List[ChunkRecord] = []
    retrieval_chunks: List[ChunkRecord] = []

    ex_counter = 0
    re_counter = 0

    for sec_title, sec_text, sec_start, _sec_end in sections:
        ex_parts = _chunk_by_tokens(sec_text, extraction_tokens, extraction_overlap)
        for part, rel_start, rel_end in ex_parts:
            chunk_id = _make_chunk_id(source_id, ex_counter, page_or_doc_idx, part, "ex")
            extraction_chunks.append(
                ChunkRecord(
                    source_id=source_id,
                    page_or_doc_idx=page_or_doc_idx,
                    chunk_id=chunk_id,
                    section_title=sec_title,
                    char_start=sec_start + rel_start,
                    char_end=sec_start + rel_end,
                    is_vision=is_vision,
                    created_at=created_at,
                    text=part,
                )
            )
            ex_counter += 1

        re_parts = _chunk_by_tokens(sec_text, retrieval_tokens, retrieval_overlap)
        for part, rel_start, rel_end in re_parts:
            chunk_id = _make_chunk_id(source_id, re_counter, page_or_doc_idx, part, "re")
            retrieval_chunks.append(
                ChunkRecord(
                    source_id=source_id,
                    page_or_doc_idx=page_or_doc_idx,
                    chunk_id=chunk_id,
                    section_title=sec_title,
                    char_start=sec_start + rel_start,
                    char_end=sec_start + rel_end,
                    is_vision=is_vision,
                    created_at=created_at,
                    text=part,
                )
            )
            re_counter += 1

    return extraction_chunks, retrieval_chunks


def profile_to_chunk_params(profile: str) -> Dict[str, int]:
    mode = str(profile or "balanced").lower()
    if mode == "fast":
        return {
            "extraction_tokens": 190,
            "extraction_overlap": 24,
            "retrieval_tokens": 330,
            "retrieval_overlap": 46,
        }
    if mode == "quality":
        return {
            "extraction_tokens": 250,
            "extraction_overlap": 36,
            "retrieval_tokens": 470,
            "retrieval_overlap": 68,
        }
    return {
        "extraction_tokens": 220,
        "extraction_overlap": 30,
        "retrieval_tokens": 400,
        "retrieval_overlap": 56,
    }


def build_chunk_tracks_for_pages(
    source_id: str,
    page_texts: List[str],
    profile: str = "balanced",
) -> Tuple[List[ChunkRecord], List[ChunkRecord]]:
    params = profile_to_chunk_params(profile)
    ex_all: List[ChunkRecord] = []
    re_all: List[ChunkRecord] = []

    for page_idx, page_text in enumerate(page_texts):
        text = str(page_text or "").strip()
        if len(text) < 40:
            continue
        ex, re_chunks = build_dual_chunks(
            source_id=source_id,
            page_or_doc_idx=page_idx,
            text=text,
            extraction_tokens=params["extraction_tokens"],
            extraction_overlap=params["extraction_overlap"],
            retrieval_tokens=params["retrieval_tokens"],
            retrieval_overlap=params["retrieval_overlap"],
            is_vision=False,
        )
        ex_all.extend(ex)
        re_all.extend(re_chunks)

    return ex_all, re_all

