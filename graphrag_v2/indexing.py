from __future__ import annotations

import pickle
import re
from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from .types import ChunkRecord
from .utils import normalize_ws

try:
    from rank_bm25 import BM25Okapi
except Exception:  # pragma: no cover
    BM25Okapi = None


@dataclass
class RetrievalIndex:
    chunk_records: List[ChunkRecord]
    chunk_texts: List[str]
    embeddings: np.ndarray
    bm25_tokens: List[List[str]]
    bm25_model: object
    metric_map: Dict[str, List[int]]
    entity_map: Dict[str, List[int]]


def _tokenize(text: str) -> List[str]:
    return [t for t in re.findall(r"[a-z0-9]{2,}", str(text or "").lower()) if t not in {"the", "and", "for", "with", "this", "that"}]


METRIC_HINTS = {
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
    "cost",
    "growth",
}


def _build_metric_map(chunk_texts: List[str]) -> Dict[str, List[int]]:
    metric_map: Dict[str, List[int]] = {}
    for idx, text in enumerate(chunk_texts):
        toks = _tokenize(text)
        seen = set()
        for tok in toks:
            if tok not in METRIC_HINTS:
                continue
            if tok in seen:
                continue
            seen.add(tok)
            metric_map.setdefault(tok, []).append(idx)
    return metric_map


def _build_entity_map(chunk_texts: List[str]) -> Dict[str, List[int]]:
    entity_map: Dict[str, List[int]] = {}
    for idx, text in enumerate(chunk_texts):
        for token in re.findall(r"\b[A-Z][A-Za-z0-9&\.\-]{2,}\b", str(text or "")):
            key = normalize_ws(token).lower()
            if not key:
                continue
            entity_map.setdefault(key, []).append(idx)
    return entity_map


def build_retrieval_index(
    embedder,
    retrieval_chunks: List[ChunkRecord],
) -> RetrievalIndex:
    chunk_texts = [c.text for c in retrieval_chunks]
    embeddings = np.array([])
    if chunk_texts:
        embeddings = np.array(embedder.encode(chunk_texts, show_progress_bar=False))
    bm25_tokens = [_tokenize(t) for t in chunk_texts]
    bm25_model = BM25Okapi(bm25_tokens) if (BM25Okapi is not None and bm25_tokens) else None
    metric_map = _build_metric_map(chunk_texts)
    entity_map = _build_entity_map(chunk_texts)
    return RetrievalIndex(
        chunk_records=retrieval_chunks,
        chunk_texts=chunk_texts,
        embeddings=embeddings,
        bm25_tokens=bm25_tokens,
        bm25_model=bm25_model,
        metric_map=metric_map,
        entity_map=entity_map,
    )


def save_bm25(path: str, idx: RetrievalIndex) -> None:
    payload = {
        "bm25_tokens": idx.bm25_tokens,
        "metric_map": idx.metric_map,
        "entity_map": idx.entity_map,
    }
    with open(path, "wb") as f:
        pickle.dump(payload, f)


def load_retrieval_index(path: str) -> Dict[str, object]:
    with open(path, "rb") as f:
        return pickle.load(f)


def dense_scores(question: str, embeddings: np.ndarray, embedder) -> np.ndarray:
    if embeddings is None or getattr(embeddings, "size", 0) == 0:
        return np.array([])
    q_vec = embedder.encode([question], show_progress_bar=False)[0]
    denom = np.linalg.norm(embeddings, axis=1) * (np.linalg.norm(q_vec) + 1e-8)
    return (embeddings @ q_vec) / (denom + 1e-8)


def sparse_scores(question: str, bm25_model, bm25_tokens: List[List[str]]) -> np.ndarray:
    if bm25_model is None:
        return np.array([])
    q_toks = _tokenize(question)
    if not q_toks:
        return np.zeros(len(bm25_tokens), dtype=np.float32)
    scores = bm25_model.get_scores(q_toks)
    return np.array(scores, dtype=np.float32)


def mmr_select(
    candidate_ids: List[int],
    candidate_scores: Dict[int, float],
    embeddings: np.ndarray,
    top_k: int = 10,
    lambda_mult: float = 0.72,
) -> List[int]:
    if not candidate_ids:
        return []
    selected: List[int] = []
    remaining = list(candidate_ids)
    while remaining and len(selected) < top_k:
        best = None
        best_score = -1e9
        for cid in remaining:
            rel = candidate_scores.get(cid, 0.0)
            div_penalty = 0.0
            if selected and embeddings is not None and getattr(embeddings, "size", 0) > 0:
                vec = embeddings[cid]
                sims = []
                for sid in selected:
                    ref = embeddings[sid]
                    denom = (np.linalg.norm(vec) * np.linalg.norm(ref)) + 1e-8
                    sims.append(float(np.dot(vec, ref) / denom))
                div_penalty = max(sims) if sims else 0.0
            score = lambda_mult * rel - (1.0 - lambda_mult) * div_penalty
            if score > best_score:
                best_score = score
                best = cid
        if best is None:
            break
        selected.append(best)
        remaining = [x for x in remaining if x != best]
    return selected
