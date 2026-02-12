from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime
from typing import Dict, Iterable, List

import networkx as nx
import numpy as np

from .canonicalize import build_entities_table
from .indexing import RetrievalIndex, load_retrieval_index as load_bm25_payload, save_bm25
from .types import ChunkRecord, Fact

try:
    from rank_bm25 import BM25Okapi
except Exception:  # pragma: no cover
    BM25Okapi = None


RAG_STORE_DIR = os.path.join(os.getcwd(), "rag_store")


def _now_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"


def ensure_v2_dir(rag_id: str) -> str:
    rag = str(rag_id or "").strip() or "graph"
    root = os.path.join(RAG_STORE_DIR, rag, "v2")
    os.makedirs(root, exist_ok=True)
    return root


def _cache_db_path(rag_id: str) -> str:
    return os.path.join(ensure_v2_dir(rag_id), "extract_cache.sqlite")


def _cache_conn(rag_id: str):
    path = _cache_db_path(rag_id)
    conn = sqlite3.connect(path, timeout=30)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS extract_cache (
            cache_key TEXT PRIMARY KEY,
            value TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """
    )
    conn.commit()
    return conn


def cache_lookup(rag_id: str, cache_key: str) -> str:
    key = str(cache_key or "").strip()
    if not key:
        return ""
    conn = _cache_conn(rag_id)
    try:
        row = conn.execute(
            "SELECT value FROM extract_cache WHERE cache_key = ?",
            (key,),
        ).fetchone()
        return str(row[0]) if row else ""
    finally:
        conn.close()


def cache_store(rag_id: str, cache_key: str, value: str) -> None:
    key = str(cache_key or "").strip()
    if not key:
        return
    conn = _cache_conn(rag_id)
    try:
        conn.execute(
            "INSERT OR REPLACE INTO extract_cache(cache_key, value, updated_at) VALUES(?, ?, ?)",
            (key, str(value or ""), _now_iso()),
        )
        conn.commit()
    finally:
        conn.close()


def _checkpoint_path(rag_id: str) -> str:
    return os.path.join(ensure_v2_dir(rag_id), "build_checkpoint.json")


def save_build_checkpoint(rag_id: str, payload: Dict[str, object]) -> None:
    path = _checkpoint_path(rag_id)
    data = dict(payload or {})
    data["updated_at"] = _now_iso()
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def load_build_checkpoint(rag_id: str) -> Dict[str, object]:
    path = _checkpoint_path(rag_id)
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def clear_build_checkpoint(rag_id: str) -> None:
    path = _checkpoint_path(rag_id)
    if os.path.exists(path):
        os.remove(path)


def _chunk_row(chunk: ChunkRecord, track: str) -> Dict[str, object]:
    return {
        "track": track,
        "source_id": chunk.source_id,
        "page_or_doc_idx": int(chunk.page_or_doc_idx),
        "chunk_id": chunk.chunk_id,
        "section_title": chunk.section_title,
        "char_start": int(chunk.char_start),
        "char_end": int(chunk.char_end),
        "is_vision": bool(chunk.is_vision),
        "created_at": chunk.created_at,
        "text": chunk.text,
    }


def _chunk_from_row(row: Dict[str, object]) -> ChunkRecord:
    return ChunkRecord(
        source_id=str(row.get("source_id", "")),
        page_or_doc_idx=int(row.get("page_or_doc_idx", 0) or 0),
        chunk_id=str(row.get("chunk_id", "")),
        section_title=str(row.get("section_title", "")),
        char_start=int(row.get("char_start", 0) or 0),
        char_end=int(row.get("char_end", 0) or 0),
        is_vision=bool(row.get("is_vision", False)),
        created_at=str(row.get("created_at", "")),
        text=str(row.get("text", "")),
    )


def _fact_row(f: Fact) -> Dict[str, object]:
    return {
        "subject": f.subject,
        "predicate": f.predicate,
        "object": f.object,
        "value": f.value,
        "unit": f.unit,
        "timeframe": f.timeframe,
        "source_ref": f.source_ref,
        "confidence": float(f.confidence),
        "source_id": f.source_id,
        "chunk_id": f.chunk_id,
    }


def _fact_from_row(row: Dict[str, object]) -> Fact:
    return Fact(
        subject=str(row.get("subject", "")),
        predicate=str(row.get("predicate", "")),
        object=str(row.get("object", "")),
        value=str(row.get("value", "")),
        unit=str(row.get("unit", "")),
        timeframe=str(row.get("timeframe", "")),
        source_ref=str(row.get("source_ref", "")),
        confidence=float(row.get("confidence", 0.0) or 0.0),
        source_id=str(row.get("source_id", "")),
        chunk_id=str(row.get("chunk_id", "")),
    )


def _write_jsonl(path: str, rows: Iterable[Dict[str, object]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _read_jsonl(path: str) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    if not os.path.exists(path):
        return rows
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if isinstance(obj, dict):
                rows.append(obj)
    return rows


def save_v2_artifacts(
    rag_id: str,
    graph_name: str,
    graph: nx.MultiDiGraph,
    facts: List[Fact],
    extraction_chunks: List[ChunkRecord],
    retrieval_chunks: List[ChunkRecord],
    retrieval_index: RetrievalIndex,
    build_profile: str,
    models: Dict[str, str],
    source_files: List[str],
    timings: Dict[str, float],
    quality_metrics: Dict[str, object],
) -> Dict[str, object]:
    v2_dir = ensure_v2_dir(rag_id)
    now = _now_iso()
    manifest_path = os.path.join(v2_dir, "manifest.json")

    created_at = now
    if os.path.exists(manifest_path):
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                old = json.load(f)
            created_at = str(old.get("created_at", now))
        except Exception:
            created_at = now

    with open(os.path.join(v2_dir, "graph.json"), "w", encoding="utf-8") as f:
        json.dump(nx.node_link_data(graph), f)
    _write_jsonl(os.path.join(v2_dir, "facts.jsonl"), [_fact_row(x) for x in facts])
    _write_jsonl(os.path.join(v2_dir, "entities.jsonl"), build_entities_table(facts))
    _write_jsonl(
        os.path.join(v2_dir, "chunks.jsonl"),
        [_chunk_row(c, "extraction") for c in extraction_chunks] + [_chunk_row(c, "retrieval") for c in retrieval_chunks],
    )
    np.save(os.path.join(v2_dir, "embeddings.npy"), retrieval_index.embeddings)
    save_bm25(os.path.join(v2_dir, "bm25_index.pkl"), retrieval_index)
    with open(os.path.join(v2_dir, "eval_report.json"), "w", encoding="utf-8") as f:
        json.dump(quality_metrics or {}, f, indent=2)

    manifest = {
        "format_version": "2.0",
        "rag_id": rag_id,
        "graph_name": graph_name,
        "created_at": created_at,
        "updated_at": now,
        "build_profile": str(build_profile or "balanced").lower(),
        "models": dict(models or {}),
        "source_files": list(source_files or []),
        "timings": dict(timings or {}),
        "quality_metrics": dict(quality_metrics or {}),
        "counts": {
            "nodes": int(graph.number_of_nodes()),
            "edges": int(graph.number_of_edges()),
            "facts": int(len(facts)),
            "retrieval_chunks": int(len(retrieval_chunks)),
            "extraction_chunks": int(len(extraction_chunks)),
        },
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return manifest


def load_v2_artifacts(rag_id: str) -> Dict[str, object]:
    v2_dir = ensure_v2_dir(rag_id)
    manifest_path = os.path.join(v2_dir, "manifest.json")
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"v2 manifest not found for graph '{rag_id}'.")

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    with open(os.path.join(v2_dir, "graph.json"), "r", encoding="utf-8") as f:
        graph_data = json.load(f)
    graph = nx.node_link_graph(graph_data, directed=True, multigraph=True)

    facts = [_fact_from_row(x) for x in _read_jsonl(os.path.join(v2_dir, "facts.jsonl"))]
    chunk_rows = _read_jsonl(os.path.join(v2_dir, "chunks.jsonl"))
    extraction_chunks = [_chunk_from_row(r) for r in chunk_rows if str(r.get("track", "")) == "extraction"]
    retrieval_chunks = [_chunk_from_row(r) for r in chunk_rows if str(r.get("track", "")) == "retrieval"]

    emb_path = os.path.join(v2_dir, "embeddings.npy")
    embeddings = np.load(emb_path) if os.path.exists(emb_path) else np.array([])
    bm25_path = os.path.join(v2_dir, "bm25_index.pkl")
    payload = load_bm25_payload(bm25_path) if os.path.exists(bm25_path) else {}
    bm25_tokens = payload.get("bm25_tokens", []) if isinstance(payload, dict) else []
    metric_map = payload.get("metric_map", {}) if isinstance(payload, dict) else {}
    entity_map = payload.get("entity_map", {}) if isinstance(payload, dict) else {}
    bm25_model = BM25Okapi(bm25_tokens) if (BM25Okapi is not None and bm25_tokens) else None

    retrieval_index = RetrievalIndex(
        chunk_records=retrieval_chunks,
        chunk_texts=[c.text for c in retrieval_chunks],
        embeddings=embeddings,
        bm25_tokens=bm25_tokens,
        bm25_model=bm25_model,
        metric_map=metric_map if isinstance(metric_map, dict) else {},
        entity_map=entity_map if isinstance(entity_map, dict) else {},
    )
    triples = [(f.subject, f.predicate, f.object) for f in facts]

    eval_report_path = os.path.join(v2_dir, "eval_report.json")
    eval_report = {}
    if os.path.exists(eval_report_path):
        try:
            with open(eval_report_path, "r", encoding="utf-8") as f:
                eval_report = json.load(f)
        except Exception:
            eval_report = {}

    return {
        "graph": graph,
        "facts": facts,
        "triples": triples,
        "extraction_chunks": extraction_chunks,
        "retrieval_chunks": retrieval_chunks,
        "retrieval_index": retrieval_index,
        "manifest": manifest,
        "eval_report": eval_report,
    }
