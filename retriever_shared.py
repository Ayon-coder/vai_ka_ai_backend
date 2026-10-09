"""
retriever_shared.py
-------------------
Shared singleton resources for all FirebaseStudentRetriever instances.

Provides:
- One GoogleGenerativeAIEmbeddings instance reused across every request
  (avoids the ~700 ms re-initialisation overhead measured per call).
- A Firestore document cache with a 30-minute TTL so the ~1900 ms cold-fetch
  only happens once (or at most every 30 min) instead of once per provider.
- An LRU cache for query embedding vectors (saves ~550 ms for repeated
  or near-identical queries within the same server session).
- Numpy-accelerated batch cosine scoring (3.6 ms for 65 docs vs ~20 ms
  pure-Python; graceful pure-Python fallback if numpy is missing).
"""

from __future__ import annotations

import asyncio
import base64
import functools
import json
import math
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import firebase_admin
from firebase_admin import credentials, firestore
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# ---------------------------------------------------------------------------
# Embedding model singleton
# ---------------------------------------------------------------------------

_embedding_model: Optional[GoogleGenerativeAIEmbeddings] = None


def get_embedding_model() -> GoogleGenerativeAIEmbeddings:
    """Return the process-wide singleton embedding model, creating it on first call."""
    global _embedding_model
    if _embedding_model is None:
        from key_manager import gemini_keys
        _embedding_model = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001",
            google_api_key=gemini_keys.get_next_key(),
            output_dimensionality=768,
        )
    return _embedding_model


# ---------------------------------------------------------------------------
# Firestore client singleton
# ---------------------------------------------------------------------------


def get_firestore_client() -> Any:
    """Return a Firestore client, initialising Firebase Admin SDK if needed."""
    if not firebase_admin._apps:
        creds_b64 = os.getenv("FIREBASE_CREDS_BASE64")
        if not creds_b64:
            raise RuntimeError("FIREBASE_CREDS_BASE64 environment variable is not set.")
        decoded = base64.b64decode(creds_b64).decode("utf-8")
        firebase_admin.initialize_app(credentials.Certificate(json.loads(decoded)))
    return firestore.client()


# ---------------------------------------------------------------------------
# Document cache (shared across gemini + groq retrievers)
# ---------------------------------------------------------------------------

_doc_cache: Optional[List[Tuple[Dict[str, Any], str, str]]] = None
_vec_cache: Optional[List[List[float]]] = None
_cache_ts: float = 0.0
CACHE_TTL_SECONDS: int = 1800  # 30 minutes


def _extract_vector(value: Any) -> Optional[List[float]]:
    """Extract a float list from a Firestore MapVector / plain list field."""
    for attr in ("value", "values", "_value"):
        candidate = getattr(value, attr, None)
        if callable(candidate):
            candidate = candidate()
        if candidate is not None:
            return [float(x) for x in candidate]
    try:
        return [float(x) for x in value]
    except Exception:
        return None


def _load_from_firestore() -> List[Tuple[Dict[str, Any], str, str]]:
    """Fetch documents from all IEEE Student Branch collections."""
    client = get_firestore_client()
    entries: List[Tuple[Dict[str, Any], str, str]] = []

    fetch_plan = [
        ("ieee_student_branch", lambda: client.collection("ieee_student_branch").stream()),
        ("teams_overview", lambda: client.collection("teams_overview").stream()),
        ("team_members_details", lambda: client.collection("team_members_details").stream()),
        ("members", lambda: client.collection_group("members").stream()),
    ]

    for coll_name, fetcher in fetch_plan:
        try:
            for doc in fetcher():
                data = doc.to_dict() or {}
                entries.append((data, doc.id, coll_name))
        except Exception as exc:
            print(f"[retriever_shared] Warning fetching '{coll_name}': {exc}")

    return entries


def get_cached_docs(
    force_refresh: bool = False,
) -> Tuple[List[Tuple[Dict, str, str]], List[List[float]]]:
    """
    Return (entries, doc_vectors) using an in-memory cache with a 30-min TTL.

    Both lists are index-aligned: entries[i] corresponds to doc_vectors[i].
    Only documents that contain a valid embedding field are included.
    """
    global _doc_cache, _vec_cache, _cache_ts

    now = time.monotonic()
    if _doc_cache is None or (now - _cache_ts) > CACHE_TTL_SECONDS or force_refresh:
        raw_entries = _load_from_firestore()
        filtered_entries: List[Tuple[Dict, str, str]] = []
        filtered_vecs: List[List[float]] = []

        for data, doc_id, coll in raw_entries:
            emb_field = data.get("embedding")
            if emb_field is None:
                continue
            vec = _extract_vector(emb_field)
            if vec is None:
                continue
            filtered_entries.append((data, doc_id, coll))
            filtered_vecs.append(vec)

        _doc_cache = filtered_entries
        _vec_cache = filtered_vecs
        _cache_ts = now
        print(f"[retriever_shared] Cache refreshed - {len(_doc_cache)} embedded docs loaded.")

    return _doc_cache, _vec_cache  # type: ignore[return-value]


async def get_cached_docs_async(force_refresh: bool = False):
    """Async wrapper: runs the synchronous Firestore fetch off the event loop."""
    return await asyncio.to_thread(get_cached_docs, force_refresh)


# ---------------------------------------------------------------------------
# Query-vector LRU cache
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=64)
def _embed_query_cached(query: str) -> tuple:
    """
    Embed query and return the vector as a tuple (required for LRU hashability).
    Results are cached across calls within the same server process.
    """
    model = get_embedding_model()
    return tuple(model.embed_query(query))


async def embed_query_async(query: str) -> List[float]:
    """
    Async, LRU-cached query embedding.
    Runs off the FastAPI event loop to avoid blocking.
    """
    result = await asyncio.to_thread(_embed_query_cached, query)
    return list(result)


# ---------------------------------------------------------------------------
# Batch cosine similarity (numpy-accelerated, pure-Python fallback)
# ---------------------------------------------------------------------------


def batch_cosine_scores(
    query_vec: List[float],
    doc_vecs: List[List[float]],
) -> List[float]:
    """
    Compute cosine similarity of query_vec against every vector in doc_vecs.
    Uses numpy (3.6 ms for 65 docs) when available; falls back to pure Python.
    """
    if not doc_vecs:
        return []

    try:
        import numpy as np

        q = np.array(query_vec, dtype=np.float32)
        M = np.array(doc_vecs, dtype=np.float32)
        q_norm = np.linalg.norm(q)
        if q_norm == 0.0:
            return [0.0] * len(doc_vecs)
        m_norms = np.linalg.norm(M, axis=1)
        denom = m_norms * q_norm
        denom[denom == 0.0] = 1e-9
        return (M @ q / denom).tolist()

    except ImportError:
        def _cosine(a: List[float], b: List[float]) -> float:
            dot = norm_a = norm_b = 0.0
            for x, y in zip(a, b):
                dot += x * y
                norm_a += x * x
                norm_b += y * y
            if not norm_a or not norm_b:
                return 0.0
            return dot / (math.sqrt(norm_a) * math.sqrt(norm_b))

        return [_cosine(query_vec, dv) for dv in doc_vecs]
