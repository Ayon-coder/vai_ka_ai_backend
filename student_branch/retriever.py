"""
Firestore-backed semantic retrieval for the IEEE Student Branch assistant.

Member embeddings are created offline and stored in Firestore. This module
embeds an incoming question, runs a nearest-neighbour query, and returns only
directory fields that are safe for the answer-generation prompt.
"""

import asyncio
import base64
import json
import math
import os
import threading
from typing import Any

import httpx


DEFAULT_COLLECTION = "ieee-members"
DEFAULT_TOP_K = 4
DEFAULT_MIN_SIMILARITY = 0.35
EMBEDDING_MODEL = "gemini-embedding-001"
EMBEDDING_DIMENSIONS = 768
EMBEDDING_TIMEOUT = 12.0

_firebase_lock = threading.Lock()
_firestore_client = None


def _env(name: str, default: str = "") -> str:
    value = os.getenv(name, default).strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
        value = value[1:-1].strip()
    return value


def _retrieval_configured() -> bool:
    return bool(_env("FIREBASE_CREDS_BASE64") and _env("GOOGLE_API_KEY"))


def _get_firestore_client():
    """Create one Firebase Admin client per worker/process."""
    global _firestore_client
    if _firestore_client is not None:
        return _firestore_client

    with _firebase_lock:
        if _firestore_client is not None:
            return _firestore_client

        import firebase_admin
        from firebase_admin import credentials, firestore

        if not firebase_admin._apps:
            raw_credentials = _env("FIREBASE_CREDS_BASE64")
            try:
                decoded = base64.b64decode(raw_credentials).decode("utf-8")
                service_account = json.loads(decoded)
            except (ValueError, UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise RuntimeError("FIREBASE_CREDS_BASE64 is not valid service-account JSON") from exc
            firebase_admin.initialize_app(credentials.Certificate(service_account))

        _firestore_client = firestore.client()
        return _firestore_client


async def _embed_query(query: str) -> list[float]:
    """Create a query vector with the same model and dimensions used at ingestion."""
    api_key = _env("GOOGLE_API_KEY")
    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"{EMBEDDING_MODEL}:embedContent"
    )
    payload = {
        "model": f"models/{EMBEDDING_MODEL}",
        "content": {"parts": [{"text": query}]},
        "taskType": "RETRIEVAL_QUERY",
        "outputDimensionality": EMBEDDING_DIMENSIONS,
    }

    async with httpx.AsyncClient(timeout=EMBEDDING_TIMEOUT) as client:
        response = await client.post(
            url,
            headers={"x-goog-api-key": api_key},
            json=payload,
        )
        response.raise_for_status()
        values = response.json().get("embedding", {}).get("values")

    if not isinstance(values, list) or len(values) != EMBEDDING_DIMENSIONS:
        raise RuntimeError("Embedding provider returned an unexpected vector")
    return [float(value) for value in values]


def _vector_values(value: Any) -> list[float]:
    """Convert Firestore Vector/list representations into numeric values."""
    for attribute in ("value", "values", "_value"):
        candidate = getattr(value, attribute, None)
        if callable(candidate):
            candidate = candidate()
        if candidate is not None:
            return [float(item) for item in candidate]
    return [float(item) for item in value]


def _cosine_similarity(first: list[float], second: list[float]) -> float:
    if len(first) != len(second):
        return 0.0
    first_norm = math.sqrt(sum(item * item for item in first))
    second_norm = math.sqrt(sum(item * item for item in second))
    if not first_norm or not second_norm:
        return 0.0
    return sum(a * b for a, b in zip(first, second)) / (first_norm * second_norm)


def _similarity_threshold() -> float:
    try:
        return max(
            0.0,
            min(
                1.0,
                float(_env("STUDENT_BRANCH_MIN_SIMILARITY", str(DEFAULT_MIN_SIMILARITY))),
            ),
        )
    except ValueError:
        return DEFAULT_MIN_SIMILARITY


def _top_k() -> int:
    try:
        return max(1, min(10, int(_env("STUDENT_BRANCH_TOP_K", str(DEFAULT_TOP_K)))))
    except ValueError:
        return DEFAULT_TOP_K


def _member_result(data: dict[str, Any], similarity: float) -> dict[str, Any]:
    """Keep only directory fields that the public chatbot may use."""
    metadata = data.get("metadata")
    member = metadata if isinstance(metadata, dict) else data
    result = {
        "id": member.get("id", data.get("id")),
        "name": member.get("name", data.get("name")),
        "team": member.get("team", data.get("team")),
        "department": member.get("department", data.get("department")),
        "college_email": member.get("college_email", data.get("college_email")),
        "linkedin_url": member.get("linkedin_url", data.get("linkedin_url")),
        "bio": member.get("bio", data.get("bio")),
        "keywords_traits": member.get("keywords_traits", data.get("keywords_traits")),
        "inspiration_drive": member.get("inspiration_drive", data.get("inspiration_drive")),
        "quote_motto": member.get("quote_motto", data.get("quote_motto")),
        "content": data.get("content", ""),
        "similarity": round(float(similarity), 4),
    }
    return {key: value for key, value in result.items() if value is not None and value != ""}


def _search_firestore(query_vector: list[float]) -> list[dict[str, Any]]:
    """Run native Firestore vector search, with an exact-search fallback."""
    from google.cloud.firestore_v1.base_vector_query import DistanceMeasure
    from google.cloud.firestore_v1.vector import Vector

    client = _get_firestore_client()
    collection = client.collection(_env("FIREBASE_COLLECTION", DEFAULT_COLLECTION))
    top_k = _top_k()

    try:
        vector_query = collection.find_nearest(
            vector_field="embedding",
            query_vector=Vector(query_vector),
            distance_measure=DistanceMeasure.COSINE,
            limit=top_k,
            distance_result_field="vector_distance",
        )
        scored = []
        for document in vector_query.stream():
            data = document.to_dict() or {}
            distance = data.get("vector_distance")
            similarity = 1.0 - float(distance) if distance is not None else 0.0
            scored.append(_member_result(data, similarity))
    except Exception as exc:
        # This fallback is useful while the Firestore vector index is pending.
        print(f"[Student RAG] Native vector search unavailable ({type(exc).__name__}); using exact fallback.")
        scored = []
        for document in collection.stream():
            data = document.to_dict() or {}
            embedding = data.get("embedding")
            if embedding is None:
                continue
            try:
                similarity = _cosine_similarity(query_vector, _vector_values(embedding))
            except (TypeError, ValueError):
                continue
            scored.append(_member_result(data, similarity))
        scored.sort(key=lambda item: item.get("similarity", 0.0), reverse=True)
        scored = scored[:top_k]

    threshold = _similarity_threshold()
    return [item for item in scored if item.get("similarity", 0.0) >= threshold]


async def retrieve_student_branch(query: str) -> list[dict[str, Any]]:
    """Retrieve verified member records relevant to a Student Branch question."""
    if not query or not query.strip() or not _retrieval_configured():
        return []

    try:
        query_vector = await _embed_query(query.strip())
        return await asyncio.to_thread(_search_firestore, query_vector)
    except Exception as exc:
        print(f"[Student RAG] Retrieval failed ({type(exc).__name__}).")
        return []
