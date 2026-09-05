"""
Firestore-backed semantic retrieval for the IEEE Student Branch assistant.

Member embeddings are created offline and stored in Firestore using team
subcollections:  ieee-members/{team-slug}/members/{id-name}

This module supports two retrieval modes:
1. **Team listing** — detects queries like "who are the tech team members?"
   and returns ALL members of that team (bypasses top_k limit).
2. **Semantic search** — embeds the query and runs nearest-neighbour search
   across all team subcollections via a Firestore collection-group query.
"""

import asyncio
import base64
import json
import math
import os
import re
import threading
from typing import Any

import httpx


DEFAULT_COLLECTION = "ieee-members"
DEFAULT_BASIC_INFO_COLLECTION = "ieee-basic-info"
DEFAULT_TOP_K = 4
DEFAULT_MIN_SIMILARITY = 0.35
EMBEDDING_MODEL = "gemini-embedding-001"
EMBEDDING_DIMENSIONS = 768
EMBEDDING_TIMEOUT = 12.0

# Robust regex patterns for teams with common typos and aliases
TEAM_PATTERNS: dict[str, list[str]] = {
    "tech-team": [
        r"\b(tech|teach|tek|teck|technical|technology|developers?|coding|programmer|programming|web\s*dev|fullstack)\b",
    ],
    "pr-team": [
        r"\b(pr|public\s*relations?|outreach|relations|spokesperson)\b",
    ],
    "design-team": [
        r"\b(design|designers?|graphics?|ui\s*/?\s*ux|creatives?|poster|posters)\b",
    ],
    "content-team": [
        r"\b(content|writers?|writing|editorial|blogs?|articles?)\b",
    ],
    "media-team": [
        r"\b(media|videos?|photo|photos|photography|photographers?|camera|social\s*media)\b",
    ],
    "core-others": [
        r"\b(core|leads?|heads?|officers?|executives?|cabinet|management|committee|board|excom|eb|chair|vice\s*chair|secretary|treasurer)\b",
    ],
}

_ALL_TEAMS_PATTERNS = [
    r"\b(all\s+teams?|all\s+members?|every\s+member|list\s+(all\s+)?teams?|what\s+teams?|how\s+many\s+teams?|who\s+all\s+are\s+in|entire\s+team|everyone|all\s+committee|whole\s+team)\b",
    r"\b(what\s+are\s+(the\s+)?teams?|which\s+teams?|teams?\s+(are\s+there|exist|available)|teams?\s+in\s+ieee|ieee\s+teams?)\b",
    r"\b(tell\s+me\s+about\s+(the\s+)?teams?|about\s+teams?|who\s+are\s+the\s+teams?|teams?\s+and\s+(their\s+)?members?|list\s+of\s+teams?|teams?\s+overview|different\s+teams?|various\s+teams?)\b",
    r"^(teams?|our\s+teams?|show\s+teams?|give\s+teams?)$",
]

_LISTING_SIGNALS = [
    "members", "names", "list", "who are", "who is in", "who all",
    "people in", "show me", "tell me about", "everyone in",
    "all the", "give me the", "give me", "how many", "team", "domain", "dept", "department"
]

_firebase_lock = threading.Lock()
_firestore_client = None


def _env(name: str, default: str = "") -> str:
    value = os.getenv(name, default).strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
        value = value[1:-1].strip()
    return value


_google_key_lock = threading.Lock()
_google_key_index = 0


def _get_google_api_keys() -> list[str]:
    """Parse comma-separated Google API keys into a list."""
    raw = _env("GOOGLE_API_KEY")
    return [k.strip(' "\'') for k in raw.split(",") if k.strip()]


def _get_round_robin_google_keys() -> list[str]:
    """Return keys rotated by round-robin index for load balancing and failover."""
    keys = _get_google_api_keys()
    if not keys:
        return []
    global _google_key_index
    with _google_key_lock:
        idx = _google_key_index % len(keys)
        _google_key_index = (idx + 1) % len(keys)
    return keys[idx:] + keys[:idx]


def _retrieval_configured() -> bool:
    return bool(_env("FIREBASE_CREDS_BASE64") and _get_google_api_keys())


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


# ── Team-listing detection ────────────────────────────────────────────

def _detect_team_listing(query: str) -> str | None:
    """Return the Firestore team-slug if the query is asking about a team's members.

    Returns:
    - 'ALL_TEAMS' if asking for all teams/members
    - 'tech-team', 'pr-team', etc. if asking for a specific team
    - None if query is not team-related (e.g. asking about a person)
    """
    q = query.lower().strip()

    # Check for all-teams queries
    for pat in _ALL_TEAMS_PATTERNS:
        if re.search(pat, q, re.IGNORECASE):
            return "ALL_TEAMS"

    # Check for specific team patterns
    has_signal = any(signal in q for signal in _LISTING_SIGNALS)

    for slug, patterns in TEAM_PATTERNS.items():
        for pat in patterns:
            if re.search(pat, q, re.IGNORECASE):
                # If it explicitly matches team keyword AND has a signal or simply mentions team
                if has_signal or "team" in q or "domain" in q or "group" in q:
                    return slug
                # For words like 'tech', 'pr', 'design' combined with question words
                if any(w in q for w in ["who", "what", "which", "list", "tell", "show", "give"]):
                    return slug

    return None


# ── In-Memory Directory & Vector Cache ────────────────────────────────
# Eliminates 1.5–2.5s of Firestore network streaming on every single query.

_cache_lock = threading.Lock()
_teams_cache: dict[str, list[dict[str, Any]]] = {}
_teams_metadata_cache: dict[str, dict[str, Any]] = {}
_all_members_cache: list[dict[str, Any]] = []
_all_vectors_cache: list[tuple[dict[str, Any], list[float]]] = []
_basic_info_cache: dict[str, Any] = {}
_cache_timestamp: float = 0.0
CACHE_TTL = 300.0  # 5 minutes


def _load_cache_if_needed(force: bool = False):
    """Load team docs, member embeddings, and basic-info pool into RAM dynamically."""
    global _teams_cache, _teams_metadata_cache, _all_members_cache, _all_vectors_cache, _basic_info_cache, _cache_timestamp
    import time
    now = time.time()
    if not force and _all_vectors_cache and (now - _cache_timestamp < CACHE_TTL):
        return

    with _cache_lock:
        if not force and _all_vectors_cache and (now - _cache_timestamp < CACHE_TTL):
            return
        try:
            client = _get_firestore_client()
            
            # 1. Fetch from members collection (extracts members AND parent team descriptions)
            collection_name = _env("FIREBASE_COLLECTION", DEFAULT_COLLECTION)
            col_ref = client.collection(collection_name)
            
            new_teams = {}
            new_teams_metadata = {}
            new_members = []
            new_vectors = []

            for team_doc in col_ref.stream():
                team_data = team_doc.to_dict() or {}
                ts = team_doc.id
                m_list = team_data.get("members", [])
                
                # Format team members
                formatted_team = [_member_result(m, similarity=1.0) for m in m_list]
                new_teams[ts] = formatted_team

                # Extract team description, about, and answer directly from ieee-members parent document
                new_teams_metadata[ts] = {
                    "team_name": team_data.get("team_name") or ts.replace("-", " ").title(),
                    "description": team_data.get("description") or team_data.get("answer") or team_data.get("about") or "",
                    "answer": team_data.get("answer") or "",
                    "about": team_data.get("about") or "",
                    "slug": ts,
                    "total_members": team_data.get("total_members") or len(formatted_team),
                    "member_names": team_data.get("member_names") or [m.get("name") for m in formatted_team if m.get("name")],
                }

                # Fetch subcollection for vectors and complete team roster
                members_col = team_doc.reference.collection("members")
                for doc in members_col.stream():
                    d = doc.to_dict() or {}
                    res_m = _member_result(d, similarity=1.0)
                    new_members.append(res_m)
                    if not any(m.get("name") == res_m.get("name") for m in new_teams[ts]):
                        new_teams[ts].append(res_m)
                    emb = d.get("embedding")
                    if emb is not None:
                        try:
                            vec = _vector_values(emb)
                            new_vectors.append((d, vec))
                        except (TypeError, ValueError):
                            pass

                if not new_teams_metadata[ts]["member_names"]:
                    new_teams_metadata[ts]["member_names"] = [m.get("name") for m in new_teams[ts] if m.get("name")]
                new_teams_metadata[ts]["total_members"] = len(new_teams[ts])

            # 2. Fetch basic info collection if configured/present (merges without forcing single source)
            basic_info_col_name = _env("BASIC_INFO_COLLECTION", DEFAULT_BASIC_INFO_COLLECTION)
            new_basic_info = {}
            try:
                info_col_ref = client.collection(basic_info_col_name)
                for doc in info_col_ref.stream():
                    doc_data = doc.to_dict() or {}
                    new_basic_info[doc.id] = doc_data
                    # If this doc provides or enriches a team description, merge it dynamically
                    if doc.id in new_teams_metadata:
                        for k in ("description", "answer", "about", "team_name"):
                            if doc_data.get(k):
                                new_teams_metadata[doc.id][k] = doc_data[k]
                    elif doc.id not in ("teams", "branch"):
                        new_teams_metadata[doc.id] = {
                            "team_name": doc_data.get("team_name") or doc.id.replace("-", " ").title(),
                            "description": doc_data.get("description") or doc_data.get("answer") or doc_data.get("about") or "",
                            "answer": doc_data.get("answer") or "",
                            "about": doc_data.get("about") or "",
                            "slug": doc.id,
                        }
                if new_basic_info:
                    print(f"[Student RAG Cache] Merged {len(new_basic_info)} document(s) from '{basic_info_col_name}'.")
            except Exception as info_err:
                print(f"[Student RAG Cache] Note: '{basic_info_col_name}' pool not available or skipped ({info_err}). Using '{collection_name}' as primary.")

            # Fallback to local basic_info.json only if no team descriptions were found anywhere
            has_descriptions = any(bool(t.get("description")) for t in new_teams_metadata.values())
            if not has_descriptions:
                try:
                    for search_dir in [
                        os.path.join(os.path.dirname(__file__), "..", "..", "vector"),
                        os.path.join(os.path.dirname(__file__), "..", "vector"),
                        os.path.join(os.getcwd(), "vector"),
                    ]:
                        local_path = os.path.join(search_dir, "basic_info.json")
                        if os.path.exists(local_path):
                            with open(local_path, "r", encoding="utf-8") as f:
                                b_data = json.load(f)
                            teams_arr = b_data.get("team", [])
                            for t in teams_arr:
                                t_name = t.get("name", "")
                                t_slug = re.sub(r'[^a-z0-9]+', '-', t_name.lower()).strip('-')
                                if t_slug in new_teams_metadata:
                                    new_teams_metadata[t_slug]["description"] = t.get("answer", "")
                                    new_teams_metadata[t_slug]["answer"] = t.get("answer", "")
                            break
                except Exception:
                    pass

            _teams_cache = new_teams
            _teams_metadata_cache = new_teams_metadata
            _all_members_cache = new_members
            _all_vectors_cache = new_vectors
            _basic_info_cache = new_basic_info
            _cache_timestamp = now
            print(f"[Student RAG Cache] Refreshed {len(_all_vectors_cache)} member vectors across {len(_teams_cache)} teams in RAM.")
        except Exception as e:
            print(f"[Student RAG Cache] Warning: cache refresh failed ({e}); falling back.")


def _fetch_team_members(team_slug: str, query: str = "") -> list[dict[str, Any]]:
    """Fetch ALL members of a specific team alongside its official description in 0ms."""
    _load_cache_if_needed()

    overview_records = []

    if team_slug == "ALL_TEAMS":
        # Dynamic team listing from loaded metadata (no hardcoding)
        teams_list = []
        for ts, t_meta in _teams_metadata_cache.items():
            teams_list.append({
                "name": t_meta.get("team_name") or ts.replace("-", " ").title(),
                "slug": ts,
                "description": t_meta.get("description") or t_meta.get("answer") or t_meta.get("about") or "",
            })
        if not teams_list and "teams" in _basic_info_cache:
            teams_list = _basic_info_cache["teams"].get("teams", [])

        team_names_str = ", ".join([t['name'] for t in teams_list if t.get('name')])
        overview_records.append({
            "type": "all_teams_overview",
            "team_name": "IEEE Student Branch AOT Teams",
            "description": f"IEEE Student Branch AOT includes the following teams: {team_names_str}.",
            "all_teams": teams_list,
            "similarity": 1.0,
        })

        # Only provide individual member profiles if the user explicitly asked for members
        wants_members = any(w in query.lower() for w in ["member", "members", "who all", "everyone", "people", "names", "who is", "roster"])
        if wants_members:
            member_results = []
            if _all_members_cache:
                member_results = _all_members_cache
            else:
                for members in _teams_cache.values():
                    member_results.extend(members)
            return overview_records + member_results
        return overview_records

    # Specific team requested: get from _teams_metadata_cache (from ieee-members parent doc) or _basic_info_cache
    team_info = _teams_metadata_cache.get(team_slug) or _basic_info_cache.get(team_slug)
    if team_info:
        desc = team_info.get("description") or team_info.get("answer") or team_info.get("about") or ""
        overview_records.append({
            "type": "team_overview",
            "team_name": team_info.get("team_name") or team_slug.replace("-", " ").title(),
            "description": desc,
            "answer": team_info.get("answer") or desc,
            "about": team_info.get("about") or desc,
            "similarity": 1.0,
        })

    members = _teams_cache.get(team_slug, [])
    if not members:
        # Fallback to direct Firestore if cache miss
        try:
            client = _get_firestore_client()
            collection_name = _env("FIREBASE_COLLECTION", DEFAULT_COLLECTION)
            team_doc_ref = client.collection(collection_name).document(team_slug)
            team_snap = team_doc_ref.get()
            if team_snap.exists:
                team_data = team_snap.to_dict() or {}
                m_list = team_data.get("members")
                if m_list and isinstance(m_list, list):
                    members = [_member_result(m, similarity=1.0) for m in m_list]
                else:
                    for doc in team_doc_ref.collection("members").stream():
                        d = doc.to_dict() or {}
                        members.append(_member_result(d, similarity=1.0))
        except Exception:
            pass

    # Only return member records if the user explicitly asked for members/roster
    wants_members = any(w in query.lower() for w in ["member", "members", "who all", "everyone", "people", "names", "who is", "who are", "roster", "leads", "head", "chair", "lead"])
    if wants_members:
        return overview_records + members
    return overview_records


# ── Embedding with persistent HTTP Client ──────────────────────────────

_embedding_client_per_loop: dict[int, httpx.AsyncClient] = {}


def _get_embedding_client() -> httpx.AsyncClient:
    try:
        loop = asyncio.get_running_loop()
        loop_id = id(loop)
    except RuntimeError:
        loop_id = 0

    with _cache_lock:
        client = _embedding_client_per_loop.get(loop_id)
        if client is None or client.is_closed:
            client = httpx.AsyncClient(
                timeout=httpx.Timeout(EMBEDDING_TIMEOUT, connect=3.0),
                limits=httpx.Limits(max_keepalive_connections=10, max_connections=20, keepalive_expiry=60.0),
            )
            _embedding_client_per_loop[loop_id] = client
        return client


async def _embed_query(query: str) -> list[float]:
    """Create a query vector with connection reuse, round-robin key rotation, and 429 failover."""
    keys_to_try = _get_round_robin_google_keys()
    if not keys_to_try:
        raise RuntimeError("No GOOGLE_API_KEY configured")

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

    client = _get_embedding_client()
    last_err = None
    for api_key in keys_to_try:
        try:
            response = await client.post(
                url,
                headers={"x-goog-api-key": api_key},
                json=payload,
            )
            if response.status_code == 200:
                values = response.json().get("embedding", {}).get("values")
                if not isinstance(values, list) or len(values) != EMBEDDING_DIMENSIONS:
                    raise RuntimeError("Embedding provider returned an unexpected vector")
                return [float(value) for value in values]
            elif response.status_code == 429:
                masked_key = f"...{api_key[-4:]}" if len(api_key) > 4 else "key"
                print(f"[Google 429] Rate limit on embedding key {masked_key}; trying next key...")
                last_err = RuntimeError(f"Google embedding rate limit 429 on {masked_key}")
                continue
            else:
                response.raise_for_status()
        except Exception as exc:
            last_err = exc
            continue

    if last_err:
        raise last_err
    raise RuntimeError("All Google API keys failed for embedding")


# ── Similarity helpers ────────────────────────────────────────────────

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


# ── Result formatting ─────────────────────────────────────────────────

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


# ── Fast In-Memory Vector Search (0.2ms) ──────────────────────────────

def _search_firestore(query_vector: list[float]) -> list[dict[str, Any]]:
    """Run cosine similarity over in-memory cached vectors (sub-millisecond speed)."""
    _load_cache_if_needed()
    top_k = _top_k()

    if _all_vectors_cache:
        scored = []
        for doc_data, doc_vec in _all_vectors_cache:
            sim = _cosine_similarity(query_vector, doc_vec)
            scored.append(_member_result(doc_data, sim))
        scored.sort(key=lambda item: item.get("similarity", 0.0), reverse=True)
        scored = scored[:top_k]
    else:
        # Fallback to direct Firestore collection group if cache empty
        scored = []
        try:
            client = _get_firestore_client()
            members_group = client.collection_group("members")
            for document in members_group.stream():
                data = document.to_dict() or {}
                emb = data.get("embedding")
                if emb is None:
                    continue
                try:
                    sim = _cosine_similarity(query_vector, _vector_values(emb))
                    scored.append(_member_result(data, sim))
                except (TypeError, ValueError):
                    continue
            scored.sort(key=lambda item: item.get("similarity", 0.0), reverse=True)
            scored = scored[:top_k]
        except Exception as exc:
            print(f"[Student RAG] Direct search error: {exc}")

    threshold = _similarity_threshold()
    return [item for item in scored if item.get("similarity", 0.0) >= threshold]


# ── Public entry point ────────────────────────────────────────────────

_BRANCH_PATTERNS = [
    r"\b(what\s+is\s+ieee(\s+sb)?(\s+aot)?|about\s+ieee(\s+sb)?(\s+aot)?|tell\s+me\s+about\s+ieee(\s+sb)?(\s+aot)?|intro(duction)?\s+to\s+ieee(\s+sb)?(\s+aot)?|overview\s+of\s+ieee(\s+sb)?(\s+aot)?)\b",
    r"\b(how\s+(can\s+i|to)\s+(get\s+involved|join|be\s+part\s+of|participate|contribute)|get\s+involved|join\s+ieee|membership|how\s+to\s+register)\b",
    r"^(ieee(\s+sb)?(\s+aot)?|ieee\s+student\s+branch(\s+aot)?|academy\s+of\s+technology)$",
]


def _detect_branch_query(query: str) -> bool:
    q = query.lower().strip()
    if any(w in q for w in ["team", "teams", "member", "members", "lead", "leads"]):
        if not any(w in q for w in ["join", "involved", "participate", "register"]):
            return False
    return any(re.search(pat, q, re.IGNORECASE) for pat in _BRANCH_PATTERNS)


def _fetch_branch_info() -> list[dict[str, Any]]:
    """Fetch official IEEE Student Branch AOT overview and team list dynamically."""
    _load_cache_if_needed()

    # Dynamic team listing from loaded metadata (from ieee-members / ieee-basic-info)
    teams_list = [
        {
            "name": t_meta.get("team_name") or ts.replace("-", " ").title(),
            "slug": ts,
            "description": t_meta.get("description") or t_meta.get("answer") or t_meta.get("about") or "",
        }
        for ts, t_meta in _teams_metadata_cache.items()
    ]
    if not teams_list and "teams" in _basic_info_cache:
        teams_list = _basic_info_cache["teams"].get("teams", [])

    # Check if a custom branch document exists in Firestore (ieee-basic-info or ieee-members)
    branch_doc = _basic_info_cache.get("branch") or _teams_metadata_cache.get("branch") or {}

    title = branch_doc.get("title") or branch_doc.get("name") or "IEEE Student Branch, Academy of Technology (IEEE SB AOT)"
    description = branch_doc.get("description") or branch_doc.get("about") or (
        "IEEE Student Branch AOT is the official student branch of IEEE at the Academy of Technology (AOT), "
        "fostering technological innovation, engineering excellence, and professional growth through hands-on projects, workshops, and events."
    )
    how_to_join = branch_doc.get("how_to_get_involved") or branch_doc.get("get_involved") or (
        "Students can get involved with IEEE Student Branch AOT by participating in technical workshops, "
        "competing in hackathons and events, joining annual recruitment drives, or connecting with current team members."
    )

    return [{
        "type": "branch_overview",
        "title": title,
        "description": description,
        "how_to_get_involved": how_to_join,
        "all_teams": teams_list,
        "similarity": 1.0,
    }]


_config_warning_shown = False
_GREETING_PATTERNS = [
    r"^(hi|hello|hey|hola|good\s*(morning|afternoon|evening)|yo|greetings|howdy)\b",
]


async def retrieve_student_branch(query: str) -> list[dict[str, Any]]:
    """Retrieve verified member records relevant to a Student Branch question.

    Optimized paths:
    1. Greetings → skip retrieval & embedding entirely (0ms).
    2. Team-listing queries → basic-info + directory RAM cache lookup (0ms).
    3. Branch overview → basic-info RAM cache lookup (0ms).
    4. Member search → Single embedding call + in-memory cosine search (0.2ms).
    """
    global _config_warning_shown
    q = query.strip()
    if not q:
        return []

    # Fast path 0: Short pure greetings don't need any directory retrieval
    if len(q) < 30 and any(re.search(pat, q, re.IGNORECASE) for pat in _GREETING_PATTERNS):
        return []

    if not _retrieval_configured():
        if not _config_warning_shown:
            _config_warning_shown = True
            missing = [
                name
                for name in ("FIREBASE_CREDS_BASE64", "GOOGLE_API_KEY")
                if not _env(name)
            ]
            print(
                "[Student RAG] DISABLED — missing env var(s): "
                f"{', '.join(missing)}. Member answers will not be grounded. "
                "Set them in the deployment environment to enable retrieval."
            )
        return []

    try:
        # Fast path 1: Team listing & descriptions from in-memory cache (0ms, no network)
        team_slug = _detect_team_listing(q)
        if team_slug:
            results = await asyncio.to_thread(_fetch_team_members, team_slug, q)
            if results:
                return results

        # Fast path 2: Branch overview or get involved questions from ieee-basic-info
        if _detect_branch_query(q):
            branch_records = await asyncio.to_thread(_fetch_branch_info)
            if branch_records:
                return branch_records

        # Fast path 3: Single embedding call + instant in-memory vector match
        query_vector = await _embed_query(q)
        return await asyncio.to_thread(_search_firestore, query_vector)
    except Exception as exc:
        print(f"[Student RAG] Retrieval failed ({type(exc).__name__}): {exc}")
        return []
