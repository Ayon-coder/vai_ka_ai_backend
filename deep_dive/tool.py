"""
deep_dive/tool.py
-----------------
Search layer for the IEEE Deep Dive assistant.

Fetches supporting sources for a technical query from DuckDuckGo (primary) with
Google as a resilience fallback, then normalises the results into the shape the
synthesis prompt expects: ``{"title", "link", "snippet"}``.
"""

import asyncio
import html
import re
from urllib.parse import urlparse

from ddgs import DDGS
from googlesearch import search as gsearch

# Characters kept per snippet. Large enough to preserve the technical
# explanation the synthesis prompt needs to cite, small enough to keep the
# <IEEE_SOURCES> block within the context budget.
SNIPPET_MAX_CHARS = 600

# Number of sources handed to the synthesis prompt.
MAX_RESULTS = 3

# Below this many hits the strict `site:` filter is considered too narrow and
# the query is widened.
MIN_ACCEPTABLE_RESULTS = 2

# Raw hits requested per DuckDuckGo call (we over-fetch, then de-duplicate).
DDGS_MAX_RESULTS = 5

# Network budgets, in seconds.
#
# NOTE: `timeout` is a constructor argument -- DDGS(proxy, timeout, *, verify).
# It is NOT accepted by DDGS.text(). Passing it to .text() is silently swallowed
# by that method's `**kwargs` catch-all on ddgs 9.x (so the budget is never
# applied) and raises TypeError on earlier releases.
DDGS_TIMEOUT = 6
GOOGLE_TIMEOUT = 4.0


# ── Text normalisation ────────────────────────────────────────────────────────

_WHITESPACE_RE = re.compile(r"\s+")
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_FILE_EXT_RE = re.compile(r"\.(?:html?|php|aspx?|jsp|pdf)$", re.IGNORECASE)

# Typographic characters that add no meaning but waste tokens and can confuse
# downstream JSON/prompt handling. Dashes are deliberately left alone: they can
# be semantically meaningful in technical text (e.g. numeric ranges).
_CHAR_MAP = {
    "\u201c": '"', "\u201d": '"', "\u201e": '"', "\u201f": '"',
    "\u2018": "'", "\u2019": "'", "\u201a": "'", "\u201b": "'",
    "\u2032": "'", "\u2033": '"',
    "\u2026": "...",
    "\u00a0": " ", "\u2009": " ", "\u200a": " ", "\u202f": " ",
    "\u200b": "", "\u200c": "", "\u200d": "", "\ufeff": "",
}


def _sanitize(text):
    """Unescape HTML entities, normalise quotes, and collapse whitespace runs."""
    if not text:
        return ""
    text = html.unescape(str(text))
    for bad, good in _CHAR_MAP.items():
        text = text.replace(bad, good)
    text = _CONTROL_RE.sub(" ", text)
    return _WHITESPACE_RE.sub(" ", text).strip()


def _truncate(text, limit=SNIPPET_MAX_CHARS):
    """
    Trim `text` to `limit` characters on a word boundary, appending an ellipsis
    only when content was actually removed.
    """
    if not text or len(text) <= limit:
        return text

    cut = text[:limit]
    pivot = cut.rfind(" ")
    # Only honour the word boundary if it doesn't discard too much of the cut.
    if pivot > limit * 0.6:
        cut = cut[:pivot]
    return cut.rstrip(" ,;:.-") + "..."


def _title_from_url(link):
    """
    Derive a readable title from a URL path.

    Uses only information present in the URL itself — nothing is invented — so a
    bare link from Google still gets a meaningful label instead of a generic one.
    """
    try:
        parsed = urlparse(link)
    except ValueError:
        return "IEEE Source"

    segments = [s for s in parsed.path.split("/") if s]
    if segments:
        label = _FILE_EXT_RE.sub("", segments[-1])
        label = _sanitize(label.replace("-", " ").replace("_", " "))
        if label:
            return f"{label} ({parsed.netloc})" if parsed.netloc else label
    return parsed.netloc or "IEEE Source"


# ── DuckDuckGo ────────────────────────────────────────────────────────────────

def _normalize_ddgs_row(row):
    """Map one raw DuckDuckGo row to our result shape, or None if unusable."""
    if not isinstance(row, dict):
        return None

    link = _sanitize(row.get("href") or "")
    if not link:
        return None

    return {
        "title": _sanitize(row.get("title") or "") or _title_from_url(link),
        "link": link,
        "snippet": _truncate(_sanitize(row.get("body") or "")),
    }


def _ddgs_sync(query):
    """Blocking DuckDuckGo text search. Returns [] on any failure."""
    try:
        # timeout goes on the constructor -- see DDGS_TIMEOUT note above.
        with DDGS(timeout=DDGS_TIMEOUT) as ddgs:
            rows = ddgs.text(query, max_results=DDGS_MAX_RESULTS)
    except Exception as e:
        # ddgs >= 9 raises DDGSException("No results found.") rather than
        # returning an empty list, so "no results" is not an error condition.
        print(f"[Search] DDGS failed for {query!r} ({type(e).__name__}): {e}")
        return []

    results = []
    for row in rows or []:
        item = _normalize_ddgs_row(row)
        if item:
            results.append(item)
    return results


async def fetch_ddgs(query):
    """Run one DuckDuckGo text search off the event loop."""
    return await asyncio.to_thread(_ddgs_sync, query)


# ── Google (fallback) ─────────────────────────────────────────────────────────

def _google_sync(query):
    """Blocking Google search. Returns bare links with URL-derived titles."""
    results = []
    try:
        for link in gsearch(query, num_results=MAX_RESULTS):
            link = _sanitize(link)
            if not link:
                continue
            results.append({
                "title": _title_from_url(link),
                "link": link,
                # Google gives us no extract. Say so plainly rather than
                # inventing content the synthesis step might cite as fact.
                "snippet": "No text extract available for this result; "
                           "see the linked IEEE page for details.",
            })
    except Exception as e:
        print(f"[Search] Google failed ({type(e).__name__}): {e}")
    return results


async def fetch_google(query):
    """Run one Google search off the event loop, capped at GOOGLE_TIMEOUT."""
    try:
        return await asyncio.wait_for(
            asyncio.to_thread(_google_sync, query), timeout=GOOGLE_TIMEOUT
        )
    except asyncio.TimeoutError:
        print(f"[Search] Google timed out (exceeded {GOOGLE_TIMEOUT}s)")
        return []


# ── Orchestration ─────────────────────────────────────────────────────────────

def _build_query_tiers(query):
    """
    Progressively broader queries, tried in order until enough hits come back.

    A strict ``site:ieee.org`` filter misses anything published on IEEE's other
    hosts (ieeexplore.ieee.org, computer.org, edu.ieee.org, ...), which is why
    tier 1 alone can return zero results for perfectly valid technical topics.
    """
    cleaned = _sanitize(query)
    return [
        f"site:ieee.org {cleaned}",
        f"{cleaned} (site:ieee.org OR site:ieeexplore.ieee.org)",
        f"{cleaned} IEEE",
    ]


def _collect(results, seen, incoming):
    """
    Append de-duplicated items from `incoming` into `results`, stopping at
    MAX_RESULTS. Tolerates an Exception in place of a result list (from
    asyncio.gather(return_exceptions=True)).
    """
    if isinstance(incoming, BaseException):
        print(f"[Search] Task failed ({type(incoming).__name__}): {incoming}")
        return

    for item in incoming or []:
        # Normalise for comparison so http/https and trailing-slash variants of
        # the same page don't both occupy a source slot.
        key = re.sub(r"^https?://(?:www\.)?", "", item["link"].rstrip("/").lower())
        if key in seen:
            continue
        seen.add(key)
        results.append(item)
        if len(results) >= MAX_RESULTS:
            return


async def search_ieee(query):
    """
    Search IEEE sources for `query` and return up to MAX_RESULTS entries shaped
    as ``{"title", "link", "snippet"}``.

    Strategy:
      1. Tier 1 (``site:ieee.org``) runs concurrently with a Google search, so
         the fallback costs no extra wall-clock time.
      2. If DuckDuckGo returns fewer than MIN_ACCEPTABLE_RESULTS hits, widen the
         query through the remaining tiers.
      3. Only if still short, top up with the Google links.
    """
    if not query or not str(query).strip():
        return []

    tiers = _build_query_tiers(query)

    ddgs_results, google_results = await asyncio.gather(
        fetch_ddgs(tiers[0]),
        fetch_google(tiers[0]),
        return_exceptions=True,
    )

    results, seen = [], set()
    _collect(results, seen, ddgs_results)

    # Widen only when the strict site: filter came up short.
    for tier in tiers[1:]:
        if len(results) >= MIN_ACCEPTABLE_RESULTS:
            break
        print(f"[Search] Only {len(results)} hit(s); widening to {tier!r}")
        _collect(results, seen, await fetch_ddgs(tier))

    # Google links are the last resort — they carry no extract.
    if len(results) < MAX_RESULTS:
        _collect(results, seen, google_results)

    print(f"[Search] {len(results)} source(s) for {str(query)[:60]!r}")
    return results[:MAX_RESULTS]
