"""
context_builder.py
------------------
Builds a compact "context vector" from a conversation window using regex-based
extraction instead of raw message dumps. This drastically reduces token usage
while preserving the key semantic signals the LLM needs:

  - IEEE standard references  (e.g. IEEE 802.11, IEEE 1394)
  - Technical acronyms        (e.g. OFDM, QoS, MAC, LTE, MIMO)
  - High-frequency tech terms (e.g. "wireless", "protocol", "latency")
  - Mentioned years           (e.g. 2019, 2023)
  - Numeric specs             (e.g. 2.4 GHz, 100 Mbps)
"""

import re
from collections import Counter


# ── Regex Patterns ─────────────────────────────────────────────────────────────

# IEEE standard numbers: "IEEE 802.11", "IEEE 1394", "IEEE 754", etc.
_IEEE_STD = re.compile(r'\bIEEE[\s\u00a0]*\d+[\w.\-]*(?:/\w+)?', re.IGNORECASE)

# Technical acronyms: 2-6 uppercase letters (avoids plain English articles)
_ACRONYM = re.compile(r'\b[A-Z]{2,6}\b')

# 4+ letter words (lowercased) used for term-frequency extraction
_WORD = re.compile(r'\b[a-zA-Z]{4,}\b')

# Calendar years 1900-2099
_YEAR = re.compile(r'\b(19|20)\d{2}\b')

# Numeric specs with optional SI unit suffix
_SPEC = re.compile(
    r'\b\d+(?:\.\d+)?\s*(?:GHz|MHz|kHz|Mbps|Gbps|kbps|dBm?|Watt?|ms|ns|µs|KB|MB|GB)\b',
    re.IGNORECASE
)

# ── Stop-sets (words to ignore during frequency analysis) ─────────────────────

_STOP_ACRONYMS: set[str] = {
    "I", "A", "AN", "THE", "OR", "AND", "BUT", "IF", "IN", "IS", "IT",
    "OF", "ON", "TO", "BE", "OK", "NO", "YES", "DO", "WE", "MY", "AT",
    "BY", "SO", "UP", "AS", "RE", "US", "AM", "PM",
}

_STOP_WORDS: set[str] = {
    "this", "that", "with", "from", "have", "been", "they", "what", "when",
    "where", "will", "also", "some", "more", "than", "such", "like", "into",
    "then", "over", "only", "about", "other", "after", "which", "their",
    "there", "these", "those", "your", "does", "used", "user", "ieee",
    "standard", "research", "explain", "tell", "give", "please", "hello",
    "using", "based", "between", "provide", "related", "describe", "question",
    "answer", "system", "different", "various", "information", "define",
    "discuss", "refer", "according", "compare", "overview", "example",
}


# ── Public API ─────────────────────────────────────────────────────────────────

def build_context_vector(context_window: list[dict]) -> str:
    """
    Analyse all *prior* messages in the context window (everything except the
    most recent user query) and return a single compact string that encodes the
    key topics discussed so far.

    Returns an empty string when there are no prior turns.

    Example output:
        [Context Vector]
        Standards: IEEE 802.11, IEEE 802.3 | Acronyms: OFDM, QoS, MIMO
        Topics: wireless, channel, access, protocol, frequency
        Specs: 2.4 GHz, 100 Mbps | Years: 2019, 2022
    """
    # Only inspect history before the current query
    prior = context_window[:-1]
    if not prior:
        return ""

    # Aggregate all text from prior turns
    corpus = " ".join(msg.get("content", "") for msg in prior)

    # ── Extract signals ──────────────────────────────────────────────────────

    ieee_stds = _unique_ordered(_IEEE_STD.findall(corpus))

    raw_acronyms = _ACRONYM.findall(corpus)
    acronyms = [
        a for a in raw_acronyms if a not in _STOP_ACRONYMS
    ]
    top_acronyms = [t for t, _ in Counter(acronyms).most_common(6)]

    raw_words = _WORD.findall(corpus.lower())
    filtered = [w for w in raw_words if w not in _STOP_WORDS]
    # Keep only terms that appear at least twice (genuine topics, not noise)
    tech_terms = [
        t for t, c in Counter(filtered).most_common(12) if c >= 2
    ][:6]

    years = _unique_ordered(_YEAR.findall(corpus))[:4]

    specs = _unique_ordered(_SPEC.findall(corpus))[:4]

    # ── Build the vector string ──────────────────────────────────────────────

    lines: list[str] = []

    if ieee_stds:
        lines.append(f"Standards : {', '.join(ieee_stds[:4])}")
    if top_acronyms:
        lines.append(f"Acronyms  : {', '.join(top_acronyms)}")
    if tech_terms:
        lines.append(f"Topics    : {', '.join(tech_terms)}")
    if specs:
        lines.append(f"Specs     : {', '.join(specs)}")
    if years:
        lines.append(f"Years     : {', '.join(years)}")

    if not lines:
        return ""

    return "[Context Vector]\n" + "\n".join(lines) + "\n"


def compress_assistant_msg(content: str, max_chars: int = 250) -> str:
    """
    Compress an assistant message to `max_chars` characters.
    Tries to cut at the last complete sentence within the limit.
    """
    if len(content) <= max_chars:
        return content

    truncated = content[:max_chars]
    # Try to find the last sentence boundary within the truncated text
    last_period = max(
        truncated.rfind(". "),
        truncated.rfind(".\n"),
        truncated.rfind("! "),
        truncated.rfind("? "),
    )
    if last_period != -1:
        truncated = truncated[: last_period + 1]

    return truncated.rstrip() + " …"


def build_slim_history(context_window: list[dict], max_prior_turns: int = 2) -> list[dict]:
    """
    Returns a slimmed version of the context window suitable for the synthesis
    prompt.  Strategy:
      - Keep the last `max_prior_turns` full user+assistant pairs verbatim
        (these are the most relevant for coherence).
      - Older assistant messages are compressed via compress_assistant_msg.
      - The current (last) user message is NOT included — the caller appends it
        augmented with the IEEE search context.

    This keeps the prompt sharp without losing recent conversational flow.
    """
    prior = context_window[:-1]  # exclude current user query
    if not prior:
        return []

    # Index of where "recent" turns start
    cutoff = max(0, len(prior) - max_prior_turns * 2)

    slim: list[dict] = []
    for i, msg in enumerate(prior):
        if msg["role"] == "assistant" and i < cutoff:
            slim.append({
                "role": "assistant",
                "content": compress_assistant_msg(msg["content"]),
            })
        else:
            slim.append(msg)

    return slim


# ── Helpers ────────────────────────────────────────────────────────────────────

def _unique_ordered(seq) -> list[str]:
    """Return unique items preserving first-occurrence order."""
    seen: set[str] = set()
    result: list[str] = []
    for item in seq:
        key = item.strip().upper()
        if key not in seen:
            seen.add(key)
            result.append(item.strip())
    return result
