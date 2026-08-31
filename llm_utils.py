"""
llm_utils.py
------------
Shared helpers for turning raw Groq chat-completion payloads into clean,
user-facing text.

Why this exists
---------------
The primary model (``qwen/qwen3.6-27b``) is a *reasoning* model: it emits a
chain-of-thought trace before its actual answer. Two things follow from that:

1. Without ``reasoning_format: "parsed"`` the trace is inlined into
   ``message.content`` wrapped in ``<think>...</think>`` tags, so the raw
   chain-of-thought is what reaches the user.
2. The trace consumes the completion budget. If ``max_completion_tokens`` is too
   low the response comes back with ``finish_reason: "length"`` and an **empty**
   ``content`` (or a truncated, unterminated ``<think>`` block).

These helpers defend against both, so no chain-of-thought ever reaches a client
regardless of which model answered or how the completion terminated.

This lives at the top level rather than inside ``app.py`` because ``app.py``
imports ``student_branch.chat``, which also needs these helpers — importing them
from ``app.py`` would create a circular import.
"""

import re

# A complete <think>...</think> block.
_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)

# An unterminated <think> block — the completion was cut off mid-reasoning.
_THINK_OPEN_RE = re.compile(r"<think>.*\Z", re.DOTALL | re.IGNORECASE)

# A stray closing tag whose opening tag was never emitted.
_THINK_CLOSE_RE = re.compile(r"\A.*?</think>", re.DOTALL | re.IGNORECASE)


def strip_reasoning_tags(text):
    """
    Remove any ``<think>`` chain-of-thought markup from ``text``.

    Handles all three malformed shapes seen in practice:
      * ``<think>reasoning</think>answer``  -> ``answer``
      * ``reasoning</think>answer``         -> ``answer``  (opener lost)
      * ``<think>reasoning``                -> ``""``      (truncated)
    """
    if not text:
        return ""

    # 1. Well-formed blocks first.
    text = _THINK_BLOCK_RE.sub("", text)

    # 2. Orphan closing tag: everything before it was reasoning.
    if "</think>" in text.lower():
        text = _THINK_CLOSE_RE.sub("", text)

    # 3. Orphan opening tag: everything after it was reasoning.
    text = _THINK_OPEN_RE.sub("", text)

    return text.strip()


def extract_message(res):
    """
    Return the assistant message dict from a chat-completion payload, or ``{}``
    when the payload is missing/malformed.
    """
    if not isinstance(res, dict):
        return {}
    try:
        message = res["choices"][0]["message"]
    except (KeyError, IndexError, TypeError):
        return {}
    return message if isinstance(message, dict) else {}


def clean_llm_content(res):
    """
    Return ONLY ``message.content``, with any ``<think>`` markup stripped.

    Use this for *classification* calls (moderation, category routing) where the
    answer is matched against known labels. Never fall back to the reasoning
    trace here: a trace naturally restates every candidate label ("is this SAFE
    or GIBBERISH?", "GREETING, TECHNICAL, STUDENT_BRANCH or REJECTED?"), so
    substring-matching it would yield essentially random verdicts.

    Returns ``""`` when there is no visible content, letting callers apply an
    explicit default.
    """
    message = extract_message(res)
    if not message:
        return ""
    return strip_reasoning_tags(message.get("content") or "")


def clean_llm_response(res):
    """
    Extract the clean, user-facing answer from a Groq chat-completion response.

    Order of preference:
      1. ``message.content`` with any ``<think>`` markup stripped.
      2. ``message.reasoning`` (also stripped) — a last resort for when the model
         spent its entire budget reasoning and never produced a visible answer.

    Use this for prose answers shown to the user. For label/verdict extraction
    use :func:`clean_llm_content` instead.

    Returns ``""`` when nothing usable is present, so callers can detect failure
    with a simple falsy check.
    """
    content = clean_llm_content(res)
    if content:
        return content

    # No visible answer. Falling back to the reasoning trace is preferable to
    # showing the user an empty message bubble.
    return strip_reasoning_tags(extract_message(res).get("reasoning") or "")


def finalize_response(res):
    """
    Normalise ``res`` in place so the client only ever sees clean text.

    Writes the cleaned answer back into ``choices[0].message.content`` and drops
    the verbose ``reasoning`` trace — the frontend reads only
    ``choices[0].message.content`` and ``sources``, and the trace can be several
    kilobytes of dead weight.

    Returns ``(res, cleaned_text)``.
    """
    cleaned = clean_llm_response(res)
    message = extract_message(res)
    if message:
        message["content"] = cleaned
        message.pop("reasoning", None)
    return res, cleaned


def finish_reason(res):
    """Return the finish_reason of the first choice, or ``""`` if unavailable."""
    if not isinstance(res, dict):
        return ""
    try:
        return res["choices"][0].get("finish_reason") or ""
    except (KeyError, IndexError, TypeError, AttributeError):
        return ""
