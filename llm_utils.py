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
    Remove any reasoning/thinking markup or plaintext thinking traces from ``text``.
    """
    if not text:
        return ""

    # 1. Well-formed <think>...</think> blocks.
    text = _THINK_BLOCK_RE.sub("", text)

    # 2. Orphan closing tag: everything before it was reasoning.
    if "</think>" in text.lower():
        text = _THINK_CLOSE_RE.sub("", text)

    # 3. Orphan opening tag: everything after it was reasoning.
    text = _THINK_OPEN_RE.sub("", text)

    # 4. Plaintext thinking traces (e.g. "Here's a thinking process:\n...")
    if re.search(r"^\s*(?:Here'?s\s+a\s+thinking\s+process|Thinking\s+Process|Thought\s+Process|Let'?s\s+think)", text, re.IGNORECASE):
        # Check if there is an explicit final answer label
        split_match = re.search(
            r'(?:Output(?:\s*generation)?|Final\s*Answer|Response|Assistant|Answer):\s*(.*)',
            text,
            re.DOTALL | re.IGNORECASE
        )
        if split_match and split_match.group(1).strip():
            text = split_match.group(1).strip()
        else:
            # Check for double-newline separating thoughts from final response
            parts = [p.strip() for p in re.split(r'\n\s*\n', text) if p.strip()]
            # If the last paragraph looks like a normal answer without thinking markers
            if parts and not re.search(r'\b(analyze|user asks|user input|step \d|rule \d|let me|i will|check rules)\b', parts[-1], re.I):
                text = parts[-1]
            else:
                text = ""

    return text.strip(' "\'\n')


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
    Return ONLY ``message.content``, with any reasoning traces stripped.
    """
    message = extract_message(res)
    if not message:
        return ""
    return strip_reasoning_tags(message.get("content") or "")


def clean_llm_response(res):
    """
    Extract the clean, user-facing answer from a chat-completion response.
    Never returns raw internal reasoning scratchpads.
    """
    content = clean_llm_content(res)
    if content:
        return content

    # If message.content is empty, do NOT leak message.reasoning to the user.
    return ""


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
