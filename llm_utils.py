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


# ── Raw-link stripping ───────────────────────────────────────────────────────
# Student Branch answers must never carry raw URLs: the UI already renders every
# verified LinkedIn profile as an interactive reference card (the ``sources``
# array), so a URL in the message text is a duplicate of that card. The system
# prompt asks the model not to emit URLs, but prompts are advisory — this is the
# hard guarantee.

_URL_BODY = r'(?:https?://|www\.)[^\s<>()\[\]{}"\'`]+'

# Words a model puts in front of a URL: "LinkedIn:", "**Profile** -", "link:".
_LINK_LABEL = (
    r'\*{0,2}\b(?:linked\s?-?in|profiles?|links?|urls?|websites?|socials?|handles?)\b'
    r'(?:\s+(?:profile|page|link|url))?\*{0,2}'
)

# [label](url) / [label](<url> "title") -> keep the human-readable label only.
_MD_LINK_RE = re.compile(r'\[([^\]\n]*)\]\(\s*<?' + _URL_BODY + r'>?[^)\n]*\)')

# <https://...> autolinks and bare URLs.
_ANGLE_URL_RE = re.compile(r'<\s*' + _URL_BODY + r'\s*>')
_BARE_URL_RE = re.compile(_URL_BODY)

# Wrappers left empty once the URL inside them is gone: "()", "[ ]", "<>".
_EMPTY_WRAPPER_RE = re.compile(r'\(\s*\)|\[\s*\]|<\s*>')

# A parenthetical that only introduced the URL: "(LinkedIn:", "(see profile -".
_PAREN_LABEL_RE = re.compile(
    r'\(\s*(?:see\s+|his\s+|her\s+|their\s+)?' + _LINK_LABEL + r'\s*[:\-\u2013\u2014]?\s*\)',
    re.IGNORECASE,
)

# A whole line that was only a labelled link: "- LinkedIn: <url>", "**Profile:**".
_LABEL_ONLY_LINE_RE = re.compile(
    r'^\s*(?:[-*\u2022+]|\d+[.)])?\s*' + _LINK_LABEL + r'\s*[:\-\u2013\u2014]?\s*$',
    re.IGNORECASE,
)

# A line reduced to bullet/emphasis punctuation once its URL was removed.
_PUNCT_ONLY_LINE_RE = re.compile(
    r'^\s*(?:[-*\u2022+]|\d+[.)])?\s*(?:\*{1,2}|[:\-\u2013\u2014,;.])*\s*$'
)

# A short trailing clause left pointing at nothing ("... Suman leads Tech. See:").
_TRAILING_CLAUSE_RE = re.compile(r'(?<=[.!?])[ \t]+[^.!?\n]{0,80}$')

# Any URL — used to decide whether a line needs tidying at all.
_HAS_LINK_RE = re.compile(r'(?:https?://|www\.)|\]\(', re.IGNORECASE)


def _tidy_link_line(line):
    """Clean a single line that had a URL removed from it. Returns ``None`` to
    signal the line existed only to carry that URL and should be dropped."""
    line = _PAREN_LABEL_RE.sub("", line)
    line = _EMPTY_WRAPPER_RE.sub("", line)
    line = re.sub(r'[ \t]{2,}', ' ', line)
    line = re.sub(r'[ \t]+([,.;:!?])', r'\1', line).rstrip()

    # The URL's introducer is now dangling ("... team. LinkedIn:"). Drop the
    # whole fragment when it is short enough to be pure link scaffolding,
    # otherwise just shed the orphaned separator.
    if re.search(r'[,;:\-\u2013\u2014]$', line):
        tail = _TRAILING_CLAUSE_RE.search(line)
        if tail and len(tail.group(0).split()) <= 8:
            line = line[:tail.start()].rstrip()
        else:
            line = re.sub(r'[ \t]*[,;:\-\u2013\u2014]+[ \t]*$', '', line).rstrip()

    if _LABEL_ONLY_LINE_RE.match(line) or _PUNCT_ONLY_LINE_RE.match(line):
        return None
    return line


def strip_raw_links(text):
    """
    Remove raw URLs — and the label remnants they leave behind — from ``text``.

    Markdown links keep their label, so ``[LinkedIn](https://...)`` becomes
    ``LinkedIn`` and the sentence around it still reads correctly. Lines and
    parentheticals that existed *only* to carry the URL are dropped outright.
    Lines with no URL are returned byte-for-byte untouched.
    """
    if not text:
        return ""

    lines = []
    for line in text.split("\n"):
        if not _HAS_LINK_RE.search(line):
            lines.append(line)  # nothing to do — never reformat clean prose
            continue

        line = _MD_LINK_RE.sub(lambda m: m.group(1).strip(), line)
        line = _ANGLE_URL_RE.sub("", line)
        line = _BARE_URL_RE.sub("", line)

        cleaned = _tidy_link_line(line)
        if cleaned is not None:
            lines.append(cleaned)

    return re.sub(r'\n{3,}', '\n\n', "\n".join(lines)).strip()


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
