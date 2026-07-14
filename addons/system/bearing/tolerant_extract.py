"""Tolerant <bearing_update> JSON recovery.

The bound model persistently emits its bearing_update envelope tangled inside
its <think> reasoning — dangling/missing close tags, an embedded </think>,
markdown fences, trailing commas or prose — so the strict ``extract_envelope``
returns parse_error on every attempt and the bearing never updates. This module
recovers the JSON object from those real shapes.

PARSING ONLY. It produces a dict; the structural verifier (bearing semantics,
owned elsewhere) still runs on that dict unchanged. Recovering an envelope is
not the same as accepting it — a recovered-but-invalid update is still rejected
by verify_structural, just with a real structural reason instead of a blanket
"json decode failed".
"""
from __future__ import annotations

import json
import re
from typing import Any


# Internal/structural tags that may be tangled into the envelope body. Scrubbed
# before JSON extraction so an interleaved </think> doesn't break the parse.
_TAG_SCRUB = re.compile(
    r"</?(?:think|analysis|reasoning|bearing_update)\b[^>]*>",
    re.IGNORECASE,
)
# Markdown code fences the model sometimes wraps the JSON in.
_FENCE_SCRUB = re.compile(r"```[a-zA-Z0-9_]*")
# Trailing commas before a closing } or ] — the most common JSON malformation.
_TRAILING_COMMA = re.compile(r",(\s*[}\]])")
_OPEN_RE = re.compile(r"<bearing_update\b[^>]*>", re.IGNORECASE)
_CLOSE_RE = re.compile(r"</bearing_update>", re.IGNORECASE)
_THINK_CLOSE_RE = re.compile(r"</think>", re.IGNORECASE)


def _first_json_object(s: str) -> str | None:
    """Return the first balanced ``{...}`` object, string-aware (braces inside
    JSON strings don't count). None if there is no ``{`` or it never balances
    (truncated)."""
    start = s.find("{")
    if start < 0:
        return None
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(s)):
        c = s[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
        elif c == '"':
            in_str = True
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return s[start : i + 1]
    return None


def _loads_tolerant(obj_str: str) -> Any:
    try:
        return json.loads(obj_str)
    except json.JSONDecodeError:
        try:
            return json.loads(_TRAILING_COMMA.sub(r"\1", obj_str))
        except json.JSONDecodeError:
            return None


def looks_like_attempt(text: str) -> bool:
    """True if there is a ``<bearing_update>`` open tag followed by a ``{`` — a
    real (if broken) envelope attempt, as opposed to a bare prose mention of the
    tag. Lets the caller reject a broken attempt without spuriously rejecting a
    prose mention."""
    if not isinstance(text, str):
        return False
    m = _OPEN_RE.search(text)
    if m is None:
        return False
    return "{" in text[m.end():]


def recover_bearing_json(text: str) -> dict[str, Any] | None:
    """Best-effort recovery of the bearing_update JSON dict, or None.

    Steps: find the ``<bearing_update>`` open tag; take the body up to its
    close tag (or to a tangled ``</think>``, or end-of-text on truncation);
    scrub interleaved internal tags + markdown fences; extract the first
    balanced JSON object; parse, tolerating trailing commas.
    """
    if not isinstance(text, str):
        return None
    open_m = _OPEN_RE.search(text)
    if open_m is None:
        return None
    body = text[open_m.end():]
    # Bound the body at the first sensible terminator.
    close_m = _CLOSE_RE.search(body)
    if close_m is not None:
        body = body[: close_m.start()]
    else:
        think_m = _THINK_CLOSE_RE.search(body)
        if think_m is not None:
            body = body[: think_m.start()]
    body = _TAG_SCRUB.sub("", body)
    body = _FENCE_SCRUB.sub("", body)
    obj = _first_json_object(body)
    if not obj:
        return None
    parsed = _loads_tolerant(obj)
    return parsed if isinstance(parsed, dict) else None
