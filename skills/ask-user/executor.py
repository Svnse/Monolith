"""ask_user — structured clarifying-question tool.

The model emits ask_user with a question + 2-4 option labels. This executor
validates the payload, generates a question_id, and calls the host UI's
on_ask_user callback (set up in ui/pages/chat.py) to render the question as
a persistent panel with clickable option buttons.

Pause/resume architecture:
  * This executor returns synchronously with a "[ask_user: pending, ...]"
    sentinel string. The model's tool loop sees the result and ends the turn
    (no further tool_call envelopes).
  * The chat surface renders the question panel below the input box (mirror
    of the existing _action_review_panel pattern used for /approve).
  * When the user clicks an option button, the chat handler synthesizes the
    next user message as a structured `[ASK_USER_ANSWER]` block carrying the
    question_id and the chosen answer(s).
  * On the next turn, the model receives the answer in its user message
    stream and continues from there.

Only one question may be pending at a time. If the model emits a second
ask_user while one is pending, the callback signals "busy" and this
executor returns an error envelope so the model sees the failure.
"""
from __future__ import annotations

import re
import time
import uuid
from typing import Any


_HEADER_MAX = 20
_QUESTION_MAX = 500
_LABEL_MAX = 80
_DESCRIPTION_MAX = 300
_MIN_OPTIONS = 2
_MAX_OPTIONS = 4
_WHITESPACE_RE = re.compile(r"\s+")


def _clean(text: Any, max_len: int) -> str:
    """Collapse internal whitespace, strip, truncate to max_len."""
    if text is None:
        return ""
    s = _WHITESPACE_RE.sub(" ", str(text)).strip()
    if len(s) > max_len:
        s = s[:max_len].rstrip()
    return s


def _normalize_options(raw: Any) -> tuple[list[dict[str, str]] | None, str | None]:
    """Validate + clean options. Returns (options, error). One of them is None."""
    if not isinstance(raw, list):
        return None, "options must be a list of {label, description?} objects"
    if len(raw) < _MIN_OPTIONS:
        return None, f"options needs at least {_MIN_OPTIONS} entries (got {len(raw)})"
    if len(raw) > _MAX_OPTIONS:
        return None, f"options allows at most {_MAX_OPTIONS} entries (got {len(raw)})"
    cleaned: list[dict[str, str]] = []
    seen_labels: set[str] = set()
    for i, item in enumerate(raw):
        if not isinstance(item, dict):
            return None, f"options[{i}] must be an object with a label"
        label = _clean(item.get("label"), _LABEL_MAX)
        if not label:
            return None, f"options[{i}] missing non-empty label"
        if label.lower() in seen_labels:
            return None, f"options[{i}] duplicate label: {label!r} (must be unique)"
        seen_labels.add(label.lower())
        description = _clean(item.get("description"), _DESCRIPTION_MAX)
        cleaned.append({"label": label, "description": description})
    return cleaned, None


def run(cmd: dict, ctx: Any) -> str:
    """Render an ask_user question. See module docstring for the contract.

    Entry-point name `run` is the contract `_load_dynamic_executor` looks for
    when loading skill executor modules.
    """
    # Validate arguments
    question = _clean(cmd.get("question"), _QUESTION_MAX)
    if not question:
        return "[ask_user: error - 'question' is required and non-empty]"
    if not question.endswith("?"):
        question = f"{question}?"  # forgiving: append ? rather than reject
    options, opt_err = _normalize_options(cmd.get("options"))
    if opt_err is not None or options is None:
        return f"[ask_user: error - {opt_err}]"
    header = _clean(cmd.get("header"), _HEADER_MAX) or None
    multi_select = bool(cmd.get("multi_select", False))

    question_id = f"q-{int(time.time())}-{uuid.uuid4().hex[:8]}"
    payload = {
        "question_id": question_id,
        "question": question,
        "options": options,
        "header": header,
        "multi_select": multi_select,
    }

    callback = getattr(ctx, "on_ask_user", None) if ctx is not None else None
    if callback is None:
        # Headless mode (tests, MCP without UI): no panel, just signal that we
        # would have asked. Caller can read the question_id in the result data.
        return (
            f"[ask_user: no UI host available (callback unset); "
            f"question_id={question_id}; would-have-asked: {question}]"
        )

    try:
        result = callback(payload)
    except Exception as exc:
        return f"[ask_user: error - host callback raised: {type(exc).__name__}: {exc}]"

    # Callback contract: returns True on accept, False or "busy" if a
    # question is already pending, or a string starting with "error" on
    # other failure.
    if result is False:
        return (
            f"[ask_user: error - another question is already pending. "
            f"Wait for it to be answered before emitting a new ask_user.]"
        )
    if isinstance(result, str) and result.lower().startswith("error"):
        return f"[ask_user: error - {result}]"

    opt_summary = " | ".join(o["label"] for o in options)
    return (
        f"[ask_user: question rendered, awaiting user answer "
        f"(question_id={question_id}; options: {opt_summary})]"
    )
