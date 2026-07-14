"""BRANCH classify pass — name the problem-type before the task is solved.

The classifier reads natural task text and emits ONE type from the closed
problem-type enum (core/problem_types.py). It is a SEPARATE, stateless call —
the freeze/order primitive the no-KV-cache API forces: you can only steer the
trajectory from the top, so the type is decided in its own call and then placed
at the head of the solve prompt, rather than declared mid-stream where it could
not steer.

Stateless by construction (the monothink._call_llm pattern): no system prompt,
no identity, no history — just the classify prompt as a single user message,
non-streaming, content-only (reasoning_content dropped so a thinking backend
cannot crowd out the one-line verdict). The model call is injectable so the
parse logic is unit-tested without a network.
"""
from __future__ import annotations

import re
from typing import Callable

from core import problem_types as pt

_TYPE_LINE_RE = re.compile(r"^\s*TYPE:\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE)

_CLASSIFY_TEMPLATE = """You are a problem-type classifier. Read the TASK and decide which single problem type from the CLOSED LIST below most closely matches what the task is asking for — the kind of quantity or judgment sought, not the surface topic.

Output exactly one line and nothing else:
TYPE: <type_id>
using a type_id from the list verbatim. If no listed type fits, output `TYPE: other: <short phrase>`.

CLOSED LIST:
{menu}

TASK:
{task}"""


def build_classify_prompt(task: str) -> str:
    return _CLASSIFY_TEMPLATE.format(menu=pt.compose_type_menu(), task=task)


def parse_type(raw: str) -> str | None:
    """Extract the type from a classifier response. Prefers the last
    ``TYPE: <x>`` line; falls back to normalizing the whole text."""
    matches = _TYPE_LINE_RE.findall(raw or "")
    if matches:
        return pt.normalize_type(matches[-1])
    return pt.normalize_type(raw)


def _default_call(prompt: str) -> str | None:
    """Stateless single-message cloud call from llm config (monothink pattern).
    Returns raw text or None; broad except so classify never raises."""
    try:
        from core.config import get_config
        from engine.llm import OpenAICompatLLM
        cfg = get_config().llm.model_dump()
        api_base = str(cfg.get("api_base", "") or "").strip()
        api_model = str(cfg.get("api_model", "") or "").strip()
        if not api_base or not api_model:
            return None
        client = OpenAICompatLLM(api_base, str(cfg.get("api_key", "") or ""), api_model)
        parts: list[str] = []
        for chunk in client.create_chat_completion(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0, max_tokens=2048, stream=False,
        ):
            delta = (chunk.get("choices", [{}]) or [{}])[0].get("delta", {})
            if delta.get("content"):
                parts.append(delta["content"])
        return "".join(parts)
    except Exception:
        return None


def classify(task: str, *, call: Callable[[str], str | None] | None = None) -> str | None:
    """Return a normalized problem-type id (or ``other:<...>`` / None) for a task."""
    prompt = build_classify_prompt(task)
    raw = (call or _default_call)(prompt)
    return parse_type(raw or "")
