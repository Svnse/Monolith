"""[OBSERVED STATE] — model identity grounding for the system prompt.

Without this block, the LLM confabulates which model/backend it's running
on. Probes confirmed the failure mode: when missing, the model invents
plausible-but-wrong answers about its own runtime ("I'm running Llama 3"
when actually running a DeepSeek model on a remote endpoint). The block
is small (5–8 lines) and provides the LLM with verifiable identity facts
it can quote back when asked.

Source data: the LLMConfig snapshot already passed to interceptors as
`config`. No world_state dependency for MVP — the block is composed from
config fields that don't change between turns within a session, so the
KV-cache prefix benefit is preserved naturally.

Block authority: declared as outranking external claims. The model is
told to cite values verbatim, not paraphrase them — paraphrasing is what
opens the door to confabulation.

Flag: MONOLITH_OBSERVED_STATE_V1 (default ON). Set =0 to disable.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from core.self_description import format_self_description_block

_BLOCK_TAG = "[OBSERVED STATE - describe_self v1]"
_FLAG_ENV = "MONOLITH_OBSERVED_STATE_V1"


def _flag_enabled() -> bool:
    raw = str(os.environ.get(_FLAG_ENV, "1")).strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _short(value: Any, limit: int = 200) -> str:
    text = str(value or "").strip()
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


def _gguf_basename(path: Any) -> str:
    """Reduce a GGUF path to just the file name.

    Keeps the prompt line short and strips machine-specific path noise
    that doesn't help the LLM identify the model.
    """
    if not path:
        return ""
    try:
        return Path(str(path)).name
    except Exception:
        return _short(path, limit=80)


def format_observed_state_block(config: dict | None = None) -> str:
    """Build the [OBSERVED STATE] block from a config snapshot.

    Delegates to core.self_description.format_self_description_block so the
    model sees one scoped describe_self payload (identity_material +
    current_model_execution + claim_scope) rather than two parallel grounding
    surfaces. Returns empty string when config is missing or has no backend
    identity to ground, so the caller can skip injection in those cases.

    Aligned with the 2026-05-14 hardening: one queryable runtime-fact
    surface, not two. The local-render fallback below kicks in only if
    describe_self produces nothing for this config — defense in depth.
    """
    if not isinstance(config, dict) or not config:
        return ""

    backend = _short(config.get("backend"))
    api_model = _short(config.get("api_model"))
    gguf_name = _gguf_basename(config.get("gguf_path"))

    # Skip when there's no backend identity to ground — an empty block is
    # worse than no block (it primes the model to expect grounding info
    # that isn't there, which can drive confabulation harder).
    if not (backend or api_model or gguf_name):
        return ""

    return format_self_description_block(config)


def observed_state_interceptor(
    messages: list[dict], config: dict
) -> list[dict] | None:
    """Inject [OBSERVED STATE] before the latest user turn.

    Returns None when:
      - flag MONOLITH_OBSERVED_STATE_V1 is off
      - the formatter returns empty (no backend identity available yet)
      - the block is already present (defense vs double-fire within a turn)
      - no non-ephemeral user message exists

    Pattern matches continuity / context_refresh / effort / adaptive_budget:
    insert ephemeral user message immediately before the latest user turn,
    so the model reads it as immediate context for the request.
    """
    if not _flag_enabled():
        return None
    block = format_observed_state_block(config)
    if not block:
        return None
    for msg in messages:
        if _BLOCK_TAG in str(msg.get("content", "")):
            return None
    last_user_idx = -1
    for i in range(len(messages) - 1, -1, -1):
        msg = messages[i]
        if msg.get("role") == "user" and not msg.get("ephemeral"):
            last_user_idx = i
            break
    if last_user_idx < 0:
        return None
    result = list(messages)
    result.insert(
        last_user_idx,
        {
            "role": "user",
            "content": block,
            "ephemeral": True,
            "source": "observed_state",
        },
    )
    return result


def contribute_section(messages: list[dict], config: dict):
    """Section-contributor variant for the ephemeral_coalescer."""
    from core.ephemeral_coalescer import SectionResult
    if not _flag_enabled():
        return None
    block = format_observed_state_block(config)
    if not block:
        return None
    return SectionResult(name="observed_state", text=block)
