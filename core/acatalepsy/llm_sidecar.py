"""LLM sidecar adapter — bridges the chat backend to ``LLMCallable``.

Per Acatalepsy v1 design call (Claude+GPT, 2026-05-13): the auditor is
a non-live, batch epistemic job; it should NOT share the main chat
engine/thread path. This sidecar instantiates a separate, lightweight
client against the same backend Monolith is using, with auditor-tuned
defaults:

  - temperature=0 (determinism — same slice should produce same candidates)
  - bounded timeout (default 90s — never hang on flaky backend)
  - bounded max_tokens (default 8192 — large enough for ~50 candidates)
  - no tool use, no system extras — pure JSON-emit task
  - no Qt/UI dependency (runs in plain Python thread)

Scope (v1):
  - Supports openai-compat backends only (api_base + api_model + api_key)
  - gguf local-model sidecar would require loading a second Llama instance
    — too heavy for v1. Documented as a v1.5+ extension.

The sidecar reads ``LLMConfig`` for backend coordinates but ignores the
chat-side ``system_prompt`` (the auditor has its own six-section prompt
from core/acatalepsy/auditor.py).
"""
from __future__ import annotations

import json
from typing import Any, Callable

from core.acatalepsy.auditor import LLMCallable


__all__ = (
    "SidecarUnsupportedBackend",
    "SidecarConfigError",
    "make_auditor_llm",
)


# Auditor-tuned defaults. Override via env or kwargs if needed.
AUDITOR_TEMPERATURE: float = 0.0
AUDITOR_TIMEOUT_SECS: float = 180.0
# Bumped to 32K (was 8K) to give reasoning models room for both the
# internal think pass AND the JSON output. DeepSeek-v4-pro spends
# considerable completion budget on `reasoning_content` before
# emitting `.content`; 8K was not enough for non-trivial audit slices.
AUDITOR_MAX_TOKENS: int = 32768


class SidecarUnsupportedBackend(RuntimeError):
    """Raised when the sidecar can't run against the current backend.

    v1 supports only openai-compat (api_base + api_model + api_key). For
    gguf local-model backends, run an HTTP server in front of the model
    (llama-cpp-python server, vLLM, etc.) and point api_base at it.
    """


class SidecarConfigError(RuntimeError):
    """Raised when LLMConfig is missing fields required by the sidecar
    (api_base or api_model)."""


def _load_current_llm_config() -> dict[str, Any]:
    """Read the current LLMConfig (api_base / api_model / api_key /
    backend). Lazy import — keeps the sidecar importable without a live
    config in test contexts.
    """
    from core.config import get_config
    cfg = get_config().llm
    return {
        "backend": cfg.backend,
        "api_provider": cfg.api_provider,
        "api_base": cfg.api_base.strip(),
        "api_model": cfg.api_model.strip(),
        "api_key": cfg.api_key.strip(),
    }


def _build_openai_compat_callable(
    *,
    api_base: str,
    api_model: str,
    api_key: str,
    temperature: float,
    timeout_secs: float,
    max_tokens: int,
) -> LLMCallable:
    """Create an LLMCallable backed by the openai SDK pointing at api_base."""
    try:
        # The OpenAI SDK is an optional Monolith dependency.
        # (used by the chat engine for cloud-backend dispatch).
        from openai import OpenAI
    except ImportError as exc:
        raise SidecarConfigError(
            "openai SDK not available — install `openai` to use the sidecar."
        ) from exc

    client = OpenAI(
        base_url=api_base or None,
        api_key=api_key or "sk-noauth",  # local servers often accept any string
        timeout=timeout_secs,
    )

    def call(*, system_prompt: str, user_content: str) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]
        # max_tokens kept conservative; auditor output is JSON, not prose.
        kwargs: dict[str, Any] = {
            "model": api_model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        # Some openai-compat servers (e.g., DeepSeek) reject `max_tokens`
        # or `temperature=0`; strip on TypeError and retry once.
        try:
            response = client.chat.completions.create(**kwargs)
        except TypeError as exc:
            # Best-effort retry without contentious params.
            msg = str(exc)
            if "max_tokens" in msg:
                kwargs.pop("max_tokens", None)
                response = client.chat.completions.create(**kwargs)
            elif "temperature" in msg:
                kwargs.pop("temperature", None)
                response = client.chat.completions.create(**kwargs)
            else:
                raise
        # Some servers wrap content in choices[0].message.content; standard
        # openai SDK gives that shape directly.
        choice = response.choices[0]
        text = getattr(choice.message, "content", None) or ""
        # Reasoning-model fallback: DeepSeek-v4-pro and similar return a
        # `reasoning_content` field alongside `.content`. If the model
        # exhausted its completion budget inside the think pass and
        # emitted no `.content`, the reasoning_content may still hold
        # the answer (or at least useful diagnostic). Surface it as
        # fallback rather than treating "" as a hard failure.
        if not text:
            rc = getattr(choice.message, "reasoning_content", None)
            if isinstance(rc, str) and rc.strip():
                text = rc
        return text

    return call


def make_auditor_llm(
    *,
    temperature: float = AUDITOR_TEMPERATURE,
    timeout_secs: float = AUDITOR_TIMEOUT_SECS,
    max_tokens: int = AUDITOR_MAX_TOKENS,
    config_override: dict[str, Any] | None = None,
) -> LLMCallable:
    """Build an LLMCallable for the auditor.

    Reads the current LLMConfig (or ``config_override`` for tests),
    validates it's an openai-compat backend, returns a callable
    conforming to the auditor.LLMCallable protocol.

    Raises:
        SidecarUnsupportedBackend — backend is gguf/local (v1 unsupported)
        SidecarConfigError — required fields missing
    """
    cfg = config_override if config_override is not None else _load_current_llm_config()

    backend = str(cfg.get("backend") or "").strip().lower()
    if backend in ("gguf",) and not cfg.get("api_base"):
        raise SidecarUnsupportedBackend(
            f"v1 sidecar does not support backend={backend!r} without an api_base. "
            f"Run a local HTTP server (llama-cpp-python server / vLLM) and point "
            f"api_base at it."
        )

    api_base = str(cfg.get("api_base") or "").strip()
    api_model = str(cfg.get("api_model") or "").strip()
    api_key = str(cfg.get("api_key") or "").strip()

    if not api_base:
        raise SidecarConfigError(
            "sidecar requires api_base — set it in LLMConfig (cloud provider URL "
            "or local server endpoint)."
        )
    if not api_model:
        raise SidecarConfigError(
            "sidecar requires api_model — set it in LLMConfig (e.g. 'deepseek-v4-pro')."
        )

    return _build_openai_compat_callable(
        api_base=api_base,
        api_model=api_model,
        api_key=api_key,
        temperature=float(temperature),
        timeout_secs=float(timeout_secs),
        max_tokens=int(max_tokens),
    )
