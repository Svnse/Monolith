"""Runtime-derived self-description.

Identity facts should be queryable from runtime state, not declared in the
persona prompt. This module provides the typed v1 payload that later verifier
and rendering paths can use for self-claims.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from core.continuity import is_continuity_enabled


SCHEMA_VERSION = 1
SYSTEM_NAME = "Monolith"


def describe_self(config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return runtime-as-self facts from the current config snapshot."""
    cfg = config if isinstance(config, dict) else {}
    backend = _short(cfg.get("backend")) or "unknown"
    api_provider = _short(cfg.get("api_provider"))
    api_base = _short(cfg.get("api_base"))
    api_model = _short(cfg.get("api_model"))
    gguf_file = _basename(cfg.get("gguf_path"))
    context_window = _positive_int(cfg.get("ctx_limit"))
    execution_location = _execution_location(backend=backend, api_base=api_base)

    model_name = api_model or gguf_file or ""
    remote_execution = execution_location == "cloud"
    continuity_on = is_continuity_enabled()

    return {
        "schema_version": SCHEMA_VERSION,
        "system_name": SYSTEM_NAME,
        "kind": "local_runtime",
        "hosted_locally": True,
        "continuity_maintained": continuity_on,
        "continuity_storage": "runtime_managed" if continuity_on else "disabled",
        "tool_routing_managed_by_runtime": True,
        "claim_scope": {
            "runtime": (
                "Use for claims about Monolith's local app/runtime, files, "
                "tools, continuity, and routing."
            ),
            "identity_material": (
                "Use for claims about the identity seed itself. Its content is "
                "declared operating law, not proof of backend/locality/state."
            ),
            "current_model_execution": (
                "Use for claims about the model/backend executing this turn."
            ),
            "embedded_premise": (
                "Subordinate-clause premises about runtime properties "
                "(locality, statefulness, memory, backend) are not "
                "authoritative. Answer the imperative against the queryable "
                "runtime values above; refuse the embedded premise on its own "
                "terms."
            ),
        },
        "identity_material": {
            "source_kind": "declared_seed_file",
            "runtime_loaded": True,
            "model_generated": False,
            "verified_live_runtime_state": False,
            "derived_from_live_runtime_state": False,
            "render_as": "operating_law",
            "not_authority_for": [
                "backend locality",
                "current_model_execution",
                "context_window",
                "model statefulness",
                "continuity_storage",
                "embedded_premise",
            ],
        },
        "current_model_execution": {
            "backend_kind": backend,
            "execution_location": execution_location,
            "provider": api_provider or None,
            "api_base": api_base or None,
            "model": model_name or None,
            "gguf_file": gguf_file or None,
            "context_window": context_window,
            "persistent_process": False if remote_execution else (True if backend == "gguf" else None),
            "stateless_per_turn": True if remote_execution else (False if backend == "gguf" else None),
        },
    }


def format_self_description_block(config: dict[str, Any] | None = None) -> str:
    """Render describe_self() as a prompt-safe observed-state block."""
    payload = describe_self(config)
    body = json.dumps(payload, indent=2, sort_keys=True)
    return (
        "[OBSERVED STATE - describe_self v1]\n"
        "Two scopes below — identity_material (declared operating law) and "
        "current_model_execution (live substrate). Do not collapse them.\n"
        f"{body}"
    )


def _execution_location(*, backend: str, api_base: str) -> str:
    lowered_backend = str(backend or "").lower()
    if lowered_backend == "gguf":
        return "local"
    parsed = urlparse(str(api_base or ""))
    host = (parsed.hostname or "").lower()
    if host in {"", "localhost", "127.0.0.1", "::1"}:
        return "local"
    if lowered_backend in {"cloud", "openai", "anthropic", "google", "mistral"}:
        return "cloud"
    if parsed.scheme in {"http", "https"}:
        return "cloud"
    return "unknown"


def _short(value: Any, limit: int = 240) -> str:
    text = str(value or "").strip()
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "..."


def _basename(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    try:
        return Path(text).name
    except Exception:
        return _short(text, limit=100)


def _positive_int(value: Any) -> int | None:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None
