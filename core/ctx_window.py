"""Cloud context-window resolution.

Single source of truth for "how big is this cloud model's context window?"
Most OpenAI-compatible /v1/models endpoints don't expose context_length, so
this module layers a fetch attempt on top of a small inference table.

Priority on the cloud-load path:
    1. /v1/models record exposes context_length / context_window      -> use it
    2. Inference table (provider+model heuristics, e.g. DeepSeek-V4)  -> use it
    3. Registry family default (model_registry.json)                  -> use it
    4. None  -> caller falls back to whatever ctx_limit it had

The runtime ctx_limit on the engine is treated separately: it acts as a
USER-EXPLICIT cost cap that lowers the resolved ceiling, never raises it.
A stale persisted 8192 must NOT impersonate ground truth.
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any
from urllib.parse import urlparse


_DEEPSEEK_V4_CONTEXT = 1_000_000
_DEEPSEEK_LEGACY_CONTEXT = 131_072
_MODELS_ENDPOINT_TIMEOUT = 4.0
_CTX_LENGTH_KEYS = (
    "context_length",
    "context_window",
    "max_context_length",
    "max_context_tokens",
    "n_ctx",
    "max_input_tokens",
    "max_position_embeddings",
)


def _extract_ctx_length(record: Any) -> int | None:
    if not isinstance(record, dict):
        return None
    for key in _CTX_LENGTH_KEYS:
        if key in record:
            try:
                value = int(record[key])
            except (TypeError, ValueError):
                continue
            if value > 0:
                return value
    return None


def fetch_cloud_context_window(
    *,
    api_base: str | None,
    api_key: str | None = None,
    model: str | None = None,
    timeout: float = _MODELS_ENDPOINT_TIMEOUT,
) -> int | None:
    """Query the OpenAI-compatible /models endpoint for a model's window.

    Returns the published context length when the provider exposes it on the
    model record, otherwise None. Most providers omit this field; callers
    should fall back to infer_cloud_context_window or the registry default.
    """
    base = (api_base or "").strip()
    if not base:
        return None
    if "://" not in base:
        base = f"https://{base}"
    base = base.rstrip("/")
    if base.endswith("/v1"):
        url = f"{base}/models"
    elif "/v1" in base:
        url = f"{base}/models"
    else:
        url = f"{base}/v1/models"
    request = urllib.request.Request(url, method="GET")
    request.add_header("Accept", "application/json")
    if api_key:
        request.add_header("Authorization", f"Bearer {api_key}")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8", errors="ignore") or "{}")
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, ValueError, OSError):
        return None
    records: list[Any] = []
    if isinstance(payload, list):
        records = payload
    elif isinstance(payload, dict):
        data = payload.get("data") or payload.get("models") or []
        if isinstance(data, list):
            records = data
    target = (model or "").strip().lower()
    if target:
        for record in records:
            if not isinstance(record, dict):
                continue
            rid = str(record.get("id") or record.get("name") or "").strip().lower()
            if rid == target:
                hit = _extract_ctx_length(record)
                if hit:
                    return hit
    for record in records:
        hit = _extract_ctx_length(record)
        if hit:
            return hit
    return None


def infer_cloud_context_window(
    *,
    provider: str | None = None,
    api_base: str | None = None,
    model: str | None = None,
) -> int | None:
    """Infer a cloud model's published context window when /models omits it.

    OpenAI-compatible providers often return only IDs from /models. Keep this
    intentionally narrow: an unknown provider returns None so callers fall
    through to the registry default rather than to a stale local guess.
    """
    provider_norm = str(provider or "").strip().lower()
    model_norm = str(model or "").strip().lower()
    base_norm = str(api_base or "").strip().lower()
    host = ""
    if base_norm:
        parsed = urlparse(base_norm if "://" in base_norm else f"https://{base_norm}")
        host = parsed.netloc or parsed.path

    is_deepseek = (
        "deepseek" in provider_norm
        or "deepseek" in host
        or model_norm.startswith("deepseek-")
    )
    if not is_deepseek:
        return None

    if model_norm in {
        "deepseek-v4-flash",
        "deepseek-v4-pro",
        "deepseek-chat",
        "deepseek-reasoner",
    }:
        return _DEEPSEEK_V4_CONTEXT
    if "deepseek-v4" in model_norm:
        return _DEEPSEEK_V4_CONTEXT
    return _DEEPSEEK_LEGACY_CONTEXT


def resolve_cloud_window(
    *,
    api_base: str | None,
    api_key: str | None,
    api_provider: str | None,
    api_model: str | None,
    registry_window: int | None = None,
) -> tuple[int | None, str]:
    """Compose the lookup chain. Returns (window_or_None, source_label)."""
    fetched = fetch_cloud_context_window(api_base=api_base, api_key=api_key, model=api_model)
    if fetched:
        return fetched, "fetched_models_endpoint"
    inferred = infer_cloud_context_window(provider=api_provider, api_base=api_base, model=api_model)
    if inferred:
        return inferred, "inferred_provider_table"
    if registry_window:
        return int(registry_window), "registry_family_default"
    return None, "unresolved"
