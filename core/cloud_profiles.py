"""Pure, Qt-free logic for named cloud backend profiles.

Operates on a config dict shaped like ``LLMConfig.model_dump()`` (the panel's
``self.config``). All mutators return a NEW dict; nothing here does I/O or loads
a model. A switched cloud profile sets ``backend="cloud"`` — verified against
``engine.llm.LLMEngine.load_model``, where only ``backend == "gguf"`` is the
local-file path and everything else routes through ``HttpModelLoader`` ->
``make_cloud_llm`` (which URL-detects the provider).
"""
from __future__ import annotations

import copy
from typing import Any
from urllib.parse import urlparse

# Cloud profiles route through the HTTP loader. "cloud" matches the user's
# existing working config and is understood by core.self_description as remote.
CLOUD_BACKEND = "cloud"

_LABELS_BY_HOST = {
    "api.anthropic.com": "Anthropic",
    "api.deepseek.com": "DeepSeek",
    "api.openai.com": "OpenAI",
    "api.mistral.ai": "Mistral",
}


def _normalize_base(api_base: str) -> str:
    return (api_base or "").strip().rstrip("/")


def profile_id(api_provider: str, api_base: str) -> str:
    return f"{(api_provider or '').strip()}|{_normalize_base(api_base)}"


def _id_of(profile: dict) -> str:
    return profile.get("id") or profile_id(profile.get("api_provider", ""), profile.get("api_base", ""))


def derive_label(profile: dict) -> str:
    label = (profile.get("label") or "").strip()
    if label:
        return label
    base = _normalize_base(profile.get("api_base", ""))
    host = (urlparse(base).hostname or base).lower()
    if host in _LABELS_BY_HOST:
        return _LABELS_BY_HOST[host]
    provider = (profile.get("api_provider") or "").strip()
    return provider.capitalize() if provider else (host or "Profile")


def list_profiles(config: dict) -> list[dict]:
    out: list[dict] = []
    for raw in (config.get("cloud_profiles") or []):
        if not isinstance(raw, dict):
            continue
        prof = dict(raw)
        prof["id"] = _id_of(prof)
        prof["label"] = derive_label(prof)
        out.append(prof)
    return out


def active_id(config: dict) -> str:
    return str(config.get("active_cloud_profile") or "")


def is_cloud_active(config: dict) -> bool:
    """True when a cloud profile is active AND the live backend is the cloud
    backend. The precondition for writing success metadata back to the active
    profile — guards against stamping a cloud profile with a local-load's model
    (active_cloud_profile is not cleared when the user switches to local GGUF)."""
    return bool(active_id(config)) and config.get("backend") == CLOUD_BACKEND


def _find(config: dict, target_id: str) -> dict | None:
    for raw in (config.get("cloud_profiles") or []):
        if isinstance(raw, dict) and _id_of(raw) == target_id:
            return raw
    return None


def activate(config: dict, target_id: str) -> dict:
    cfg = copy.deepcopy(config)
    prof = _find(cfg, target_id)
    if prof is None:
        return cfg  # safe no-op
    cfg["api_provider"] = prof.get("api_provider") or "openai"
    cfg["api_base"] = prof.get("api_base") or ""
    cfg["api_key"] = prof.get("api_key") or ""
    cfg["api_model"] = prof.get("last_model") or prof.get("api_model") or ""
    cfg["backend"] = CLOUD_BACKEND
    cfg["active_cloud_profile"] = target_id
    return cfg


def _unique_profile_id(profiles: list, cfg: dict, label: str) -> str:
    """Mint an id for a NEW profile. The first profile for an endpoint keeps the clean
    provider|base id (back-compat); a second profile on the same endpoint disambiguates by
    the user-given name so the two coexist with distinct ids."""
    existing_ids = {_id_of(p) for p in profiles if isinstance(p, dict)}
    base = profile_id(cfg.get("api_provider", ""), cfg.get("api_base", ""))
    if base not in existing_ids:
        return base
    cand = f"{base}|{label}"
    uid, n = cand, 1
    while uid in existing_ids:
        n += 1
        uid = f"{cand}#{n}"
    return uid


def upsert_from_current(config: dict, label: str) -> dict:
    """Save the current connection as a NAMED profile. Identity is the NAME the user gives:
    a NEW name ADDS a profile (multiple per endpoint allowed); an EXISTING name UPDATES it in
    place (preserving cached models). Activates the saved profile."""
    cfg = copy.deepcopy(config)
    label = (label or "").strip() or derive_label(cfg)
    profiles = [p for p in (cfg.get("cloud_profiles") or []) if isinstance(p, dict)]
    existing = next((p for p in profiles if (p.get("label") or derive_label(p)) == label), None)
    pid = _id_of(existing) if existing is not None else _unique_profile_id(profiles, cfg, label)
    new_profile = {
        "id": pid,
        "label": label,
        "api_provider": cfg.get("api_provider", "") or "openai",
        "api_base": cfg.get("api_base", "") or "",
        "api_key": cfg.get("api_key", "") or "",
        "api_model": cfg.get("api_model", "") or "",
        "last_model": cfg.get("api_model", "") or "",
        "models": (existing or {}).get("models", []),
        "last_success_at": (existing or {}).get("last_success_at"),
    }
    profiles = [p for p in profiles if _id_of(p) != pid]
    profiles.append(new_profile)
    cfg["cloud_profiles"] = profiles
    cfg["active_cloud_profile"] = pid
    return cfg


def suggest_label(config: dict) -> str:
    """A default profile name that does NOT collide with an existing one, so 'Save as profile'
    + accept-the-default ADDS a new profile instead of silently overwriting. Type an existing
    name to update that profile instead."""
    base = derive_label(config)
    existing = {(p.get("label") or derive_label(p))
                for p in (config.get("cloud_profiles") or []) if isinstance(p, dict)}
    if base not in existing:
        return base
    n = 2
    while f"{base} {n}" in existing:
        n += 1
    return f"{base} {n}"


def delete(config: dict, target_id: str) -> dict:
    cfg = copy.deepcopy(config)
    cfg["cloud_profiles"] = [p for p in (cfg.get("cloud_profiles") or [])
                             if isinstance(p, dict) and _id_of(p) != target_id]
    if active_id(cfg) == target_id:
        cfg["active_cloud_profile"] = ""
    return cfg


def record_success(config: dict, target_id: str, model: str, when_iso: str) -> dict:
    cfg = copy.deepcopy(config)
    prof = _find(cfg, target_id)
    if prof is not None:
        prof["last_success_at"] = when_iso
        if model:
            prof["last_model"] = model
    return cfg
