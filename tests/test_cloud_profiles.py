from __future__ import annotations

from core import cloud_profiles as cp


def _seed():
    return {
        "backend": "cloud", "api_provider": "anthropic",
        "api_base": "https://api.anthropic.com", "api_key": "sk-ant",
        "api_model": "claude-opus-4-8",
        "active_cloud_profile": "anthropic|https://api.anthropic.com",
        "cloud_profiles": [
            {"id": "anthropic|https://api.anthropic.com", "api_provider": "anthropic",
             "api_base": "https://api.anthropic.com", "api_key": "sk-ant",
             "api_model": "claude-opus-4-8", "last_model": "claude-opus-4-8",
             "models": ["claude-opus-4-8"], "last_success_at": None},
            {"id": "openai|https://api.deepseek.com", "api_provider": "openai",
             "api_base": "https://api.deepseek.com", "api_key": "sk-ds",
             "api_model": "deepseek-v4-pro", "last_model": "deepseek-v4-pro",
             "models": ["deepseek-v4-pro"], "last_success_at": None},
        ],
    }


def test_profile_id_normalizes_trailing_slash():
    assert cp.profile_id("openai", "https://api.deepseek.com/") == "openai|https://api.deepseek.com"


def test_derive_label_from_host_when_absent():
    assert cp.derive_label({"api_base": "https://api.deepseek.com"}) == "DeepSeek"
    assert cp.derive_label({"api_base": "https://api.anthropic.com"}) == "Anthropic"
    assert cp.derive_label({"label": "My Box", "api_base": "https://x"}) == "My Box"


def test_list_profiles_fills_id_and_label_backward_compat():
    # entry with neither id nor label (the user's real data shape)
    cfg = {"cloud_profiles": [{"api_provider": "openai", "api_base": "https://api.deepseek.com"}]}
    out = cp.list_profiles(cfg)
    assert out[0]["id"] == "openai|https://api.deepseek.com"
    assert out[0]["label"] == "DeepSeek"


def test_activate_mirrors_to_toplevel_and_sets_backend_cloud():
    cfg = cp.activate(_seed(), "openai|https://api.deepseek.com")
    assert cfg["api_base"] == "https://api.deepseek.com"
    assert cfg["api_key"] == "sk-ds"
    assert cfg["api_model"] == "deepseek-v4-pro"        # uses last_model
    assert cfg["api_provider"] == "openai"
    assert cfg["backend"] == "cloud"                    # verified routing value
    assert cfg["active_cloud_profile"] == "openai|https://api.deepseek.com"


def test_activate_missing_id_is_safe_noop():
    seed = _seed()
    out = cp.activate(seed, "nope|nope")
    assert out["active_cloud_profile"] == seed["active_cloud_profile"]  # unchanged


def test_upsert_from_current_adds_and_activates():
    cfg = {"api_provider": "openai", "api_base": "https://api.mistral.ai",
           "api_key": "sk-m", "api_model": "mistral-large", "cloud_profiles": []}
    out = cp.upsert_from_current(cfg, "Mistral")
    assert out["active_cloud_profile"] == "openai|https://api.mistral.ai"
    prof = [p for p in out["cloud_profiles"] if p["id"] == "openai|https://api.mistral.ai"][0]
    assert prof["label"] == "Mistral"
    assert prof["last_model"] == "mistral-large"


def test_upsert_new_name_same_endpoint_adds_not_replaces():
    # E's bug: saving with a NEW NAME must ADD a profile, even on the same endpoint --
    # NOT silently replace the existing one. Identity is the user-given name.
    out = cp.upsert_from_current(_seed(), "Claude (work)")  # anthropic endpoint, new name
    labels = [p.get("label") or cp.derive_label(p) for p in out["cloud_profiles"]]
    assert "Claude (work)" in labels
    assert len(out["cloud_profiles"]) == 3                     # anthropic + deepseek + new
    assert len({p["id"] for p in cp.list_profiles(out)}) == 3  # distinct ids (menu can switch each)
    # the original anthropic profile is untouched (cached models preserved)
    orig = [p for p in out["cloud_profiles"] if p["id"] == "anthropic|https://api.anthropic.com"][0]
    assert orig["models"] == ["claude-opus-4-8"]


def test_upsert_same_name_updates_in_place():
    # Same name -> update the existing profile, no new row.
    out = cp.upsert_from_current(_seed(), "My Claude")        # new name -> adds (3rd)
    n = len(out["cloud_profiles"])
    out["api_model"] = "claude-opus-4-9"
    out = cp.upsert_from_current(out, "My Claude")            # same name -> update in place
    assert len(out["cloud_profiles"]) == n
    prof = [p for p in out["cloud_profiles"] if (p.get("label") == "My Claude")][0]
    assert prof["last_model"] == "claude-opus-4-9"


def test_upsert_first_profile_per_endpoint_keeps_clean_id():
    # back-compat: the first profile for an endpoint keeps the provider|base id.
    cfg = {"api_provider": "openai", "api_base": "https://api.mistral.ai",
           "api_key": "sk-m", "api_model": "mistral-large", "cloud_profiles": []}
    out = cp.upsert_from_current(cfg, "Mistral")
    assert out["active_cloud_profile"] == "openai|https://api.mistral.ai"


def test_suggest_label_is_unique_against_existing():
    seed = _seed()   # already has an "Anthropic" + "DeepSeek"
    # current top-level endpoint is anthropic -> derived "Anthropic" collides -> suggest "Anthropic 2"
    assert cp.suggest_label(seed) == "Anthropic 2"


def test_delete_removes_and_clears_active_if_active():
    out = cp.delete(_seed(), "anthropic|https://api.anthropic.com")
    assert all(p["id"] != "anthropic|https://api.anthropic.com" for p in out["cloud_profiles"])
    assert out["active_cloud_profile"] == ""   # was active -> cleared


def test_is_cloud_active_requires_active_and_cloud_backend():
    # Guard for write-back: only when a cloud profile is active AND the live
    # backend is cloud. Prevents stamping a cloud profile with a local load.
    assert cp.is_cloud_active({"active_cloud_profile": "x|y", "backend": "cloud"}) is True
    assert cp.is_cloud_active({"active_cloud_profile": "x|y", "backend": "gguf"}) is False
    assert cp.is_cloud_active({"active_cloud_profile": "x|y", "backend": "gguf_api"}) is False
    assert cp.is_cloud_active({"active_cloud_profile": "", "backend": "cloud"}) is False


def test_record_success_writes_back():
    out = cp.record_success(_seed(), "openai|https://api.deepseek.com",
                            "deepseek-v4-flash", "2026-06-08T00:00:00Z")
    prof = [p for p in out["cloud_profiles"] if p["id"] == "openai|https://api.deepseek.com"][0]
    assert prof["last_success_at"] == "2026-06-08T00:00:00Z"
    assert prof["last_model"] == "deepseek-v4-flash"
