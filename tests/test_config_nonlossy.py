from __future__ import annotations

import yaml
import pytest

from core import config as C


@pytest.fixture
def cfg_file(tmp_path, monkeypatch):
    path = tmp_path / "config.yaml"
    monkeypatch.setattr(C, "CONFIG_PATH", path)
    # Reset the module-level cache so each test starts clean.
    monkeypatch.setattr(C, "_config_cache", None)
    monkeypatch.setattr(C, "_config_mtime", None)
    monkeypatch.setattr(C, "_config_env", None)
    monkeypatch.setattr(C, "_env_mtime", None)
    return path


def test_update_preserves_unknown_sibling_keys(cfg_file):
    cfg_file.write_text(yaml.safe_dump({
        "version": 1,
        "llm": {
            "api_base": "http://old",
            "cloud_profiles": [{"id": "x|y", "api_base": "y"}],
            "effort": "med",
            "__orphan__": 7,
        },
    }), encoding="utf-8")

    C.update_config_section("llm", {"api_base": "http://new"})

    written = yaml.safe_load(cfg_file.read_text(encoding="utf-8"))["llm"]
    assert written["api_base"] == "http://new"                            # update applied
    assert written["cloud_profiles"] == [{"id": "x|y", "api_base": "y"}]  # list preserved
    assert written["effort"] == "med"                                     # orphan preserved
    assert written["__orphan__"] == 7                                     # arbitrary orphan preserved


def test_update_other_section_does_not_resurrect_or_drop(cfg_file):
    cfg_file.write_text(yaml.safe_dump({
        "version": 1,
        "llm": {"__orphan__": 1},
        "theme": {"current": "midnight"},
    }), encoding="utf-8")

    C.update_config_section("theme", {"current": "dawn"})

    data = yaml.safe_load(cfg_file.read_text(encoding="utf-8"))
    assert data["theme"]["current"] == "dawn"     # update applied
    assert data["llm"]["__orphan__"] == 1         # untouched section's orphan preserved


def test_real_world_llm_shape_survives_panel_save(cfg_file):
    """Mirror the user's actual config.yaml (backend cloud, cloud_profiles with
    explicit ids + keys, plus orphan keys effort/m_mode_enabled/*_secondary) and
    simulate the panel's save (load_config -> save_config). Nothing must be lost."""
    cfg_file.write_text(yaml.safe_dump({
        "version": 1,
        "llm": {
            "backend": "cloud",
            "api_provider": "anthropic",
            "api_base": "https://api.anthropic.com",
            "api_model": "claude-opus-4-8",
            "api_key": "sk-ant-REDACTED",
            "active_cloud_profile": "anthropic|https://api.anthropic.com",
            "cloud_profiles": [
                {"id": "openai|https://api.deepseek.com", "api_provider": "openai",
                 "api_base": "https://api.deepseek.com", "api_key": "sk-ds-REDACTED",
                 "api_model": "deepseek-v4-pro", "models": ["deepseek-v4-pro"]},
                {"id": "anthropic|https://api.anthropic.com", "api_provider": "anthropic",
                 "api_base": "https://api.anthropic.com", "api_key": "sk-ant-REDACTED",
                 "api_model": "claude-opus-4-8", "models": ["claude-opus-4-8"]},
            ],
            "effort": "med",
            "m_mode_enabled": True,
            "backend_secondary": "gguf_api",
            "api_base_secondary": "",
            "gguf_path_secondary": None,
        },
    }), encoding="utf-8")

    from core.llm_config import load_config, save_config
    save_config(load_config())   # the "next save" that previously wiped orphans

    llm = yaml.safe_load(cfg_file.read_text(encoding="utf-8"))["llm"]
    assert llm["active_cloud_profile"] == "anthropic|https://api.anthropic.com"
    assert len(llm["cloud_profiles"]) == 2
    assert all(p.get("api_key") for p in llm["cloud_profiles"])   # keys survive
    assert llm["effort"] == "med"                                 # orphans survive
    assert llm["m_mode_enabled"] is True
    assert llm["backend_secondary"] == "gguf_api"
    assert "api_base_secondary" in llm
    assert "gguf_path_secondary" in llm
    assert "system_prompt" not in llm                             # still never persisted


def test_update_nonllm_section_does_not_persist_master_prompt(cfg_file):
    cfg_file.write_text(yaml.safe_dump({
        "version": 1,
        "llm": {"api_base": "x"},
        "theme": {"current": "midnight"},
    }), encoding="utf-8")

    C.update_config_section("theme", {"current": "dawn"})

    written = yaml.safe_load(cfg_file.read_text(encoding="utf-8"))
    # system_prompt is rebuilt from the master prompt on load, so it must never
    # be persisted to config.yaml — regardless of which section triggered the write.
    assert "system_prompt" not in written.get("llm", {})


def test_llmconfig_roundtrips_cloud_profiles(cfg_file):
    cfg_file.write_text(yaml.safe_dump({
        "version": 1,
        "llm": {
            "active_cloud_profile": "openai|https://api.deepseek.com",
            "cloud_profiles": [
                {"id": "openai|https://api.deepseek.com",
                 "api_provider": "openai", "api_base": "https://api.deepseek.com",
                 "api_model": "deepseek-v4-pro", "models": ["deepseek-v4-pro"],
                 "last_success_at": None},
            ],
        },
    }), encoding="utf-8")

    llm = C.get_config(force_reload=True).llm
    assert llm.active_cloud_profile == "openai|https://api.deepseek.com"
    assert llm.cloud_profiles[0]["api_model"] == "deepseek-v4-pro"
    # load_config (the panel's reader) must also surface them.
    from core.llm_config import load_config
    assert load_config()["cloud_profiles"][0]["id"] == "openai|https://api.deepseek.com"
