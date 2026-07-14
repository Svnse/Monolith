from __future__ import annotations

import copy
import json
import os
import threading
from pathlib import Path
from typing import Any, Callable

import yaml
from pydantic import BaseModel, Field

from core.llm_prompt import MASTER_PROMPT
from core.paths import CONFIG_DIR, MONOLITH_ROOT


CONFIG_VERSION = 1
CONFIG_PATH = CONFIG_DIR / "config.yaml"


class LLMConfig(BaseModel):
    backend: str = "gguf_api"
    api_provider: str = "openai"
    api_base: str = ""
    api_model: str = ""
    api_key: str = ""
    pipeline_mode: str = "standard"
    pipeline_preset_id: str = ""
    gguf_path: str | None = None
    temp: float = 1.0
    top_p: float = 0.95
    top_k: int = 20
    min_p: float = 0.0
    presence_penalty: float = 1.5
    repetition_penalty: float = 1.0
    # max_tokens field is currently inert at the per-call generation flow
    # (smart-spec re-make 2026-05-09). engine/llm.py LLMEngine.generate,
    # GeneratorWorker, and GGUFRuntime.generate no longer read or send
    # this value — backends use their own ceiling. Field kept so existing
    # config.yaml files don't fail validation; it's the landing pad for
    # the deterministic /effort surface to write into when it lands.
    max_tokens: int = 1024
    # 0 = unset; engine resolves on load via /v1/models -> inference -> registry.
    # Persisted user override is honored only when it would *lower* the resolved ceiling.
    ctx_limit: int = 0
    serve_profile: dict[str, Any] = Field(default_factory=dict)
    endpoint_profiles: list[dict[str, Any]] = Field(default_factory=list)
    active_cloud_profile: str = ""
    cloud_profiles: list[dict[str, Any]] = Field(default_factory=list)
    context_profile: str = "standard_local"
    system_prompt: str = MASTER_PROMPT


class ThemeConfig(BaseModel):
    current: str = "midnight"


class VisionConfig(BaseModel):
    model_path: str = ""
    model_root: str = str(MONOLITH_ROOT / "models" / "vision")
    prompt: str = ""
    negative_prompt: str = ""
    width: int = 512
    height: int = 512
    steps: int = 25
    guidance_scale: float = 7.5
    seed: int = -1
    scheduler: str = "dpm++"
    batch_size: int = 1
    lora_path: str = ""
    lora_scale: float = 0.8
    save_dir: str = ""


class AudioConfig(BaseModel):
    model_path: str = ""
    model_id: str = "facebook/musicgen-small"
    duration: float = 5.0
    sample_rate: int = 32000


class AppConfig(BaseModel):
    version: int = Field(default=CONFIG_VERSION, ge=1)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    theme: ThemeConfig = Field(default_factory=ThemeConfig)
    vision: VisionConfig = Field(default_factory=VisionConfig)
    audio: AudioConfig = Field(default_factory=AudioConfig)


_config_cache: AppConfig | None = None
_config_mtime: float | None = None
_config_env: str | None = None
_env_mtime: float | None = None
_lock = threading.Lock()


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _write_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = yaml.safe_dump(data, sort_keys=False)
    tmp_path = path.with_name(f"{path.name}.tmp")
    tmp_path.write_text(text, encoding="utf-8")
    try:
        os.replace(tmp_path, path)
    except Exception:
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except Exception:
            pass
        raise


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def _legacy_llm_config() -> dict[str, Any] | None:
    path = CONFIG_DIR / "llm_config.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def _legacy_theme_config() -> dict[str, Any] | None:
    path = CONFIG_DIR / "theme.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def _legacy_vision_config() -> dict[str, Any] | None:
    path = CONFIG_DIR / "vision_config.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def _legacy_audio_config() -> dict[str, Any] | None:
    candidates = [
        CONFIG_DIR / "audiogen_config.json",
        Path("config") / "audiogen_config.json",
    ]
    for path in candidates:
        if not path.exists():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return data
        except Exception:
            continue
    return None


def _migrate_from_legacy(base: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    changed = False
    data = dict(base or {})

    llm = _legacy_llm_config()
    if llm:
        data.setdefault("llm", {})
        data["llm"].update(llm)
        changed = True

    theme = _legacy_theme_config()
    if theme:
        data.setdefault("theme", {})
        if "theme" in theme:
            data["theme"]["current"] = theme.get("theme")
        changed = True

    vision = _legacy_vision_config()
    if vision:
        data.setdefault("vision", {})
        data["vision"].update(vision)
        changed = True

    audio = _legacy_audio_config()
    if audio:
        data.setdefault("audio", {})
        data["audio"].update(audio)
        changed = True

    return data, changed


def _migrate_config_dict(raw: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    data = dict(raw or {})
    changed = False
    version = int(data.get("version") or data.get("schema_version") or 0)
    if version < 1:
        data, legacy_changed = _migrate_from_legacy(data)
        changed = changed or legacy_changed
        data["version"] = CONFIG_VERSION
        changed = True
    if data.get("version") != CONFIG_VERSION:
        data["version"] = CONFIG_VERSION
        changed = True
    return data, changed


def _load_base_config() -> tuple[dict[str, Any], bool]:
    raw = _read_yaml(CONFIG_PATH)
    migrated, changed = _migrate_config_dict(raw)
    if changed or not CONFIG_PATH.exists():
        validated = AppConfig.model_validate(migrated)
        # Preserve unknown keys through the migration rewrite: deep-merge the
        # coerced known fields over the raw dict instead of writing the
        # unknown-stripping model_dump().
        full = _deep_merge(
            copy.deepcopy(migrated) if isinstance(migrated, dict) else {},
            validated.model_dump(),
        )
        _write_yaml(CONFIG_PATH, full)
        return full, True
    return migrated, changed


def _env_overlay(env_name: str | None) -> tuple[dict[str, Any], float | None]:
    if not env_name:
        return {}, None
    env_path = CONFIG_DIR / f"config.{env_name}.yaml"
    if not env_path.exists():
        return {}, None
    return _read_yaml(env_path), env_path.stat().st_mtime


def get_config(env: str | None = None, force_reload: bool = False) -> AppConfig:
    global _config_cache, _config_mtime, _config_env, _env_mtime
    env_name = env if env is not None else os.getenv("MONOLITH_ENV")
    env_name = env_name or ""
    with _lock:
        base_mtime = CONFIG_PATH.stat().st_mtime if CONFIG_PATH.exists() else None
        env_overlay, current_env_mtime = _env_overlay(env_name)

        if (
            not force_reload
            and _config_cache is not None
            and _config_mtime == base_mtime
            and _config_env == env_name
            and _env_mtime == current_env_mtime
        ):
            return _config_cache

        base, _ = _load_base_config()
        merged = dict(base)
        if env_overlay:
            _deep_merge(merged, env_overlay)

        try:
            cfg = AppConfig.model_validate(merged)
        except Exception:
            cfg = AppConfig()
        _config_cache = cfg
        _config_mtime = base_mtime
        _config_env = env_name
        _env_mtime = current_env_mtime
        return cfg


def update_config_section(section: str, values: dict[str, Any], persist: bool = True) -> AppConfig:
    if not isinstance(values, dict):
        raise ValueError("values must be a dict")
    with _lock:
        base, _ = _load_base_config()
        # Deep-merge the caller's update onto the RAW base so unknown sibling
        # keys (cloud_profiles, active_cloud_profile, effort, *_secondary, and
        # anything future) survive. Pydantic validation drops unknown keys, so
        # writing model_dump() directly silently wipes them.
        merged = copy.deepcopy(base) if isinstance(base, dict) else {}
        section_dict = merged.get(section)
        if not isinstance(section_dict, dict):
            section_dict = {}
            merged[section] = section_dict
        section_dict.update(values)
        # Validate to coerce/default KNOWN fields and raise on invalid ones.
        updated = AppConfig.model_validate(merged)
        # Land coerced known fields on top of the raw dict; unknown keys remain
        # because updated.model_dump() simply has no key for them.
        write_dict = _deep_merge(copy.deepcopy(merged), updated.model_dump())
        if persist:
            # system_prompt is always rebuilt from the master prompt on load
            # (core.llm_config.load_config), so it must never be persisted —
            # regardless of which section triggered the write. The non-lossy
            # merge would otherwise make a stray master-prompt blob sticky.
            if isinstance(write_dict.get("llm"), dict):
                write_dict["llm"].pop("system_prompt", None)
            _write_yaml(CONFIG_PATH, write_dict)
            global _config_cache, _config_mtime
            _config_cache = updated
            _config_mtime = CONFIG_PATH.stat().st_mtime if CONFIG_PATH.exists() else None
        return updated


class ConfigWatcher:
    def __init__(self, poll_interval: float = 1.0, env: str | None = None):
        self._poll_interval = poll_interval
        self._env = env
        self._callbacks: list[Callable[[AppConfig, AppConfig | None], None]] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._last_config: AppConfig | None = None

    def on_change(self, callback: Callable[[AppConfig, AppConfig | None], None]) -> None:
        self._callbacks.append(callback)

    def start(self) -> None:
        if self._thread.is_alive():
            return
        self._last_config = get_config(env=self._env)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=1.0)

    def _run(self) -> None:
        while not self._stop.wait(self._poll_interval):
            cfg = get_config(env=self._env)
            if self._last_config is None:
                self._last_config = cfg
                continue
            if cfg.model_dump() == self._last_config.model_dump():
                continue
            previous = self._last_config
            self._last_config = cfg
            for callback in list(self._callbacks):
                try:
                    callback(cfg, previous)
                except Exception:
                    continue
