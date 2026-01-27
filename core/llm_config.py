import json
from pathlib import Path

DEFAULT_CONFIG = {
    "gguf_path": None,
    "temp": 0.7,
    "top_p": 0.9,
    "max_tokens": 2048,
    "ctx_limit": 8192,
    "system_prompt": "You are Monolith. Be precise.",
    "context_injection": "",
}

CONFIG_PATH = Path("ui/addons/configs/llm_config.json")


def load_config():
    config = DEFAULT_CONFIG.copy()
    if CONFIG_PATH.exists():
        try:
            with CONFIG_PATH.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
                if isinstance(data, dict):
                    config.update(data)
        except Exception:
            pass
    config.setdefault("context_injection", "")
    return config


def save_config(config):
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CONFIG_PATH.open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)
