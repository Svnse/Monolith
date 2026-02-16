import json
from core.paths import CONFIG_DIR

THEME_CONFIG_PATH = CONFIG_DIR / "theme.json"
DEFAULT_THEME = "midnight"


def load_theme_config() -> dict:
    config = {"theme": DEFAULT_THEME}
    if THEME_CONFIG_PATH.exists():
        try:
            with THEME_CONFIG_PATH.open("r", encoding="utf-8") as f:
                data = json.load(f)
                config.update(data)
        except Exception:
            pass
    return config


def save_theme_config(config: dict) -> None:
    THEME_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with THEME_CONFIG_PATH.open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
