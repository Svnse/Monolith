from core.config import get_config, update_config_section

DEFAULT_THEME = "midnight"


def load_theme_config() -> dict:
    cfg = get_config().theme
    return {"theme": cfg.current or DEFAULT_THEME}


def save_theme_config(config: dict) -> None:
    if not isinstance(config, dict):
        return
    value = config.get("theme")
    if not value:
        return
    update_config_section("theme", {"current": value}, persist=True)
