import os
from pathlib import Path


def _default_monolith_root() -> Path:
    override = os.getenv("MONOLITH_ROOT")
    if override:
        return Path(override).expanduser()

    if os.name == "nt":
        appdata = os.getenv("APPDATA")
        if appdata:
            return Path(appdata) / "Monolith"
        return Path.home() / "AppData" / "Roaming" / "Monolith"

    legacy_root = Path.home() / "Monolith"
    if legacy_root.exists():
        # Backward compatibility for existing Linux/macOS installs.
        return legacy_root

    xdg_data_home = os.getenv("XDG_DATA_HOME")
    if xdg_data_home:
        return Path(xdg_data_home).expanduser() / "Monolith"
    return Path.home() / ".local" / "share" / "Monolith"


MONOLITH_ROOT = _default_monolith_root()

CONFIG_DIR = MONOLITH_ROOT / "config"
ARCHIVE_DIR = MONOLITH_ROOT / "chats"
LOG_DIR = MONOLITH_ROOT / "logs"
ADDON_CONFIG_DIR = MONOLITH_ROOT / "addons" / "configs"

for _dir in (MONOLITH_ROOT, CONFIG_DIR, ARCHIVE_DIR, LOG_DIR, ADDON_CONFIG_DIR):
    _dir.mkdir(parents=True, exist_ok=True)
