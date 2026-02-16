import os
from pathlib import Path


if "MONOLITH_ROOT" in os.environ:
    MONOLITH_ROOT = Path(os.environ["MONOLITH_ROOT"]).expanduser()
elif os.name == "nt":
    appdata = os.getenv("APPDATA")
    if appdata:
        MONOLITH_ROOT = Path(appdata) / "Monolith"
    else:
        MONOLITH_ROOT = Path.home() / "AppData" / "Roaming" / "Monolith"
else:
    MONOLITH_ROOT = Path.home() / "Monolith"

CONFIG_DIR = MONOLITH_ROOT / "config"
ARCHIVE_DIR = MONOLITH_ROOT / "chats"
LOG_DIR = MONOLITH_ROOT / "logs"
ADDON_CONFIG_DIR = MONOLITH_ROOT / "addons" / "configs"

for _dir in (MONOLITH_ROOT, CONFIG_DIR, ARCHIVE_DIR, LOG_DIR, ADDON_CONFIG_DIR):
    _dir.mkdir(parents=True, exist_ok=True)
