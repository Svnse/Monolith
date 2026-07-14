import os
from pathlib import Path


def _is_relative_to(path: Path, base: Path) -> bool:
    try:
        path.relative_to(base)
        return True
    except ValueError:
        return False


def _default_root() -> Path:
    if os.name == "nt":
        appdata = os.getenv("APPDATA")
        if appdata:
            return Path(appdata) / "Monolith"
        return Path.home() / "AppData" / "Roaming" / "Monolith"
    return Path.home() / "Monolith"


def _resolve_root() -> Path:
    default_root = _default_root().expanduser().resolve()
    raw_root = os.getenv("MONOLITH_ROOT")
    if not raw_root:
        return default_root

    candidate = Path(raw_root).expanduser()
    if not candidate.is_absolute():
        return default_root

    try:
        resolved = candidate.resolve()
    except OSError:
        return default_root

    allow_unanchored = str(os.getenv("MONOLITH_ALLOW_UNANCHORED_ROOT", "")).strip().lower() in {"1", "true", "yes"}
    if allow_unanchored:
        return resolved

    anchors = [Path.home().expanduser().resolve()]
    if os.name == "nt":
        appdata = os.getenv("APPDATA")
        if appdata:
            anchors.append(Path(appdata).expanduser().resolve())

    if any(_is_relative_to(resolved, anchor) for anchor in anchors):
        return resolved
    return default_root


def ensure_safe_local_path(path: Path) -> Path:
    candidate = Path(path).expanduser()
    parent = candidate.parent
    parent.mkdir(parents=True, exist_ok=True)

    if parent.is_symlink():
        raise OSError(f"Refusing to use symlinked directory: {parent}")

    if candidate.exists():
        if candidate.is_symlink():
            raise OSError(f"Refusing to use symlinked file: {candidate}")
        if not candidate.is_file():
            raise OSError(f"Refusing to use non-file path: {candidate}")

    return candidate


MONOLITH_ROOT = _resolve_root()
if os.name == "nt":
    appdata = os.getenv("APPDATA")
    # Keep Windows default behavior stable when env root is not set.
    if "MONOLITH_ROOT" not in os.environ and appdata:
        MONOLITH_ROOT = Path(appdata).expanduser().resolve() / "Monolith"

SKILLS_DIR = Path(__file__).parent.parent / "skills"

CONFIG_DIR = MONOLITH_ROOT / "config"
ARCHIVE_DIR = MONOLITH_ROOT / "chats"
LOG_DIR = MONOLITH_ROOT / "logs"
ADDON_CONFIG_DIR = MONOLITH_ROOT / "addons" / "configs"
ADDON_MANIFEST = MONOLITH_ROOT / "addons" / "manifest.json"
NOTES_DIR = MONOLITH_ROOT / "notes"
ARTIFACTS_DIR = MONOLITH_ROOT / "artifacts"

for _dir in (MONOLITH_ROOT, CONFIG_DIR, ARCHIVE_DIR, LOG_DIR, ADDON_CONFIG_DIR, NOTES_DIR, ARTIFACTS_DIR):
    _dir.mkdir(parents=True, exist_ok=True)
