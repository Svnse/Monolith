"""
Deterministic environment resolution for loop prompt injection.
"""

from __future__ import annotations

import importlib.util
import platform
import sys
from pathlib import Path
from typing import Any

_STDLIB_PROBES = [
    "curses", "readline", "tkinter", "msvcrt",
    "resource", "fcntl", "termios", "pty",
]


def resolve_environment(workspace_root: str | Path | None = None) -> dict[str, Any]:
    ws = Path(str(workspace_root or ".")).expanduser().resolve()
    plat = platform.system().lower()

    stdlib: dict[str, bool] = {}
    for mod in _STDLIB_PROBES:
        stdlib[mod] = importlib.util.find_spec(mod) is not None

    tree: list[str] = []
    if ws.is_dir():
        for item in sorted(ws.iterdir(), key=lambda p: p.name.lower()):
            if item.name.startswith(".") or item.name == "__pycache__":
                continue
            suffix = "/" if item.is_dir() else ""
            tree.append(f"{item.name}{suffix}")

    return {
        "platform": plat,
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}",
        "stdlib": stdlib,
        "workspace_root": str(ws),
        "workspace_files": tree,
    }


def render_env_block(env: dict[str, Any]) -> str:
    stdlib = env.get("stdlib", {})
    available = [m for m, v in stdlib.items() if v]
    unavailable = [m for m, v in stdlib.items() if not v]
    files = env.get("workspace_files", [])
    lines = [
        f"platform: {env.get('platform', 'unknown')}",
        f"python: {env.get('python_version', 'unknown')}",
        f"stdlib available: {', '.join(available) if available else '(none probed)'}",
        f"stdlib unavailable: {', '.join(unavailable) if unavailable else '(none)'}",
        f"workspace: {env.get('workspace_root', '.')}",
        f"files: {', '.join(files[:20]) if files else '(empty)'}",
    ]
    return "\n".join(lines)

