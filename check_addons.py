from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ui.addons.builtin import build_builtin_registry


def main() -> int:
    registry = build_builtin_registry()
    for spec in registry.all():
        print(f"ID: {spec.id}, Title: {spec.title}, Kind: {spec.kind}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
