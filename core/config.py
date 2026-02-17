from __future__ import annotations

import os
from pathlib import Path


DEFAULT_WORKSPACE_ROOT = (
    Path(os.getenv("MONOLITH_WORKSPACE"))
    if os.getenv("MONOLITH_WORKSPACE")
    else Path.home() / "AppData" / "Roaming" / "Monolith" / "workspace"
).resolve()

DEFAULT_WORKSPACE_ROOT.mkdir(parents=True, exist_ok=True)
