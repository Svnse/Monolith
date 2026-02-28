from __future__ import annotations

import os
from pathlib import Path

from core.paths import MONOLITH_ROOT

DEFAULT_WORKSPACE_ROOT = (
    Path(os.getenv("MONOLITH_WORKSPACE"))
    if os.getenv("MONOLITH_WORKSPACE")
    else MONOLITH_ROOT / "workspace"
).resolve()

DEFAULT_WORKSPACE_ROOT.mkdir(parents=True, exist_ok=True)
