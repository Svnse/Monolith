"""First-run Workshop seeding: copy bundled starter .monoline flows into the shared worlds dir
IF it is empty. Pure file copy -- imports NO Monoline module (INV-#0). Best-effort: never raises
into startup."""
from __future__ import annotations

import shutil
from pathlib import Path

from core.paths import MONOLITH_ROOT

BUNDLED_SEEDS_DIR = Path(__file__).resolve().parent.parent / "assets" / "workshop_seeds"
DEFAULT_WORLDS_DIR = MONOLITH_ROOT / "monoline" / "worlds"


def seed_workshop_flows(*, worlds_dir: Path = DEFAULT_WORLDS_DIR,
                        seeds_dir: Path = BUNDLED_SEEDS_DIR) -> int:
    """Copy bundled *.monoline seeds into worlds_dir IF it currently has none. Returns the count
    copied (0 if the dir already has flows, or on any error). Never overwrites user files."""
    try:
        worlds_dir = Path(worlds_dir)
        worlds_dir.mkdir(parents=True, exist_ok=True)
        if any(worlds_dir.glob("*.monoline")):
            return 0  # user already has flows -> do nothing
        if not Path(seeds_dir).exists():
            return 0
        n = 0
        for seed in sorted(Path(seeds_dir).glob("*.monoline")):
            dest = worlds_dir / seed.name
            if not dest.exists():
                shutil.copy2(seed, dest)
                n += 1
        return n
    except Exception:
        return 0  # best-effort; must never break startup
