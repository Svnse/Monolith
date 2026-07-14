"""Bearing plane registration — visibility surface for subsystem_map.

Bearing isn't a scaffold-content plane like effort/conversation/reasoning/
linguency. It carries no mode content; it's an on/off substrate. But
registering it as a PlaneConfig makes it appear in subsystem_map and
keeps the runtime's introspection (scratchpad op=introspect) able to
enumerate Bearing alongside the scaffold planes.

Both modes are silent — PlaneLoader.load_mode_content always returns
None for "on" and "off" because Bearing has no mode .md files. The
real injection runs through `compiler.bearing_interceptor`, NOT through
PlaneLoader.interceptor.

If V1+ ever adds Bearing scaffold variants (e.g. an "ambient" mode that
narrates Bearing differently), the silent_modes set can shrink to just
{"off"} and scaffold .md files can land in scaffolds_dir.
"""
from __future__ import annotations

from pathlib import Path

from core.plane_loader import PlaneConfig, PlaneLoader

from . import kill_switch

PLANE_NAME = "bearing"

# Dummy scaffolds_dir — PlaneLoader requires a path but no scaffold files
# need to exist there for V0. Both modes are silent, so load_mode_content
# always returns None regardless of dir contents.
_DUMMY_SCAFFOLDS_DIR = Path(__file__).parent / "_no_scaffolds"

PLANE_CONFIG = PlaneConfig(
    plane_name=PLANE_NAME,
    valid_modes=frozenset({"on", "off"}),
    default_mode="on",
    scaffolds_dir=_DUMMY_SCAFFOLDS_DIR,
    flag_env=kill_switch.FLAG_ENV,
    silent_modes=frozenset({"on", "off"}),
)

_loader = PlaneLoader(PLANE_CONFIG)


def valid_modes():
    return _loader.valid_modes()


def is_enabled() -> bool:
    """Convenience re-export so callers can ask the plane directly."""
    return kill_switch.is_enabled()
