"""Boundary audit — Bearing addon respects territorial constraints.

The Bearing V0 plan §2 establishes that Bearing must NOT touch:
  - core.turn_pipeline_events (Turn Pipeline event kinds)
  - core.turn_trace (Turn Pipeline canonical store)
  - core.acu_store (MonoBase epistemic state)
  - core.acatalepsy (MonoBase canonical log)

The classifier hook is the only exception, and it goes in the OTHER
direction (kernel → addon via DI handle, not addon → kernel).

This test scans every .py file under addons/system/bearing/ for any
forbidden import. Mechanical verification prevents drift.
"""
from __future__ import annotations

from pathlib import Path

FORBIDDEN_PREFIXES = (
    "core.turn_pipeline_events",
    "core.turn_trace",
    "core.acu_store",
    "core.acatalepsy",
    "core.pipeline_registry",
    "core.pipeline_policies",
    "monokernel",
    "engine.agent_server",
)

# emit_fault is allowed: the Bearing plan §7 explicitly routes escalation
# through core.fault_response.emit_fault. The other core/fault_response
# functions are also allowed transitively.
ALLOWED_CORE_IMPORTS = {
    "core.paths",
    "core.plane_loader",
    "core.fault_response",  # emit_fault path — explicit per plan §7
}


def _walk_addon_files() -> list[Path]:
    root = Path("addons/system/bearing")
    return sorted(root.rglob("*.py"))


def test_addon_files_exist() -> None:
    files = _walk_addon_files()
    assert len(files) > 0, "Expected addons/system/bearing/*.py to exist"


def test_no_forbidden_kernel_imports() -> None:
    violations: list[str] = []
    for path in _walk_addon_files():
        src = path.read_text(encoding="utf-8")
        for line in src.splitlines():
            stripped = line.strip()
            if not (stripped.startswith("import ") or stripped.startswith("from ")):
                continue
            # Skip relative imports
            if "from ." in stripped or stripped.startswith("from .."):
                continue
            for forbidden in FORBIDDEN_PREFIXES:
                if forbidden in stripped:
                    violations.append(f"{path}: {stripped}")
    assert violations == [], (
        "Bearing addon imports from forbidden kernel paths:\n  " + "\n  ".join(violations)
    )


def test_only_allowed_core_imports() -> None:
    """Bearing may import from a small, explicit set of core modules.
    Anything else from core/* should be flagged."""
    seen_core_imports: set[str] = set()
    for path in _walk_addon_files():
        src = path.read_text(encoding="utf-8")
        for line in src.splitlines():
            stripped = line.strip()
            if not stripped.startswith("from core."):
                continue
            # Extract the 'core.something' part
            parts = stripped.split()
            if len(parts) < 2:
                continue
            module = parts[1]
            seen_core_imports.add(module)
    extra = seen_core_imports - ALLOWED_CORE_IMPORTS
    assert extra == set(), (
        f"Bearing addon imports unexpected core modules: {sorted(extra)}.\n"
        f"Allowed: {sorted(ALLOWED_CORE_IMPORTS)}.\n"
        f"If a new core import is intentional, update ALLOWED_CORE_IMPORTS "
        f"in this test AND document in the Bearing plan §9."
    )


def test_no_bearing_sqlite_files() -> None:
    """Bearing must persist only to bearing.json + bearing.audit.jsonl —
    no .sqlite, .db, or .sqlite3 references."""
    for path in _walk_addon_files():
        src = path.read_text(encoding="utf-8")
        for needle in (".sqlite", ".db'", ".db\""):
            assert needle not in src.lower() or "sqlite3" in src.lower() and "import" not in src, (
                f"{path} appears to reference a SQLite file via {needle!r}"
            )


def test_bearing_module_count() -> None:
    """Plan §3 caps V0 at the listed files. Mechanically verify no unexpected
    modules snuck in. tolerant_extract.py added 2026-06-02 (intentional): a
    parsing-only recoverer for <bearing_update> envelopes the bound model emits
    tangled inside <think>; it does not touch bearing semantics. staleness.py
    added 2026-06-06 (intentional): the detector-agnostic channel-staleness
    closure spine (pure state machine, no I/O); see
    docs/superpowers/specs/2026-06-06-bearing-staleness-loop-design.md.
    frame_observe.py added 2026-06-14 (intentional): observable_frame_fastpath_v0
    — OBSERVE-only ledger of <frame>/<bearing_update> emission (pure detection +
    safe JSONL append to CONFIG_DIR/frame.ledger.jsonl, no bearing mutation).
    The later drift, frame-selection/fidelity, and CorrectionCard modules are
    intentional Bearing extensions.  The import-boundary tests above still
    mechanically enforce their territorial constraints."""
    files = {p.name for p in _walk_addon_files()}
    expected = {
        "__init__.py",
        "schema.py",
        "store.py",
        "compiler.py",
        "structural_verifier.py",
        "grounding_verifier.py",
        "audit.py",
        "provider.py",
        "plane.py",
        "kill_switch.py",
        "updater.py",
        "tolerant_extract.py",
        "staleness.py",
        "frame_observe.py",
        "drift.py",
        "drift_observe.py",
        "frame_shift.py",
        "monoframe.py",
        "stateless_reframe.py",
        "frame_selection.py",
        "frame_fidelity.py",
        "frame_fidelity_jobs.py",
        "correction_card.py",
        "correction_runner.py",
        "correction_store.py",
        "correction_synthesis.py",
    }
    extra = files - expected
    missing = expected - files
    assert missing == set(), f"Missing expected Bearing files: {missing}"
    assert extra == set(), (
        f"Unexpected files under addons/system/bearing/: {extra}. "
        f"Plan §3 caps V0 file count. If adding a new file is intentional, "
        f"update the plan + this test."
    )


def test_no_promotion_file_present() -> None:
    """Plan §3: promotion.py is V1, explicitly excluded from V0."""
    assert not Path("addons/system/bearing/promotion.py").exists()


def test_no_peer_models_file_present() -> None:
    """Plan §3: peer_models.py is V5, explicitly excluded from V0."""
    assert not Path("addons/system/bearing/peer_models.py").exists()


def test_no_cost_surface_file_present() -> None:
    """Plan §3: cost_surface.py is V2, explicitly excluded from V0."""
    assert not Path("addons/system/bearing/cost_surface.py").exists()
