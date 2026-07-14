from __future__ import annotations

import json
import sys
from pathlib import Path

from core.workflow_registry import GENESIS_ID, Workflow, WorkflowRegistry


def _write_blueprint(d: Path, wid: str, name: str, mtime: float) -> Path:
    p = d / f"{wid}.monoline"
    p.write_text(json.dumps({
        "schema_version": 1, "id": wid, "name": name,
        "description": f"desc {name}", "modified_at": mtime,
        "blocks": [], "connections": [], "composites": [],
    }), encoding="utf-8")
    import os
    os.utime(p, (mtime, mtime))
    return p


def test_empty_dir_lists_only_genesis(tmp_path):
    reg = WorkflowRegistry(workflows_dir=tmp_path / "absent")
    items = reg.list_workflows()
    assert len(items) == 1
    assert items[0].id == GENESIS_ID
    assert items[0].kind == "native"
    assert items[0].source_path is None


def test_lists_blueprints_genesis_first_newest_first(tmp_path):
    _write_blueprint(tmp_path, "alpha", "Alpha", 1000.0)
    _write_blueprint(tmp_path, "beta", "Beta", 2000.0)
    reg = WorkflowRegistry(workflows_dir=tmp_path)
    items = reg.list_workflows()
    assert items[0].id == GENESIS_ID
    assert [w.id for w in items[1:]] == ["beta", "alpha"]
    assert items[1].kind == "monoline"
    assert items[1].name == "Beta"
    assert items[1].source_path is not None


def test_genesis_id_never_shadowed_by_disk_file(tmp_path):
    _write_blueprint(tmp_path, GENESIS_ID, "Fake Genesis", 9999.0)
    reg = WorkflowRegistry(workflows_dir=tmp_path)
    ids = [w.id for w in reg.list_workflows()]
    assert ids.count(GENESIS_ID) == 1
    assert reg.get(GENESIS_ID).kind == "native"


def test_corrupt_file_tolerated(tmp_path):
    (tmp_path / "bad.monoline").write_text("{not json", encoding="utf-8")
    _write_blueprint(tmp_path, "good", "Good", 1000.0)
    reg = WorkflowRegistry(workflows_dir=tmp_path)
    ids = [w.id for w in reg.list_workflows()]
    assert "good" in ids and len(ids) == 2  # genesis + good


def test_get_collapses_unset_and_unknown_to_genesis(tmp_path):
    reg = WorkflowRegistry(workflows_dir=tmp_path)
    assert reg.get("").id == GENESIS_ID
    assert reg.get(GENESIS_ID).id == GENESIS_ID
    assert reg.get("nonexistent") is None  # caller falls back to Genesis


def test_import_does_not_pull_monoline():
    # INV-#0 arm (a): importing the registry must not load any Monoline module.
    # MUST be checked in a FRESH interpreter: the in-process sys.modules is shared across
    # the whole pytest session, and once any Monoline run happens the bridge's one-time swap
    # leaves core.monoline_* RESIDENT by design (so lazy `from core.monoline_X import ...`
    # resolves). The real invariant is "the registry's own import graph is Monoline-free",
    # which a clean subprocess proves precisely; an ambient scan is order-dependent.
    import subprocess
    code = (
        "import sys;"
        "import core.workflow_registry;"
        "leaked=[m for m in sys.modules if m.startswith('monoline') "
        "or m.startswith('core.monoline_')];"
        "print('LEAK' if leaked else 'CLEAN', leaked)"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert out.stdout.startswith("CLEAN"), out.stdout
