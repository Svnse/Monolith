from __future__ import annotations
import json
import types
from pathlib import Path
import pytest
import core.turn_trace as tt
import core.subagent as _sa
from engine import monoline_bridge as br
from core.workflow_registry import Workflow
from tests._monoline_requirement import requires_monoline


pytestmark = requires_monoline


# ---------------------------------------------------------------------------
# Shared test helpers (Task 1 setup; used by Tasks 2+)
# ---------------------------------------------------------------------------

def _world(tmp_path, *, provider=None):
    """A 1-llm-block echo world. provider=None -> default 'local'; 'monolith' -> atom path."""
    llm = {"id": "assistant", "kind": "llm"}
    if provider:
        llm["config"] = {"provider": provider}
    p = Path(tmp_path) / "echo.monoline"
    p.write_text(json.dumps({
        "name": "Echo",  # no schema_version -> lenient build path (world actually runs)
        "blocks": [
            {"id": "input", "kind": "port", "config": {"direction": "in", "label": "request", "source": "user_input"}},
            llm,
            {"id": "output", "kind": "port", "config": {"direction": "out", "label": "response", "source": "subgraph"}}],
        "connections": [["input.value", "assistant.prompt"], ["assistant.response", "output.value"]],
    }), encoding="utf-8")
    return Workflow(id="echo", name="Echo", description="", kind="monoline", source_path=p)


def _run(tmp_path, world, **kw):
    br.run_monoline_world(world, user_input=kw.pop("user_input", "hi"), parent_turn_id="",
                          spawn_budget=None, should_cancel=lambda: False, is_busy=lambda: False,
                          on_step=None, should_stop=None, **kw)


# ---------------------------------------------------------------------------

def test_projection_flag(monkeypatch):
    monkeypatch.delenv("MONOLINE_PROJECTION", raising=False)
    assert br._projection_enabled() is False
    monkeypatch.setenv("MONOLINE_PROJECTION", "1")
    assert br._projection_enabled() is True
    monkeypatch.setenv("MONOLINE_PROJECTION", "off")
    assert br._projection_enabled() is False


# ---------------------------------------------------------------------------
# Task 2 — provider-aware single frame
# ---------------------------------------------------------------------------

def test_native_block_one_frame(tmp_path, monkeypatch):
    monkeypatch.setattr(tt, "_flag_enabled", lambda: True)
    tt.set_db_path(Path(tmp_path) / "tt.sqlite3")
    m = br.load_monoline()
    monkeypatch.setattr(m["engine"], "engine_call", lambda msgs, cfg: "OUT")
    try:
        _run(tmp_path, _world(tmp_path))  # default provider -> local/native
        root = tt.latest_governance_root()
        kids = [k for k in tt.list_child_frames(root)
                if (getattr(k, "metadata", {}) or {}).get("kind") == "monoline_block"]
        assert len(kids) == 1  # native llm block: exactly one bridge frame
    finally:
        tt.set_db_path(None)


def test_monolith_block_not_double_framed(tmp_path, monkeypatch):
    monkeypatch.setattr(tt, "_flag_enabled", lambda: True)
    tt.set_db_path(Path(tmp_path) / "tt.sqlite3")
    monkeypatch.setattr(_sa, "run_subagent",
                        lambda messages, config, **kw: types.SimpleNamespace(text="OUT", halt_reason=None))
    m = br.load_monoline()
    try:
        _run(tmp_path, _world(tmp_path, provider="monolith"))
        root = tt.latest_governance_root()
        bridge_frames = [k for k in tt.list_child_frames(root)
                         if (getattr(k, "metadata", {}) or {}).get("kind") == "monoline_block"]
        assert bridge_frames == []  # monolith block: bridge skips its frame (atom owns it)
    finally:
        tt.set_db_path(None)


# ---------------------------------------------------------------------------
# Task 3 — overlay discriminators + payload-first restructure
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Task 4 — fault detectors folded into payload
# ---------------------------------------------------------------------------

# Unbalanced <think> tag (open=1, close=0) → deterministically trips detect_think_leak.
# Balanced tags (e.g. <think>...</think>) would NOT fire — the detector catches IMBALANCE only.
LEAK = "<think>hidden visible answer"


def test_detectors_fold_into_payload(tmp_path, monkeypatch):
    monkeypatch.setattr(tt, "_flag_enabled", lambda: True)
    monkeypatch.setenv("MONOLINE_PROJECTION", "1")
    tt.set_db_path(Path(tmp_path) / "tt.sqlite3")
    m = br.load_monoline()
    monkeypatch.setattr(m["engine"], "engine_call", lambda msgs, cfg: LEAK)
    try:
        _run(tmp_path, _world(tmp_path))
        root = tt.latest_governance_root()
        ev = [e for e in tt.list_governance_events(root)
              if getattr(e, "event_kind", "") == "monoline_block"
              and (getattr(e, "payload", {}) or {}).get("step_kind") == "call_llm"][0]
        assert ev.payload.get("detectors"), "expected detector findings in payload"
        # CRITICAL no-leak invariant on the ACTIVE-projection path: even with a real
        # think_leak finding (a KNOWN_KIND), nothing reaches Genesis's fault readers —
        # an emit_fault regression would write a fault_kind-NOT-NULL row and trip these.
        import core.fault_response as fr
        assert fr.read_recent(limit=50) == []
        assert tt.list_faults_since("1970-01-01T00:00:00+00:00") == []
    finally:
        tt.set_db_path(None)


def test_detectors_off_when_flag_off(tmp_path, monkeypatch):
    monkeypatch.setattr(tt, "_flag_enabled", lambda: True)
    monkeypatch.delenv("MONOLINE_PROJECTION", raising=False)
    tt.set_db_path(Path(tmp_path) / "tt.sqlite3")
    m = br.load_monoline()
    monkeypatch.setattr(m["engine"], "engine_call", lambda msgs, cfg: LEAK)
    try:
        _run(tmp_path, _world(tmp_path))
        root = tt.latest_governance_root()
        for e in tt.list_governance_events(root):
            assert "detectors" not in (getattr(e, "payload", {}) or {})
    finally:
        tt.set_db_path(None)


# ---------------------------------------------------------------------------

def test_verdict_folds_and_genesis_verdict_read_untouched(tmp_path, monkeypatch):
    monkeypatch.setattr(tt, "_flag_enabled", lambda: True)
    monkeypatch.setenv("MONOLINE_PROJECTION", "1")
    tt.set_db_path(Path(tmp_path) / "tt.sqlite3")
    before = tt.get_last_verification_result()  # None on a fresh tmp DB
    m = br.load_monoline()
    monkeypatch.setattr(m["engine"], "engine_call", lambda msgs, cfg: "a plain answer")
    try:
        _run(tmp_path, _world(tmp_path))
        root = tt.latest_governance_root()
        ev = [e for e in tt.list_governance_events(root)
              if getattr(e, "event_kind", "") == "monoline_block"
              and (getattr(e, "payload", {}) or {}).get("step_kind") == "call_llm"][0]
        assert "verdict" in ev.payload and ev.payload["verdict"].get("verdict")
        # Genesis's global verdict read is unaffected (we never wrote VerifierVerdictEvent):
        assert tt.get_last_verification_result() == before
    finally:
        tt.set_db_path(None)


# ---------------------------------------------------------------------------
# Task 6 — Workshop renders verdict + detectors
# ---------------------------------------------------------------------------

import os as _os
_os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def test_block_rows_show_verdict_and_detectors():
    """The unified RunView must surface a block's verdict + detector kinds on expand. (The
    capability moved here from the old WorkshopPane tree -- the bridge folds the same payload
    verdict/detectors onto the BlockFinished event.)"""
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication
    _app = QApplication.instance() or QApplication([])

    from core.run_model import (BlockFinished, RunBlockSpec, RunModelBuilder, RunStarted)
    from ui.components.run_view import RunView
    b = RunModelBuilder()
    b.apply(RunStarted(run_id="r1", flow_id="f", name="F", user_input="hi",
                       graph=[RunBlockSpec(id="assistant", label="assistant", kind="llm")],
                       wires=[]))
    b.apply(BlockFinished(
        run_id="r1", block_id="assistant", label="assistant", kind="llm",
        outputs={"response": "x"}, started_at=1.0, completed_at=2.0, status="done",
        verdict={"verdict": "warn", "reasons": ["x"]},
        detectors=[{"kind": "think_leak", "name": "d", "evidence": "<think>"}]))
    view = RunView(b.model)
    try:
        view.show()
        view.expand_row("assistant")
        blob = view.row_detail_text("assistant")
        assert "warn" in blob, f"verdict not surfaced: {blob!r}"
        assert "think_leak" in blob, f"detector kind not surfaced: {blob!r}"
    finally:
        view.deleteLater()


# ---------------------------------------------------------------------------

def test_overlay_discriminators_and_genesis_isolation(tmp_path, monkeypatch):
    monkeypatch.setattr(tt, "_flag_enabled", lambda: True)
    tt.set_db_path(Path(tmp_path) / "tt.sqlite3")
    import core.fault_response as fr
    m = br.load_monoline()
    def _boom(msgs, cfg):
        raise RuntimeError("kaboom")
    monkeypatch.setattr(m["engine"], "engine_call", _boom)  # force the llm block to error
    try:
        _run(tmp_path, _world(tmp_path))
        root = tt.latest_governance_root()
        events = [e for e in tt.list_governance_events(root)
                  if getattr(e, "event_kind", "") == "monoline_block"]
        assert events, "expected a monoline_block overlay row"
        for e in events:
            assert getattr(e, "fault_kind", "SENTINEL") is None        # never a real fault_kind
            assert getattr(e, "source_kind", "") == "kernel"            # not 'producer' (observation is authority_tier vocab)
        errs = [e for e in events if (getattr(e, "payload", {}) or {}).get("ok") is False]
        assert errs and (errs[0].payload.get("error"))                 # error lives in payload
        assert fr.read_recent(limit=50) == []                          # Genesis fault reader sees nothing
        assert tt.list_faults_since("1970-01-01T00:00:00+00:00") == []
    finally:
        tt.set_db_path(None)


# ---------------------------------------------------------------------------
# Task 7 — integration guards: flag-off zero-rows + Genesis reader isolation
# ---------------------------------------------------------------------------

def test_flag_off_writes_zero_rows(tmp_path, monkeypatch):
    # MONOLITH_TURN_TRACE_V1 off -> record_* no-op -> zero rows, regardless of MONOLINE_PROJECTION
    monkeypatch.setattr(tt, "_flag_enabled", lambda: False)
    monkeypatch.setenv("MONOLINE_PROJECTION", "1")
    tt.set_db_path(Path(tmp_path) / "tt.sqlite3")
    m = br.load_monoline()
    monkeypatch.setattr(m["engine"], "engine_call", lambda msgs, cfg: "OUT")
    try:
        _run(tmp_path, _world(tmp_path))
        assert tt.latest_governance_root() in (None, "")   # nothing recorded
    finally:
        tt.set_db_path(None)


def test_full_run_genesis_readers_isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(tt, "_flag_enabled", lambda: True)
    monkeypatch.setenv("MONOLINE_PROJECTION", "1")
    tt.set_db_path(Path(tmp_path) / "tt.sqlite3")
    import core.fault_response as fr
    from core import stats_store
    m = br.load_monoline()
    monkeypatch.setattr(m["engine"], "engine_call", lambda msgs, cfg: "<think>x visible")
    try:
        _run(tmp_path, _world(tmp_path))
        root = tt.latest_governance_root()
        assert any(getattr(e, "event_kind", "") == "monoline_block"
                   for e in tt.list_governance_events(root))     # Monoline rows EXIST
        # ...but every Genesis-facing reader is empty:
        assert fr.read_recent(limit=50) == []
        assert tt.list_faults_since("1970-01-01T00:00:00+00:00") == []
        assert tt.get_last_verification_result() is None
        # stats get_tool_usage must NOT list the phantom 'monoline_bridge' tool (locks producer->kernel):
        usage = stats_store.StatsStore().get_tool_usage("all")
        assert all(u.get("tool") != "monoline_bridge" for u in usage)
    finally:
        tt.set_db_path(None)
