"""Tests for the author_workshop_card L1 model tool.

TDD: these tests are written BEFORE the implementation.

Coverage:
- Governance: in L1_PRINCIPAL_TOOLS only; execute_author_workshop_card with L3 ctx → deny.
- Happy path: valid blueprint → file saved, registry lists it, success string.
- Fix-loop closes (proof): broken blueprint (wired inert loop) → error, NO file;
  then FIXED blueprint → saved; saved card runs via run_monoline_world with stub engine.
- Malformed blueprint: connection to missing block → blueprint-error string, no crash, no file.
- Collision refusal: existing id → refuses, no clobber.
- id sanitizer: genesis, empty/whitespace, path separators, .. → rejected.
- Per-turn cap: > _MAX_CARD_AUTHORS_PER_TURN → cap message.
"""
from __future__ import annotations

import json
import os
import types
from pathlib import Path

import pytest

from tests._monoline_requirement import requires_monoline


pytestmark = requires_monoline

# ---------------------------------------------------------------------------
# Helpers: blueprint payloads
# ---------------------------------------------------------------------------

def _valid_blueprint(name: str = "My Card") -> dict:
    """A sound port→text→port blueprint (no llm: deterministic, model-free run)."""
    return {
        "name": name,
        "blocks": [
            {"id": "in", "kind": "port",
             "config": {"direction": "in", "label": "request", "source": "user_input"}},
            {"id": "txt", "kind": "text",
             "config": {"content": "Hello {{request}}"}},
            {"id": "out", "kind": "port",
             "config": {"direction": "out", "label": "response", "source": "subgraph"}},
        ],
        "connections": [
            ["in.value", "txt.request"],
            ["txt.text", "out.value"],
        ],
    }


def _broken_blueprint(name: str = "Broken Card") -> dict:
    """A blueprint with a wired inert 'loop' block — validate_preset will reject it."""
    return {
        "name": name,
        "blocks": [
            {"id": "in", "kind": "port",
             "config": {"direction": "in", "label": "value", "source": "user_input"}},
            {"id": "loop1", "kind": "loop", "label": "My Loop"},
            {"id": "out", "kind": "port",
             "config": {"direction": "out", "label": "response", "source": "subgraph"}},
        ],
        "connections": [
            ["in.value", "loop1.input"],
            ["loop1.output", "out.value"],
        ],
    }


def _malformed_blueprint(name: str = "Bad Conn") -> dict:
    """A blueprint whose connection references a block that does not exist."""
    return {
        "name": name,
        "blocks": [
            {"id": "in", "kind": "port",
             "config": {"direction": "in", "label": "request", "source": "user_input"}},
            {"id": "out", "kind": "port",
             "config": {"direction": "out", "label": "response", "source": "subgraph"}},
        ],
        "connections": [
            ["in.value", "ghost.prompt"],  # 'ghost' does not exist
        ],
    }


# ---------------------------------------------------------------------------
# Stub factory: mimics the SimpleNamespace stub used in test_chat_run_workshop.py
# ---------------------------------------------------------------------------

def _make_stub(workflows_dir: Path) -> types.SimpleNamespace:
    """Minimal PageChat stub for _on_author_workshop_card tests."""
    return types.SimpleNamespace(
        _workflow_registry=types.SimpleNamespace(
            workflows_dir=workflows_dir,
            get=lambda n: None,
            list_workflows=lambda: [],
        ),
        _MAX_CARD_AUTHORS_PER_TURN=3,
        _card_author_turn_count=0,
    )


# ============================================================================
# 1.  Governance: L1_PRINCIPAL_TOOLS + execute_author_workshop_card deny below L1
# ============================================================================

def test_author_workshop_card_in_l1_only():
    from core.skill_runtime import (
        L1_PRINCIPAL_TOOLS, L2_WORKER_TOOLS, L3_LEAF_TOOLS,
    )
    assert "author_workshop_card" in L1_PRINCIPAL_TOOLS
    assert "author_workshop_card" not in L2_WORKER_TOOLS
    assert "author_workshop_card" not in L3_LEAF_TOOLS


def test_execute_author_workshop_card_deny_without_hook(tmp_path):
    """Below L1 (no on_author_workshop_card) → deny string."""
    from core.skill_runtime import (
        execute_author_workshop_card, ToolExecutionContext, L3_LEAF_TOOLS,
    )
    ctx = ToolExecutionContext(
        archive_dir=tmp_path,
        level=3,
        allowed_tools=L3_LEAF_TOOLS,
        on_author_workshop_card=None,
    )
    result = execute_author_workshop_card({"tool": "author_workshop_card", "name": "x"}, ctx)
    assert "only the principal" in result.lower() or "l1" in result.lower()


# ============================================================================
# 2.  Happy path: valid blueprint → file saved, registry lists it
# ============================================================================

def test_author_workshop_card_happy_path(tmp_path, monkeypatch):
    import engine.monoline_bridge as br
    from ui.pages.chat import PageChat
    from core.workflow_registry import WorkflowRegistry

    # skip the real sys.modules swap
    m = br.load_monoline()
    stub = _make_stub(tmp_path)

    result = PageChat._on_author_workshop_card(stub, {
        "name": "my-card",
        "blueprint": _valid_blueprint("My Card"),
    })

    assert "saved" in result.lower(), f"expected success; got: {result!r}"
    assert "my-card" in result

    saved_files = list(tmp_path.glob("*.monoline"))
    assert len(saved_files) == 1, f"expected 1 file; found {saved_files}"

    data = json.loads(saved_files[0].read_text(encoding="utf-8"))
    assert data["id"] == "my-card"

    # Registry should list the new card
    reg = WorkflowRegistry(workflows_dir=tmp_path)
    ids = {w.id for w in reg.list_workflows()}
    assert "my-card" in ids


# ============================================================================
# 3.  Fix-loop closure (the proof)
#     broken blueprint → error + NO file; then FIXED blueprint → saved + runs
# ============================================================================

def test_fix_loop_closes(tmp_path, monkeypatch):
    """Model-free proxy: broken→error→fixed→saved sequence, then the card runs."""
    import engine.monoline_bridge as br
    from ui.pages.chat import PageChat
    from core.workflow_registry import WorkflowRegistry, Workflow

    m = br.load_monoline()
    stub = _make_stub(tmp_path)

    # --- Step 1: send broken blueprint ---
    broken_result = PageChat._on_author_workshop_card(stub, {
        "name": "fix-loop-card",
        "blueprint": _broken_blueprint("Fix Loop Card"),
    })
    # Must be a plain string (flows to _queue_tool_followup)
    assert isinstance(broken_result, str), "result must be a string for the tool-loop"
    # Must contain validation error info
    assert "validation failed" in broken_result.lower() or "no executable handler" in broken_result.lower(), \
        f"expected inert-block error; got: {broken_result!r}"
    # Must NOT have written a file
    assert list(tmp_path.glob("*.monoline")) == [], "broken blueprint must not be saved"

    # --- Step 2: re-author with fixed blueprint ---
    fixed_result = PageChat._on_author_workshop_card(stub, {
        "name": "fix-loop-card",
        "blueprint": _valid_blueprint("Fix Loop Card"),
    })
    assert "saved" in fixed_result.lower(), f"expected success after fix; got: {fixed_result!r}"
    assert "fix-loop-card" in fixed_result

    saved_files = list(tmp_path.glob("*.monoline"))
    assert len(saved_files) == 1, "fixed blueprint must produce exactly one saved file"

    # --- Step 3: saved card appears in registry ---
    reg = WorkflowRegistry(workflows_dir=tmp_path)
    wf = reg.get("fix-loop-card")
    assert wf is not None, "registry must list the saved card"
    assert wf.kind == "monoline"

    # --- Step 4: saved card runs via run_monoline_world with stub engine ---
    import engine.monoline_bridge as bridge

    # Patch the native engine_call to echo — the card is port→text→port (no llm),
    # so this path won't even be reached; the run must succeed regardless.
    def _echo_engine(messages, cfg):
        for msg in reversed(messages):
            if str(msg.get("role", "")).lower() == "user":
                return f"echo:{msg.get('content', '')}"
        return "echo:"

    monkeypatch.setattr(m["engine"], "engine_call", _echo_engine)

    import core.turn_trace as tt
    monkeypatch.setattr(tt, "_flag_enabled", lambda: False)

    run = bridge.run_monoline_world(
        wf, user_input="hello",
        parent_turn_id="", spawn_budget=None,
        should_cancel=lambda: False, is_busy=lambda: False,
        on_step=None, should_stop=None,
    )
    # The text block outputs "Hello hello" (content="Hello {{request}}", input="hello")
    output = run.result.output if hasattr(run, "result") else str(run)
    assert output, f"expected non-empty output; run={run!r}"


# ============================================================================
# 4.  Malformed blueprint: connection to missing block → blueprint-error, no crash
# ============================================================================

def test_malformed_blueprint_no_crash(tmp_path):
    from ui.pages.chat import PageChat
    import engine.monoline_bridge as br
    br.load_monoline()
    stub = _make_stub(tmp_path)

    result = PageChat._on_author_workshop_card(stub, {
        "name": "bad-conn",
        "blueprint": _malformed_blueprint("Bad Conn"),
    })
    assert isinstance(result, str)
    assert "blueprint error" in result.lower(), f"expected blueprint error; got: {result!r}"
    assert list(tmp_path.glob("*.monoline")) == [], "malformed blueprint must not be saved"


# ============================================================================
# 5.  Collision refusal: existing id → refuses, no clobber
# ============================================================================

def test_collision_refused(tmp_path):
    from ui.pages.chat import PageChat
    import engine.monoline_bridge as br
    br.load_monoline()
    stub = _make_stub(tmp_path)

    # First author — must succeed
    r1 = PageChat._on_author_workshop_card(stub, {
        "name": "collide-card",
        "blueprint": _valid_blueprint("Collide Card"),
    })
    assert "saved" in r1.lower(), f"first author should succeed; got: {r1!r}"
    original = (tmp_path / "collide-card.monoline").read_text(encoding="utf-8")

    # Second author of SAME id — must refuse
    r2 = PageChat._on_author_workshop_card(stub, {
        "name": "collide-card",
        "blueprint": _valid_blueprint("Collide Card v2"),
    })
    assert "already exists" in r2.lower(), f"expected collision refusal; got: {r2!r}"
    # File must not be clobbered
    assert (tmp_path / "collide-card.monoline").read_text(encoding="utf-8") == original


# ============================================================================
# 6.  id sanitizer: reserved/dangerous ids → rejected
# ============================================================================

@pytest.mark.parametrize("bad_name,reason", [
    ("genesis", "reserved sentinel"),
    ("", "empty"),
    ("   ", "whitespace-only"),
    ("../etc/passwd", "path traversal"),
    ("a/b", "forward slash"),
    ("a\\b", "backslash"),
    ("a..b", "contains '..' — any double-dot sequence rejected"),
])
def test_id_sanitizer_rejects(tmp_path, bad_name, reason):
    from ui.pages.chat import PageChat
    import engine.monoline_bridge as br
    br.load_monoline()
    stub = _make_stub(tmp_path)

    result = PageChat._on_author_workshop_card(stub, {
        "name": bad_name,
        "blueprint": _valid_blueprint("Card"),
    })
    assert isinstance(result, str)
    # empty name case will hit a different error path; just verify no file was saved
    saved = list(tmp_path.glob("*.monoline"))
    assert saved == [], f"id sanitizer ({reason!r}): file must not be saved; got {saved}"


def test_id_sanitizer_allows_valid_name(tmp_path):
    from ui.pages.chat import PageChat
    import engine.monoline_bridge as br
    br.load_monoline()
    stub = _make_stub(tmp_path)

    result = PageChat._on_author_workshop_card(stub, {
        "name": "My Valid Card 123",
        "blueprint": _valid_blueprint("My Valid Card"),
    })
    assert "saved" in result.lower(), f"valid name should succeed; got: {result!r}"
    # id should be slugified
    assert "my-valid-card-123" in result


# ============================================================================
# 7.  Per-turn cap
# ============================================================================

def test_per_turn_cap(tmp_path):
    from ui.pages.chat import PageChat
    import engine.monoline_bridge as br
    br.load_monoline()

    # cap = 2 for this test
    stub = _make_stub(tmp_path)
    stub._MAX_CARD_AUTHORS_PER_TURN = 2
    stub._card_author_turn_count = 0

    # Author two distinct cards (cap=2, both should succeed)
    r1 = PageChat._on_author_workshop_card(stub, {
        "name": "card-one", "blueprint": _valid_blueprint("Card One"),
    })
    assert "saved" in r1.lower(), f"1st author should succeed; got: {r1!r}"

    r2 = PageChat._on_author_workshop_card(stub, {
        "name": "card-two", "blueprint": _valid_blueprint("Card Two"),
    })
    assert "saved" in r2.lower(), f"2nd author should succeed; got: {r2!r}"

    # Third call must hit the cap
    r3 = PageChat._on_author_workshop_card(stub, {
        "name": "card-three", "blueprint": _valid_blueprint("Card Three"),
    })
    assert "limit" in r3.lower() or "cap" in r3.lower(), \
        f"3rd call (cap=2) should be refused; got: {r3!r}"
    # Only 2 files should exist
    assert len(list(tmp_path.glob("*.monoline"))) == 2


# ============================================================================
# 8.  INV-#1: Genesis card unaffected; no import monolith
# ============================================================================

def test_genesis_unaffected(tmp_path):
    from core.workflow_registry import WorkflowRegistry, GENESIS_ID, GENESIS
    reg = WorkflowRegistry(workflows_dir=tmp_path)
    wfs = reg.list_workflows()
    genesis = next((w for w in wfs if w.id == GENESIS_ID), None)
    assert genesis is not None, "Genesis must always be present"
    assert genesis.kind == "native"


def test_no_import_monolith():
    """The implementation must not add 'import monolith' anywhere (INV-#0)."""
    import importlib.util
    # Just verify the module doesn't contain 'import monolith' at the top level
    import core.skill_runtime as sr_mod
    import ui.pages.chat as chat_mod
    for mod in (sr_mod, chat_mod):
        src = Path(mod.__file__).read_text(encoding="utf-8")
        assert "import monolith" not in src, \
            f"{mod.__file__}: 'import monolith' found (INV-#0 violation)"
