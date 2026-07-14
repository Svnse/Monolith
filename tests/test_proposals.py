"""Tests for core.proposals + scratchpad propose_amendment / list_proposals ops."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from core import proposals


# ── fixtures ─────────────────────────────────────────────────────────


@pytest.fixture
def tmp_store(monkeypatch, tmp_path):
    """Redirect the proposals store to a temp file for each test."""
    store_path = tmp_path / "proposals.json"
    monkeypatch.setattr(proposals, "STORE_PATH", store_path)
    yield store_path


def _import_scratchpad_executor():
    """Load the dynamic skill executor by file path (matches runtime loader)."""
    spec_path = Path(__file__).parent.parent / "skills" / "scratchpad" / "executor.py"
    spec = importlib.util.spec_from_file_location("scratchpad_exec_proposals_test", spec_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def scratchpad(tmp_store, monkeypatch):
    """Scratchpad executor with STORE_PATH redirected and model_id pinned."""
    from core import llm_config
    monkeypatch.setattr(llm_config, "get_current_model_id", lambda: "test-model")
    # Also redirect proposals.STORE_PATH inside the freshly-loaded executor module.
    mod = _import_scratchpad_executor()
    monkeypatch.setattr(mod._proposals, "STORE_PATH", tmp_store)
    return mod


# ── shared proposal factory ────────────────────────────────────────────


# Post-M2: identity.md amendments target the Emergent region (Origin-0 is
# frozen). Defaults use Emergent so generic queue tests exercise a valid path.
def _make_proposal(
    target="identity.md",
    section="Emergent",
    current_text="(no emergent claims yet)",
    proposed_text="- I lean on adversarial verification before declaring done.",
    rationale="Y is now an observed failure mode just like X.",
    writer_model_id="test-model",
):
    return proposals.propose_amendment(
        target=target,
        section=section,
        current_text=current_text,
        proposed_text=proposed_text,
        rationale=rationale,
        writer_model_id=writer_model_id,
    )


# ── core module tests ─────────────────────────────────────────────────


def test_propose_assigns_incrementing_ids(tmp_store):
    r1 = _make_proposal(current_text="old A", proposed_text="new A")
    r2 = _make_proposal(current_text="old B", proposed_text="new B")
    r3 = _make_proposal(current_text="old C", proposed_text="new C")
    assert r1["id"] == 1
    assert r2["id"] == 2
    assert r3["id"] == 3


def test_propose_persists_all_fields(tmp_store):
    record = _make_proposal()
    # Read raw JSON to confirm persistence independent of list helper.
    data = json.loads(tmp_store.read_text(encoding="utf-8"))
    assert len(data["proposals"]) == 1
    p = data["proposals"][0]
    assert p["id"] == 1
    assert p["target"] == "identity.md"
    assert p["section"] == "Emergent"
    assert p["current_text"] == "(no emergent claims yet)"
    assert p["proposed_text"] == "- I lean on adversarial verification before declaring done."
    assert p["rationale"] == "Y is now an observed failure mode just like X."
    assert p["status"] == "pending"
    assert p["writer_model_id"] == "test-model"
    assert "created_at" in p


def test_propose_rejects_invalid_target(tmp_store):
    with pytest.raises(ValueError, match="target must be"):
        _make_proposal(target="other_file.md")


def test_propose_rejects_empty_fields(tmp_store):
    # Missing target after strip.
    with pytest.raises(ValueError, match="required and must be non-empty"):
        _make_proposal(target="   ")
    # Missing section after strip.
    with pytest.raises(ValueError, match="required and must be non-empty"):
        _make_proposal(section="")
    # Missing current_text after strip.
    with pytest.raises(ValueError, match="required and must be non-empty"):
        _make_proposal(current_text="")
    # Missing proposed_text after strip.
    with pytest.raises(ValueError, match="required and must be non-empty"):
        _make_proposal(proposed_text="   \n\t  ")
    # Missing rationale after strip.
    with pytest.raises(ValueError, match="required and must be non-empty"):
        _make_proposal(rationale="")


def test_propose_rejects_oversize_current_text(tmp_store):
    with pytest.raises(ValueError, match="current_text exceeds"):
        _make_proposal(current_text="x" * 2001)


def test_propose_rejects_oversize_proposed_text(tmp_store):
    with pytest.raises(ValueError, match="proposed_text exceeds"):
        _make_proposal(proposed_text="x" * 2001)


def test_propose_rejects_oversize_rationale(tmp_store):
    with pytest.raises(ValueError, match="rationale exceeds"):
        _make_proposal(rationale="r" * 801)


def test_propose_rejects_no_op_when_current_equals_proposed(tmp_store):
    with pytest.raises(ValueError, match="identical"):
        _make_proposal(current_text="same text", proposed_text="same text")


def test_list_returns_newest_first(tmp_store):
    _make_proposal(current_text="old A", proposed_text="new A", section="S1")
    _make_proposal(current_text="old B", proposed_text="new B", section="S2")
    _make_proposal(current_text="old C", proposed_text="new C", section="S3")
    items = proposals.list_proposals()
    assert items[0]["section"] == "S3"
    assert items[1]["section"] == "S2"
    assert items[2]["section"] == "S1"


def test_list_caps_at_default_limit(tmp_store):
    for i in range(25):
        _make_proposal(
            current_text=f"old {i}",
            proposed_text=f"new {i}",
            section=f"S{i}",
        )
    items = proposals.list_proposals()
    assert len(items) == 20


def test_proposals_cap_at_50(tmp_store):
    for i in range(55):
        _make_proposal(
            current_text=f"old {i}",
            proposed_text=f"new {i}",
            section=f"S{i}",
        )
    data = json.loads(tmp_store.read_text(encoding="utf-8"))
    assert len(data["proposals"]) == 50
    # The oldest 5 were dropped — the lowest id in the store should be > 5.
    ids = [p["id"] for p in data["proposals"]]
    assert min(ids) == 6  # ids 1–5 dropped, 6–55 remain


def test_propose_accepts_boundary_lengths(tmp_store):
    """Exactly at the caps must succeed."""
    record = proposals.propose_amendment(
        target="system.md",
        section="RESPONSE DISCIPLINE",
        current_text="A" * 2000,
        proposed_text="B" * 2000,
        rationale="R" * 800,
        writer_model_id="m",
    )
    assert record["id"] == 1
    assert len(record["current_text"]) == 2000
    assert len(record["proposed_text"]) == 2000
    assert len(record["rationale"]) == 800


def test_propose_strips_whitespace_before_validation(tmp_store):
    """Leading/trailing whitespace on required fields is stripped before checking."""
    record = proposals.propose_amendment(
        target="  identity.md  ",
        section="  Tone  ",
        current_text="  old  ",
        proposed_text="  new  ",
        rationale="  because  ",
        writer_model_id="m",
    )
    assert record["target"] == "identity.md"
    assert record["section"] == "Tone"
    assert record["current_text"] == "old"
    assert record["proposed_text"] == "new"
    assert record["rationale"] == "because"


def test_list_proposals_empty_store(tmp_store):
    assert proposals.list_proposals() == []


def test_id_survives_cap_drop(tmp_store):
    """After 50-cap drop, new entries continue incrementing from max+1."""
    for i in range(51):
        _make_proposal(
            current_text=f"old {i}",
            proposed_text=f"new {i}",
            section=f"S{i}",
        )
    # Now add one more — id should be 52, not 1.
    r = _make_proposal(current_text="final old", proposed_text="final new", section="Final")
    assert r["id"] == 52


# ── scratchpad op layer tests ─────────────────────────────────────────


def test_scratchpad_op_propose_amendment_round_trip(tmp_store, scratchpad):
    out = scratchpad.run(
        {
            "op": "propose_amendment",
            "target": "identity.md",
            "section": "Emergent",
            "current_text": "(no emergent claims yet)",
            "proposed_text": "- I lean on adversarial verification before declaring done.",
            "rationale": "Y is now an observed failure mode just like X.",
        },
        None,
    )
    assert "queued as proposal id=1" in out
    assert "identity.md" in out
    assert "Emergent" in out
    # Verify the store via list.
    items = proposals.list_proposals()
    assert len(items) == 1
    assert items[0]["writer_model_id"] == "test-model"


def test_scratchpad_op_list_proposals(tmp_store, scratchpad):
    # Write two proposals via op.
    scratchpad.run(
        {
            "op": "propose_amendment",
            "target": "identity.md",
            "section": "Alpha",
            "current_text": "old alpha",
            "proposed_text": "new alpha",
            "rationale": "Alpha needs updating.",
        },
        None,
    )
    scratchpad.run(
        {
            "op": "propose_amendment",
            "target": "system.md",
            "section": "Beta",
            "current_text": "old beta",
            "proposed_text": "new beta",
            "rationale": "Beta needs updating.",
        },
        None,
    )
    out = scratchpad.run({"op": "list_proposals"}, None)
    assert "[list_proposals: 2 queued]" in out
    # Newest first — Beta (id=2) should appear before Alpha (id=1).
    beta_pos = out.find("Beta")
    alpha_pos = out.find("Alpha")
    assert beta_pos < alpha_pos
    assert "pending" in out
    assert "system.md" in out
    assert "identity.md" in out


def test_scratchpad_op_propose_rejects_invalid_target(tmp_store, scratchpad):
    out = scratchpad.run(
        {
            "op": "propose_amendment",
            "target": "README.md",
            "section": "S",
            "current_text": "old",
            "proposed_text": "new",
            "rationale": "reason",
        },
        None,
    )
    assert "propose_amendment" in out
    assert "target must be" in out


def test_scratchpad_op_propose_rejects_missing_field(tmp_store, scratchpad):
    out = scratchpad.run(
        {
            "op": "propose_amendment",
            "target": "identity.md",
            # section missing
            "current_text": "old",
            "proposed_text": "new",
            "rationale": "reason",
        },
        None,
    )
    assert "propose_amendment" in out
    assert "required" in out.lower() or "section" in out.lower()


def test_scratchpad_op_list_proposals_empty(tmp_store, scratchpad):
    out = scratchpad.run({"op": "list_proposals"}, None)
    assert "[list_proposals: 0 queued]" in out


def test_scratchpad_op_propose_stamps_writer_model_id(tmp_store, scratchpad):
    scratchpad.run(
        {
            "op": "propose_amendment",
            "target": "system.md",
            "section": "MODEL ID CHECK",
            "current_text": "old content",
            "proposed_text": "new content",
            "rationale": "Verifying writer_model_id is stamped from get_current_model_id.",
        },
        None,
    )
    items = proposals.list_proposals()
    assert items[0]["writer_model_id"] == "test-model"
