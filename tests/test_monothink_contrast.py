from __future__ import annotations

import json

import pytest

from core import monothink as mt
from core import monothink_contrast as mc
from core import turn_trace as tt


@pytest.fixture
def trace_db(tmp_path, monkeypatch):
    db = tmp_path / "turn_trace.sqlite3"
    tt.set_db_path(db)
    monkeypatch.setenv("MONOLITH_TURN_TRACE_V1", "1")
    yield db
    tt.set_db_path(None)


def _case(case_id: str, *, role: str, tag: str = "premise_unchecked") -> mc.ContrastCase:
    return mc.ContrastCase(
        case_id=case_id,
        tag=tag,
        turn_id=f"T-{case_id}",
        rating=30,
        created_ts=1.0,
        base_scaffold_sha="base",
        minimized_input="original user ask",
        failed_trace=f"failed {case_id}",
        corrected_shape=f"corrected {case_id}",
        separation_predicate=f"SECRET_PREDICATE_{case_id}",
        role=role,
    )


def test_store_assigns_both_roles_and_roundtrips(tmp_path) -> None:
    store = mc.ContrastStore(tmp_path / "cases.jsonl")

    first = store.record_from_rating(
        turn_id="T1",
        tag="premise_unchecked",
        rating=35,
        base_scaffold_sha="sha1",
        minimized_input="ask",
        failed_trace="bad trace",
        tag_gloss="gloss",
        build_case=lambda *_: mc.BuiltCase("correct move", "did it correct?"),
    )
    second = store.record_from_rating(
        turn_id="T2",
        tag="premise_unchecked",
        rating=35,
        base_scaffold_sha="sha1",
        minimized_input="ask",
        failed_trace="bad trace 2",
        tag_gloss="gloss",
        build_case=lambda *_: mc.BuiltCase("correct move 2", "did it correct 2?"),
    )

    assert {first.role, second.role} == {"canary", "prompt"}
    loaded = store.all()
    assert [c.case_id for c in loaded] == [first.case_id, second.case_id]


def test_prompt_render_excludes_canaries_and_predicates(tmp_path) -> None:
    store = mc.ContrastStore(tmp_path / "cases.jsonl")
    store.append(_case("prompt", role="prompt"))
    store.append(_case("canary", role="canary"))

    block = mc.render_contrast_block(mc.select_for_prompt(store, "premise_unchecked"))

    assert "corrected prompt" in block
    assert "corrected canary" not in block
    assert "SECRET_PREDICATE" not in block


def test_evaluate_separation_scores_behavior_not_scaffold_text(tmp_path) -> None:
    store = mc.ContrastStore(tmp_path / "cases.jsonl")
    store.append(_case("canary", role="canary"))

    def run_scaffold(scaffold: str, _inp: str) -> str:
        return "corrected behavior present" if "new" in scaffold else "old failure"

    verdict = mc.evaluate_separation(
        tag="premise_unchecked",
        old_scaffold="old scaffold",
        candidate_scaffold="new scaffold",
        store=store,
        run_scaffold=run_scaffold,
        judge=lambda _predicate, trace: "corrected behavior" in trace,
        k=1,
    )

    assert verdict.admit is True
    assert verdict.target_gain == 1.0
    assert verdict.per_case["canary"]["predicate_sha"]


def test_evaluate_separation_no_canary_rejects_shadow_only(tmp_path) -> None:
    store = mc.ContrastStore(tmp_path / "cases.jsonl")
    store.append(_case("prompt", role="prompt"))

    verdict = mc.evaluate_separation(
        tag="premise_unchecked",
        old_scaffold="old",
        candidate_scaffold="new",
        store=store,
        run_scaffold=lambda *_: "",
        judge=lambda *_: False,
        k=1,
    )

    assert verdict.admit is False
    assert verdict.reason == "no-canary-case-for-tag"


def test_record_outcome_forwards_replay_input(trace_db, monkeypatch) -> None:
    captured = {}

    def fake_hook(turn_id, rating_value, failure_tags, think_block=None, replay_input=None, rater_note=None):
        captured.update(
            turn_id=turn_id,
            rating=rating_value,
            tags=failure_tags,
            think=think_block,
            replay=replay_input,
            note=rater_note,
        )

    monkeypatch.setattr("core.monothink.maybe_evolve_after_rating", fake_hook)
    tt.record_outcome(tt.OutcomeTraceRecord(
        turn_id="T-replay",
        recorded_at="2026-06-26T00:00:00Z",
        kind="rating",
        rating_value=42,
        reason="r",
        metadata={
            "failure_tags": ["premise_unchecked"],
            "think_block": "failed trace",
            "replay_input": "user ask",
        },
    ))

    assert captured["replay"] == "user ask"
    assert captured["think"] == "failed trace"


def test_apply_path_logs_shadow_summary_without_blocking(tmp_path, monkeypatch) -> None:
    scaffold = tmp_path / "monothink.md"
    journal = tmp_path / "monothink.journal.jsonl"
    scaffold.write_text(
        "# MonoThink - seed\n\n"
        "## Audit\n\n"
        "1. Think.\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(mt, "_SCAFFOLD_PATH", scaffold)
    monkeypatch.setattr(mt, "_JOURNAL_PATH", journal)
    monkeypatch.setattr(mt, "_lookup_turn_monothink_active", lambda _turn_id: True)
    monkeypatch.setenv("MONOLITH_MONOTHINK_EVOLVE_V1", "1")
    monkeypatch.setenv("MONOLITH_MONOTHINK_ASYNC", "0")
    monkeypatch.setenv("MONOLITH_MONOTHINK_CONTRAST_V1", "1")
    monkeypatch.setattr(mt, "_contrast_prompt_block", lambda _tag: (None, ""))
    monkeypatch.setattr(mt, "_record_contrast_case", lambda **_kwargs: None)
    monkeypatch.setattr(mt, "_contrast_shadow_summary", lambda **_kwargs: {
        "contrast_shadow": True,
        "contrast_would_admit": False,
        "contrast_target_gain": 0.0,
        "contrast_case_count": 0,
        "contrast_reason": "no-canary-case-for-tag",
    })
    monkeypatch.setattr(mt, "_call_llm", lambda _prompt: (
        "=== PATCH ===\n"
        "PRIMARY_FAILURE_TAG: premise_unchecked\n"
        "PRIMARY_FAILURE: premise used unchecked\n"
        "TARGET_SECTION: Audit\n"
        "PROPOSED_SCOPE: line\n"
        "DEFERRED_CONCERNS:\n"
        "- none\n"
        "PATCH_MODE: replace_lines\n"
        "TARGET_LINES: 5-5\n"
        "BEFORE:\n"
        "1. Think.\n"
        "AFTER:\n"
        "1. Think carefully.\n"
    ))

    entry = mt.maybe_evolve_after_rating(
        "T-shadow",
        33,
        ["premise_unchecked"],
        think_block="failed trace",
        replay_input="user ask",
    )

    assert entry["applied"] is True
    assert entry["contrast_shadow"] is True
    assert entry["contrast_reason"] == "no-canary-case-for-tag"
    assert "Think carefully" in scaffold.read_text(encoding="utf-8")
    rows = [
        json.loads(line)
        for line in journal.read_text(encoding="utf-8").splitlines()
        if line.strip() and '"kind": "bootstrap"' not in line
    ]
    assert rows[-1]["contrast_shadow"] is True
