from __future__ import annotations

import json

import pytest

from core import monothink as mt
from core import monothink_deferred_ledger as ledger


@pytest.fixture(autouse=True)
def _sync(monkeypatch):
    monkeypatch.setenv("MONOLITH_MONOTHINK_ASYNC", "0")
    monkeypatch.setenv("MONOLITH_MONOTHINK_EVOLVE_V1", "1")


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    scaffold = tmp_path / "monothink.md"
    journal = tmp_path / "monothink.journal.jsonl"
    ledger_path = tmp_path / "deferred_ledger.jsonl"
    monitor_path = tmp_path / "monitor.jsonl"
    scaffold.write_text("# MonoThink - seed\n\n## Audit\n\n1. Think.\n", encoding="utf-8")
    monkeypatch.setattr(mt, "_SCAFFOLD_PATH", scaffold)
    monkeypatch.setattr(mt, "_JOURNAL_PATH", journal)
    monkeypatch.setattr(mt, "_lookup_turn_monothink_active", lambda _tid: True)
    monkeypatch.setattr(ledger, "LEDGER_PATH", ledger_path)
    monkeypatch.setattr(ledger, "MONITOR_PATH", monitor_path)
    return {
        "scaffold": scaffold,
        "journal": journal,
        "ledger": ledger_path,
        "monitor": monitor_path,
    }


def _decision(
    *,
    mutation: str = "reserve",
    evidence: str = "single_instance",
    signature_status: str = "provisional",
    fix_kind: str = "enforcement_gap",
    already: str = "yes(Audit)",
    patch: bool = False,
) -> str:
    base = (
        "=== DECISION ===\n"
        "PRIMARY_FAILURE_TAG: premise_unchecked\n"
        "PRIMARY_FAILURE: premise was not checked\n"
        "FAILURE_SIGNATURE: comparative-superiority-without-evidence\n"
        f"SIGNATURE_STATUS: {signature_status}\n"
        "LEDGER_LOOKUP: (premise_unchecked, Audit, comparative-superiority-without-evidence) -> count=1\n"
        f"EVIDENCE_CLASS: {evidence}\n"
        f"ALREADY_COVERED: {already}\n"
        f"FIX_KIND: {fix_kind}\n"
        f"MUTATION_DECISION: {mutation}\n"
        "DECISION_REASON: rule exists or evidence is insufficient\n"
    )
    if not patch:
        return base + (
            "PATCH_MODE: no_change\n"
            "DEFERRED_CONCERN: enforcement_gap at Audit / comparative-superiority-without-evidence\n"
        )
    return base + (
        "PATCH_MODE: replace_lines\n"
        "\n=== PATCH ===\n"
        "TARGET_SECTION: Audit\n"
        "TARGET_LINES: 5-5\n"
        "PROPOSED_SCOPE: line\n"
        "STRUCTURAL_LOCATION: Audit rule failed to demand comparison.\n"
        "PREDICTED_EFFECT: future traces compare the premise before concluding.\n"
        "FALSIFIER: same signature recurs in Audit after this edit.\n"
        "APPLIED_AT_TURN: T-apply\n"
        "PATCH_MODE: replace_lines\n"
        "BEFORE:\n"
        "1. Think.\n"
        "AFTER:\n"
        "1. Think with a premise check.\n"
    )


def test_parse_decision_reserve_schema() -> None:
    parsed, err = mt._parse_evolution_response(_decision())
    assert err is None
    assert parsed["schema_version"] == "decision_v2"
    assert parsed["mutation_decision"] == "reserve"
    assert parsed["patch_mode"] == "no_change"
    assert parsed["failure_signature"] == "comparative-superiority-without-evidence"


def test_reserve_writes_ledger_without_scaffold_edit(isolated, monkeypatch) -> None:
    monkeypatch.setattr(mt, "_call_llm", lambda _prompt: _decision())

    entry = mt.maybe_evolve_after_rating(
        "T-reserve",
        40,
        ["premise_unchecked"],
        think_block="trace span",
        rater_note="this should reserve",
    )

    assert entry["reject_reason"] == "reserved_to_ledger"
    assert entry["applied"] is False
    assert isolated["scaffold"].read_text(encoding="utf-8").endswith("1. Think.\n")
    rows = [json.loads(l) for l in isolated["ledger"].read_text(encoding="utf-8").splitlines()]
    assert rows[-1]["failure_signature"] == "comparative-superiority-without-evidence"
    assert rows[-1]["status"] == "open"


def test_single_instance_apply_is_blocked_and_reserved(isolated, monkeypatch) -> None:
    monkeypatch.setattr(mt, "_call_llm", lambda _prompt: _decision(
        mutation="apply",
        evidence="single_instance",
        signature_status="canonical",
        fix_kind="scaffold_gap",
        already="no",
        patch=True,
    ))

    entry = mt.maybe_evolve_after_rating("T-single", 35, ["premise_unchecked"], think_block="trace span")

    assert entry["reject_reason"] == "promotion_gate:single_instance_reserved"
    assert entry["applied"] is False
    assert isolated["scaffold"].read_text(encoding="utf-8").endswith("1. Think.\n")


def test_repeated_canonical_pattern_can_apply(isolated, monkeypatch) -> None:
    store = ledger.DeferredLedger(isolated["ledger"])
    store.record_reservation(
        tag="premise_unchecked",
        section="Audit",
        failure_signature="comparative-superiority-without-evidence",
        signature_status="canonical",
        turn_id="T-prior",
        trace_span="prior trace span",
        rater_note=None,
        rated_index=1,
    )
    monkeypatch.setattr(mt, "_call_llm", lambda _prompt: _decision(
        mutation="apply",
        evidence="repeated_pattern",
        signature_status="canonical",
        fix_kind="scaffold_gap",
        already="no",
        patch=True,
    ))

    entry = mt.maybe_evolve_after_rating("T-apply", 35, ["premise_unchecked"], think_block="new trace span")

    assert entry["applied"] is True
    assert entry["ledger_status"] == "promoted"
    assert "premise check" in isolated["scaffold"].read_text(encoding="utf-8")
    assert isolated["monitor"].exists()
