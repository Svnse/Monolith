from __future__ import annotations

import importlib.util
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from core import continuity, proposals, review_loop
from core.acu_store import ACUStore


NOW = datetime(2026, 5, 20, 12, 0, tzinfo=timezone.utc)


@pytest.fixture()
def review_env(monkeypatch, tmp_path):
    monkeypatch.setattr(review_loop, "_STATE_PATH", tmp_path / "review_loop.json")
    monkeypatch.setattr(review_loop, "_OBSERVATIONS_PATH", tmp_path / "review_observations.json")
    monkeypatch.setattr(continuity, "_STORE_PATH", tmp_path / "continuity.json")
    monkeypatch.setattr(proposals, "STORE_PATH", tmp_path / "proposals.json")
    from core import db_connect as _dbc
    monkeypatch.setattr(_dbc, "DB_PATH", tmp_path / "acatalepsy.sqlite3", raising=True)
    monkeypatch.setenv("MONOLITH_DB_AUTHORIZER_STRICT", "1")
    from core.acatalepsy import schema, intake, canonical_log
    schema.migrate()
    for mod in (intake, canonical_log):
        tl = getattr(mod, "_tl", None)
        if tl is not None:
            for a in ("writer_conn", "reader_conn", "writer", "reader"):
                if hasattr(tl, a):
                    delattr(tl, a)
    return tmp_path


def _old(hours: int) -> str:
    return (NOW - timedelta(hours=hours)).isoformat()


def _make_acu(text: str, *, source: str = "auditor_monolith", hours_old: int = 72) -> int:
    # Direct insert: the fixture needs an ACU with a specific source + age for the
    # backlog logic, so it bypasses the intake atomicity gate (which would reject
    # free-text). Returns the new acu_id.
    from core import db_connect
    from core.acatalepsy.normalize import normalize_canonical
    ts = _old(hours_old)
    conn = db_connect.connect_acatalepsy(role="migration")
    try:
        cur = conn.execute(
            "INSERT INTO acus(canonical, source, provenance, l_level, reinforcement, "
            "evidence_spans, created_at, last_seen, last_touched_ts, state, cf_version) "
            "VALUES(?,?,?,?,?,?,?,?,?,?,?)",
            (normalize_canonical(text) or text, source, "self", "L1", 1, "[]",
             ts, ts, ts, "active", 1),
        )
        conn.commit()
        return int(cur.lastrowid)
    finally:
        conn.close()


def _make_proposal(*, hours_old: int = 120) -> int:
    rec = proposals.propose_amendment(
        target="identity.md",
        section="Emergent",  # post-M2: Origin-0 is frozen; amendments target Emergent
        current_text="(no emergent claims yet)",
        proposed_text="- a self-derived claim queued for review",
        rationale="needs review",
        writer_model_id="test-model",
    )
    data = json.loads(proposals.STORE_PATH.read_text(encoding="utf-8"))
    for item in data["proposals"]:
        if item.get("id") == rec["id"]:
            item["created_at"] = _old(hours_old)
    proposals.STORE_PATH.write_text(json.dumps(data), encoding="utf-8")
    return int(rec["id"])


def _make_pin(text: str, *, category: str = "pending", hours_old: int = 120, evidence: str | None = None) -> int:
    rec = continuity.pin(text, category=category, source="i_inferred", evidence=evidence)
    data = json.loads(continuity._STORE_PATH.read_text(encoding="utf-8"))
    for pin in data["active"]:
        if pin["id"] == rec["id"]:
            pin["created_at"] = _old(hours_old)
    continuity._STORE_PATH.write_text(json.dumps(data), encoding="utf-8")
    return int(rec["id"])


def test_classifies_audit_subkind_with_priority() -> None:
    assert review_loop.classify_audit_subkind("continuity pins are not injected") == "bug"
    assert review_loop.classify_audit_subkind("identity contradiction in runtime substrate") == "ontology"
    assert review_loop.classify_audit_subkind("routing queue has no cadence") == "architecture"
    assert review_loop.classify_audit_subkind("not injected identity contradiction") == "bug"
    assert review_loop.classify_audit_subkind("useful note") == "general"


def test_backlog_sources_surface_without_source_mutation(review_env) -> None:
    acu_id = _make_acu("continuity pins are not injected despite runtime asserting maintained")
    proposal_id = _make_proposal()
    pin_id = _make_pin("Follow up on the review loop")

    items = review_loop.list_review_items(limit=None, now=NOW)
    ids = {item["id"] for item in items}

    assert f"acu:{acu_id}" in ids
    assert f"proposal:{proposal_id}" in ids
    assert f"pin:{pin_id}" in ids

    store = ACUStore()
    try:
        acu = store.get_by_id(acu_id)
    finally:
        store.close()
    assert acu["veracity"] == 5.0
    assert continuity.read()["counts"]["active"] == 1
    assert proposals.list_proposals()[0]["status"] == "pending"


def test_continuity_pin_rule_and_evidence_filters(review_env) -> None:
    rule_pin = _make_pin("Do not treat stale as false", category="anchor", hours_old=24 * 8)
    evidence_pin = _make_pin("Tool-result receipt behavior matters", category="lesson", hours_old=24 * 8, evidence="die probe")
    young_anchor = _make_pin("Do not over-rotate", category="anchor", hours_old=2)

    items = review_loop.list_review_items(limit=None, now=NOW)
    ids = {item["id"] for item in items}

    assert f"pin:{rule_pin}" in ids
    assert f"pin:{evidence_pin}" in ids
    assert f"pin:{young_anchor}" not in ids


def test_monolith_cannot_resolve_or_dismiss_any_bug(review_env) -> None:
    # 2026-06-19 chokepoint fix: monolith may NOT resolve/dismiss ANY review item
    # (owned OR external) — those silently remove a real claim from E's queue, a
    # judgment call that stays E's, refused at review_mark's authz so it holds on
    # every caller (incl. the scratchpad review_mark op). A future re-verified-
    # resolve re-introduces resolve behind an executable check, not this grant.
    external_id = _make_acu("external claim says feature is broken", source="auditor_claude")
    owned_id = _make_acu("trace shows feature is broken", source="auditor_claude")

    for acu_id in (external_id, owned_id):
        for action in ("resolve", "dismiss"):
            with pytest.raises(review_loop.ReviewAuthorizationError):
                review_loop.review_mark(f"acu:{acu_id}", action, actor="monolith", now=NOW)

    # snooze/escalate remain available to monolith
    ok = review_loop.review_mark(f"acu:{owned_id}", "snooze", actor="monolith", now=NOW)
    assert ok["ok"] is True and ok["action"] == "snooze"


def test_snooze_and_escalate_adjust_effective_severity(review_env) -> None:
    proposal_id = _make_proposal()
    item_id = f"proposal:{proposal_id}"

    review_loop.review_mark(
        item_id,
        "snooze",
        actor="e",
        snoozed_until=(NOW - timedelta(hours=1)).isoformat(),
        now=NOW,
    )
    item = review_loop.get_review_item(item_id, now=NOW)
    assert item is not None
    assert item["severity"] == 4
    assert item["effective_severity"] == 5

    review_loop.review_mark(item_id, "escalate", actor="e", now=NOW)
    item = review_loop.get_review_item(item_id, now=NOW)
    assert item is not None
    assert item["effective_severity"] == 5
    assert item["status"] == "escalated"


def test_review_queue_injection_cadence_cap_and_counter(review_env) -> None:
    for _ in range(5):
        _make_proposal()

    sec = review_loop.contribute_section(
        [{"role": "system", "content": "s"}, {"role": "user", "content": "hi"}],
        {},
    )
    assert sec is not None
    assert "[REVIEW QUEUE]" in sec.text
    assert "(+2 more unresolved)" in sec.text

    no_sec = review_loop.contribute_section(
        [{"role": "user", "content": f"u{i}"} for i in range(2)],
        {},
    )
    assert no_sec is None

    later_sec = review_loop.contribute_section(
        [{"role": "user", "content": f"u{i}"} for i in range(16)],
        {},
    )
    assert later_sec is not None


def test_observation_store_surfaces_after_threshold(review_env) -> None:
    record = review_loop.record_observation(
        "Die-roll probe result needs review",
        reason="probe lane",
        severity=2,
        subkind="probe",
        now=NOW - timedelta(hours=80),
    )
    items = review_loop.list_review_items(limit=None, now=NOW)
    assert any(item["id"] == f"observation:{record['id']}" for item in items)


def test_scratchpad_review_ops_round_trip(review_env, monkeypatch) -> None:
    _make_proposal()
    spec_path = Path(__file__).parent.parent / "skills" / "scratchpad" / "executor.py"
    spec = importlib.util.spec_from_file_location("scratchpad_exec_review_test", spec_path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    monkeypatch.setattr(mod._review, "_STATE_PATH", review_loop._STATE_PATH)
    monkeypatch.setattr(mod._review, "_OBSERVATIONS_PATH", review_loop._OBSERVATIONS_PATH)

    out = mod.run({"op": "review_read", "limit": 3}, None)
    assert "[review_read: 1 item(s)]" in out
    assert "proposal:1" in out

    denied = mod.run({"op": "review_mark", "item_id": "proposal:1", "action": "resolve"}, None)
    assert "unauthorized" in denied

    ok = mod.run({"op": "review_mark", "item_id": "proposal:1", "action": "escalate"}, None)
    assert "[review_mark: escalate proposal:1 as monolith]" in ok


def test_tool_validation_allows_review_fields() -> None:
    from core.tool_validation import validate_tool_arguments

    errors = validate_tool_arguments(
        "scratchpad",
        {
            "op": "review_mark",
            "item_id": "acu:25",
            "action": "snooze",
            "snooze_hours": 12,
            "note": "checking later",
        },
    )
    assert errors == []


def test_rule_language_uses_word_boundaries() -> None:
    # Regression: pin 2's "user-controlled" and "User picks" must not match "use".
    # The substring matcher caused pin 2 to surface as anchor_rule incorrectly.
    pin_2_text = (
        "Cognitive scaffolding = /effort tier (user-controlled depth dial), "
        "not task-type. User picks the tier; model loads matching prompts/effort/<tier>.md."
    )
    assert review_loop._has_rule_language(pin_2_text) is False

    # Standalone keywords still match.
    assert review_loop._has_rule_language("use the new flow") is True
    assert review_loop._has_rule_language("do not perform it") is True
    assert review_loop._has_rule_language("the rule is: ship first") is True

    # Other common substring false-positives stay rejected.
    assert review_loop._has_rule_language("shoulder injuries are common") is False
    assert review_loop._has_rule_language("nevertheless, ship it") is False
    assert review_loop._has_rule_language("preferences are mixed") is False


def test_audit_subkind_uses_word_boundaries() -> None:
    # "missing" inside "dismissing" must not trigger bug classification.
    assert review_loop.classify_audit_subkind("dismissing the report") == "general"
    # but the standalone keyword still matches.
    assert review_loop.classify_audit_subkind("the feature is missing") == "bug"


def test_monolith_owned_uses_word_boundaries() -> None:
    # "state" inside "stateful" should not flag a generic claim as Monolith-owned.
    assert review_loop._is_monolith_owned_bug(
        origin_source="auditor_claude",
        text="stateful service is broken",
    ) is False
    # standalone "state" still counts (e.g., references Monolith's /state surface).
    assert review_loop._is_monolith_owned_bug(
        origin_source="auditor_claude",
        text="state shows the value diverged",
    ) is True
    # auditor_monolith source always carves out, regardless of text.
    assert review_loop._is_monolith_owned_bug(
        origin_source="auditor_monolith",
        text="unrelated text",
    ) is True
