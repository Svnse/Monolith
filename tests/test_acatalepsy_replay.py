"""Replay reducer — A1 acceptance criterion (second half: state is replayable).

The store is not event-sourced; canonical_log is a parallel audit floor. This
verifies the spine's events are reconstruction-complete: replaying the spine
event stream into a FRESH store rebuilds identical acus/acu_relations state,
without joining any other table.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest


def _clear(*mods):
    for mod in mods:
        tl = getattr(mod, "_tl", None)
        if tl is not None:
            for attr in ("writer_conn", "reader_conn", "writer", "reader"):
                if hasattr(tl, attr):
                    delattr(tl, attr)


def test_replay_reconstructs_identical_state(tmp_path: Path, monkeypatch):
    from core import db_connect as _dbc
    monkeypatch.setenv("MONOLITH_DB_AUTHORIZER_STRICT", "1")

    # ── Build the SOURCE store A via the real intake path ──
    path_a = tmp_path / "a.sqlite3"
    monkeypatch.setattr(_dbc, "DB_PATH", path_a, raising=True)
    from core.acatalepsy import schema, intake, canonical_log
    schema.migrate()
    _clear(intake, canonical_log)

    intake.ingest_l1(raw_form="a | r | b", provenance="user")          # novel
    intake.ingest_l1(raw_form="a | r | b", provenance="user")          # match -> reinforce
    intake.ingest_l1(raw_form="a | r | c", provenance="user")          # partial (shares 'a') + edge
    intake.ingest_l1(raw_form="z | q | w", provenance="self")          # novel, unrelated

    conn_a = _dbc.connect_acatalepsy(role="migration")
    rows = conn_a.execute("SELECT kind, payload FROM canonical_log ORDER BY event_id").fetchall()
    events = [
        {"kind": r["kind"], "payload": json.loads(r["payload"]) if r["payload"] else None}
        for r in rows
    ]
    a_acus = {
        r["canonical"]: (r["reinforcement"], r["cid"] is not None)
        for r in conn_a.execute(
            "SELECT canonical, reinforcement, cid FROM acus WHERE merged_into IS NULL").fetchall()
    }
    a_edges = conn_a.execute("SELECT COUNT(*) FROM acu_relations").fetchone()[0]
    conn_a.close()

    # world-fact user claims crystallized (cid set); the self claim stayed L1.
    assert a_acus == {
        "a | r | b": (4, True), "a | r | c": (2, True), "z | q | w": (1, False)}
    assert a_edges == 1

    # ── Replay into a FRESH store B ──
    path_b = tmp_path / "b.sqlite3"
    monkeypatch.setattr(_dbc, "DB_PATH", path_b, raising=True)
    schema.migrate()
    from core.acatalepsy import replay
    conn_b = _dbc.connect_acatalepsy(role="migration")
    try:
        replay.replay_spine(events, conn=conn_b)
        conn_b.commit()
        b_acus = {
            r["canonical"]: (r["reinforcement"], r["cid"] is not None)
            for r in conn_b.execute(
                "SELECT canonical, reinforcement, cid FROM acus WHERE merged_into IS NULL").fetchall()
        }
        b_edges = conn_b.execute("SELECT COUNT(*) FROM acu_relations").fetchone()[0]
    finally:
        conn_b.close()

    assert b_acus == a_acus      # identical canonical -> reinforcement + cid presence
    assert b_edges == a_edges    # the PARTIAL overlaps edge reconstructed


def test_replay_match_dedups_identical_spans(tmp_path: Path, monkeypatch):
    # Reinforcement with identical evidence (differing only in ts) must dedup on
    # replay exactly as intake's _append_span does — else replayed state diverges.
    from core import db_connect as _dbc
    monkeypatch.setenv("MONOLITH_DB_AUTHORIZER_STRICT", "1")
    monkeypatch.setattr(_dbc, "DB_PATH", tmp_path / "r.sqlite3", raising=True)
    from core.acatalepsy import schema, replay
    schema.migrate()
    events = [
        {"kind": "l1_novel_survive", "payload": {
            "acu_id": 1, "canonical_form": "sky | is | blue", "provenance": "self",
            "kind": "world-fact", "cf_version": 1, "reinforcement": 1,
            "span": {"text": "T", "provenance": "self", "ts": "T1"}}},
        {"kind": "l1_match_reinforce", "payload": {
            "acu_id": 1, "weight": 1, "canonical_form": "sky | is | blue",
            "span": {"text": "T", "provenance": "self", "ts": "T2"}}},
    ]
    conn = _dbc.connect_acatalepsy(role="migration")
    try:
        replay.replay_spine(events, conn=conn)
        conn.commit()
        row = conn.execute(
            "SELECT reinforcement, evidence_spans FROM acus WHERE canonical='sky | is | blue'"
        ).fetchone()
    finally:
        conn.close()
    assert row["reinforcement"] == 2
    assert len(json.loads(row["evidence_spans"])) == 1  # identical span deduped


def test_replay_reconstructs_self_l2_identity_memory(tmp_path: Path, monkeypatch):
    # A self claim that crystallized to L2 via 3 distinct source_events must replay to
    # the same L2/cid state (entry_crystallize applied directly) WITH its distinct
    # spans preserved (widened dedup) — reconstruction-complete for the new self path.
    monkeypatch.setenv("MONOLITH_SELF_IDENTITY_L2_V1", "1")
    monkeypatch.setenv("MONOLITH_DB_AUTHORIZER_STRICT", "1")
    from core import db_connect as _dbc
    canon = "monolith | values | honesty"

    path_a = tmp_path / "a.sqlite3"
    monkeypatch.setattr(_dbc, "DB_PATH", path_a, raising=True)
    from core.acatalepsy import schema, intake, canonical_log
    schema.migrate()
    _clear(intake, canonical_log)
    for ev in (1, 2, 3):
        intake.ingest_l1(raw_form=canon, provenance="self", source_event=ev)

    conn_a = _dbc.connect_acatalepsy(role="migration")
    events = [
        {"kind": r["kind"], "payload": json.loads(r["payload"]) if r["payload"] else None}
        for r in conn_a.execute(
            "SELECT kind, payload FROM canonical_log ORDER BY event_id").fetchall()
    ]
    a_row = conn_a.execute(
        "SELECT l_level, cid, evidence_spans FROM acus WHERE canonical=?", (canon,)).fetchone()
    conn_a.close()
    assert a_row["l_level"] == "L2" and a_row["cid"] is not None
    assert len({s.get("source_event") for s in json.loads(a_row["evidence_spans"])}) == 3

    # Replay into a FRESH store B (entry_crystallize applies directly; flag-independent).
    path_b = tmp_path / "b.sqlite3"
    monkeypatch.setattr(_dbc, "DB_PATH", path_b, raising=True)
    schema.migrate()
    from core.acatalepsy import replay
    conn_b = _dbc.connect_acatalepsy(role="migration")
    try:
        replay.replay_spine(events, conn=conn_b)
        conn_b.commit()
        b_row = conn_b.execute(
            "SELECT l_level, cid, evidence_spans FROM acus WHERE canonical=?", (canon,)).fetchone()
    finally:
        conn_b.close()
    assert b_row["l_level"] == "L2"
    assert b_row["cid"] == a_row["cid"]                       # identity reconstructed
    assert len({s.get("source_event") for s in json.loads(b_row["evidence_spans"])}) == 3


def test_replay_rederives_eqid_on_crystallize(tmp_path, monkeypatch):
    from core import db_connect as _dbc
    monkeypatch.setattr(_dbc, "DB_PATH", tmp_path / "replay_eqid.sqlite3", raising=True)
    monkeypatch.setenv("MONOLITH_DB_AUTHORIZER_STRICT", "1")
    from core.acatalepsy import schema, replay, eqid
    schema.migrate()

    events = [
        {"kind": "l1_novel_survive",
         "payload": {"acu_id": 1, "canonical_form": "paris | capital_of | france",
                     "provenance": "world", "kind": "world-fact", "cf_version": 1,
                     "reinforcement": 1, "span": {"text": "x", "provenance": "world", "ts": "t"}}},
        {"kind": "entry_crystallize",
         "payload": {"acu_id": 1, "cid": "cid:sha256:aaa",
                     "canonical_form": "paris | capital_of | france", "prior_l_level": "L1"}},
    ]
    conn = _dbc.connect_acatalepsy(role="migration")
    try:
        replay.replay_spine(events, conn=conn)
        conn.commit()
        row = conn.execute(
            "SELECT cid, eqid FROM acus WHERE canonical='paris | capital_of | france'"
        ).fetchone()
    finally:
        conn.close()
    assert row["cid"] == "cid:sha256:aaa"
    assert row["eqid"] == eqid.compute_eqid_for_form("paris | capital_of | france")
