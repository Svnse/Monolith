"""Spine replay reducer — reconstructs state from canonical_log events.

The substrate is NOT event-sourced: it writes ``acus``/``acu_relations`` state
directly, with ``canonical_log`` as a parallel audit floor. This module proves
the A1 invariant — that the spine's events are *reconstruction-complete* — by
replaying the spine event stream into a FRESH (already-migrated) store and
rebuilding identical state, without joining any other table.

Scope: the spine event kinds emitted by ``intake`` —
``l1_novel_survive`` / ``l1_partial_edge_induced`` / ``l1_match_reinforce``.
``entry_crystallize`` / ``entry_merge`` are handled once their callers exist
(crystallize/backfill are not wired in this tranche). Pre-spine legacy events
are not reconstruction-complete and are out of scope.

An id-map (original acu_id -> reconstructed id) bridges the autoincrement ids
in the source events to the fresh store's ids, for edges and reinforcement
targeting.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

from core.acatalepsy import eqid as _eqid
from core.acatalepsy.normalize import parse_triple

__all__ = ("replay_spine",)

_SPINE_KINDS = frozenset({
    "l1_novel_survive", "l1_partial_edge_induced", "l1_match_reinforce",
    "entry_crystallize",
})


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ev(event):
    """Return (kind, payload) from an Event object or a {kind,payload} dict."""
    if isinstance(event, dict):
        return event.get("kind"), event.get("payload")
    return getattr(event, "kind", None), getattr(event, "payload", None)


def _triple_json(canonical_form: str) -> str | None:
    triple = parse_triple(canonical_form)
    if triple is None:
        return None
    return json.dumps({
        "entity_a": triple.entity_a, "relation": triple.relation,
        "entity_b": triple.entity_b, "qualifiers": triple.qualifiers,
    })


def _insert_stub(conn, payload: dict) -> int:
    cf = payload["canonical_form"]
    prov = payload.get("provenance", "self")
    span = payload.get("span")
    spans = [span] if span else []
    now = _now_iso()
    cur = conn.execute(
        "INSERT INTO acus(canonical, source, provenance, kind, l_level, reinforcement, "
        "evidence_spans, canonical_triple, valid_from, created_at, last_seen, "
        "last_touched_ts, cf_version, state) "
        "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (
            cf, prov, prov, payload.get("kind"), "L1", int(payload.get("reinforcement", 1)),
            json.dumps(spans), _triple_json(cf), now, now, now, now,
            int(payload.get("cf_version", 1)), "active",
        ),
    )
    return int(cur.lastrowid)


def replay_spine(events, *, conn) -> dict:
    """Reconstruct acus + acu_relations into ``conn`` from spine ``events``.

    The caller owns the transaction (commit) on ``conn``. Returns a summary of
    how many events of each kind were applied.
    """
    id_map: dict[int, int] = {}
    counts = {"novel": 0, "match": 0, "partial": 0, "edges": 0,
              "crystallized": 0, "skipped": 0}

    for event in events:
        kind, payload = _ev(event)
        if kind not in _SPINE_KINDS or not payload:
            counts["skipped"] += 1
            continue

        if kind == "l1_novel_survive":
            new_id = _insert_stub(conn, payload)
            id_map[int(payload["acu_id"])] = new_id
            counts["novel"] += 1

        elif kind == "l1_partial_edge_induced":
            new_id = _insert_stub(
                conn, {**payload, "acu_id": payload["new_acu_id"]}
            )
            id_map[int(payload["new_acu_id"])] = new_id
            target = id_map.get(int(payload["existing_acu_id"]))
            if target is not None:
                now = _now_iso()
                conn.execute(
                    "INSERT INTO acu_relations(source_id, target_id, relation, score, "
                    "created_at, updated_at) VALUES(?,?,?,?,?,?)",
                    (new_id, target, payload.get("relation", "overlaps"), 1.0, now, now),
                )
                counts["edges"] += 1
            counts["partial"] += 1

        elif kind == "l1_match_reinforce":
            orig = int(payload["acu_id"])
            target = id_map.get(orig)
            if target is None:
                row = conn.execute(
                    "SELECT id FROM acus WHERE canonical=? AND merged_into IS NULL",
                    (payload.get("canonical_form"),),
                ).fetchone()
                target = int(row["id"]) if row else None
            if target is not None:
                weight = int(payload.get("weight", 1))
                row = conn.execute(
                    "SELECT reinforcement, evidence_spans FROM acus WHERE id=?", (target,)
                ).fetchone()
                spans = json.loads(row["evidence_spans"]) if row["evidence_spans"] else []
                span = payload.get("span")
                if span:
                    # Dedup by (text, provenance, source_event) to match intake's
                    # _append_span — NOT full-dict (the `ts` differs, which would never
                    # dedup). source_event keeps distinct OCCASIONS distinct on replay,
                    # so reconstructed evidence_spans match the live store (and the
                    # distinct-source-event count a self ACU read survives replay).
                    key = (span.get("text"), span.get("provenance"), span.get("source_event"))
                    if not any(
                        (s.get("text"), s.get("provenance"), s.get("source_event")) == key
                        for s in spans
                    ):
                        spans.append(span)
                conn.execute(
                    "UPDATE acus SET reinforcement=?, evidence_spans=?, last_touched_ts=? "
                    "WHERE id=?",
                    (int(row["reinforcement"] or 0) + weight, json.dumps(spans),
                     _now_iso(), target),
                )
                counts["match"] += 1

        elif kind == "entry_crystallize":
            orig = int(payload["acu_id"])
            target = id_map.get(orig)
            if target is None:
                row = conn.execute(
                    "SELECT id FROM acus WHERE canonical=? AND merged_into IS NULL",
                    (payload.get("canonical_form"),),
                ).fetchone()
                target = int(row["id"]) if row else None
            if target is not None:
                conn.execute(
                    "UPDATE acus SET cid=?, eqid=?, l_level='L2', promoted_to_l2_ts=? WHERE id=?",
                    (payload.get("cid"),
                     _eqid.compute_eqid_for_form(payload.get("canonical_form", "")),
                     _now_iso(), target),
                )
                counts["crystallized"] += 1

    return counts
