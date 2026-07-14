"""L1 comparison pass — the one-writer intake for the substrate spine.

Every claim entering the substrate flows through ``ingest_l1``. The pass is
decided PURELY by normalizer output (the Kind branch does not exist until L2,
so it cannot be consulted):

  MATCH    same normalized canonical form exists -> reinforce it (provenance-
           weighted), append a structured evidence span; no new row. A LOCKED
           match (e.g. Origin-0) is recognized but left immutable.
  PARTIAL  same parsed subject, different relation/object -> new L1 stub plus
           one neutral ``overlaps`` edge in acu_relations (the first CCG writer).
           No edge-typing (contradicts/refines is a Truth-branch judgment,
           deferred).
  NOVEL    nothing matches -> new L1 stub.

This unifies the historical write paths so claims dedup/reinforce consistently.
``veracity`` is never touched (dormant). Each outcome emits a reconstruction-
complete canonical_log event, written on the SAME connection/transaction as the
mutation so the event can never be lost relative to the state change (the A1
replay/audit invariant).

Transaction ownership: if the caller passes ``conn`` it owns the sentinel and
commit (so the ACU + event land atomically with the caller's own writes, e.g.
the decision row); otherwise a thread-local writer connection is used and
committed here.
"""
from __future__ import annotations

import json
import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime, timezone

from core.db_connect import authorized_write, connect_acatalepsy
from core.acatalepsy import canonical_log as _canonical_log
from core.acatalepsy.atomicity import is_atomic
from core.acatalepsy.kind import classify_kind
from core.acatalepsy.normalize import CF_VERSION, normalize_canonical, parse_triple

__all__ = ("IntakeResult", "ingest_l1")

# Provenance-weighted reinforcement: world corroboration counts more than the
# model's own echo (a soft expression of the Mad-Cow asymmetry).
_PROV_WEIGHT = {"world": 3, "user": 2, "self": 1}


@dataclass(frozen=True)
class IntakeResult:
    acu_id: int
    outcome: str                 # 'match' | 'partial' | 'novel' | 'rejected'
    edge_id: int | None = None
    reason: str | None = None    # rejection reason when outcome == 'rejected'


_tl = threading.local()


def _writer_conn() -> sqlite3.Connection:
    conn = getattr(_tl, "writer", None)
    if conn is None:
        conn = connect_acatalepsy(role="memory_writer")
        _tl.writer = conn
    return conn


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_spans(raw: object) -> list[dict]:
    if not raw:
        return []
    try:
        data = json.loads(raw)
    except (TypeError, ValueError):
        return []
    return [s for s in data if isinstance(s, dict)] if isinstance(data, list) else []


# Cap on the per-ACU evidence-span array. Widening the dedup key by source_event
# (below) lets distinct occasions accumulate distinct spans; this bound stops a
# heavily-reinforced claim from growing the array without limit. 10 is well above
# the self->L2 distinct-event gate (3), so the gate is never starved.
_MAX_SPANS = 10


def _append_span(spans: list[dict], span: dict) -> list[dict]:
    # Dedup by (text, provenance, source_event). source_event distinguishes
    # OCCASIONS: the same claim re-asserted on a different turn/event is a new,
    # countable span (the distinct-source-event signal self->L2 reads), while a
    # verbatim re-assertion within the same occasion still dedups. NONE-SAFE: every
    # caller that passes no source_event yields key (text, provenance, None), i.e.
    # byte-identical to the prior (text, provenance) dedup.
    key = (span.get("text"), span.get("provenance"), span.get("source_event"))
    if any((s.get("text"), s.get("provenance"), s.get("source_event")) == key for s in spans):
        return spans
    out = [*spans, span]
    return out[-_MAX_SPANS:] if len(out) > _MAX_SPANS else out


def _triple_json(triple) -> str | None:
    if triple is None:
        return None
    return json.dumps({
        "entity_a": triple.entity_a,
        "relation": triple.relation,
        "entity_b": triple.entity_b,
        "qualifiers": triple.qualifiers,
    })


def _maybe_crystallize(conn, acu_id: int) -> None:
    """Crystallization trigger: once Kind is resolved and Mad Cow passes, cool the
    L1 stub into a content-addressed L2 ACU (emitting a replayable event). Fires in
    the same transaction as the intake write. No-op when ineligible (e.g. a self
    claim still awaiting user/world confirmation)."""
    from core.acatalepsy import crystallize as _cryst
    if not _cryst.is_crystallize_eligible(acu_id, conn):
        return
    res = _cryst.crystallize(acu_id, conn=conn)
    if res.crystallized:
        row = conn.execute(
            "SELECT canonical, provenance, kind, evidence_spans FROM acus WHERE id=?",
            (int(acu_id),),
        ).fetchone()
        # Observability: stamp provenance/kind + the distinct-source-event count so
        # every self->L2 identity-memory promotion is visible in canonical_log /
        # MonoSearch (the self path is the new, flag-gated behaviour worth watching).
        payload = {
            "acu_id": int(acu_id), "cid": res.cid,
            "canonical_form": row["canonical"], "prior_l_level": "L1",
            "provenance": row["provenance"], "kind": row["kind"],
            "distinct_source_events": _cryst._distinct_source_events(
                _load_spans(row["evidence_spans"])),
        }
        _canonical_log.append_on(conn, "entry_crystallize", payload, acu_id=int(acu_id))


def _record_affect(conn, acu_id, triple, provenance) -> None:
    """For an `emotional`-kind claim, append an affect reading (the Affect lane).
    Affect is modelled, not truth-checked."""
    from core.acatalepsy import affect
    if triple is None:
        return
    aff = affect.extract_affect(triple)
    if aff is None:
        return
    valence, arousal, intensity = aff
    affect.append_reading(
        conn, subject=triple.entity_a, valence=valence, arousal=arousal,
        intensity=intensity, target=triple.entity_b, source=provenance, acu_id=acu_id,
    )


def ingest_l1(
    *,
    raw_form: str,
    provenance: str,
    evidence_span: str | None = None,
    source_event: int | None = None,
    conn: sqlite3.Connection | None = None,
) -> IntakeResult:
    """Run the L1 comparison pass for one incoming claim. See module docstring."""
    provenance = (provenance or "self").strip().lower()
    weight = _PROV_WEIGHT.get(provenance, 1)
    cf = normalize_canonical(raw_form)
    gate = is_atomic(cf)
    triple = parse_triple(cf)
    span = {
        "text": evidence_span if evidence_span is not None else raw_form,
        "provenance": provenance,
        "source_event": source_event,   # per-occasion id; distinct-count feeds self->L2
        "ts": _now_iso(),
    }

    own = conn is None
    if own:
        conn = _writer_conn()

    # authorized_write is re-entrant; harmless if the caller already holds it.
    with authorized_write(f"l1_intake:{provenance}"):
        result = _run_intake(conn, cf, gate, triple, span, provenance, weight, source_event)
        if own:
            conn.commit()
    return result


def _run_intake(conn, cf, gate, triple, span, provenance, weight, source_event) -> IntakeResult:
    if not gate.ok:
        _canonical_log.append_on(
            conn, "auditor_atomicity_reject",
            {"canonical_form": cf, "reason": gate.reason, "provenance": provenance},
        )
        return IntakeResult(acu_id=-1, outcome="rejected", reason=gate.reason)

    match = conn.execute(
        "SELECT id, reinforcement, evidence_spans, canonical_triple, kind, locked, state "
        "FROM acus WHERE canonical=? AND merged_into IS NULL",
        (cf,),
    ).fetchone()

    if match is not None:
        acu_id = int(match["id"])
        if int(match["locked"] or 0) or match["state"] == "-inf":
            # Immutable: locked (Origin-0) or a known falsehood (-inf). Recognized
            # as MATCH but never reinforced/resurrected and never duplicated.
            return IntakeResult(acu_id=acu_id, outcome="match")
        new_reinf = int(match["reinforcement"] or 0) + weight
        spans = _append_span(_load_spans(match["evidence_spans"]), span)
        ct = match["canonical_triple"]
        if (ct is None or ct == "") and triple is not None:
            ct = _triple_json(triple)  # lazily populate legacy NULL triple
        # Infer Kind for a legacy row that lacks one (so it can crystallize once
        # user/world-confirmed).
        new_kind = match["kind"] or (classify_kind(triple) if triple is not None else None)
        conn.execute(
            "UPDATE acus SET reinforcement=?, evidence_spans=?, canonical_triple=?, "
            "kind=?, last_confirmed_at=?, last_touched_ts=? WHERE id=?",
            (new_reinf, json.dumps(spans), ct, new_kind, _now_iso(), _now_iso(), acu_id),
        )
        _canonical_log.append_on(
            conn, "l1_match_reinforce",
            {"acu_id": acu_id, "provenance": provenance, "weight": weight,
             "canonical_form": cf, "span": span},
            acu_id=acu_id,
        )
        if new_kind == "emotional":
            # Each re-assertion is another point in the affect trajectory.
            _record_affect(conn, acu_id, triple, provenance)
        _maybe_crystallize(conn, acu_id)
        # A confirmed claim that crosses the reinforcement bar via this re-assertion
        # earns L3 (trusted). No-op unless it is L2 + confirmed + reinforced.
        from core.acatalepsy import crystallize as _cryst
        _cryst.maybe_promote_l3(acu_id, conn)
        return IntakeResult(acu_id=acu_id, outcome="match")

    # ── PARTIAL vs NOVEL: look for a shared parsed subject ──
    edge_target: int | None = None
    if triple is not None:
        cand = conn.execute(
            "SELECT id FROM acus WHERE merged_into IS NULL AND canonical<>? "
            "AND json_extract(canonical_triple,'$.entity_a')=? LIMIT 1",
            (cf, triple.entity_a),
        ).fetchone()
        if cand is not None:
            edge_target = int(cand["id"])

    kind = classify_kind(triple)
    cur = conn.execute(
        "INSERT INTO acus(canonical, source, provenance, kind, l_level, reinforcement, "
        "evidence_spans, canonical_triple, valid_from, created_at, last_seen, "
        "last_touched_ts, source_event, state, cf_version) "
        "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (
            cf, provenance, provenance, kind, "L1", weight,
            json.dumps([span]), _triple_json(triple), _now_iso(), _now_iso(),
            _now_iso(), _now_iso(), source_event, "active", CF_VERSION,
        ),
    )
    acu_id = int(cur.lastrowid)

    if kind == "emotional":
        _record_affect(conn, acu_id, triple, provenance)

    if edge_target is not None:
        ecur = conn.execute(
            "INSERT INTO acu_relations(source_id, target_id, relation, score, "
            "created_at, updated_at) VALUES(?,?,?,?,?,?)",
            (acu_id, edge_target, "overlaps", 1.0, _now_iso(), _now_iso()),
        )
        edge_id = int(ecur.lastrowid)
        _canonical_log.append_on(
            conn, "l1_partial_edge_induced",
            {"new_acu_id": acu_id, "existing_acu_id": edge_target, "edge_id": edge_id,
             "relation": "overlaps", "flagged_existing": True, "canonical_form": cf,
             "provenance": provenance, "kind": kind, "cf_version": CF_VERSION,
             "reinforcement": weight, "span": span},
            acu_id=acu_id,
        )
        _maybe_crystallize(conn, acu_id)
        return IntakeResult(acu_id=acu_id, outcome="partial", edge_id=edge_id)

    _canonical_log.append_on(
        conn, "l1_novel_survive",
        {"acu_id": acu_id, "canonical_form": cf, "provenance": provenance,
         "kind": kind, "cf_version": CF_VERSION, "reinforcement": weight, "span": span},
        acu_id=acu_id,
    )
    _maybe_crystallize(conn, acu_id)
    return IntakeResult(acu_id=acu_id, outcome="novel")
