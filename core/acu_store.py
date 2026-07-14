"""ACUStore — thin facade over the unified Acatalepsy substrate.

Historically a standalone store with its own connection + minimal schema. As of
the spine, it is a facade over the single global substrate DB (resolved via
``core.db_connect.DB_PATH``):

  * Writes funnel through the one-writer L1 intake (``intake.ingest_l1``) —
    normalized dedup, provenance-weighted reinforcement, and replay-complete
    canonical_log events. Non-atomic free-text is rejected by the atomicity gate
    rather than silently stored (this is what stopped the legacy junk).
  * Reads rank by behavioural signal (``reinforcement`` + recency), excluding
    merged/archived rows. This is the documented recall STOPGAP; its final form
    is Authority-driven (branch phase).
  * ``veracity`` is dead — never read for ranking, never written.

The ``db_path`` constructor arg is retained for signature compatibility but
ignored; tests isolate by monkeypatching ``db_connect.DB_PATH``.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from core.db_connect import authorized_write, connect_acatalepsy

# Columns returned by reads. `veracity` kept in the projection (dormant) so any
# legacy consumer reading the key still works; it is NOT used for ranking.
_READ_COLS = (
    "id, canonical, veracity, reinforcement, source, provenance, l_level, kind, "
    "domain, truth, truth_confidence, truth_checked_at, state, created_at, last_seen, "
    "last_touched_ts, confidentity, locked, lock_reason, eqid, cid"
)

# Bounded scan for the decay-aware retrieve() re-rank (no exp() in SQLite).
# Matches affect_profile's 2000-row scan bound; within it the effective re-rank
# is exact, beyond it raw-reinforcement order is the prefilter.
_DECAY_SCAN_CAP = 2000

# Free-text source -> provenance (self|user|world).
_PROV = {
    "identity_origin_0": "user",
    "user_stated": "user",
    "user": "user",
    "world": "world",
    "tool": "world",
}


def _provenance(source: str) -> str:
    return _PROV.get((source or "").strip().lower(), "self")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class ACUStore:
    DB_PATH = None  # legacy attribute; substrate path lives in db_connect.DB_PATH

    def __init__(self, db_path: str | None = None) -> None:  # db_path ignored
        # NOTE: does NOT migrate here. Schema is ensured by bootstrap_acatalepsy()
        # at app start (production) and explicitly by test harnesses — so simply
        # constructing an ACUStore can never mutate the live DB.
        pass

    # ── write ───────────────────────────────────────────────────────────

    def ingest(self, canonical: str, source: str = "model",
               *, source_event: int | None = None) -> int:
        """Store/reinforce one claim via the L1 intake. Returns the acu_id, or
        -1 if the claim is empty or fails the atomicity gate.

        ``source_event`` is a per-occasion id (e.g. the outer-turn counter); when a
        self-provenance claim recurs across distinct source_events it can crystallize
        to L2 identity memory. Omitting it (None) keeps legacy behaviour byte-identical.
        """
        from core.acatalepsy import intake
        res = intake.ingest_l1(raw_form=canonical, provenance=_provenance(source),
                               source_event=source_event)
        return res.acu_id

    def ingest_many(self, claims: list[str], source: str = "model",
                    *, source_event: int | None = None) -> list[int]:
        return [self.ingest(c, source, source_event=source_event)
                for c in claims if c and c.strip()]

    def ingest_locked(
        self,
        canonical: str,
        *,
        source: str = "identity_origin_0",
        lock_reason: str = "origin_0",
        confidentity: float = 1.0,
    ) -> int:
        """Store/mark an immutable Origin-0 ACU (normalized, locked, no veracity)."""
        from core.acatalepsy.atomicity import is_atomic
        from core.acatalepsy.normalize import normalize_canonical, parse_triple
        raw = (canonical or "").strip()
        if not raw:
            return -1
        norm = normalize_canonical(raw)
        # Atomic triples store the normalized form (so they dedup with intake);
        # human-readable identity prose keeps its original casing for display.
        # Use the real atomicity gate (not just pipe-count) so a prose paragraph
        # that happens to contain pipes isn't mistaken for a triple.
        atomic = is_atomic(norm).ok
        cf = norm if atomic else raw
        triple = parse_triple(norm) if atomic else None
        ct = json.dumps({
            "entity_a": triple.entity_a, "relation": triple.relation,
            "entity_b": triple.entity_b, "qualifiers": triple.qualifiers,
        }) if triple else None
        now = _now()
        conn = connect_acatalepsy(role="memory_writer")
        try:
            with authorized_write("acu_store:ingest_locked"):
                existing = conn.execute(
                    "SELECT id FROM acus WHERE canonical=? AND merged_into IS NULL", (cf,)
                ).fetchone()
                if existing is None and not atomic:
                    # Prose: dedup on the normalized key so casing/whitespace drift
                    # on re-edited identity text can't insert a duplicate locked row.
                    for r in conn.execute(
                        "SELECT id, canonical FROM acus WHERE locked=1 AND merged_into IS NULL"
                    ).fetchall():
                        if normalize_canonical(r["canonical"]) == norm:
                            existing = r
                            break
                if existing:
                    conn.execute(
                        "UPDATE acus SET source=?, confidentity=MAX(COALESCE(confidentity,0),?), "
                        "locked=1, lock_reason=?, last_seen=?, last_touched_ts=? WHERE id=?",
                        (source, float(confidentity), lock_reason, now, now, int(existing["id"])),
                    )
                    conn.commit()
                    return int(existing["id"])
                cur = conn.execute(
                    "INSERT INTO acus(canonical, source, provenance, l_level, reinforcement, "
                    "evidence_spans, canonical_triple, confidentity, locked, lock_reason, "
                    "valid_from, created_at, last_seen, last_touched_ts, state, cf_version) "
                    "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (cf, source, "user", "L1", 1, "[]", ct, float(confidentity), 1,
                     lock_reason, now, now, now, now, "active", 1),
                )
                conn.commit()
                return int(cur.lastrowid)
        finally:
            conn.close()

    def contradict(self, acu_id: int) -> None:
        """Retired. veracity is dead; contradiction handling moves to the Truth
        branch (a typed CCG `contradicts` edge + `state='-inf'`). No-op."""
        return None

    # ── read ────────────────────────────────────────────────────────────

    @staticmethod
    def _reader():
        return connect_acatalepsy(role="reader")

    def retrieve(self, limit: int = 20, *, now=None) -> list[dict[str, Any]]:
        """Top active ACUs by behavioural signal (reinforcement + recency).
        STOPGAP ranking — final form is Authority-driven (branch phase).

        When MONOLITH_ACU_DECAY_V1 is on, rank by EFFECTIVE (time-decayed)
        reinforcement instead of raw: an un-reinforced old claim loses ranking
        weight without being deleted. SQLite has no exp() to ORDER BY, so we
        fetch the active set ordered by raw reinforcement (bounded by
        _DECAY_SCAN_CAP, like affect's profile scan) and re-rank in Python.
        Within the cap this is exact (effective<=raw); beyond it the raw order
        is the prefilter — the same bounded-scan tradeoff affect already makes.
        Flag off → byte-identical raw ordering.
        """
        from core.acatalepsy import decay
        conn = self._reader()
        try:
            if decay.decay_enabled():
                cur = conn.execute(
                    f"SELECT {_READ_COLS} FROM acus "
                    "WHERE merged_into IS NULL AND state='active' "
                    "ORDER BY reinforcement DESC, last_touched_ts DESC, last_seen DESC LIMIT ?",
                    (int(_DECAY_SCAN_CAP),),
                )
                rows = [dict(row) for row in cur.fetchall()]
                rows.sort(
                    key=lambda r: decay.effective_reinforcement(r, now=now),
                    reverse=True,
                )
                return rows[: int(limit)]
            cur = conn.execute(
                f"SELECT {_READ_COLS} FROM acus "
                "WHERE merged_into IS NULL AND state='active' "
                "ORDER BY reinforcement DESC, last_touched_ts DESC, last_seen DESC LIMIT ?",
                (int(limit),),
            )
            return [dict(row) for row in cur.fetchall()]
        finally:
            conn.close()

    def search(self, query: str, limit: int = 10) -> list[dict[str, Any]]:
        conn = self._reader()
        try:
            cur = conn.execute(
                f"SELECT {_READ_COLS} FROM acus "
                "WHERE canonical LIKE ? AND merged_into IS NULL AND state='active' "
                "ORDER BY reinforcement DESC LIMIT ?",
                (f"%{query}%", int(limit)),
            )
            return [dict(row) for row in cur.fetchall()]
        finally:
            conn.close()

    def count(self) -> int:
        conn = self._reader()
        try:
            return int(conn.execute("SELECT COUNT(*) FROM acus").fetchone()[0])
        finally:
            conn.close()

    def get_by_id(self, acu_id: int) -> dict[str, Any] | None:
        conn = self._reader()
        try:
            row = conn.execute(
                f"SELECT {_READ_COLS} FROM acus WHERE id=?", (int(acu_id),)
            ).fetchone()
            return dict(row) if row else None
        finally:
            conn.close()

    def close(self) -> None:
        return None
