"""acatalepsy-acus adapter (spec §5, row `acatalepsy-acus`).

Wraps ``ACUStore.search`` / ``get_by_id`` / ``retrieve`` (core.acu_store). Those
reads return plain dict rows whose keys are exactly ``acu_store._READ_COLS`` —
which this adapter required be extended to also project ``eqid`` and ``cid``
(both real TEXT columns on the ``acus`` table; see schema.py ``_SPINE_COLUMNS``).

Two source-faithfulness rules baked in (spec §5/§5.1):
  * recurrence_key = ``eqid`` if present else ``cid`` else ``None``. Both are
    dormant/empty on the cold day-1 corpus (no cids/eqids minted yet), so the
    adapter contributes ZERO recurrence day-1 — that is correct, not a bug. Once
    acatalepsy accrual turns on it gets denser for free.
  * The adapter EXCLUDES ``locked=1`` Origin-0 rows itself — identity owns those
    (the LOCKED ACUs); MonoSearch must not double-serve their ``acu:<id>``. The
    store has no filter for this, so we fetch then drop in Python.

provenance comes from the row's own ``provenance`` column (already in
``_READ_COLS``); ``ts`` is parsed from the ISO ``last_seen`` (the most recent
touch — the temporal anchor for decay/filters). evidence_tier = DERIVED (an ACU
is an interpreted claim, not a literal event record).
"""
from __future__ import annotations

from datetime import datetime
from typing import Any

from core.acu_store import ACUStore
from core.monosearch.adapter import SourceAdapter
from core.monosearch.record import EvidenceTier, Provenance, Record

_ID_PREFIX = "acu:"

_PROVENANCE = {
    "self": Provenance.SELF,
    "user": Provenance.USER,
    "world": Provenance.WORLD,
}


def _iso_to_epoch(s: str | None) -> float | None:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s).timestamp()
    except (ValueError, TypeError):
        return None


class AcuAdapter(SourceAdapter):
    name = "acatalepsy-acus"
    evidence_tier = EvidenceTier.DERIVED

    def _reader(self) -> ACUStore:
        # Construct per call (ACUStore is a thin stateless facade over the global
        # substrate; its __init__ does no work). Constructing fresh keeps the read
        # bound to the live db_connect.DB_PATH and honours test isolation.
        return ACUStore()

    # ── internal helpers ──────────────────────────────────────────────

    @staticmethod
    def _is_locked(row: dict[str, Any]) -> bool:
        # Origin-0 LOCKED rows are owned by identity; exclude them here. The store
        # cannot push this filter, so we fetch then drop in Python (spec §5.1).
        return bool(row.get("locked"))

    def _provenance(self, row: dict[str, Any]) -> Provenance:
        prov = (row.get("provenance") or "").strip().lower()
        return _PROVENANCE.get(prov, Provenance.SELF)

    def _recurrence_key(self, row: dict[str, Any]) -> str | None:
        # SEAL: pure self-reinforcement (kind=self AND provenance=self) is NOT
        # salience-eligible. Self-identity-memory L2 ACUs mint cid/eqid, which would
        # otherwise become a recurrence_key and surface a self-repeated belief on the
        # live `recurring()` model tool at FULL parity (salience deliberately bypasses
        # the self-suppression provenance multiplier). Returning None keeps such a row
        # lookup-only. user/world-sourced self-facts keep their key (externally grounded).
        if ((str(row.get("kind", "") or "").strip().lower() == "self")
                and (str(row.get("provenance", "") or "").strip().lower() == "self")):
            return None
        # eqid (deterministic inverse-relation equivalence group) is the right
        # aggregation unit when present; cid (crystallized claim id) is the
        # fallback. Both are dormant on the cold corpus -> None (lookup-only,
        # not salience-eligible). No LLM, no hashing — the store mints these.
        eqid = (row.get("eqid") or "").strip()
        if eqid:
            return eqid
        cid = (row.get("cid") or "").strip()
        if cid:
            return cid
        return None

    def _to_record(self, row: dict[str, Any]) -> Record:
        return Record(
            namespaced_id=f"{_ID_PREFIX}{row['id']}",
            source=self.name,
            provenance=self._provenance(row),
            recurrence_key=self._recurrence_key(row),
            text=str(row.get("canonical") or ""),
            metadata={
                "kind": row.get("kind"),
                "l_level": row.get("l_level"),
                "state": row.get("state"),
                "truth": row.get("truth"),
                "provenance": row.get("provenance"),
                "reinforcement": row.get("reinforcement"),
                "eqid": row.get("eqid"),
                "cid": row.get("cid"),
            },
            ts=_iso_to_epoch(row.get("last_seen")),
            evidence_tier=EvidenceTier.DERIVED,
        )

    # ── public contract ───────────────────────────────────────────────

    def search(self, query: str, filters: dict, limit: int) -> list[Record]:
        # ACUStore.search can't push filters/since -> fetch then filter in Python.
        # Over-fetch so the locked-exclusion doesn't shrink results below `limit`.
        rows = self._reader().search(query or "", limit=max(int(limit) * 2, int(limit)))
        out = [self._to_record(r) for r in rows if not self._is_locked(r)]
        return out[: int(limit)]

    def get(self, namespaced_id: str) -> Record | None:
        if not namespaced_id.startswith(_ID_PREFIX):
            return None
        try:
            acu_id = int(namespaced_id[len(_ID_PREFIX):])
        except ValueError:
            return None
        row = self._reader().get_by_id(acu_id)
        if row is None or self._is_locked(row):
            return None
        return self._to_record(row)

    def list(self, filters: dict, limit: int) -> list[Record]:
        # The iteration path salience.rebuild uses. retrieve() returns the active,
        # ranked set (excludes merged/archived); over-fetch then drop locked rows.
        rows = self._reader().retrieve(limit=max(int(limit) * 2, int(limit)))
        out = [self._to_record(r) for r in rows if not self._is_locked(r)]
        return out[: int(limit)]
