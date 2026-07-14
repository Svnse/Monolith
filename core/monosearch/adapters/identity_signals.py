"""identity_signals adapter (spec §5.2 — the `pulling` source). Wraps the
curiosity + emergence signals in core.identity_milestones. This is what makes
`pulling` answerable: it surfaces the REAL current-state signal content (the
per-pull / per-candidate canonical claim), NOT a "run the skill" nudge.

Two signals live in the milestone ledger (deterministic detectors write them;
this adapter only reads):
  * curiosity (`get_latest_curiosity_signal`): {pull_count, top:[{id, canonical,
    pull_strength, confidentity, stability, provenance}], message, detected_at}
  * emergence (`get_latest_emergence_signal`): {candidate_count, new_acu_count,
    threshold, top:[{id, canonical, confidentity, provenance, reinforcement}],
    message, detected_at}

Mapping decisions:
  * ONE Record per top item (curiosity pull / emergence candidate), id =
    `curiosity:<acu_id>` / `emergence:<acu_id>` from the item's own `id` (an ACU
    PK). Per-item is what actually answers `pulling`.
  * text = the item's CANONICAL claim — the real content. The aggregate
    `message` is a count-summary nudge; it is preserved in metadata, never used
    as text.
  * provenance = the item's OWN `provenance` field, mapped
    self/user/world -> SELF/USER/WORLD (default SELF for absent/unknown). The
    producers (core.curiosity / core.identity_emergence) populate it from the
    underlying ACU row, so a claim grounded in user or world signal is mapped
    faithfully rather than hardcoded SELF.
  * evidence_tier = DERIVED (the signal is a computed advisory, not a literal
    record of what happened).
  * recurrence_key = None for every record — these are CURRENT-STATE signals
    (the latest pull set), not recurring events. `pulling` reads the signal's
    own pull-strength ordering; it does NOT go through the salience ledger.
  * ts = epoch(signal['detected_at']) shared across that signal's records.

The ledger getters read JSON on each call (load_ledger), so this is always the
live current signal — no caching, no copy.
"""
from __future__ import annotations

from datetime import datetime

from core import identity_milestones as _m
from core.monosearch.adapter import SourceAdapter
from core.monosearch.record import EvidenceTier, Provenance, Record

_CURIOSITY_PREFIX = "curiosity:"
_EMERGENCE_PREFIX = "emergence:"

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


class IdentitySignalAdapter(SourceAdapter):
    name = "identity_signals"
    evidence_tier = EvidenceTier.DERIVED

    # ── helpers ───────────────────────────────────────────────────────────

    def _recurrence_key(self, item: dict) -> None:
        # Current-state signals are NOT recurrence events: each is the single
        # latest pull set, not a stream of repeating units. Always None ⇒ never
        # salience-eligible; `pulling` uses the signal's own ordering instead.
        return None

    def _provenance(self, item: dict) -> Provenance:
        # Map the item's OWN provenance (the producers populate it from the
        # underlying ACU row) self/user/world -> SELF/USER/WORLD. Default SELF
        # for absent/unknown values — faithful map, not a hardcoded constant.
        prov = str(item.get("provenance", "self")).lower()
        return _PROVENANCE.get(prov, Provenance.SELF)

    def _to_record(self, item: dict, *, prefix: str, signal_kind: str,
                   signal: dict, ts: float | None) -> Record | None:
        acu_id = item.get("id")
        if acu_id is None:
            return None  # never emit "<kind>:None"
        canonical = str(item.get("canonical", "")).strip()
        meta = {
            "signal_kind": signal_kind,
            "acu_id": acu_id,
            "signal_message": signal.get("message"),
            "detected_at": signal.get("detected_at"),
        }
        if signal_kind == "curiosity":
            meta.update({
                "pull_count": signal.get("pull_count"),
                "pull_strength": item.get("pull_strength"),
                "confidentity": item.get("confidentity"),
                "stability": item.get("stability"),
            })
        else:  # emergence
            meta.update({
                "candidate_count": signal.get("candidate_count"),
                "new_acu_count": signal.get("new_acu_count"),
                "threshold": signal.get("threshold"),
                "confidentity": item.get("confidentity"),
                "reinforcement": item.get("reinforcement"),
            })
        return Record(
            namespaced_id=f"{prefix}{acu_id}",
            source=self.name,
            provenance=self._provenance(item),
            recurrence_key=self._recurrence_key(item),
            text=canonical,
            metadata=meta,
            ts=ts,
            evidence_tier=EvidenceTier.DERIVED,
        )

    def _all_records(self) -> list[Record]:
        """Every current-signal record, top-ordering preserved within each
        signal (curiosity first, then emergence)."""
        out: list[Record] = []
        cur = _m.get_latest_curiosity_signal()
        if isinstance(cur, dict):
            ts = _iso_to_epoch(cur.get("detected_at"))
            for item in cur.get("top") or []:
                if isinstance(item, dict):
                    rec = self._to_record(item, prefix=_CURIOSITY_PREFIX,
                                          signal_kind="curiosity", signal=cur, ts=ts)
                    if rec is not None:
                        out.append(rec)
        em = _m.get_latest_emergence_signal()
        if isinstance(em, dict):
            ts = _iso_to_epoch(em.get("detected_at"))
            for item in em.get("top") or []:
                if isinstance(item, dict):
                    rec = self._to_record(item, prefix=_EMERGENCE_PREFIX,
                                          signal_kind="emergence", signal=em, ts=ts)
                    if rec is not None:
                        out.append(rec)
        return out

    # ── SourceAdapter interface ───────────────────────────────────────────

    def search(self, query: str, filters: dict, limit: int) -> list[Record]:
        recs = self._all_records()
        q = (query or "").strip().lower()
        if q:
            recs = [r for r in recs if q in r.text.lower()]
        since = (filters or {}).get("since")
        if since:
            cutoff = _iso_to_epoch(since)
            if cutoff is not None:
                recs = [r for r in recs if r.ts is None or r.ts >= cutoff]
        return recs[:limit]

    def get(self, namespaced_id: str) -> Record | None:
        if not (namespaced_id.startswith(_CURIOSITY_PREFIX)
                or namespaced_id.startswith(_EMERGENCE_PREFIX)):
            return None
        for r in self._all_records():
            if r.namespaced_id == namespaced_id:
                return r
        return None

    def list(self, filters: dict, limit: int) -> list[Record]:
        # The salience.rebuild iteration path. Returns current-signal records;
        # all carry recurrence_key=None, so nothing is salience-eligible (correct
        # — `pulling` is current-state, not a recurrence ledger feed).
        return self._all_records()[:limit]
