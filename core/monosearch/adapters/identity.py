"""identity adapter (spec §5.2). The self-knowledge / `unresolved` surface.

Two surfaces, one adapter (no double-serving anywhere):
  (a) the identity.md CORPUS — region-aware paragraphs from `core.identity` /
      `core.identity_regions`, as Records id `identity:<region>/<slug>`.
  (b) the LOCKED Origin-0 ACU rows — the `acus` rows with `locked=1`, which the
      acus adapter EXCLUDES. Identity OWNS them: Records id `acu:<id>`.

provenance is a constant SELF (the corpus and the locked seed are both the
runtime's own self-description). evidence_tier is DERIVED — identity is the
seed/interpretation layer, not a literal event record. recurrence_key is always
None: this is a lookup-only self-knowledge surface; every paragraph and every
locked claim is unique, so none is salience-eligible (LESSON 1 — return None for
records that are each unique).

The locked rows are read DIRECTLY (SELECT-only, `WHERE locked=1`) rather than
through `ACUStore.search`/`retrieve` — those rank by behavioural signal and never
filter on `locked`, so a direct read is the faithful map of "the Origin-0 seed
rows" (LESSON 2). We do NOT edit acu_store: the acus agent owns `_READ_COLS`.
"""
from __future__ import annotations

from datetime import datetime

from core.identity import load_identity
from core.identity_regions import EMERGENT_BEGIN, EMERGENT_END, split_regions
from core.monosearch.adapter import SourceAdapter
from core.monosearch.record import EvidenceTier, Provenance, Record

_CORPUS_PREFIX = "identity:"
_ACU_PREFIX = "acu:"

# Only the seed rows identity owns. The acus adapter serves everything else.
_LOCKED_READ_COLS = "id, canonical, source, provenance, kind, state, created_at, locked, lock_reason"


def _iso_to_epoch(s: str | None) -> float | None:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s).timestamp()
    except (ValueError, TypeError):
        return None


def _slug(section: str, index: int) -> str:
    """Deterministic, human-legible slug from a section header + paragraph index.
    Same identity text always yields the same id (LESSON: ids must be stable)."""
    base = "".join(c.lower() if c.isalnum() else "-" for c in (section or "").strip())
    base = "-".join(p for p in base.split("-") if p) or "identity"
    return f"{base}-{index}"


def _iter_corpus_paragraphs(text: str):
    """Yield (region, section, paragraph_index, body) for each prose paragraph.

    Region split mirrors `identity_regions.split_regions` (origin0 vs emergent);
    section/paragraph splitting mirrors `identity_acus.extract_origin0_claims` so
    the corpus surface stays a faithful map of how identity is already parsed.

    `split_regions` slices the emergent region at the START of the EMERGENT:BEGIN
    line, so that region's first line literally IS the sentinel. Any line whose
    stripped form equals a region control sentinel (EMERGENT:BEGIN/EMERGENT:END)
    is skipped — it is structural marker, not self-description content.
    """
    _SENTINELS = {EMERGENT_BEGIN, EMERGENT_END}
    origin0, emergent = split_regions(text)
    for region, region_text in (("origin0", origin0), ("emergent", emergent)):
        section = "Identity"
        idx = 0
        paragraph: list[str] = []

        def flush():
            nonlocal idx
            if not paragraph:
                return None
            body = " ".join(p.strip() for p in paragraph if p.strip()).strip()
            paragraph.clear()
            if not body:
                return None
            out = (region, section, idx, body)
            idx += 1
            return out

        for raw_line in region_text.splitlines():
            line = raw_line.strip()
            if line in _SENTINELS:
                # Region control sentinel — structural, not content. Treat it as
                # a paragraph boundary so it neither joins nor becomes a record.
                got = flush()
                if got:
                    yield got
                continue
            if not line:
                got = flush()
                if got:
                    yield got
                continue
            if line.startswith("#"):
                got = flush()
                if got:
                    yield got
                section = line.lstrip("#").strip() or section
                continue
            paragraph.append(line)
        got = flush()
        if got:
            yield got


class IdentityAdapter(SourceAdapter):
    name = "identity"
    evidence_tier = EvidenceTier.DERIVED

    # ── shared per-source rules ────────────────────────────────────────────

    def _provenance(self, _native=None) -> Provenance:
        # Constant SELF: both the corpus and the locked Origin-0 seed are the
        # runtime's own self-description.
        return Provenance.SELF

    def _recurrence_key(self, _native=None) -> None:
        # Lookup-only self-knowledge surface; each paragraph / locked claim is
        # unique, so none is salience-eligible.
        return None

    # ── corpus surface ─────────────────────────────────────────────────────

    def _corpus_record(self, region: str, section: str, index: int, body: str) -> Record:
        return Record(
            namespaced_id=f"{_CORPUS_PREFIX}{region}/{_slug(section, index)}",
            source=self.name,
            provenance=self._provenance(),
            recurrence_key=self._recurrence_key(),
            text=body,
            metadata={"region": region, "section": section, "kind": "corpus"},
            ts=None,  # identity.md prose carries no per-paragraph timestamp
            evidence_tier=EvidenceTier.DERIVED,
        )

    def _corpus_records(self) -> list[Record]:
        text = load_identity()
        return [
            self._corpus_record(region, section, index, body)
            for (region, section, index, body) in _iter_corpus_paragraphs(text)
        ]

    # ── locked-ACU surface (the rows the acus adapter excludes) ─────────────

    def _to_record(self, row) -> Record:
        # row is an sqlite3.Row from the locked-acu read.
        return Record(
            namespaced_id=f"{_ACU_PREFIX}{row['id']}",
            source=self.name,
            provenance=self._provenance(row),
            recurrence_key=self._recurrence_key(row),
            text=str(row["canonical"]),
            metadata={
                "kind": row["kind"],
                "state": row["state"],
                "source": row["source"],
                "provenance": row["provenance"],
                "lock_reason": row["lock_reason"],
            },
            ts=_iso_to_epoch(row["created_at"]),
            evidence_tier=EvidenceTier.DERIVED,
        )

    def _read_locked_rows(self, limit: int | None = None) -> list:
        """Direct SELECT-only read of the LOCKED Origin-0 rows (the acus adapter
        drops these; identity owns them). Adopts the agent_server SELECT-only
        pattern: a fixed column allowlist, no user-supplied SQL, reader role.
        Returns [] gracefully if the acus table is absent (cold day-1)."""
        from core.db_connect import connect_acatalepsy
        # Own EVERY locked row so the partition is exhaustive: the acus adapter
        # excludes ANY truthy `locked` (bool(row['locked'])), so a locked+merged
        # row must still be owned HERE — do NOT re-add a `merged_into IS NULL`
        # condition or such a row would be owned by neither adapter.
        sql = (
            f"SELECT {_LOCKED_READ_COLS} FROM acus "
            "WHERE locked=1 ORDER BY id ASC"
        )
        params: tuple = ()
        if limit is not None:
            sql += " LIMIT ?"
            params = (int(limit),)
        conn = connect_acatalepsy(role="reader")
        try:
            return list(conn.execute(sql, params).fetchall())
        except Exception:
            return []  # no acus table yet (cold start) — nothing to own
        finally:
            conn.close()

    def _locked_records(self, limit: int | None = None) -> list[Record]:
        return [self._to_record(r) for r in self._read_locked_rows(limit)]

    def _get_locked(self, acu_id: int) -> Record | None:
        from core.db_connect import connect_acatalepsy
        conn = connect_acatalepsy(role="reader")
        try:
            row = conn.execute(
                f"SELECT {_LOCKED_READ_COLS} FROM acus "
                "WHERE id=? AND locked=1",
                (int(acu_id),),
            ).fetchone()
        except Exception:
            row = None
        finally:
            conn.close()
        return self._to_record(row) if row is not None else None

    # ── public API ─────────────────────────────────────────────────────────

    def search(self, query: str, filters: dict, limit: int) -> list[Record]:
        q = (query or "").strip().lower()
        recs = self.list(filters, limit if not q else max(limit, 500))
        if q:
            recs = [r for r in recs if q in r.text.lower()]
        return recs[:limit]

    def get(self, namespaced_id: str) -> Record | None:
        if namespaced_id.startswith(_ACU_PREFIX):
            try:
                acu_id = int(namespaced_id[len(_ACU_PREFIX):])
            except ValueError:
                return None
            return self._get_locked(acu_id)
        if namespaced_id.startswith(_CORPUS_PREFIX):
            for r in self._corpus_records():
                if r.namespaced_id == namespaced_id:
                    return r
            return None
        return None

    def list(self, filters: dict, limit: int) -> list[Record]:
        # Iteration path salience.rebuild uses. Both surfaces; recurrence_key is
        # None throughout, so identity contributes nothing to the salience ledger
        # (correct — it is a reference surface, not a recurring-signal one).
        out: list[Record] = list(self._corpus_records())
        if len(out) < limit:
            out.extend(self._locked_records(limit - len(out)))
        return out[:limit]
