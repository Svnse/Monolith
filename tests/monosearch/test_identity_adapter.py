"""IdentityAdapter (spec §5.2 — identity source).

Two surfaces, one adapter:
  (a) identity.md corpus paragraphs  -> Record id `identity:<region>/<slug>`
  (b) the LOCKED Origin-0 ACU rows   -> Record id `acu:<id>`  (the rows the
      acus adapter EXCLUDES; identity OWNS them — no double-serving).

provenance=SELF (constant), evidence_tier=DERIVED, recurrence_key=None
(self-knowledge / lookup-only surface; each entry is unique).

The corpus tests are pure (monkeypatch `load_identity`). One end-to-end test
hits the REAL locked-ACU read path: it seeds a real `acus` table via the
acatalepsy schema/isolation pattern (monkeypatch `core.db_connect.DB_PATH`),
inserts a `locked=1` row, and asserts the adapter surfaces it as `acu:<id>`
while NOT surfacing an unlocked row.
"""
from __future__ import annotations

import threading

from core.monosearch.adapters.identity import IdentityAdapter
from core.monosearch.record import EvidenceTier, Provenance


_SAMPLE_IDENTITY = """\
# Monolith — origin 0

## What I am

I am Monolith. An AI operating system.
Not E, not an emulation of E.

## What I refuse

I do not perform E.

EMERGENT:BEGIN

## What I learned

Push-back as audit is load-bearing.
"""


# ── corpus surface (pure) ──────────────────────────────────────────────────

def test_corpus_paragraphs_become_records(monkeypatch):
    import core.monosearch.adapters.identity as mod
    monkeypatch.setattr(mod, "load_identity", lambda: _SAMPLE_IDENTITY)
    a = IdentityAdapter()
    recs = a.list({}, 100)
    corpus = [r for r in recs if r.namespaced_id.startswith("identity:")]
    assert corpus, "no corpus records produced"
    for r in corpus:
        assert r.source == "identity"
        assert r.provenance is Provenance.SELF
        assert r.evidence_tier == EvidenceTier.DERIVED
        assert r.recurrence_key is None  # lookup-only, each unique
        assert r.namespaced_id.startswith("identity:")
        assert r.text


def test_corpus_skips_region_control_sentinel(monkeypatch):
    # split_regions slices the emergent region at the START of the EMERGENT:BEGIN
    # line, so that line literally IS the sentinel — it must NOT become a record
    # (it is structural marker, not self-description content).
    import core.monosearch.adapters.identity as mod
    from core.identity_regions import EMERGENT_BEGIN, EMERGENT_END
    monkeypatch.setattr(mod, "load_identity", lambda: _SAMPLE_IDENTITY)
    a = IdentityAdapter()
    corpus = [r for r in a.list({}, 100) if r.namespaced_id.startswith("identity:")]
    for r in corpus:
        assert r.text.strip() != EMERGENT_BEGIN
        assert r.text.strip() != EMERGENT_END
        assert EMERGENT_BEGIN not in r.text
        assert EMERGENT_END not in r.text
    # the real emergent CONTENT still surfaces (skipping the sentinel must not
    # drop the paragraph that follows it).
    assert any("Push-back as audit" in r.text for r in corpus)


def test_corpus_id_encodes_region_and_slug(monkeypatch):
    import core.monosearch.adapters.identity as mod
    monkeypatch.setattr(mod, "load_identity", lambda: _SAMPLE_IDENTITY)
    a = IdentityAdapter()
    recs = [r for r in a.list({}, 100) if r.namespaced_id.startswith("identity:")]
    ids = {r.namespaced_id for r in recs}
    # origin0 + emergent regions both appear; id form is identity:<region>/<slug>
    regions = {rid.split(":", 1)[1].split("/", 1)[0] for rid in ids}
    assert "origin0" in regions
    assert "emergent" in regions
    # the emergent paragraph is tagged emergent, not origin0
    emergent = [r for r in recs if r.namespaced_id.split(":", 1)[1].startswith("emergent/")]
    assert any("Push-back as audit" in r.text for r in emergent)
    # origin0 paragraphs carry the seed prose
    origin0 = [r for r in recs if r.namespaced_id.split(":", 1)[1].startswith("origin0/")]
    assert any("I am Monolith" in r.text for r in origin0)


def test_corpus_ids_are_deterministic(monkeypatch):
    import core.monosearch.adapters.identity as mod
    monkeypatch.setattr(mod, "load_identity", lambda: _SAMPLE_IDENTITY)
    a = IdentityAdapter()
    ids1 = [r.namespaced_id for r in a.list({}, 100) if r.namespaced_id.startswith("identity:")]
    ids2 = [r.namespaced_id for r in a.list({}, 100) if r.namespaced_id.startswith("identity:")]
    assert ids1 == ids2
    assert len(ids1) == len(set(ids1)), "corpus ids must be unique"


def test_get_corpus_round_trips(monkeypatch):
    import core.monosearch.adapters.identity as mod
    monkeypatch.setattr(mod, "load_identity", lambda: _SAMPLE_IDENTITY)
    a = IdentityAdapter()
    rec = next(r for r in a.list({}, 100) if r.namespaced_id.startswith("identity:"))
    got = a.get(rec.namespaced_id)
    assert got is not None
    assert got.namespaced_id == rec.namespaced_id
    assert got.text == rec.text


def test_search_filters_corpus_by_keyword(monkeypatch):
    import core.monosearch.adapters.identity as mod
    monkeypatch.setattr(mod, "load_identity", lambda: _SAMPLE_IDENTITY)
    a = IdentityAdapter()
    hits = a.search("perform", {}, 50)
    corpus = [r for r in hits if r.namespaced_id.startswith("identity:")]
    assert corpus, "keyword search returned no corpus hit"
    assert all("perform" in r.text.lower() for r in corpus)


def test_get_rejects_unknown_prefix(monkeypatch):
    import core.monosearch.adapters.identity as mod
    monkeypatch.setattr(mod, "load_identity", lambda: _SAMPLE_IDENTITY)
    a = IdentityAdapter()
    assert a.get("fault:10") is None
    assert a.get("identity:origin0/does-not-exist") is None


# ── locked-ACU surface (REAL end-to-end) ───────────────────────────────────

def _seed_acus_table(monkeypatch, tmp_path):
    """Create a real acatalepsy DB with a full acus schema at a temp path."""
    from core import db_connect as _dbc
    db_path = tmp_path / "test_identity_acus.sqlite3"
    monkeypatch.setattr(_dbc, "DB_PATH", db_path, raising=True)
    monkeypatch.setenv("MONOLITH_DB_AUTHORIZER_STRICT", "1")
    from core.acatalepsy import schema
    schema.migrate()
    return db_path


def test_locked_origin0_acu_rows_are_surfaced(monkeypatch, tmp_path):
    import core.monosearch.adapters.identity as mod
    monkeypatch.setattr(mod, "load_identity", lambda: _SAMPLE_IDENTITY)  # hermetic
    _seed_acus_table(monkeypatch, tmp_path)
    from core.acu_store import ACUStore
    store = ACUStore()
    locked_id = store.ingest_locked(
        "Origin 0 / What I am: I am Monolith.",
        source="identity_origin_0",
        lock_reason="origin_0",
        confidentity=1.0,
    )
    # an UNLOCKED row that identity must NOT own (the acus adapter serves it).
    # Must be an ATOMIC triple — prose like "the sky is blue" fails the
    # atomicity gate and returns id<=0, which would make the exclusion
    # assertion below vacuous.
    unlocked_id = store.ingest("sky | has_color | blue", source="model")
    store.close()
    assert unlocked_id > 0, "atomic-triple seed must pass the gate (non-vacuous)"

    a = IdentityAdapter()
    recs = a.list({}, 200)
    acu_recs = [r for r in recs if r.namespaced_id.startswith("acu:")]
    ids = {r.namespaced_id for r in acu_recs}

    assert f"acu:{locked_id}" in ids, "locked Origin-0 row not surfaced"
    assert f"acu:{unlocked_id}" not in ids, "unlocked row leaked into identity"

    for r in acu_recs:
        assert r.source == "identity"
        assert r.provenance is Provenance.SELF
        assert r.evidence_tier == EvidenceTier.DERIVED
        assert r.recurrence_key is None
        assert r.ts is not None  # parsed from created_at


def test_get_locked_acu_by_id_and_rejects_unlocked(monkeypatch, tmp_path):
    import core.monosearch.adapters.identity as mod
    monkeypatch.setattr(mod, "load_identity", lambda: _SAMPLE_IDENTITY)  # hermetic
    _seed_acus_table(monkeypatch, tmp_path)
    from core.acu_store import ACUStore
    store = ACUStore()
    locked_id = store.ingest_locked(
        "Origin 0 / What I refuse: I do not perform E.",
        source="identity_origin_0",
        lock_reason="origin_0",
    )
    # ATOMIC triple so the seed passes the atomicity gate (id>0); prose would
    # return id<=0 and make the exclusion assertion below vacuous.
    unlocked_id = store.ingest("grass | has_color | green", source="model")
    store.close()
    assert unlocked_id > 0, "atomic-triple seed must pass the gate (non-vacuous)"

    a = IdentityAdapter()
    got = a.get(f"acu:{locked_id}")
    assert got is not None
    assert got.namespaced_id == f"acu:{locked_id}"
    assert "perform E" in got.text

    assert a.get(f"acu:{unlocked_id}") is None, "identity served an unlocked acu"


def test_locked_merged_row_is_still_owned_by_identity(monkeypatch, tmp_path):
    """Exhaustive-partition regression: identity owns EVERY locked row, including
    a locked+merged one. The acus adapter excludes ANY truthy `locked`, so if
    identity re-added a `merged_into IS NULL` condition a locked+merged row would
    be owned by NEITHER adapter. Both list() and get() must still surface it."""
    import core.monosearch.adapters.identity as mod
    monkeypatch.setattr(mod, "load_identity", lambda: _SAMPLE_IDENTITY)  # hermetic
    _seed_acus_table(monkeypatch, tmp_path)
    from core.acu_store import ACUStore
    store = ACUStore()
    locked_id = store.ingest_locked(
        "Origin 0 / What I am: I am Monolith.",
        source="identity_origin_0",
        lock_reason="origin_0",
    )
    store.close()

    # Mark the locked row as merged (the real merge path sets merged_into +
    # state='archived'); write through the SAME isolated DB the adapter reads.
    # role="migration" is the legitimate authorizer-bypass for an admin/test write.
    from core.db_connect import connect_acatalepsy
    conn = connect_acatalepsy(role="migration")
    try:
        conn.execute(
            "UPDATE acus SET merged_into=?, state='archived' WHERE id=?",
            (locked_id, locked_id),
        )
        conn.commit()
    finally:
        conn.close()

    a = IdentityAdapter()
    ids = {r.namespaced_id for r in a.list({}, 200)}
    assert f"acu:{locked_id}" in ids, "locked+merged row not owned by identity (list)"
    assert a.get(f"acu:{locked_id}") is not None, \
        "locked+merged row not owned by identity (get)"
