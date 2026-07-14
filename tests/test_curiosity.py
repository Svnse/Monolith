from __future__ import annotations

import pytest


@pytest.fixture
def substrate(monkeypatch, tmp_path):
    from core import db_connect as _dbc
    monkeypatch.setattr(_dbc, "DB_PATH", tmp_path / "acatalepsy.sqlite3", raising=True)
    monkeypatch.setenv("MONOLITH_DB_AUTHORIZER_STRICT", "1")
    monkeypatch.setenv("MONOLITH_CURIOSITY_V1", "1")
    from core.acatalepsy import schema, intake, canonical_log
    schema.migrate()
    for mod in (intake, canonical_log):
        tl = getattr(mod, "_tl", None)
        if tl is not None:
            for a in ("writer_conn", "reader_conn", "writer", "reader"):
                if hasattr(tl, a):
                    delattr(tl, a)
    from core import identity_milestones as m, identity
    monkeypatch.setattr(m, "STORE_PATH", tmp_path / "identity_milestones.json")
    monkeypatch.setattr(
        identity, "load_identity",
        lambda: "# Monolith\n\n## What I value\nPrecision over fluency. Naming over narration.",
    )
    yield tmp_path


def _ingest(canonical, source="model", times=1):
    from core.acu_store import ACUStore
    s = ACUStore()
    try:
        acu_id = -1
        for _ in range(times):
            acu_id = s.ingest(canonical, source=source)
        return acu_id
    finally:
        s.close()


def test_detect_pulls_fires_on_fresh_aligned(substrate) -> None:
    from core import curiosity
    _ingest("monolith | values | precision")       # fresh + aligned -> pull
    _ingest("weather | is | sunny")                 # fresh + unaligned -> excluded

    rep = curiosity.detect_pulls(align_threshold=0.2)

    assert rep.fired is True
    canons = [p["canonical"] for p in rep.pulls]
    assert any("precision" in c for c in canons)
    assert all("weather" not in c for c in canons)
    assert all(p["stability"] < 0.5 for p in rep.pulls)  # curiosity = fresh disposition


def test_detect_pulls_excludes_stable_claims(substrate) -> None:
    from core import curiosity
    _ingest("monolith | values | precision", times=6)  # reinforced -> stable -> emergence, NOT curiosity

    rep = curiosity.detect_pulls(align_threshold=0.2)

    assert all("precision" not in p["canonical"] for p in rep.pulls)


def test_detect_pulls_retires_after_surface_cap(substrate) -> None:
    from core import curiosity
    _ingest("monolith | values | precision")

    r1 = curiosity.detect_pulls(align_threshold=0.2, surface_cap=2)
    r2 = curiosity.detect_pulls(align_threshold=0.2, surface_cap=2)
    r3 = curiosity.detect_pulls(align_threshold=0.2, surface_cap=2)

    assert r1.fired is True
    assert r2.fired is True
    assert r3.fired is False  # retired after surfacing twice — no infinite resurfacing


def test_detect_pulls_emits_canonical_log_and_signal(substrate) -> None:
    from core import curiosity, identity_milestones as m
    from core.acatalepsy import canonical_log
    _ingest("monolith | values | precision")

    curiosity.detect_pulls(align_threshold=0.2)

    kinds = [e.kind for e in canonical_log.read_since(0, limit=500)]
    assert "curiosity_pull_detected" in kinds
    assert m.get_latest_curiosity_signal() is not None


def test_detect_pulls_silent_when_flag_off(substrate, monkeypatch) -> None:
    from core import curiosity
    monkeypatch.setenv("MONOLITH_CURIOSITY_V1", "0")
    _ingest("monolith | values | precision")
    rep = curiosity.detect_pulls(align_threshold=0.2)
    assert rep.fired is False
    # but explicit force bypasses the dark flag
    assert curiosity.detect_pulls(align_threshold=0.2, force=True).fired is True


def test_detect_pulls_excludes_claims_already_in_corpus(substrate, monkeypatch) -> None:
    """A claim already reflected verbatim in the identity corpus is not a
    curiosity pull (parity with emergence's dedup)."""
    from core import curiosity, identity
    from core.acu_store import ACUStore
    acu_id = _ingest("monolith | values | precision")
    row = None
    s = ACUStore()
    try:
        row = s.get_by_id(acu_id)
    finally:
        s.close()
    canon = row["canonical"]
    monkeypatch.setattr(identity, "load_identity", lambda: f"# Monolith\n## seed\n{canon}\n")
    rep = curiosity.detect_pulls(align_threshold=0.0, force=True)  # only corpus-dedup can exclude
    assert all(p["canonical"] != canon for p in rep.pulls)


def test_detect_pulls_clears_stale_signal_when_detection_empties(substrate) -> None:
    """Regression (2026-06-03 curiosity incident): a later detection that finds
    nothing must clear the stored signal. Otherwise the Observer keeps surfacing
    a ghost ("7 pulls") that contradicts the live tool ("0 pulls")."""
    from core import curiosity, identity_milestones as m
    _ingest("monolith | values | precision")
    r1 = curiosity.detect_pulls(align_threshold=0.2, force=True, surface_cap=1)
    assert r1.fired is True
    assert m.get_latest_curiosity_signal() is not None        # signal stored on fire
    # Surfaced once -> retired at cap=1 -> next real detection finds nothing.
    r2 = curiosity.detect_pulls(align_threshold=0.2, force=True, surface_cap=1)
    assert r2.fired is False
    assert m.get_latest_curiosity_signal() is None            # <-- stale signal cleared


def test_detect_pulls_empty_does_not_clear_on_disabled_flag(substrate, monkeypatch) -> None:
    """The dark-flag early-return means we never looked — it must NOT clobber a
    previously-stored signal (only a real, looked-and-found-nothing run clears)."""
    from core import curiosity, identity_milestones as m
    _ingest("monolith | values | precision")
    curiosity.detect_pulls(align_threshold=0.2, force=True)   # store a signal
    assert m.get_latest_curiosity_signal() is not None
    monkeypatch.setenv("MONOLITH_CURIOSITY_V1", "0")
    rep = curiosity.detect_pulls(align_threshold=0.2)         # disabled: didn't look
    assert rep.fired is False
    assert m.get_latest_curiosity_signal() is not None        # signal preserved


def test_latest_surfaceable_signal_keeps_live_pull(substrate) -> None:
    """A signal with at least one not-yet-retired pull is still surfaceable."""
    from core import curiosity
    _ingest("monolith | values | precision")
    curiosity.detect_pulls(align_threshold=0.2, force=True, surface_cap=3)  # surfaced=1 < 3
    assert curiosity.latest_surfaceable_signal(surface_cap=3) is not None


def test_latest_surfaceable_signal_hides_fully_retired(substrate) -> None:
    """A signal whose pulls have all retired is a ghost — suppress it even
    though the raw ledger still holds it (read-site liveness gate). Pure read."""
    from core import curiosity, identity_milestones as m
    _ingest("monolith | values | precision")
    # Surface up to the cap so the pull retires.
    curiosity.detect_pulls(align_threshold=0.2, force=True, surface_cap=2)  # surfaced 0->1
    curiosity.detect_pulls(align_threshold=0.2, force=True, surface_cap=2)  # surfaced 1->2 (==cap)
    assert m.get_latest_curiosity_signal() is not None          # raw ledger still has it
    assert curiosity.latest_surfaceable_signal(surface_cap=2) is None  # but it's a ghost


def test_observer_does_not_surface_retired_curiosity_signal(substrate) -> None:
    """End-to-end: the Observer's curiosity read is gated on liveness, so a
    fully-retired signal never reaches the [OBSERVER] block."""
    from core import curiosity
    from addons.system.observer import runtime
    _ingest("monolith | values | precision")
    for _ in range(3):                                          # default cap=3 -> retire
        curiosity.detect_pulls(align_threshold=0.2, force=True)
    assert runtime._latest_curiosity_signal() is None


def test_pull_graduates_to_emergence_when_reinforced(substrate) -> None:
    """A claim transitions curiosity -> emergence as it reinforces (the
    transitive proof of the unification)."""
    from core import curiosity, identity_emergence as em
    _ingest("monolith | values | precision")  # fresh
    assert curiosity.detect_pulls(align_threshold=0.2, force=True).fired is True
    assert em.detect_emergence(min_new_acus=1, threshold_confidentity=0.2, force=True).fired is False

    _ingest("monolith | values | precision", times=6)  # reinforce -> stable
    cur = curiosity.detect_pulls(align_threshold=0.2, force=True)
    assert all("precision" not in p["canonical"] for p in cur.pulls)  # left curiosity
    assert em.detect_emergence(min_new_acus=1, threshold_confidentity=0.2, force=True).fired is True  # entered emergence
