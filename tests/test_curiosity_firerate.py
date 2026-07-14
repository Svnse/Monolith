"""DONE-GATE for M3 (spec §3). M3 is done when the curiosity loop FIRES on
fresh identity-relevant claims AND emergence is EMPTY on the same fresh data —
proving the two dispositions are exclusive, not duplicative.
"""
from __future__ import annotations

import pytest

_IDENTITY_CLAIMS = [
    "monolith | values | precision",
    "monolith | holds | continuity",
    "monolith | credits | observation",
    "monolith | respects | authorship",
]
_STRONG_TOKENS = ["precision", "continuity", "observation", "authorship"]
_NOISE_CLAIMS = ["weather | is | sunny", "python | has | gil", "server | returned | error"]
_NOISE_TOKENS = ["sunny", "gil", "error"]


@pytest.fixture
def real_seed_substrate(monkeypatch, tmp_path):
    from core import db_connect as _dbc
    monkeypatch.setattr(_dbc, "DB_PATH", tmp_path / "acatalepsy.sqlite3", raising=True)
    monkeypatch.setenv("MONOLITH_DB_AUTHORIZER_STRICT", "1")
    monkeypatch.setenv("MONOLITH_CURIOSITY_V1", "1")
    monkeypatch.setenv("MONOLITH_IDENTITY_EMERGENCE_V1", "1")
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
    monkeypatch.setattr(identity, "load_identity", lambda: identity._DEFAULT_IDENTITY)
    yield tmp_path


def test_fresh_identity_claims_fire_curiosity_and_not_emergence(real_seed_substrate) -> None:
    from core.acu_store import ACUStore
    from core import curiosity, identity_emergence as em

    s = ACUStore()
    try:
        for c in _IDENTITY_CLAIMS + _NOISE_CLAIMS:
            s.ingest(c, source="model")  # fresh (reinforcement=1)
    finally:
        s.close()

    cur = curiosity.detect_pulls(align_threshold=0.2, force=True)
    assert cur.fired is True, "curiosity must fire on fresh identity-relevant claims"
    text = " ".join(p["canonical"] for p in cur.pulls).lower()
    assert sum(t in text for t in _STRONG_TOKENS) >= 3, "too few identity pulls surfaced"
    assert sum(t in text for t in _NOISE_TOKENS) == 0, "noise leaked into pulls"

    # Exclusivity: the SAME fresh data must produce NO emergence candidates
    # (nothing is stable enough to consolidate yet).
    emr = em.detect_emergence(min_new_acus=1, force=True)
    assert emr.fired is False, "fresh claims must not be emergence candidates (dispositions exclusive)"
