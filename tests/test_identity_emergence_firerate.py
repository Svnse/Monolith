"""DONE-GATE for M2 (per spec §11). M2 is not 'done' because unit tests pass —
it is done when this shows the propose-only loop actually FIRES on realistic
identity-relevant claims (and stays silent on noise) under the shipped default
backend. If lexical fires near-zero here, flip MONOLITH_IDENTITY_ALIGN_EMBED.
"""
from __future__ import annotations

import pytest


# Realistic self-derived claims a model would form ABOUT Monolith, scored
# against the ACTUAL shipped Origin-0 seed (core.identity._DEFAULT_IDENTITY).
_IDENTITY_CLAIMS = [
    "monolith | values | precision",
    "monolith | values | provenance",
    "monolith | holds | continuity",
    "monolith | uses | recall",
    "monolith | credits | observation",
    "monolith | respects | authorship",
]
_STRONG_TOKENS = ["precision", "provenance", "continuity", "recall", "observation", "authorship"]

_NOISE_CLAIMS = [
    "weather | is | sunny",
    "python | has | gil",
    "user | clicked | button",
    "server | returned | error",
]
_NOISE_TOKENS = ["sunny", "gil", "button", "error"]


@pytest.fixture
def real_seed_substrate(monkeypatch, tmp_path):
    from core import db_connect as _dbc
    monkeypatch.setattr(_dbc, "DB_PATH", tmp_path / "acatalepsy.sqlite3", raising=True)
    monkeypatch.setenv("MONOLITH_DB_AUTHORIZER_STRICT", "1")
    monkeypatch.setenv("MONOLITH_IDENTITY_EMERGENCE_V1", "1")
    from core.acatalepsy import schema, intake, canonical_log
    schema.migrate()
    for mod in (intake, canonical_log):
        tl = getattr(mod, "_tl", None)
        if tl is not None:
            for a in ("writer_conn", "reader_conn", "writer", "reader"):
                if hasattr(tl, a):
                    delattr(tl, a)
    from core import identity_milestones as m
    monkeypatch.setattr(m, "STORE_PATH", tmp_path / "identity_milestones.json")
    # Use the REAL shipped Origin-0 seed as the corpus.
    from core import identity
    monkeypatch.setattr(identity, "load_identity", lambda: identity._DEFAULT_IDENTITY)
    yield tmp_path


def test_lexical_default_fires_on_identity_and_is_silent_on_noise(real_seed_substrate) -> None:
    from core.acu_store import ACUStore
    from core import identity_emergence as em

    s = ACUStore()
    try:
        for c in _IDENTITY_CLAIMS:
            for _ in range(6):  # reinforce into the STABLE disposition (emergence)
                s.ingest(c, source="model")
        for c in _NOISE_CLAIMS:
            s.ingest(c, source="model")  # fresh + unaligned
    finally:
        s.close()

    # Shipped default backend + default threshold (no overrides except min_new).
    rep = em.detect_emergence(min_new_acus=1)

    assert rep.fired is True, "loop must fire on realistic identity-relevant claims"

    candidate_text = " ".join(c["canonical"] for c in rep.candidates).lower()
    strong_hits = sum(tok in candidate_text for tok in _STRONG_TOKENS)
    noise_hits = sum(tok in candidate_text for tok in _NOISE_TOKENS)

    # A plausible fraction of clearly identity-relevant claims must surface...
    assert strong_hits >= 4, f"only {strong_hits}/6 identity claims surfaced — loop too silent"
    # ...and noise must not leak into the Emergent proposal stream.
    assert noise_hits == 0, f"{noise_hits} noise claim(s) leaked into candidates"
