from __future__ import annotations

from pathlib import Path

from core import identity_acus
from core.acu_store import ACUStore
from core.irp import label_for_claim


def test_extract_origin0_claims_uses_markdown_sections() -> None:
    claims = identity_acus.extract_origin0_claims(
        """# Monolith

## What I value
Precision over fluency.
Provenance over assertion.

## What I refuse
I do not invent confidence.
"""
    )

    assert claims == [
        "Origin 0 / What I value: Precision over fluency. Provenance over assertion.",
        "Origin 0 / What I refuse: I do not invent confidence.",
    ]


def test_extract_origin0_claims_stops_at_emergent_begin() -> None:
    text = (
        "# Monolith\n\n## What I am\nI am Monolith.\n\n"
        "<!-- EMERGENT:BEGIN -->\n## Emergent\n"
        "I lean on adversarial verification.\n<!-- EMERGENT:END -->\n"
    )
    claims = identity_acus.extract_origin0_claims(text)
    assert any("I am Monolith." in c for c in claims)
    assert all("adversarial verification" not in c for c in claims)


def test_origin0_acus_load_locked_and_idempotent(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from core import db_connect as _dbc
    monkeypatch.setattr(_dbc, "DB_PATH", tmp_path / "acatalepsy.sqlite3", raising=True)
    monkeypatch.setenv("MONOLITH_DB_AUTHORIZER_STRICT", "1")
    from core.acatalepsy import schema, intake, canonical_log
    schema.migrate()
    for mod in (intake, canonical_log):
        tl = getattr(mod, "_tl", None)
        if tl is not None:
            for a in ("writer_conn", "reader_conn", "writer", "reader"):
                if hasattr(tl, a):
                    delattr(tl, a)
    monkeypatch.setattr(
        identity_acus,
        "load_identity",
        lambda: "# Monolith\n\n## What I value\nPrecision over fluency.",
    )

    ids = identity_acus.ensure_origin0_acus_loaded()
    assert len(ids) == 1

    store = ACUStore()
    row = store.get_by_id(ids[0])
    assert row is not None
    assert row["source"] == "identity_origin_0"
    assert row["locked"] == 1
    assert row["confidentity"] == 1.0
    assert label_for_claim(row) == "LOCKED"

    # Re-loading is idempotent: same id, no duplicate, still locked + unreinforced.
    ids2 = identity_acus.ensure_origin0_acus_loaded()
    assert ids2 == ids
    assert store.count() == 1
    assert store.get_by_id(ids[0])["reinforcement"] == 1
    store.close()
