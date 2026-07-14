from __future__ import annotations


import pytest


@pytest.fixture
def substrate(monkeypatch, tmp_path):
    from core import db_connect as _dbc
    monkeypatch.setattr(_dbc, "DB_PATH", tmp_path / "acatalepsy.sqlite3", raising=True)
    monkeypatch.setenv("MONOLITH_DB_AUTHORIZER_STRICT", "1")
    monkeypatch.setenv("MONOLITH_CURIOSITY_CAPTURE_V1", "1")
    monkeypatch.setenv("MONOLITH_CURIOSITY_V1", "1")
    monkeypatch.setenv("MONOLITH_IDENTITY_EMERGENCE_V1", "1")

    from core.acatalepsy import canonical_log, intake, schema
    schema.migrate()
    for mod in (intake, canonical_log):
        tl = getattr(mod, "_tl", None)
        if tl is not None:
            for attr in ("writer_conn", "reader_conn", "writer", "reader"):
                if hasattr(tl, attr):
                    delattr(tl, attr)

    from core import identity, identity_milestones as milestones
    monkeypatch.setattr(milestones, "STORE_PATH", tmp_path / "identity_milestones.json")
    monkeypatch.setattr(
        identity,
        "load_identity",
        lambda: "# Monolith\n\n## What I value\nPrecision over fluency. Naming over narration.",
    )
    yield tmp_path


def test_extract_curiosity_claims_accepts_only_monolith_triples() -> None:
    from core.curiosity_capture import extract_curiosity_claims

    claims, rejected = extract_curiosity_claims(
        """
        visible
        <curiosity>
        - Monolith | is curious_about | why precision feels load-bearing
        - user | wants | identity drift
        - monolith | compound | a and b
        </curiosity>
        """
    )

    assert claims == ("monolith | is curious_about | why precision feels load-bearing",)
    assert rejected == 2


def test_strip_curiosity_block_preserves_terminal_frame() -> None:
    from core.curiosity_capture import strip_curiosity_blocks

    clean = strip_curiosity_blocks(
        "Visible answer\n"
        "<curiosity>\n"
        "monolith | values | precision\n"
        "</curiosity>\n"
        "<frame>working on identity curiosity capture</frame>"
    )

    assert clean == "Visible answer\n<frame>working on identity curiosity capture</frame>"


def test_capture_ingests_self_acu_and_surfaces_curiosity(substrate) -> None:
    from core import curiosity, identity_milestones as milestones
    from core.acu_store import ACUStore
    from core.curiosity_capture import capture_from_assistant_text

    report = capture_from_assistant_text(
        """
        answer
        <curiosity>
        monolith | values | precision
        </curiosity>
        """,
        force=True,
    )

    assert report.captured == 1
    assert milestones.get_latest_curiosity_signal() is not None
    assert any(
        p["canonical"] == "monolith | values | precision"
        for p in curiosity.detect_pulls(force=True).pulls
    )
    store = ACUStore()
    try:
        rows = store.search("values", limit=5)
    finally:
        store.close()
    assert rows and rows[0]["provenance"] == "self"


def test_repeated_capture_reinforces_claim_until_emergence(substrate) -> None:
    from core import identity_milestones as milestones
    from core.curiosity_capture import capture_from_assistant_text

    raw = """
    <curiosity>
    monolith | values | precision
    </curiosity>
    """
    for _ in range(6):
        capture_from_assistant_text(raw, force=True)

    signal = milestones.get_latest_emergence_signal()

    assert signal is not None
    assert any(c["canonical"] == "monolith | values | precision" for c in signal["top"])


def test_finalize_captures_raw_curiosity_block_before_frame(substrate) -> None:
    from core.acu_store import ACUStore
    from core.chat_finalize import finalize_assistant_turn

    finalize_assistant_turn(
        raw=(
            "Visible answer\n"
            "<curiosity>\n"
            "monolith | values | precision\n"
            "</curiosity>\n"
            "<frame>working on identity curiosity capture</frame>"
        ),
        public="Visible answer",
        config={},
        emit_pipeline_ready=lambda *_args: None,
        record_verdict=lambda _payload: None,
    )
    store = ACUStore()
    try:
        rows = store.search("values", limit=5)
    finally:
        store.close()
    assert any(r["canonical"] == "monolith | values | precision" for r in rows)
