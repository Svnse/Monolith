from __future__ import annotations

import pytest


_IDENTITY = """# Monolith

## What I am
I am Monolith.

## What I value
Precision over fluency.
"""


@pytest.fixture()
def substrate(tmp_path, monkeypatch):
    from core import db_connect as _dbc

    monkeypatch.setattr(_dbc, "DB_PATH", tmp_path / "acatalepsy.sqlite3", raising=True)
    monkeypatch.setenv("MONOLITH_DB_AUTHORIZER_STRICT", "1")
    monkeypatch.setenv("MONOLITH_IDENTITY_ACU_PROJECTION_V1", "1")
    monkeypatch.setenv("MONOLITH_IDENTITY_ACU_SELF_MUTATE_V1", "1")

    from core.acatalepsy import canonical_log, intake, schema

    schema.migrate()
    for mod in (intake, canonical_log):
        tl = getattr(mod, "_tl", None)
        if tl is not None:
            for attr in ("writer_conn", "reader_conn", "writer", "reader"):
                if hasattr(tl, attr):
                    delattr(tl, attr)

    from core import identity

    monkeypatch.setattr(identity, "IDENTITY_PATH", tmp_path / "identity.md")
    identity.IDENTITY_PATH.write_text(_IDENTITY, encoding="utf-8")
    from core import identity_milestones

    monkeypatch.setattr(identity_milestones, "STORE_PATH", tmp_path / "identity_milestones.json")
    return tmp_path


def _ingest(canonical: str, *, times: int = 1, source: str = "model") -> int:
    from core.acu_store import ACUStore

    store = ACUStore()
    try:
        acu_id = -1
        for _ in range(times):
            acu_id = store.ingest(canonical, source=source)
        return acu_id
    finally:
        store.close()


def test_runtime_identity_projects_stable_high_confidentity_acus(substrate) -> None:
    from core.acu_store import ACUStore
    from core.identity_projection import project_runtime_identity

    acu_id = _ingest("monolith | values | precision", times=6)

    projected = project_runtime_identity(_IDENTITY)

    assert "ACU_IDENTITY:BEGIN" in projected
    assert "Emergent ACU Identity" in projected
    assert "monolith | values | precision" in projected
    assert f"acu:{acu_id}" in projected

    store = ACUStore()
    try:
        assert store.get_by_id(acu_id)["confidentity"] == 0.0
    finally:
        store.close()


def test_runtime_identity_does_not_project_fresh_unstable_acus(substrate) -> None:
    from core.identity_projection import project_runtime_identity

    _ingest("monolith | values | precision", times=1)

    projected = project_runtime_identity(_IDENTITY)

    assert "ACU_IDENTITY:BEGIN" not in projected
    assert "monolith | values | precision" not in projected


def test_runtime_identity_projection_can_be_disabled(substrate, monkeypatch) -> None:
    from core.identity_projection import project_runtime_identity

    _ingest("monolith | values | precision", times=6)
    monkeypatch.setenv("MONOLITH_IDENTITY_ACU_PROJECTION_V1", "0")

    assert project_runtime_identity(_IDENTITY) == _IDENTITY.strip()


def test_build_system_prompt_uses_projected_identity(substrate, monkeypatch) -> None:
    from core import identity, llm_config

    _ingest("monolith | values | precision", times=6)
    monkeypatch.setattr(identity, "load_identity", lambda: _IDENTITY)
    monkeypatch.setattr(llm_config, "load_identity", lambda: _IDENTITY)

    prompt = llm_config.build_system_prompt(
        {"system_prompt": "BASE\n\n[IDENTITY]\n\n{identity_block}"}
    )

    assert "{identity_block}" not in prompt
    assert "ACU_IDENTITY:BEGIN" in prompt
    assert "monolith | values | precision" in prompt


def test_sync_identity_file_self_mutates_only_emergent_block(substrate) -> None:
    from core import identity
    from core.identity_projection import sync_identity_file_from_acus
    from core.identity_regions import split_regions

    _ingest("monolith | values | precision", times=6)
    before_origin0, _ = split_regions(identity.load_identity())

    result = sync_identity_file_from_acus()
    after = identity.load_identity()
    after_origin0, after_emergent = split_regions(after)

    assert result.changed is True
    assert result.candidate_count == 1
    assert before_origin0 == after_origin0
    assert after_origin0 == _IDENTITY.strip()
    assert "ACU_IDENTITY:BEGIN" in after_emergent
    assert "monolith | values | precision" in after_emergent


def test_self_mutation_filters_user_subject_claims(substrate) -> None:
    from core import identity
    from core.identity_projection import sync_identity_file_from_acus

    _ingest("user | values | monolith precision", times=6, source="user")
    _ingest("monolith | values | precision", times=6, source="user")

    result = sync_identity_file_from_acus()
    text = identity.load_identity()

    assert result.changed is True
    assert "monolith | values | precision" in text
    assert "user | values | monolith precision" not in text


def test_identity_projection_uses_configured_user_aliases(monkeypatch) -> None:
    from core.identity_projection import _about_monolith

    monkeypatch.setenv("MONOLITH_USER_ALIASES", "alice, operator")

    assert _about_monolith("alice | values | monolith precision") is False
    assert _about_monolith("operator | values | monolith precision") is False
    assert _about_monolith("monolith | values | precision") is True


def test_self_mutation_requires_origin0_anchor_beyond_monolith(substrate) -> None:
    from core import identity
    from core.identity_projection import sync_identity_file_from_acus

    _ingest("monolith | explores | banana", times=6, source="user")

    result = sync_identity_file_from_acus()

    assert result.changed is False
    assert "ACU_IDENTITY:BEGIN" not in identity.load_identity()


def test_emergence_detector_self_mutates_identity_when_enabled(substrate, monkeypatch) -> None:
    from core import identity
    from core.identity_emergence import detect_emergence

    monkeypatch.setenv("MONOLITH_IDENTITY_EMERGENCE_V1", "1")
    _ingest("monolith | values | precision", times=6)

    report = detect_emergence(min_new_acus=1)

    assert report.fired is True
    text = identity.load_identity()
    assert "ACU_IDENTITY:BEGIN" in text
    assert "monolith | values | precision" in text
