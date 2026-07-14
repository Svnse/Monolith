from __future__ import annotations

from datetime import datetime, timedelta, timezone
import inspect

import pytest

from core import continuity
from core import runtime_state_projection as rsp
from core.runtime_state_lanes import LANES, LANE_ORDER, lead_phrase


@pytest.fixture
def tmp_store(monkeypatch, tmp_path):
    store_path = tmp_path / "continuity.json"
    monkeypatch.setattr(continuity, "_STORE_PATH", store_path)
    yield store_path


def _user(text: str = "tell me about restore monolith runtime state please"):
    return [{"role": "user", "content": text}]


def test_lane_registry_order_and_lead_phrases_are_fixed() -> None:
    assert LANE_ORDER == (
        "identity_material",
        "continuity",
        "recall",
        "current_model_execution",
        "temporal_context",
        "temporal_relative",
    )
    assert [lane.lead_phrase for lane in LANES] == [
        "Identity material, as operating law:",
        "What I carry forward from continuity:",
        "From recalled memory:",
        "Current execution facts:",
        "Current local time:",
        "Time elapsed:",
    ]


def test_identity_lane_is_pinned_third_person_operating_law() -> None:
    assert list(inspect.signature(rsp.render_identity_lane).parameters) == []
    line = rsp.render_identity_lane()
    assert line == (
        "Identity material, as operating law: "
        "the Monolith identity seed governs posture and commitments."
    )
    lowered = line.lower()
    assert "i am monolith" not in lowered
    for term in (
        "verified",
        "live runtime",
        "backend",
        "local",
        "cloud",
        "model",
        "context window",
        "running on",
        "hosted",
        "currently executing",
    ):
        assert term not in lowered


def test_runtime_state_lane_order_is_byte_stable(monkeypatch, tmp_store) -> None:
    monkeypatch.setenv("MONOLITH_TEMPORAL_CONTEXT_V1", "1")
    continuity.pin("prefer implementation over proposal loops", category="lesson")
    from core import acu_retrieval, irp
    monkeypatch.setattr(
        acu_retrieval,
        "retrieve_relevant_acus",
        lambda prompt: [{"canonical": "monolith | uses | runtime state projection"}],
    )
    monkeypatch.setattr(irp, "label_text", lambda canonical, **kwargs: f"LABELED {canonical}")
    now = datetime(2026, 6, 2, 12, 10, tzinfo=timezone(timedelta(hours=-4), "EDT"))
    config = {
        "backend": "cloud",
        "api_provider": "openai",
        "api_base": "https://api.deepseek.com",
        "api_model": "deepseek-v4-pro",
        "ctx_limit": 8192,
    }

    block_a = rsp.render_runtime_state(_user(), config, now=now)
    block_b = rsp.render_runtime_state(_user(), config, now=now)

    assert block_a == block_b
    assert block_a.startswith("[RUNTIME STATE] - ambient runtime state; NOT this turn's request.")
    assert block_a.rstrip().endswith("[/RUNTIME STATE]")
    positions = [block_a.index(lead_phrase(name)) for name in (
        "identity_material",
        "continuity",
        "recall",
        "current_model_execution",
        "temporal_context",
    )]
    assert positions == sorted(positions)
    assert "Current local time: 2026-06-02 12:10 EDT (Tuesday)" in block_a


def test_continuity_lane_clears_working_memory_on_first_turn(tmp_store) -> None:
    continuity.set_working_memory("from prior session", "model-a")
    continuity.pin("audit tasks stay read-only", category="lesson")

    block = rsp.render_runtime_state(_user(), {}, now=datetime(2026, 6, 2, 12, 0))

    assert "What I carry forward from continuity:" in block
    assert "lesson(1): audit tasks stay read-only" in block
    assert continuity.get_working_memory() is None


def test_recall_lane_uses_registered_lead_phrase(monkeypatch) -> None:
    from core import acu_retrieval

    monkeypatch.setattr(
        acu_retrieval,
        "retrieve_relevant_acus",
        lambda prompt: [{"canonical": "monolith | uses | runtime state projection", "locked": 1}],
    )

    block = rsp.render_runtime_state(_user(), {}, now=datetime(2026, 6, 2, 12, 0))

    assert "From recalled memory:" in block
    # Label is authority-derived now (a locked claim -> [LOCKED], the deference tier);
    # the old veracity-based label_text never emitted [VERIFIED] for confirmed facts.
    assert "[LOCKED] monolith | uses | runtime state projection" in block


def test_current_execution_lane_omits_secrets() -> None:
    block = rsp.render_runtime_state(
        _user(),
        {
            "backend": "cloud",
            "api_provider": "openai",
            "api_base": "https://api.deepseek.com",
            "api_model": "deepseek-v4-pro",
            "api_key": "secret",
            "ctx_limit": 8192,
        },
        now=datetime(2026, 6, 2, 12, 0),
    )

    assert "Current execution facts:" in block
    assert "backend=cloud" in block
    assert "model=deepseek-v4-pro" in block
    assert "context_window=8192" in block
    assert "secret" not in block


def test_projection_has_no_llm_summarization_path() -> None:
    source = inspect.getsource(rsp)
    for forbidden in (
        "llm_call",
        "create_chat_completion",
        "GeneratorWorker",
        "engine.llm",
        "OpenAICompatLLM",
    ):
        assert forbidden not in source


def test_contribute_section_returns_runtime_state_section() -> None:
    section = rsp.contribute_section(_user(), {})
    assert section is not None
    assert section.name == "runtime_state"
    assert "[RUNTIME STATE]" in section.text


# ── temporal_relative lane (when-plane relational time) ──────────────


def _multi_turn():
    return [
        {"role": "user", "content": "first question about restore monolith"},
        {"role": "assistant", "content": "answer"},
        {"role": "user", "content": "second question about restore monolith runtime"},
    ]


def test_relative_lane_disabled_by_flag(tmp_store, monkeypatch):
    monkeypatch.delenv("MONOLITH_RELATIVE_TIME_V1", raising=False)
    continuity.set_last_turn_at("2026-06-02T11:55:00+00:00")
    now = datetime(2026, 6, 2, 12, 0, tzinfo=timezone.utc)
    assert rsp.render_relative_time_lane(_multi_turn(), now=now) == ""


def test_relative_lane_empty_without_marker(tmp_store, monkeypatch):
    monkeypatch.setenv("MONOLITH_RELATIVE_TIME_V1", "1")
    now = datetime(2026, 6, 2, 12, 0, tzinfo=timezone.utc)
    assert rsp.render_relative_time_lane(_multi_turn(), now=now) == ""


def test_relative_lane_subsequent_turn_shows_last_turn(tmp_store, monkeypatch):
    monkeypatch.setenv("MONOLITH_RELATIVE_TIME_V1", "1")
    continuity.set_last_turn_at("2026-06-02T11:55:00+00:00")  # 5 min before now
    now = datetime(2026, 6, 2, 12, 0, tzinfo=timezone.utc)
    line = rsp.render_relative_time_lane(_multi_turn(), now=now)  # 2 user msgs = not first turn
    assert lead_phrase("temporal_relative") in line
    assert "last turn" in line and "5m ago" in line
    assert "previous session" not in line


def test_relative_lane_first_turn_shows_session_gap(tmp_store, monkeypatch):
    monkeypatch.setenv("MONOLITH_RELATIVE_TIME_V1", "1")
    continuity.set_last_turn_at("2026-05-30T12:00:00+00:00")  # 3 days before now
    now = datetime(2026, 6, 2, 12, 0, tzinfo=timezone.utc)
    line = rsp.render_relative_time_lane(_user(), now=now)  # 1 user msg = first turn
    assert "3d" in line and "previous session" in line
    assert "last turn" not in line


def test_runtime_state_on_commit_records_turn_timestamp(tmp_store, monkeypatch):
    monkeypatch.setenv("MONOLITH_RELATIVE_TIME_V1", "1")
    section = rsp.contribute_section(_user(), {})
    assert section is not None and section.on_commit is not None
    assert continuity.get_last_turn_at() is None
    section.on_commit()
    assert continuity.get_last_turn_at() is not None
