"""Tests for core/ephemeral_coalescer.py.

The coalescer keeps dynamic state at one moving insertion point before the
latest non-ephemeral user message, with a total-char budget that drops
lowest-priority sections under pressure.

Audit defects #6 (KV-cache prefix invalidation) + #8 (no ephemeral budget).
"""
from __future__ import annotations

from pathlib import Path

import pytest

from core import ephemeral_coalescer as ec
from core.ephemeral_coalescer import SectionResult, ephemeral_coalescer_interceptor


def _msg_user(text: str = "real"):
    return {"role": "user", "content": text}


def _msg_system(text: str = "sys"):
    return {"role": "system", "content": text}


def _msg_ephemeral(content: str = "x", source: str = "x"):
    return {"role": "user", "content": content, "ephemeral": True, "source": source}


@pytest.fixture
def fake_contributors(monkeypatch):
    """Install controllable contributors via monkeypatch on _contributors."""
    contributors: list = []
    monkeypatch.setattr(ec, "_contributors", lambda: contributors)
    return contributors


def _add(contribs, name: str, text: str, on_commit=None) -> None:
    contribs.append((name, lambda messages, config: SectionResult(name=name, text=text, on_commit=on_commit)))


# ── basic mechanics ──────────────────────────────────────────────────


def test_no_contributors_returns_none(fake_contributors):
    messages = [_msg_system(), _msg_user("real")]
    assert ephemeral_coalescer_interceptor(messages, {}) is None


def test_single_contributor_inserts_one_block(fake_contributors):
    _add(fake_contributors, "runtime_state", "[RUNTIME STATE]\npin A")
    messages = [_msg_system(), _msg_user("real")]

    result = ephemeral_coalescer_interceptor(messages, {})

    assert result is not None
    assert len(result) == len(messages) + 1
    user_idx = next(i for i, m in enumerate(result) if m.get("content") == "real")
    inserted = result[user_idx - 1]
    assert inserted["role"] == "user"
    assert inserted["ephemeral"] is True
    assert inserted["source"] == "ephemeral_coalescer"
    assert "[RUNTIME STATE]" in inserted["content"]
    assert inserted["sections"] == ("runtime_state",)


def test_multiple_sections_joined_in_priority_order(fake_contributors):
    # Add in reverse drop-order to verify the coalescer sorts.
    _add(fake_contributors, "context_refresh", "[SYSTEM REMINDER]\nrefresh")
    _add(fake_contributors, "rating_telemetry", "[RATING TELEMETRY]\nrolling avg")
    _add(fake_contributors, "runtime_state", "[RUNTIME STATE]\npin A")
    messages = [_msg_system(), _msg_user("real")]

    result = ephemeral_coalescer_interceptor(messages, {})
    inserted = result[1]
    body = inserted["content"]

    # Priority high -> low: runtime_state, rating_telemetry, context_refresh.
    pos_c = body.index("[RUNTIME STATE]")
    pos_r = body.index("[RATING TELEMETRY]")
    pos_e = body.index("[SYSTEM REMINDER]")
    assert pos_c < pos_r < pos_e
    assert inserted["sections"] == ("runtime_state", "rating_telemetry", "context_refresh")


def test_inserted_before_latest_non_ephemeral_user(fake_contributors):
    _add(fake_contributors, "runtime_state", "[RUNTIME STATE] pin A")
    messages = [
        _msg_system(),
        _msg_user("first"),
        {"role": "assistant", "content": "answer"},
        _msg_user("second"),
        _msg_ephemeral(content="ephemeral after", source="x"),
    ]

    result = ephemeral_coalescer_interceptor(messages, {})

    second_idx = next(i for i, m in enumerate(result) if m.get("content") == "second")
    assert result[second_idx - 1]["source"] == "ephemeral_coalescer"
    assert result[-1].get("source") == "x"


# ── budget ──────────────────────────────────────────────────────────


def test_budget_drops_lowest_priority_first(fake_contributors, monkeypatch):
    monkeypatch.setenv("MONOLITH_EPHEMERAL_BUDGET_CHARS", "50")
    _add(fake_contributors, "runtime_state", "C" * 30)
    _add(fake_contributors, "context_refresh", "E" * 30)  # total 60 > 50; context_refresh drops
    messages = [_msg_system(), _msg_user("real")]

    result = ephemeral_coalescer_interceptor(messages, {})
    inserted = result[1]

    assert "C" * 30 in inserted["content"]
    assert "E" * 30 not in inserted["content"]
    assert inserted["sections"] == ("runtime_state",)


def test_budget_zero_drops_all_returns_none(fake_contributors, monkeypatch):
    monkeypatch.setenv("MONOLITH_EPHEMERAL_BUDGET_CHARS", "0")
    _add(fake_contributors, "runtime_state", "[RUNTIME STATE] pin A")
    messages = [_msg_system(), _msg_user("real")]

    assert ephemeral_coalescer_interceptor(messages, {}) is None


def test_budget_invalid_env_falls_to_default(fake_contributors, monkeypatch):
    monkeypatch.setenv("MONOLITH_EPHEMERAL_BUDGET_CHARS", "not-a-number")
    _add(fake_contributors, "runtime_state", "[RUNTIME STATE] pin A")
    messages = [_msg_system(), _msg_user("real")]

    # Default budget is generous (4000); section fits.
    result = ephemeral_coalescer_interceptor(messages, {})
    assert result is not None


# ── on_commit lifecycle ─────────────────────────────────────────────


def test_on_commit_fires_for_kept_sections(fake_contributors, monkeypatch):
    monkeypatch.setenv("MONOLITH_EPHEMERAL_BUDGET_CHARS", "3000")
    committed: list = []
    _add(fake_contributors, "runtime_state", "X", on_commit=lambda: committed.append("c"))
    _add(fake_contributors, "context_refresh", "Y", on_commit=lambda: committed.append("e"))
    messages = [_msg_system(), _msg_user("real")]

    ephemeral_coalescer_interceptor(messages, {})

    assert sorted(committed) == ["c", "e"]


def test_on_commit_does_not_fire_for_dropped_sections(fake_contributors, monkeypatch):
    monkeypatch.setenv("MONOLITH_EPHEMERAL_BUDGET_CHARS", "5")
    committed: list = []
    _add(fake_contributors, "runtime_state", "X" * 4, on_commit=lambda: committed.append("c"))
    _add(fake_contributors, "context_refresh", "Y" * 4, on_commit=lambda: committed.append("e"))
    messages = [_msg_system(), _msg_user("real")]

    ephemeral_coalescer_interceptor(messages, {})

    # runtime_state (priority 0) keeps; context_refresh drops.
    assert committed == ["c"]


def test_on_commit_does_not_fire_when_no_non_ephemeral_user(fake_contributors):
    """When-plane fix #7: on_commit must NOT fire when there is no non-ephemeral
    user message to insert before. The coalescer returns None in that case, so a
    section's on_commit (e.g. context_refresh's gate-advance) must not run —
    honoring the SectionResult contract: 'runs only if ... the final block lands
    in the message list.'"""
    committed: list = []
    _add(fake_contributors, "context_refresh", "X", on_commit=lambda: committed.append("c"))
    messages = [_msg_system(), _msg_ephemeral(content="ephemeral only", source="other")]

    result = ephemeral_coalescer_interceptor(messages, {})

    assert result is None
    assert committed == []


def test_on_commit_failure_does_not_break_chain(fake_contributors, monkeypatch):
    monkeypatch.setenv("MONOLITH_EPHEMERAL_BUDGET_CHARS", "3000")
    committed: list = []

    def crash() -> None:
        raise RuntimeError("boom")

    _add(fake_contributors, "runtime_state", "X", on_commit=crash)
    _add(fake_contributors, "context_refresh", "Y", on_commit=lambda: committed.append("e"))
    messages = [_msg_system(), _msg_user("real")]

    result = ephemeral_coalescer_interceptor(messages, {})

    assert result is not None
    assert committed == ["e"]


# ── isolation + defense ─────────────────────────────────────────────


def test_contributor_raising_does_not_break_others(fake_contributors):
    def crash(messages, config):
        raise RuntimeError("boom")

    fake_contributors.append(("crasher", crash))
    _add(fake_contributors, "runtime_state", "[RUNTIME STATE] pin A")
    messages = [_msg_system(), _msg_user("real")]

    result = ephemeral_coalescer_interceptor(messages, {})

    assert result is not None
    assert "[RUNTIME STATE]" in result[1]["content"]


def test_already_injected_returns_none(fake_contributors):
    _add(fake_contributors, "runtime_state", "[RUNTIME STATE] pin A")
    messages = [
        _msg_system(),
        _msg_ephemeral(content="prior coalesced", source="ephemeral_coalescer"),
        _msg_user("real"),
    ]

    assert ephemeral_coalescer_interceptor(messages, {}) is None


def test_no_non_ephemeral_user_returns_none(fake_contributors):
    _add(fake_contributors, "runtime_state", "[RUNTIME STATE] pin A")
    messages = [
        _msg_system(),
        _msg_ephemeral(content="ephemeral only", source="other"),
    ]

    assert ephemeral_coalescer_interceptor(messages, {}) is None


def test_empty_section_text_ignored(fake_contributors):
    fake_contributors.append(("empty", lambda m, c: SectionResult(name="empty", text="")))
    fake_contributors.append(("whitespace", lambda m, c: SectionResult(name="whitespace", text="   ")))
    _add(fake_contributors, "runtime_state", "[RUNTIME STATE] pin A")
    messages = [_msg_system(), _msg_user("real")]

    result = ephemeral_coalescer_interceptor(messages, {})

    assert result is not None
    assert result[1]["sections"] == ("runtime_state",)


def test_contributor_returning_non_section_result_ignored(fake_contributors):
    fake_contributors.append(("wrong_type", lambda m, c: "not a SectionResult"))
    _add(fake_contributors, "runtime_state", "[RUNTIME STATE] pin A")
    messages = [_msg_system(), _msg_user("real")]

    result = ephemeral_coalescer_interceptor(messages, {})

    assert result is not None
    assert result[1]["sections"] == ("runtime_state",)


# ── integration smoke test (real contributors exist) ────────────────


def test_all_real_modules_expose_contribute_section(monkeypatch):
    """Smoke: each registered contributor module has contribute_section."""
    monkeypatch.setenv("MONOLITH_VERIFIER_V1", "1")  # standard env

    from core import (
        confidence_trajectory,
        context_refresh,
        last_turn,
        rating_telemetry,
        review_loop,
        runtime_state_projection,
    )
    from addons.system import observer

    messages = [_msg_system(), _msg_user("hello")]

    for mod in (
        runtime_state_projection,
        review_loop,
        observer,
        last_turn,
        confidence_trajectory,
        rating_telemetry,
        context_refresh,
    ):
        assert hasattr(mod, "contribute_section"), f"{mod.__name__} missing contribute_section"
        result = mod.contribute_section(messages, {})
        assert result is None or isinstance(result, ec.SectionResult)


def test_real_contributor_registry_uses_runtime_state_not_old_dynamic_sections():
    names = tuple(name for name, _fn in ec._contributors())
    assert names == (
        "runtime_state",
        "active_agents",
        "agent_recap",
        "review_loop",
        "self_check",
        "observer",
        "last_turn",
        "confidence_trajectory",
        "rating_telemetry",
        "context_refresh",
    )
    assert "continuity" not in names
    assert "observed_state" not in names
    assert "acu_recall" not in names
    assert "temporal_context" not in names
    assert "bearing" not in names
    assert "command_feedback" not in names


def test_bootstrap_keeps_direct_inject_blocks_outside_coalescer():
    source = (Path(__file__).resolve().parents[1] / "bootstrap.py").read_text(encoding="utf-8")

    def pos(fragment: str) -> int:
        idx = source.find(fragment)
        assert idx >= 0, f"missing bootstrap fragment: {fragment}"
        return idx

    coalescer = pos("register_interceptor(ephemeral_coalescer_interceptor)")
    assert pos("register_interceptor(prompt_interceptor)") < coalescer
    assert pos("register_interceptor(monothink_interceptor)") < coalescer
    assert pos("register_interceptor(tool_interceptor)") < coalescer
    assert pos("register_interceptor(_bearing_addon.interceptor)") < coalescer
    assert pos("register_interceptor(command_feedback_interceptor)") < coalescer


def test_temporal_context_block_format():
    """Sanity: the temporal block carries an ISO-style date + day name."""
    from core import temporal_context

    block = temporal_context.render_temporal_block()
    assert "[TEMPORAL CONTEXT]" in block
    assert "current_time:" in block
    # Must contain a 4-digit year and a day name
    import re
    assert re.search(r"\b20\d\d-\d\d-\d\d\b", block)
    # One of the seven day names
    assert any(d in block for d in (
        "Monday", "Tuesday", "Wednesday", "Thursday",
        "Friday", "Saturday", "Sunday",
    ))


def test_temporal_context_disabled_by_flag(monkeypatch):
    """When MONOLITH_TEMPORAL_CONTEXT_V1=0, contribute_section returns None."""
    monkeypatch.setenv("MONOLITH_TEMPORAL_CONTEXT_V1", "0")
    from core import temporal_context
    assert temporal_context.contribute_section([], {}) is None
