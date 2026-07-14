from __future__ import annotations

import json

import pytest

from core import continuity


@pytest.fixture
def tmp_store(monkeypatch, tmp_path):
    """Redirect the continuity store to a temp file for each test."""
    store_path = tmp_path / "continuity.json"
    monkeypatch.setattr(continuity, "_STORE_PATH", store_path)
    yield store_path


# ── store helpers ────────────────────────────────────────────────────


def test_get_working_memory_empty_initially(tmp_store) -> None:
    assert continuity.get_working_memory() is None


def test_set_working_memory_persists(tmp_store) -> None:
    continuity.set_working_memory("mid-derivation state", "model_a")
    slot = continuity.get_working_memory()
    assert slot is not None
    assert slot["text"] == "mid-derivation state"
    assert slot["writer_model_id"] == "model_a"


def test_set_working_memory_overwrites(tmp_store) -> None:
    continuity.set_working_memory("first", "model_a")
    continuity.set_working_memory("second", "model_b")
    slot = continuity.get_working_memory()
    assert slot["text"] == "second"
    assert slot["writer_model_id"] == "model_b"


def test_clear_working_memory_nulls_slot(tmp_store) -> None:
    continuity.set_working_memory("something", "model_a")
    continuity.clear_working_memory()
    assert continuity.get_working_memory() is None


def test_clear_working_memory_on_empty_is_noop(tmp_store) -> None:
    continuity.clear_working_memory()
    assert continuity.get_working_memory() is None


def test_working_memory_slot_is_not_a_pin(tmp_store) -> None:
    """Setting working_memory must not consume a pin slot or affect pins."""
    pin_result = continuity.pin("an anchor", category="anchor")
    pin_id = pin_result["id"]
    continuity.set_working_memory("scratch", "model_a")
    # Read store JSON directly — robust against unknown reader-helper names.
    store = json.loads(tmp_store.read_text(encoding="utf-8"))
    active_ids = [
        p["id"] for p in store.get("active", [])
        if not p.get("retired_at") and not p.get("retire_reason")
    ]
    assert pin_id in active_ids


def test_working_memory_schema_is_two_fields_only(tmp_store) -> None:
    """Slot record must have exactly {text, writer_model_id} — no others."""
    continuity.set_working_memory("x", "model_a")
    slot = continuity.get_working_memory()
    assert set(slot.keys()) == {"text", "writer_model_id"}


# ── session-clear hook on first-turn ─────────────────────────────────


def test_continuity_interceptor_clears_working_memory_on_first_turn(tmp_store) -> None:
    """First-turn-of-session interceptor must clear working_memory."""
    continuity.set_working_memory("from prior session", "model_a")
    # Simulate first turn: exactly one non-ephemeral user message.
    messages = [{"role": "user", "content": "first user msg", "ephemeral": False}]
    continuity.continuity_interceptor(messages, {})
    assert continuity.get_working_memory() is None


def test_continuity_interceptor_does_not_clear_on_later_turns(tmp_store) -> None:
    """Mid-session turns must not touch working_memory."""
    continuity.set_working_memory("active state", "model_a")
    # Two non-ephemeral user messages → not first turn.
    messages = [
        {"role": "user", "content": "first", "ephemeral": False},
        {"role": "assistant", "content": "reply"},
        {"role": "user", "content": "second", "ephemeral": False},
    ]
    continuity.continuity_interceptor(messages, {})
    slot = continuity.get_working_memory()
    assert slot is not None
    assert slot["text"] == "active state"


def test_contribute_section_clears_working_memory_on_first_turn(tmp_store) -> None:
    """Live path (contribute_section) must clear working_memory on first turn."""
    continuity.set_working_memory("from prior session", "model_a")
    # Simulate first turn: exactly one non-ephemeral user message.
    messages = [{"role": "user", "content": "first user msg", "ephemeral": False}]
    continuity.contribute_section(messages, {})
    assert continuity.get_working_memory() is None


def test_session_clear_fires_regardless_of_continuity_flag(
    tmp_store, monkeypatch
) -> None:
    """Lock 1: session-boundary clear fires even when MONOLITH_CONTINUITY_BOOT_V1=0.

    The flag gates block injection, not the clear. WM must not survive into
    the next session regardless of whether the CONTINUITY block is rendered.
    """
    monkeypatch.setenv("MONOLITH_CONTINUITY_BOOT_V1", "0")
    continuity.set_working_memory("from prior session", "model_a")
    messages = [{"role": "user", "content": "first user msg", "ephemeral": False}]
    continuity.contribute_section(messages, {})
    assert continuity.get_working_memory() is None


# ── current-model-id resolver ────────────────────────────────────────


def test_get_current_model_id_prefers_api_model(monkeypatch) -> None:
    """When api_model is set (cloud), it wins over gguf_path."""
    from core import llm_config
    from core.config import LLMConfig

    fake_cfg = type("FakeRoot", (), {"llm": LLMConfig(api_model="deepseek-chat", gguf_path="/path/to/local.gguf")})()
    monkeypatch.setattr(llm_config, "get_config", lambda: fake_cfg)
    assert llm_config.get_current_model_id() == "deepseek-chat"


def test_get_current_model_id_falls_back_to_gguf_path(monkeypatch) -> None:
    """When api_model is empty, gguf_path is used."""
    from core import llm_config
    from core.config import LLMConfig

    fake_cfg = type("FakeRoot", (), {"llm": LLMConfig(api_model="", gguf_path="omnicoder-9b-q4_k_m.gguf")})()
    monkeypatch.setattr(llm_config, "get_config", lambda: fake_cfg)
    assert llm_config.get_current_model_id() == "omnicoder-9b-q4_k_m.gguf"


def test_get_current_model_id_empty_returns_unknown(monkeypatch) -> None:
    """Both empty/None → unknown sentinel."""
    from core import llm_config
    from core.config import LLMConfig

    fake_cfg = type("FakeRoot", (), {"llm": LLMConfig(api_model="", gguf_path=None)})()
    monkeypatch.setattr(llm_config, "get_config", lambda: fake_cfg)
    assert llm_config.get_current_model_id() == "unknown"


# ── scratchpad ops ───────────────────────────────────────────────────


def _import_scratchpad_executor():
    """Load the dynamic skill executor by file path (matches runtime loader)."""
    import importlib.util
    from pathlib import Path
    spec_path = Path(__file__).parent.parent / "skills" / "scratchpad" / "executor.py"
    spec = importlib.util.spec_from_file_location("scratchpad_exec_test", spec_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def scratchpad(tmp_store, monkeypatch):
    """Scratchpad executor with current_model_id pinned to 'test-model'."""
    from core import llm_config
    monkeypatch.setattr(llm_config, "get_current_model_id", lambda: "test-model")
    return _import_scratchpad_executor()


def test_working_memory_set_persists(scratchpad) -> None:
    out = scratchpad.run({"op": "working_memory_set", "text": "in-flight state"}, None)
    assert "working_memory_set" in out
    slot = continuity.get_working_memory()
    assert slot["text"] == "in-flight state"
    assert slot["writer_model_id"] == "test-model"


def test_working_memory_set_strips_whitespace(scratchpad) -> None:
    scratchpad.run({"op": "working_memory_set", "text": "  padded\n\n  "}, None)
    slot = continuity.get_working_memory()
    assert slot["text"] == "padded"


def test_working_memory_set_rejects_empty_after_strip(scratchpad) -> None:
    out = scratchpad.run({"op": "working_memory_set", "text": "   \n\t  "}, None)
    assert "empty" in out.lower()
    assert continuity.get_working_memory() is None


def test_working_memory_set_rejects_over_1000_chars(scratchpad) -> None:
    out = scratchpad.run({"op": "working_memory_set", "text": "x" * 1001}, None)
    assert "1000" in out
    assert continuity.get_working_memory() is None


def test_working_memory_set_accepts_exactly_1000_chars(scratchpad) -> None:
    out = scratchpad.run({"op": "working_memory_set", "text": "x" * 1000}, None)
    assert "1000 chars written" in out
    slot = continuity.get_working_memory()
    assert len(slot["text"]) == 1000


def test_working_memory_set_preserves_multiline_shape(scratchpad) -> None:
    """Multiline content stays multiline — no normalization."""
    multiline = "line one\n\nline two with a\nparagraph break\n\nline three"
    scratchpad.run({"op": "working_memory_set", "text": multiline}, None)
    slot = continuity.get_working_memory()
    assert slot["text"] == multiline


def test_working_memory_get_returns_text_and_writer(scratchpad) -> None:
    scratchpad.run({"op": "working_memory_set", "text": "state"}, None)
    out = scratchpad.run({"op": "working_memory_get"}, None)
    assert "state" in out
    assert "test-model" in out


def test_working_memory_get_empty_returns_empty_envelope(scratchpad) -> None:
    out = scratchpad.run({"op": "working_memory_get"}, None)
    assert "empty" in out.lower()


def test_working_memory_clear_nulls_slot(scratchpad) -> None:
    scratchpad.run({"op": "working_memory_set", "text": "state"}, None)
    out = scratchpad.run({"op": "working_memory_clear"}, None)
    assert out == "[working_memory_clear: ok]"
    assert continuity.get_working_memory() is None


def test_working_memory_clear_on_empty_is_idempotent(scratchpad) -> None:
    out = scratchpad.run({"op": "working_memory_clear"}, None)
    assert continuity.get_working_memory() is None
    assert out == "[working_memory_clear: ok]"


def test_working_memory_set_clear_get_round_trip(scratchpad) -> None:
    scratchpad.run({"op": "working_memory_set", "text": "round-trip-marker"}, None)
    scratchpad.run({"op": "working_memory_clear"}, None)
    out = scratchpad.run({"op": "working_memory_get"}, None)
    assert "empty" in out.lower()
    assert "round-trip-marker" not in out


# ── build_system_prompt + inject read-path ───────────────────────────
# inject_working_memory_into_prompt is the injection site after Task 6 fix.
# Tests call inject_working_memory_into_prompt(build_system_prompt()) to
# mirror the production pipeline in engine/llm.py:_compile_system_prompt.


def test_build_system_prompt_injects_working_memory_on_match(tmp_store, monkeypatch) -> None:
    from core import llm_config
    monkeypatch.setattr(llm_config, "get_current_model_id", lambda: "model_a")
    continuity.set_working_memory("test-payload-12345", "model_a")
    prompt = llm_config.inject_working_memory_into_prompt(llm_config.build_system_prompt())
    assert "[WORKING MEMORY]" in prompt
    assert "test-payload-12345" in prompt


def test_build_system_prompt_skips_working_memory_on_mismatch(tmp_store, monkeypatch) -> None:
    from core import llm_config
    monkeypatch.setattr(llm_config, "get_current_model_id", lambda: "model_b")
    continuity.set_working_memory("stale-payload-67890", "model_a")
    prompt = llm_config.inject_working_memory_into_prompt(llm_config.build_system_prompt())
    assert "stale-payload-67890" not in prompt
    assert "[WORKING MEMORY]" not in prompt


def test_build_system_prompt_atomically_nulls_slot_on_mismatch(tmp_store, monkeypatch) -> None:
    from core import llm_config
    monkeypatch.setattr(llm_config, "get_current_model_id", lambda: "model_b")
    continuity.set_working_memory("payload", "model_a")
    llm_config.inject_working_memory_into_prompt(llm_config.build_system_prompt())
    # Mismatch path must atomically null the slot — verified by reading the
    # store after the build.
    assert continuity.get_working_memory() is None


def test_build_system_prompt_skips_when_slot_empty(tmp_store, monkeypatch) -> None:
    from core import llm_config
    monkeypatch.setattr(llm_config, "get_current_model_id", lambda: "model_a")
    # No working_memory set — slot is None.
    prompt = llm_config.inject_working_memory_into_prompt(llm_config.build_system_prompt())
    assert "[WORKING MEMORY]" not in prompt


def test_build_system_prompt_injects_above_identity_block(tmp_store, monkeypatch) -> None:
    """Positional invariant: WORKING MEMORY appears before [IDENTITY] in the prompt."""
    from core import llm_config
    monkeypatch.setattr(llm_config, "get_current_model_id", lambda: "model_a")
    continuity.set_working_memory("positioned-marker", "model_a")
    prompt = llm_config.inject_working_memory_into_prompt(llm_config.build_system_prompt())
    wm_idx = prompt.find("[WORKING MEMORY]")
    id_idx = prompt.find("[IDENTITY]")
    assert wm_idx >= 0
    assert id_idx >= 0
    assert wm_idx < id_idx


# ── end-to-end integration ───────────────────────────────────────────


def test_e2e_sentinel_round_trip_via_set_then_build_prompt(tmp_store, monkeypatch) -> None:
    """Round-trip: scratchpad writes via working_memory_set; the full pipeline
    (inject_working_memory_into_prompt(build_system_prompt())) sees the sentinel
    when model_id matches — mirrors engine/llm.py:_compile_system_prompt.
    """
    from core import llm_config
    monkeypatch.setattr(llm_config, "get_current_model_id", lambda: "sentinel-model")
    sp = _import_scratchpad_executor()
    sp.run({"op": "working_memory_set", "text": "SENTINEL-ROUND-TRIP-2026-05-15"}, None)
    prompt = llm_config.inject_working_memory_into_prompt(llm_config.build_system_prompt())
    assert "SENTINEL-ROUND-TRIP-2026-05-15" in prompt
    assert "[WORKING MEMORY]" in prompt


def test_e2e_swap_clears_slot_lazily(tmp_store, monkeypatch) -> None:
    """Model A writes; model B's inject call nulls the slot (lazy swap-clear)."""
    from core import llm_config
    sp = _import_scratchpad_executor()

    # Model A writes.
    monkeypatch.setattr(llm_config, "get_current_model_id", lambda: "model_a")
    sp.run({"op": "working_memory_set", "text": "model-a-state"}, None)
    assert continuity.get_working_memory() is not None

    # Swap to model B. inject_working_memory_into_prompt must lazy-clear.
    monkeypatch.setattr(llm_config, "get_current_model_id", lambda: "model_b")
    prompt = llm_config.inject_working_memory_into_prompt(llm_config.build_system_prompt())
    assert "model-a-state" not in prompt
    assert continuity.get_working_memory() is None  # atomically nulled


def test_e2e_session_boundary_clears_slot(tmp_store, monkeypatch) -> None:
    """First-turn-of-session interceptor must clear working_memory."""
    from core import llm_config
    monkeypatch.setattr(llm_config, "get_current_model_id", lambda: "model_a")
    sp = _import_scratchpad_executor()
    sp.run({"op": "working_memory_set", "text": "prior-session-state"}, None)

    # New session — first user message arrives.
    messages = [{"role": "user", "content": "new session start", "ephemeral": False}]
    continuity.continuity_interceptor(messages, {})

    assert continuity.get_working_memory() is None


def test_e2e_carry_forward_no_op_preserves_slot(tmp_store, monkeypatch) -> None:
    """No-op turn (no set/clear call) leaves the slot intact for next read."""
    from core import llm_config
    monkeypatch.setattr(llm_config, "get_current_model_id", lambda: "model_a")
    sp = _import_scratchpad_executor()
    sp.run({"op": "working_memory_set", "text": "persists-across-turns"}, None)

    # Two pipeline calls (simulating two consecutive same-model turns with no
    # intervening write or clear — mirrors _compile_system_prompt × 2 turns).
    p1 = llm_config.inject_working_memory_into_prompt(llm_config.build_system_prompt())
    p2 = llm_config.inject_working_memory_into_prompt(llm_config.build_system_prompt())
    assert "persists-across-turns" in p1
    assert "persists-across-turns" in p2
    # Slot still intact after two reads.
    slot = continuity.get_working_memory()
    assert slot["text"] == "persists-across-turns"


# ── per-turn injection (the fix for the cached-prompt path) ──────────


def test_inject_working_memory_into_prompt_on_match(tmp_store, monkeypatch) -> None:
    from core import llm_config
    monkeypatch.setattr(llm_config, "get_current_model_id", lambda: "m_a")
    continuity.set_working_memory("inject-test-payload", "m_a")

    cached_prompt = (
        "Some preamble.\n"
        "═══════════════════════════════════════════════════════\n"
        "MEMORY — five surfaces, never conflate them\n"
        "═══════════════════════════════════════════════════════\n"
        "Body of MEMORY section.\n"
    )
    out = llm_config.inject_working_memory_into_prompt(cached_prompt)
    assert "[WORKING MEMORY]" in out
    assert "inject-test-payload" in out
    # Injection lands above the MEMORY header.
    assert out.find("[WORKING MEMORY]") < out.find("MEMORY — five surfaces")


def test_inject_working_memory_into_prompt_on_mismatch_clears_slot(tmp_store, monkeypatch) -> None:
    from core import llm_config
    monkeypatch.setattr(llm_config, "get_current_model_id", lambda: "m_b")
    continuity.set_working_memory("stale", "m_a")
    out = llm_config.inject_working_memory_into_prompt("dummy prompt\nMEMORY — five surfaces\n")
    assert "[WORKING MEMORY]" not in out
    assert "stale" not in out
    assert continuity.get_working_memory() is None  # atomic null


def test_inject_working_memory_into_prompt_idempotent(tmp_store, monkeypatch) -> None:
    """If [WORKING MEMORY] already in prompt, do not re-inject."""
    from core import llm_config
    monkeypatch.setattr(llm_config, "get_current_model_id", lambda: "m_a")
    continuity.set_working_memory("payload", "m_a")
    prompt = (
        "prefix\n[WORKING MEMORY]\nold content\n"
        "═══════════════════════════════════════════════════════\n"
        "MEMORY — five surfaces, never conflate them\n"
    )
    out = llm_config.inject_working_memory_into_prompt(prompt)
    # The function saw [WORKING MEMORY] already present and skipped injection.
    assert out == prompt
    assert out.count("[WORKING MEMORY]") == 1


def test_inject_working_memory_into_prompt_empty_slot_is_noop(tmp_store, monkeypatch) -> None:
    from core import llm_config
    monkeypatch.setattr(llm_config, "get_current_model_id", lambda: "m_a")
    out = llm_config.inject_working_memory_into_prompt("hello world")
    assert out == "hello world"


def test_load_config_reads_master_prompt_fresh(monkeypatch, tmp_path):
    """load_config calls load_master_prompt at each invocation, not at module
    import -- verifies substrate edits propagate without process restart."""
    from core import llm_prompt, llm_config

    call_count = {"n": 0}
    def counting_loader():
        call_count["n"] += 1
        return "[FRESH MASTER PROMPT MARKER]\n{skills_catalog}"

    monkeypatch.setattr(llm_config, "load_master_prompt", counting_loader)
    cfg = llm_config.load_config()
    assert call_count["n"] >= 1
    assert "[FRESH MASTER PROMPT MARKER]" in cfg["system_prompt"]

    # Second call must re-invoke the loader.
    call_count["n"] = 0
    llm_config.load_config()
    assert call_count["n"] >= 1
