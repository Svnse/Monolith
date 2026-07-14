"""Tests for core/command_feedback.py — the [COMMAND_FAILED] outside-in block.

Generalizes the bearing rejection template (capture-why -> feed-back -> bounded
repair) to any model command that fails in the external system. Delivery is a
direct-inject interceptor (load-bearing — never budget-dropped); capture is
per-surface. See docs/audits/MODEL_OUTPUT_BLINDSPOT_MAP.md.

The render is pure and is the testable core.
"""
from __future__ import annotations

from core.command_feedback import (
    clear_pending,
    command_feedback_interceptor,
    get_pending,
    is_non_convergent,
    record_failure,
    render_command_failed_block,
)


def test_non_convergent_when_only_reasoning_no_answer_no_action():
    # The observed hello.py failure: a 4k-char <think> block, empty public
    # answer, no tool call — "didn't say anything".
    assert is_non_convergent(public_answer="", had_tool_call=False, done_signal=False) is True
    assert is_non_convergent(public_answer="   \n  ", had_tool_call=False, done_signal=False) is True


def test_convergent_when_there_is_a_public_answer():
    assert is_non_convergent(public_answer="Done — created hello.py.", had_tool_call=False, done_signal=False) is False


def test_convergent_when_a_tool_call_fired():
    assert is_non_convergent(public_answer="", had_tool_call=True, done_signal=False) is False


def test_convergent_when_done_signal():
    assert is_non_convergent(public_answer="", had_tool_call=False, done_signal=True) is False


# --- trapped-answer recovery (2026-06-16 "behind the reasoning curtain") -------
# Root cause: the model emits unbalanced <think> tags — <think>A</think> then
# re-opens <think>B with the answer inside, never closing it. The normalizer's
# depth-counter keeps the answer in the thinking lane => empty public answer =>
# generic non-convergence nudge that took TWO re-emits to recover. A cause-aware
# classifier lets the single allowed retry get a surgical instruction.

def test_answer_trapped_when_think_tags_unbalanced():
    from core.command_feedback import answer_trapped_in_think
    raw = "<think>reasoning A</think>\n[CHANNEL: ASSISTANT]\n<think>the actual answer B, never closed"
    assert answer_trapped_in_think(raw) is True


def test_not_trapped_when_think_balanced_and_no_answer():
    # Genuine non-convergence: one closed think, nothing after. Not a trapped answer.
    from core.command_feedback import answer_trapped_in_think
    assert answer_trapped_in_think("<think>only reasoning, never answered</think>") is False


def test_not_trapped_when_no_think_tags():
    from core.command_feedback import answer_trapped_in_think
    assert answer_trapped_in_think("plain answer text, no tags") is False


def test_trapped_counts_analysis_and_reasoning_variants():
    from core.command_feedback import answer_trapped_in_think
    assert answer_trapped_in_think("<analysis>a</analysis><reasoning>answer here, unclosed") is True


def test_nudge_trapped_names_think_tags_and_says_outside():
    from core.command_feedback import build_non_convergence_nudge
    msg = build_non_convergence_nudge("what is 7+5", trapped=True)
    assert "<think>" in msg
    assert "outside" in msg.lower()
    assert "7+5" in msg  # the user's ask is woven in


def test_nudge_generic_when_not_trapped_preserves_old_contract():
    from core.command_feedback import build_non_convergence_nudge
    msg = build_non_convergence_nudge("what is 7+5", trapped=False)
    assert "only internal reasoning" in msg  # the original generic nudge text
    assert "7+5" in msg


# --- recover-and-suppress (2026-06-24): pull the trapped answer out instead of -
# firing the contaminating re-emit. The peer/training path's _clean_agent_response
# already orphan-salvages the answer; the bug is the gate re-emitting anyway via an
# ephemeral role:user nudge the model reads as the USER correcting it. When the
# answer is CONFIDENTLY recoverable we recover + suppress; otherwise regen (the
# blessed worst-case). Discriminator is tag-count only (no content heuristics):
# bias to regen, since a wrong recover silently surfaces reasoning as the answer.

def test_recover_trapped_answer_pulls_answer_after_orphan_think():
    # case (a) — the documented "behind the reasoning curtain" shape: a closed
    # reasoning block, then a re-opened (never-closed) block holding the answer.
    from core.command_feedback import recover_trapped_answer
    raw = "<think>reasoning A</think>\n[CHANNEL: ASSISTANT]\n<think>the actual answer B"
    assert recover_trapped_answer(raw) == "the actual answer B"


def test_recover_trapped_answer_empty_when_balanced_reasoning_only():
    from core.command_feedback import recover_trapped_answer
    assert recover_trapped_answer("<think>only reasoning</think>") == ""


def test_recover_trapped_answer_strips_channel_echo():
    from core.command_feedback import recover_trapped_answer
    raw = "<think>r</think>[CHANNEL: ASSISTANT] Hello there"
    assert recover_trapped_answer(raw) == "Hello there"


def test_should_recover_true_for_documented_trapped_pattern():
    # case (a): completed block + re-opened trailing block with the answer.
    from core.command_feedback import should_recover_trapped
    raw = "<think>reasoning A</think>\n[CHANNEL: ASSISTANT]\n<think>the actual answer B"
    assert should_recover_trapped(raw) is True


def test_should_recover_false_for_pure_unclosed_reasoning():
    # case (c): no completed block (closes == 0) — a genuinely truncated turn
    # that only ever reasoned. MUST regen (don't surface raw reasoning as answer).
    from core.command_feedback import should_recover_trapped
    assert should_recover_trapped("<think>just reasoning, cut off mid-thought") is False


def test_should_recover_false_when_nothing_recoverable():
    # unbalanced, but stripping yields empty text — nothing to recover, so regen.
    from core.command_feedback import should_recover_trapped
    assert should_recover_trapped("<think>A</think><think>") is False


def test_should_recover_false_when_balanced():
    # balanced tags are not "trapped" at all (handled by the answer lane, not us).
    from core.command_feedback import should_recover_trapped
    assert should_recover_trapped("<think>x</think>the answer") is False


def test_should_recover_locks_case_d_two_blocks_recovers_per_floor():
    # case (d): a completed block then a SECOND reasoning burst that got cut off,
    # with NO answer. Tag counts cannot tell answer-from-reasoning here, so the
    # floor RECOVERS (and would surface reasoning as the answer). Locked as a
    # KNOWN limitation: trapped raws are logged on fire (capture-on-fire) so real
    # (d) frequency is observable, and biasing to regen would need fragile content
    # heuristics. See advisor 2026-06-24. If real data shows (d) is common, tighten
    # the signal (e.g. require the [CHANNEL: ASSISTANT] echo) — not a content guess.
    from core.command_feedback import should_recover_trapped
    raw = "<think>A</think><think>second reasoning burst, cut off"
    assert should_recover_trapped(raw) is True


def test_render_none_when_empty():
    assert render_command_failed_block([]) is None


def test_store_roundtrip_and_clear():
    clear_pending()
    assert get_pending() == []
    record_failure(kind="tool_call", failed_rules=["x"], detail="d", offending="o")
    pend = get_pending()
    assert len(pend) == 1
    assert pend[0]["kind"] == "tool_call"
    assert pend[0]["failed_rules"] == ["x"]
    clear_pending()
    assert get_pending() == []


def test_interceptor_injects_block_and_clears_pending():
    # One repair attempt: the block lands, then pending is cleared so it does
    # not re-fire forever (bearing's one-repair discipline).
    clear_pending()
    record_failure(kind="tool_call", failed_rules=["unknown_shape"], detail="no match", offending="{bad}")
    messages = [{"role": "user", "content": "do the thing"}]
    result = command_feedback_interceptor(messages, {})
    assert result is not None
    injected = [m for m in result if "[COMMAND_FAILED]" in str(m.get("content", ""))]
    assert injected, "block must be injected"
    assert get_pending() == [], "pending must clear after one inject (one repair attempt)"


def test_interceptor_none_when_no_pending():
    clear_pending()
    assert command_feedback_interceptor([{"role": "user", "content": "hi"}], {}) is None


def test_record_failure_dedupes_identical_pending():
    # extract_commands runs more than once per turn (check + process); an
    # identical drop must not double-record into two [COMMAND_FAILED] entries.
    clear_pending()
    record_failure(kind="tool_call", failed_rules=["x"], detail="d", offending="o")
    record_failure(kind="tool_call", failed_rules=["x"], detail="d", offending="o")
    assert len(get_pending()) == 1
    # a genuinely different failure still records
    record_failure(kind="tool_call", failed_rules=["x"], detail="d", offending="other")
    assert len(get_pending()) == 2


def test_render_single_failure_includes_why_and_offending():
    pending = [{
        "kind": "tool_call",
        "failed_rules": ["unknown_shape"],
        "detail": "command did not match any known tool shape",
        "offending": '{"function": "X"}',
    }]
    block = render_command_failed_block(pending)
    assert block is not None
    assert block.startswith("[COMMAND_FAILED]")
    assert block.rstrip().endswith("[/COMMAND_FAILED]")
    # the runtime's real why, verbatim:
    assert "command did not match any known tool shape" in block
    assert '{"function": "X"}' in block      # offending input echoed back verbatim
    assert "unknown_shape" in block
