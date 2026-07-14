"""Tests for the incomplete-action nudge guard.

Regression coverage for the 2026-05-29 "ghost greeting" bug: a CONNECT turn
where Monolith narrated "Let me verify the state" before a scratchpad tool
call that fulfilled the verification. After the tool call was consumed, the
leftover preamble tripped the incomplete-action guard, firing a spurious
extra generation that degenerated into "E. Good morning."
"""
from core.incomplete_action import detect_incomplete_action

_NUDGE = "You described an action but didn't execute it. Do it now or mark it as deferred."


def test_no_nudge_when_a_tool_already_ran_this_turn():
    # The narrated action ("Let me verify the state") WAS executed via a tool
    # call earlier in the same accumulated message, so no nudge is warranted.
    text = (
        "Claude.\n\nLet me verify the state before I answer.\n\n"
        "Claude.\n\nTwo answers — the pins are present this turn."
    )
    assert detect_incomplete_action(text, tool_ran=True) is None


def test_nudge_still_fires_for_a_genuine_dangling_action():
    # Guard's real purpose: a narrated action with no tool executed and no
    # reasoning block must still be nudged.
    text = "Let me check the logs to be sure."
    assert detect_incomplete_action(text, tool_ran=False) == _NUDGE


def test_no_nudge_when_action_phrase_is_only_inside_a_think_block():
    # Action language inside the model's reasoning is deliberation, not a
    # dangling promise — it must not trip the guard.
    text = "<think>Let me verify the config first.</think>\n\nHere is the final answer."
    assert detect_incomplete_action(text, tool_ran=False) is None
