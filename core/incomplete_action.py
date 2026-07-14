from __future__ import annotations

import re

from core.cmd_parser import extract_commands

INCOMPLETE_ACTION_PATTERNS = (
    (r"[Ss]ave (?:this|it|the \w+) (?:as|to|in) [`'\"]?(?:prompts/|skills/)", "You described saving but didn't emit write_file. Complete the save now."),
    (r"(?:I'll|Let me|I will|I need to) (?:check|verify|read|scan|look at|inspect)", "You described an action but didn't execute it. Do it now or mark it as deferred."),
)


_THINK_RE = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)


def detect_incomplete_action(text, *, tool_ran=False, patterns=INCOMPLETE_ACTION_PATTERNS):
    """Return a nudge string when a response narrates an action it never executed.

    Suppressed when:
      * a tool already ran this turn (``tool_ran``): the narrated action was
        fulfilled by that tool call, even though the pre-call narration
        ("Let me verify…") still sits in the accumulated assistant message.
        Without this gate the guard re-fires after every tool-using turn —
        the 2026-05-29 ghost-greeting bug, where a spurious extra generation
        degenerated into an off-target greeting.
      * the only action phrase lives inside a ``<think>`` block: that is the
        model's deliberation, not a dangling promise to the user.
    """
    if tool_ran:
        return None
    if extract_commands(text):
        return None
    scan = _THINK_RE.sub(" ", str(text or ""))
    for pattern, nudge in patterns:
        if re.search(pattern, scan):
            return nudge
    return None
