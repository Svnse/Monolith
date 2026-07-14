"""Command feedback — the ``[COMMAND_FAILED]`` outside-in block.

Generalizes the bearing rejection template to ANY model-emitted command that
fails in the external system (parse, validation, execution, policy, infra).
The model is inside-out; the runtime knows the *why*; this block carries that
why back into the model's next turn so it can repair the action instead of
retrying blind or never learning it failed at all.

Three parts (mirrors bearing — see docs/audits/MODEL_OUTPUT_BLINDSPOT_MAP.md):
  * capture  — per-surface: a failing site calls ``record_failure(...)``.
  * delivery — ``command_feedback_interceptor`` (direct-inject, registered in
               bootstrap NOT through the coalescer budget gate — a repair
               instruction is load-bearing and must never be silently dropped).
  * contract — declared in the cached system prompt so the model reads it right.

Discipline: feed the RAW runtime detail + the offending input verbatim, not a
pre-authored canned hint — the real failure is richer than any static string.
One repair attempt per turn (the interceptor clears pending on inject).
"""
from __future__ import annotations

import os
import re
import threading


_TAG = "[COMMAND_FAILED]"
_END_TAG = "[/COMMAND_FAILED]"
_MAX_OFFENDING_CHARS = 400
_FLAG_ENV = "MONOLITH_COMMAND_FEEDBACK_V1"
_SOURCE = "command_feedback"

# Per-process pending queue: capture sites (cmd_parser, guard, …) append a
# failure during turn N's processing; the interceptor drains it into turn N+1's
# prompt. Turn-scoped lifetime, single process — a module-level list is
# sufficient (no cross-restart persistence needed for V0).
_lock = threading.Lock()
_pending: list[dict] = []


def is_enabled() -> bool:
    """Kill switch. Default ON — a repair instruction that fails to fire is the
    write-only disease this exists to cure."""
    raw = str(os.environ.get(_FLAG_ENV, "1")).strip().lower()
    return raw in {"1", "true", "yes", "on"}


def record_failure(
    *,
    kind: str,
    failed_rules: list | tuple | None = None,
    detail: str = "",
    offending: str = "",
) -> None:
    """Capture a system-side command failure for next-turn outside-in feedback.

    Called by any surface where a model-emitted command fails (parse drop,
    validation, policy block, execution error). Feed the RAW detail + the
    offending input; never a canned hint. Best-effort, never raises.
    """
    if not is_enabled():
        return
    try:
        entry = {
            "kind": str(kind or "command"),
            "failed_rules": [str(r) for r in (failed_rules or [])],
            "detail": str(detail or ""),
            "offending": str(offending or ""),
        }
        with _lock:
            if entry not in _pending:  # dedupe — extract_commands runs >1x/turn
                _pending.append(entry)
    except Exception:
        pass


def get_pending() -> list[dict]:
    with _lock:
        return [dict(f) for f in _pending]


def clear_pending() -> None:
    with _lock:
        _pending.clear()


def render_command_failed_block(pending: list[dict]) -> str | None:
    """Render queued command failures into one self-describing block.

    Returns None when there is nothing pending. Each entry surfaces its
    ``kind``, ``failed_rules``, the raw ``detail``, and the ``offending`` input
    verbatim (truncated for KV-budget safety).
    """
    if not pending:
        return None
    lines = [
        f"{_TAG} — your prior command(s) failed in the runtime, not in your "
        "reasoning. The detail below is the actual reason; one repair attempt "
        "this turn."
    ]
    for f in pending:
        if not isinstance(f, dict):
            continue
        kind = str(f.get("kind", "command") or "command")
        rules = f.get("failed_rules") or []
        detail = str(f.get("detail", "") or "")
        offending = str(f.get("offending", "") or "")[:_MAX_OFFENDING_CHARS]
        head = f"- kind: {kind}"
        if rules:
            head += " | rules: " + ", ".join(str(r) for r in rules)
        lines.append(head)
        if detail:
            lines.append(f"  detail: {detail}")
        if offending:
            lines.append("  offending:")
            lines.append(f"  {offending}")
    lines.append(_END_TAG)
    return "\n".join(lines)


def is_non_convergent(
    public_answer: str,
    *,
    had_tool_call: bool = False,
    done_signal: bool = False,
) -> bool:
    """True when a turn produced NO public answer AND NO tool action.

    The observed "didn't say anything" failure: the model spent the turn
    reasoning (or truncated mid-``<think>``) and nothing — no answer, no tool
    call — reached the user. A tool call or a [TOOL_LOOP_DONE] signal both count
    as convergence (the turn did something); a non-empty public answer is
    convergence. Empty/whitespace public + no action + no done = non-convergent.
    """
    if had_tool_call or done_signal:
        return False
    return not str(public_answer or "").strip()


_THINK_OPEN_RE = re.compile(r"<(?:think|analysis|reasoning)>", re.IGNORECASE)
_THINK_CLOSE_RE = re.compile(r"</(?:think|analysis|reasoning)>", re.IGNORECASE)


def answer_trapped_in_think(raw: str) -> bool:
    """True when a think block was left open — more ``<think>``/``<analysis>``/
    ``<reasoning>`` opens than closes — so the model's answer is stranded in the
    thinking lane and never reaches the user.

    The 2026-06-16 "behind the reasoning curtain" failure: the model emits
    ``<think>A</think>`` then re-opens ``<think>B-with-answer`` (often after a
    ``[CHANNEL: ASSISTANT]`` echo) and never closes it. The depth-counting
    normalizer keeps that trailing block in the thinking lane, so the public
    answer is empty. Distinct from genuine non-convergence (balanced tags — the
    model truly only reasoned and never answered).
    """
    text = str(raw or "")
    return len(_THINK_OPEN_RE.findall(text)) > len(_THINK_CLOSE_RE.findall(text))


_CHANNEL_ECHO_RE = re.compile(r"\[(?:CHANNEL|AGENT)\b[^\]]*\]", re.IGNORECASE)


def recover_trapped_answer(raw: str) -> str:
    """Pull the public answer out of a turn whose ``<think>`` tags are unbalanced.

    Mirrors ``engine.agent_server._clean_agent_response`` so the recovered text
    matches what an external peer already receives: ``strip_tag_blocks``'s
    orphan-pass removes balanced think blocks AND the trailing orphan ``<think>``
    tag, preserving the answer text after it; then channel/agent echoes and the
    ``[TOOL_LOOP_DONE]`` sentinel are stripped. Returns ``""`` when nothing public
    remains (e.g. balanced reasoning-only, or an empty trailing block).
    """
    from core.internal_tags import EXTERNAL_STRIP_TAGS, strip_tag_blocks

    text = strip_tag_blocks(str(raw or ""), EXTERNAL_STRIP_TAGS)
    text = _CHANNEL_ECHO_RE.sub("", text)
    text = text.replace("[TOOL_LOOP_DONE]", "")
    return text.strip()


def should_recover_trapped(raw: str) -> bool:
    """True when an unbalanced-``<think>`` turn has a CONFIDENTLY recoverable
    public answer — recover it and suppress the contaminating re-emit; otherwise
    regen (the blessed worst-case).

    Confident = a completed reasoning block (``>=1`` close) THEN a re-opened
    trailing block (``opens > closes``) whose stripped content is non-empty (the
    documented 2026-06-16 "behind the reasoning curtain" shape). Tag-count only —
    no content heuristics. Biased to ``False`` (regen): a wrong *recover* silently
    surfaces reasoning as the answer (corrupts a rated turn), whereas a wrong
    *regen* is a visible, skippable re-answer. So pure-unclosed reasoning
    (``closes == 0``, genuine truncation) and the nothing-recoverable case both
    regen. Known limitation: when the trailing block is reasoning-not-answer, tag
    counts can't tell, so it recovers — trapped raws are logged on fire so real
    frequency is observable. See advisor 2026-06-24.
    """
    text = str(raw or "")
    opens = len(_THINK_OPEN_RE.findall(text))
    closes = len(_THINK_CLOSE_RE.findall(text))
    if not (closes >= 1 and opens > closes):
        return False
    return bool(recover_trapped_answer(text))


def build_non_convergence_nudge(ask: str, *, trapped: bool) -> str:
    """Re-prompt for a turn that produced no public answer and no tool action.

    When the answer is trapped in an unbalanced ``<think>`` (per
    ``answer_trapped_in_think``), give a *surgical* instruction so the single
    allowed retry lands a clean answer OUTSIDE the tags — the generic "stop
    deliberating" nudge took two re-emits to recover in the wild. Otherwise the
    generic nudge (the model genuinely only reasoned).
    """
    ask = str(ask or "").strip()
    asked = f' You were asked: "{ask[:200]}".' if ask else ""
    if trapped:
        return (
            "Your answer was written inside a <think> block and never reached the "
            "user — your think tags are unbalanced (an open <think> with no closing "
            "</think>). Re-emit your answer as plain text OUTSIDE any "
            "<think>/</think> tags, and close every <think> you open." + asked
        )
    return (
        "Your last turn produced only internal reasoning — no answer reached the "
        "user and no tool call fired. Stop deliberating and act now: emit the tool "
        "call (with ALL required fields) or the final answer." + asked
        + " If a prior command failed, read the [COMMAND_FAILED] / [*_REJECTED] "
        "detail, fix that specific field, and re-emit. Keep it to the action — one "
        "clean command or a direct answer, no more planning."
    )


def _last_non_ephemeral_user_idx(messages: list[dict]) -> int:
    for i in range(len(messages) - 1, -1, -1):
        msg = messages[i]
        if msg.get("role") == "user" and not msg.get("ephemeral"):
            return i
    return -1


def command_feedback_interceptor(messages: list[dict], config: dict) -> list[dict] | None:
    """Direct-inject the [COMMAND_FAILED] block before the latest user turn.

    Registered in bootstrap NOT through the coalescer budget gate — a repair
    instruction is load-bearing and must never be silently dropped. Clears
    pending on inject (one repair attempt per turn — bearing's discipline; a
    repeat failure re-records a fresh entry next turn). Returns None when
    disabled, nothing pending, already injected, or no user turn to anchor to.
    """
    if not is_enabled():
        return None
    pending = get_pending()
    if not pending:
        return None
    block = render_command_failed_block(pending)
    if not block:
        return None
    for msg in messages:
        if msg.get("source") == _SOURCE or _TAG in str(msg.get("content", "")):
            return None  # double-fire defense
    idx = _last_non_ephemeral_user_idx(messages)
    if idx < 0:
        return None
    result = list(messages)
    result.insert(
        idx,
        {"role": "user", "content": block, "ephemeral": True, "source": _SOURCE},
    )
    clear_pending()
    return result
