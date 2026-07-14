"""Fault detectors for four named runtime failure modes.

Each function returns a FaultRecord (not yet persisted) or None.
They NEVER raise — caller wraps in try/except anyway, but defensive
no-raise is the contract.

Detector functions:
  detect_markdown_corruption  -- unbalanced code fences / backticks
  detect_tool_no_fire         -- intent language without a tool_call block
  detect_think_leak           -- unbalanced think tags in response
  detect_regen_mismatch       -- tool_result refs exceeding frame count

Context dict keys used:
  frame_traces   list[dict] from turn's frame trace messages (optional)
"""
from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from core.fault_response import FaultRecord


# ── shared helpers ─────────────────────────────────────────────────────────


def _make_record(
    turn_id: str,
    fault_kind: str,
    detector_name: str,
    evidence: str | None,
    metadata: dict | None = None,
) -> "FaultRecord":
    """Build a FaultRecord without persisting it."""
    from datetime import datetime, timezone
    from core.fault_response import FaultRecord
    return FaultRecord(
        id=-1,  # not persisted yet
        turn_id=turn_id,
        fault_kind=fault_kind,
        detected_at=datetime.now(timezone.utc).isoformat(),
        detector_name=detector_name,
        evidence=evidence,
        metadata=metadata or {},
    )


# ── markdown corruption ────────────────────────────────────────────────────

# Match a fully-closed fenced block so we can scrub it before counting
# residual triple-backtick markers.
_FENCE_CLOSED_RE = re.compile(r"```[^\n`]*\n.*?```\s*?", re.MULTILINE | re.DOTALL)


def detect_markdown_corruption(
    response_text: str,
    turn_id: str,
    context: dict,
) -> "FaultRecord | None":
    """Detect unbalanced markdown control characters.

    Checks (high-precision / low-recall to avoid false-positive noise):
      1. Odd number of triple-backtick markers → unclosed code fence
      2. Odd number of single backticks (after scrubbing closed fences) →
         dangling inline code marker

    Does NOT check bold/italic markers — too many false positives from
    normal prose containing literal asterisks.
    """
    try:
        text = str(response_text or "")
        if not text:
            return None

        fires: list[str] = []

        # Check triple-backtick fence balance.
        fence_count = text.count("```")
        if fence_count % 2 == 1:
            fires.append(f"fence_imbalance(count={fence_count})")

        # Strip closed fences before counting inline backticks.
        scrubbed = _FENCE_CLOSED_RE.sub("", text)
        backtick_count = scrubbed.count("`")
        if backtick_count % 2 == 1:
            fires.append(f"single_backtick_imbalance(count={backtick_count})")

        if not fires:
            return None

        evidence = "; ".join(fires)
        return _make_record(
            turn_id=turn_id,
            fault_kind="markdown_corruption",
            detector_name="detect_markdown_corruption",
            evidence=evidence,
            metadata={"rules_fired": fires},
        )
    except Exception:
        return None


# ── tool no-fire ───────────────────────────────────────────────────────────

# Intent patterns: model says it will invoke a tool.
_TOOL_INTENT_RE = re.compile(
    r"\bI(?:'m| am|'ll| will| can)?\s+(?:going\s+to\s+)?(?:check|search|look(?:\s+up)?|"
    r"verify|run|execute|call|fetch|query|retrieve|browse|read)\b",
    re.IGNORECASE,
)

# A <tool_call> envelope was emitted.
_TOOL_CALL_PRESENT_RE = re.compile(r"<tool_call\b", re.IGNORECASE)


def detect_tool_no_fire(
    response_text: str,
    turn_id: str,
    context: dict,
) -> "FaultRecord | None":
    """Detect tool-intent language without a corresponding <tool_call> block.

    Conservative: only fires when BOTH conditions hold:
      1. Response contains intent pattern (I'll check / I will search / etc.)
      2. Response contains NO <tool_call> envelope.

    High false-positive risk for conversational mentions of searching — the
    intent regex is tuned to first-person future-tense only to reduce noise.
    """
    try:
        text = str(response_text or "")
        if not text:
            return None

        intent_match = _TOOL_INTENT_RE.search(text)
        if intent_match is None:
            return None

        # If there IS a tool_call block, the intent was fulfilled.
        if _TOOL_CALL_PRESENT_RE.search(text):
            return None

        evidence = text[max(0, intent_match.start() - 20): intent_match.end() + 40].strip()
        return _make_record(
            turn_id=turn_id,
            fault_kind="tool_no_fire",
            detector_name="detect_tool_no_fire",
            evidence=evidence,
            metadata={"intent_span": [intent_match.start(), intent_match.end()]},
        )
    except Exception:
        return None


# ── think-leak ────────────────────────────────────────────────────────────

_THINK_OPEN_RE = re.compile(r"<think\b", re.IGNORECASE)
_THINK_CLOSE_RE = re.compile(r"</think\s*>", re.IGNORECASE)

# Acatalepsy markers that should never appear in rendered output.
_ACATALEPSY_RE = re.compile(
    r"<acatalepsy\b|</acatalepsy|<ACU\b|<VIN\b|<CCG\b|<MonoVerify\b",
    re.IGNORECASE,
)

# Code regions where a tag is *mentioned* (documented), not emitted. Counting
# tags inside these inflates the imbalance — a model writing "...within
# `<think>` blocks" or a fenced example is not leaking reasoning.
_CODE_FENCE_RE = re.compile(r"```.*?```|~~~.*?~~~", re.DOTALL)
_INLINE_CODE_RE = re.compile(r"`[^`\n]*`")


def _strip_code_regions(text: str) -> str:
    """Blank out fenced blocks and inline code spans so mentioned tags don't
    count as real structural tags. Real tags live outside code formatting, so
    a genuine leak (e.g. an unclosed <think> in prose) still survives."""
    text = _CODE_FENCE_RE.sub(" ", text)
    text = _INLINE_CODE_RE.sub(" ", text)
    return text


def detect_think_leak(
    response_text: str,
    turn_id: str,
    context: dict,
) -> "FaultRecord | None":
    """Detect unbalanced or leaked think/acatalepsy content in the response.

    Checks:
      1. Unbalanced <think>...</think> tags (open count != close count) —
         a missing </think> means raw reasoning content reached the output.
      2. Acatalepsy markers in the response — should never appear post-render.

    Workers intentionally emit balanced <think>...</think> pairs during
    reasoning — so detecting PRESENCE alone would always fire.  Only an
    IMBALANCE (or acatalepsy markers) represents the failure mode.
    """
    try:
        text = str(response_text or "")
        if not text:
            return None

        fires: list[str] = []

        # Count only real structural tags — ignore ones mentioned in code spans.
        scan = _strip_code_regions(text)
        open_count = len(_THINK_OPEN_RE.findall(scan))
        close_count = len(_THINK_CLOSE_RE.findall(scan))
        if open_count != close_count:
            fires.append(
                f"think_tag_imbalance(open={open_count}, close={close_count})"
            )

        aca_match = _ACATALEPSY_RE.search(scan)
        if aca_match:
            fires.append(f"acatalepsy_marker(at={aca_match.start()})")

        if not fires:
            return None

        evidence = "; ".join(fires)
        return _make_record(
            turn_id=turn_id,
            fault_kind="think_leak",
            detector_name="detect_think_leak",
            evidence=evidence,
            metadata={
                "think_open_count": open_count,
                "think_close_count": close_count,
                "has_acatalepsy_marker": aca_match is not None,
            },
        )
    except Exception:
        return None


# ── regen mismatch ─────────────────────────────────────────────────────────

# References like "tool_result_3", "tool_result_12" in the assistant's text.
_TOOL_RESULT_REF_RE = re.compile(r"\btool_result_(\d+)\b", re.IGNORECASE)


def detect_regen_mismatch(
    response_text: str,
    turn_id: str,
    context: dict,
) -> "FaultRecord | None":
    """Detect model references to tool results that don't exist in the frame.

    Checks: any "tool_result_N" reference in the response where N exceeds
    the count of tool-result messages in context["frame_traces"].

    Context dict:
      frame_traces  list[dict] — the final_messages snapshot for this turn
                    (each dict has at minimum a "role" key). If absent,
                    detector skips rather than false-positiving.
    """
    try:
        text = str(response_text or "")
        if not text:
            return None

        frame_traces = context.get("frame_traces")
        if not isinstance(frame_traces, list):
            return None

        # Count tool-result messages in the frame.
        tool_result_count = sum(
            1 for m in frame_traces
            if isinstance(m, dict) and str(m.get("role", "")).lower() == "tool"
        )

        refs = _TOOL_RESULT_REF_RE.findall(text)
        if not refs:
            return None

        bad_refs = [int(n) for n in refs if int(n) >= tool_result_count]
        if not bad_refs:
            return None

        evidence = f"refs {sorted(bad_refs)} exceed frame tool_result_count={tool_result_count}"
        return _make_record(
            turn_id=turn_id,
            fault_kind="regen_mismatch",
            detector_name="detect_regen_mismatch",
            evidence=evidence,
            metadata={
                "bad_refs": bad_refs,
                "tool_result_count": tool_result_count,
            },
        )
    except Exception:
        return None
