"""READY-time response verifier — deterministic post-stream checks.

Runs after streaming finalizes (status → READY, no further tool
followups) and before display finalization. Observation-only in v1:
the verdict is stashed on the session payload and emitted to the
debug trace. The verifier does NOT mutate the assistant's text and
does NOT inject anything into the model's context.

This is deliberate. Self-judgment text in the prompt context causes
spiraling — advanced Monolith's audit (`self_aware_loop.py:5-9`)
documents it. The closed-loop pattern here is: deterministic checks
→ verdict on session → caller-visible (debug trace + future UI badge).
The model never reads its own verdict.

Checks (cheapest-fail-first):
  1. Structural — leaked internal tags (think/analysis/reasoning/
     monolith_cmd/tool_call/acatalepsy) in the public answer, or a
     dangling ``<tool_evidence>`` open tag without close. Defense-
     in-depth: ``AssistantStreamNormalizer`` already routes these
     into separate lanes; this check fires on regression.
  2. Empty public answer — warn-only; the existing empty-answer
     repair owns recovery.
  3. Tool evidence presence — only when callers report tools used.
     v1 callers pass ``tools_used=[]`` until per-turn tool tracking
     lands; this check is dormant.
  4. Weak completion signal — composite from ``compute_success_signal``.
     Hard-fails when public answer claims completion (``fixed``,
     ``done``, etc.) on code/debug task_types without supporting
     evidence; warns otherwise.
  5. Memory contradiction — not implemented in v1 (no mono_verify).

v1 omissions vs advanced Monolith:
  - terminal_step_reached: path_parser not ported; always None
  - memory contradiction (mono_verify): not ported
  - per-turn tool outcomes / parse retries: callsite signals not tracked

Flag: MONOLITH_VERIFIER_V1 (default ON). Set =0 to disable.
"""
from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass, field
from typing import Any

from core.turn_outcome import compute_success_signal


VERDICT_PASS = "pass"
VERDICT_WARN = "warn"
VERDICT_HARD_FAIL = "hard_fail"
VERDICT_SKIPPED = "skipped"

SEVERITY_WARN = "warn"
SEVERITY_HARD_FAIL = "hard_fail"

_FLAG_ENV = "MONOLITH_VERIFIER_V1"

_TOOL_EVIDENCE_OPEN = "<tool_evidence>"
_TOOL_EVIDENCE_CLOSE = "</tool_evidence>"

# Defense-in-depth: AssistantStreamNormalizer routes these tags into
# separate lanes (acu_text / think_text / etc), so a clean answer_text
# should never contain them. This regex fires on regression — e.g.,
# normalizer state machine bug, archive-reload bypass, or a future
# code path that constructs the public answer without going through
# the normalizer.
#
# Tag set lives in core/internal_tags.INTERNAL_LEAK_TAGS so the live-
# stream sanitizer (core/pipeline_policies/output_sanitizer.py) and
# this terminal verifier stay in lockstep without manual re-syncing.
from core.internal_tags import INTERNAL_LEAK_TAGS, make_leak_detection_pattern
_RAW_INTERNAL_TAG_RE = make_leak_detection_pattern(INTERNAL_LEAK_TAGS)
_UNCLOSED_TOOL_EVIDENCE_RE = re.compile(
    r"<tool_evidence\b[^>]*>(?!.*</tool_evidence>)",
    flags=re.IGNORECASE | re.DOTALL,
)
_ASSERTIVE_DONE_RE = re.compile(
    r"\b(?:fixed|implemented|completed|resolved|verified|tested|done|working|passes?)\b",
    flags=re.IGNORECASE,
)


@dataclass(frozen=True)
class VerificationFinding:
    code: str
    severity: str
    message: str
    source: str = "response_verifier"
    detail: dict[str, Any] = field(default_factory=dict)

    def to_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "code": self.code,
            "severity": self.severity,
            "message": self.message,
            "source": self.source,
        }
        if self.detail:
            payload["detail"] = dict(self.detail)
        return payload


@dataclass(frozen=True)
class VerificationResult:
    verdict: str
    findings: tuple[VerificationFinding, ...] = ()
    duration_ms: float = 0.0
    success_signal: dict[str, Any] | None = None

    @property
    def ok(self) -> bool:
        return self.verdict == VERDICT_PASS

    def summary_lines(self, *, limit: int = 3) -> list[str]:
        lines: list[str] = []
        for finding in self.findings[: max(0, int(limit))]:
            lines.append(f"{finding.code}: {finding.message}")
        return lines

    def to_payload(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict,
            "duration_ms": round(float(self.duration_ms), 3),
            "findings": [f.to_payload() for f in self.findings],
            "success_signal": self.success_signal,
        }


def _flag_enabled() -> bool:
    raw = str(os.environ.get(_FLAG_ENV, "1")).strip().lower()
    return raw in {"1", "true", "yes", "on"}


def verifier_enabled(config: dict[str, Any] | None = None) -> bool:
    """Single kill switch — env flag only.

    Advanced Monolith gated this behind both a slash toggle and a flag.
    Monolith doesn't have the slash toggle yet; the env flag is
    the only switch. The ``config`` parameter is reserved for the future
    slash command without changing the verifier's signature.
    """
    return _flag_enabled()


def has_complete_tool_evidence(raw_answer: str) -> bool:
    text = raw_answer or ""
    return _TOOL_EVIDENCE_OPEN in text and _TOOL_EVIDENCE_CLOSE in text


def format_public_note(result: VerificationResult) -> str:
    """Render the verdict as a human-readable note for trace/UI surfaces.

    Not auto-injected into the assistant text — call this from a UI
    surface that wants to display the warning. The verifier itself
    never mutates the model's context.
    """
    lines = result.summary_lines(limit=3)
    if not lines:
        lines = ["response verification could not prove this answer is complete"]
    body = "\n".join(f"- {line}" for line in lines)
    return "[VERIFY WARNING]\n" + body


def verify_response(
    *,
    raw_answer: str,
    public_answer: str,
    tools_used: list[str] | tuple[str, ...] | None = None,
    tool_outcomes: list[tuple[str, bool]] | tuple[tuple[str, bool], ...] | None = None,
    task_type: str = "chat",
    assumptions: list[str] | tuple[str, ...] | None = None,
    no_parse_errors: bool | None = None,
    no_consecutive_tool_failures: bool | None = None,
) -> VerificationResult:
    """Run deterministic checks in cheapest-fail-first order."""
    started = time.perf_counter()
    findings: list[VerificationFinding] = []
    tools = [str(t) for t in (tools_used or []) if str(t or "").strip()]
    outcomes = list(tool_outcomes or [])
    tt = str(task_type or "chat").strip().lower() or "chat"

    public = (public_answer or "").strip()
    raw = raw_answer or ""

    # 1. Structural.
    if _RAW_INTERNAL_TAG_RE.search(public):
        findings.append(_hard(
            "raw_internal_tag",
            "public answer still contains an internal reasoning or command tag",
        ))
    if _UNCLOSED_TOOL_EVIDENCE_RE.search(raw):
        findings.append(_hard(
            "dangling_tool_evidence",
            "tool evidence block is opened but not closed",
        ))

    # 2. Empty-answer guard. Empty public answers belong to the existing
    # empty-answer repair first; the verifier records the warning and stops.
    if not public:
        findings.append(_warn(
            "empty_public_answer",
            "public answer is empty after display normalization",
        ))
        return _finalize(findings, started, success_signal=None)

    # 3. Tool evidence presence.
    if tools and not has_complete_tool_evidence(raw):
        findings.append(_hard(
            "missing_tool_evidence",
            "tool results were used but the answer lacks a complete tool_evidence block",
            detail={"tools_used": tools},
        ))

    # 4. Task-completion sentinel. terminal_step_reached is None in v1
    # (path_parser not ported); recipe handles None as "no evidence either way."
    last_tool_ok = bool(outcomes[-1][1]) if outcomes else None
    success_signal = compute_success_signal(
        tt,
        no_parse_errors=no_parse_errors,
        no_consecutive_tool_failures=no_consecutive_tool_failures,
        no_unresolved_assumptions=(len(list(assumptions or [])) == 0),
        last_tool_ok=last_tool_ok,
        terminal_step_reached=None,
    )
    if not bool(success_signal.get("composite", False)):
        if tt in {"code", "debug"} and _ASSERTIVE_DONE_RE.search(public):
            findings.append(_hard(
                "unsupported_completion_claim",
                "answer claims completion but turn outcome lacks required success evidence",
                detail={"task_type": tt, "components": success_signal.get("components")},
            ))
        else:
            findings.append(_warn(
                "weak_completion_signal",
                "turn outcome did not fully prove task completion",
                detail={"task_type": tt, "components": success_signal.get("components")},
            ))

    return _finalize(findings, started, success_signal=success_signal)


def _hard(code: str, message: str, *, detail: dict[str, Any] | None = None) -> VerificationFinding:
    return VerificationFinding(
        code=code,
        severity=SEVERITY_HARD_FAIL,
        message=message,
        detail=dict(detail or {}),
    )


def _warn(code: str, message: str, *, detail: dict[str, Any] | None = None) -> VerificationFinding:
    return VerificationFinding(
        code=code,
        severity=SEVERITY_WARN,
        message=message,
        detail=dict(detail or {}),
    )


def _finalize(
    findings: list[VerificationFinding],
    started: float,
    *,
    success_signal: dict[str, Any] | None,
) -> VerificationResult:
    if any(f.severity == SEVERITY_HARD_FAIL for f in findings):
        verdict = VERDICT_HARD_FAIL
    elif findings:
        verdict = VERDICT_WARN
    else:
        verdict = VERDICT_PASS
    return VerificationResult(
        verdict=verdict,
        findings=tuple(findings),
        duration_ms=(time.perf_counter() - started) * 1000.0,
        success_signal=success_signal,
    )
