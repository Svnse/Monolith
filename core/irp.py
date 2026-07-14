"""IRP label contract for model-visible claim confidence.

IRP exposes coarse labels to the model while keeping numeric veracity in the
substrate. V0 is intentionally narrow: claim surfaces and Observer output only.
Normal chat text must pass through unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal


IRPLabel = Literal[
    "LOCKED",
    "VERIFIED",
    "ACCEPTED",
    "PROVISIONAL",
    "CONTESTED",
    "REJECTED",
]

IRP_SCOPES = frozenset({"claim", "observer"})

LOCKED: IRPLabel = "LOCKED"
VERIFIED: IRPLabel = "VERIFIED"
ACCEPTED: IRPLabel = "ACCEPTED"
PROVISIONAL: IRPLabel = "PROVISIONAL"
CONTESTED: IRPLabel = "CONTESTED"
REJECTED: IRPLabel = "REJECTED"


@dataclass(frozen=True)
class LabeledText:
    """One model-visible text span with its IRP label."""

    label: IRPLabel
    text: str
    scope: str

    def render(self) -> str:
        body = self.text.strip()
        if not body:
            return f"[{self.label}]"
        return f"[{self.label}] {body}"


def map_veracity_to_label(
    value: Any,
    *,
    locked: bool = False,
    rejected: bool = False,
) -> IRPLabel:
    """Map hidden substrate veracity to a coarse model-visible IRP label.

    Numeric values stay internal. The model sees only the returned label.
    """

    if locked:
        return LOCKED
    if rejected:
        return REJECTED
    try:
        score = float(value)
    except (TypeError, ValueError):
        return PROVISIONAL
    if score >= 75.0:
        return VERIFIED
    if score >= 5.0:
        return ACCEPTED
    if score > 0.0:
        return PROVISIONAL
    if score > -20.0:
        return CONTESTED
    return REJECTED


def is_locked_claim(row: dict[str, Any]) -> bool:
    """Return whether an ACU row should be treated as an immutable seed."""

    if not isinstance(row, dict):
        return False
    locked_raw = row.get("locked")
    if locked_raw is True:
        return True
    if isinstance(locked_raw, (int, float)) and locked_raw != 0:
        return True
    source = str(row.get("source") or "").strip().lower()
    if source in {"origin_0", "identity_origin_0", "identity_seed"}:
        return True
    reason = str(row.get("lock_reason") or "").strip().lower()
    return reason.startswith("origin_0")


def label_for_claim(row: dict[str, Any]) -> IRPLabel:
    """Compute the model-visible IRP label for an ACU-like row."""

    if not isinstance(row, dict):
        return PROVISIONAL
    rejected = str(row.get("state") or "").strip().lower() == "rejected"
    return map_veracity_to_label(
        row.get("veracity"),
        locked=is_locked_claim(row),
        rejected=rejected,
    )


def label_text(
    text: str,
    *,
    scope: str,
    label: IRPLabel | None = None,
    row: dict[str, Any] | None = None,
) -> str:
    """Render a labeled span for IRP scopes, leaving normal chat untouched."""

    body = str(text or "")
    if scope not in IRP_SCOPES:
        return body
    effective = label or (label_for_claim(row) if row is not None else PROVISIONAL)
    return LabeledText(label=effective, text=body, scope=scope).render()


def strip_irp_prefix(text: str) -> str:
    """Remove a leading IRP label from text for comparisons/tests."""

    body = str(text or "")
    for label in (LOCKED, VERIFIED, ACCEPTED, PROVISIONAL, CONTESTED, REJECTED):
        prefix = f"[{label}]"
        if body.startswith(prefix):
            return body[len(prefix):].lstrip()
    return body
