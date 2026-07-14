"""Capture explicit self-curiosity markers into the ACU substrate.

This is deliberately not an extractor. The model must emit atomic canonical
forms inside a private marker block:

    <curiosity>
    monolith | is curious_about | why contradiction detection feels load-bearing
    </curiosity>

Only Monolith-subject triples are accepted. The ACU intake owns normalization,
deduplication, reinforcement, evidence spans, and audit events.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass

from core.acatalepsy.atomicity import is_atomic
from core.acatalepsy.normalize import normalize_canonical, parse_triple

_FLAG_ENV = "MONOLITH_CURIOSITY_CAPTURE_V1"
_BLOCK_RE = re.compile(
    # Terminal-only by design: examples or discussion in the middle of a reply
    # must not become ACUs. If a <frame> heartbeat is also emitted, curiosity
    # sits immediately before it so both terminal surfaces can coexist.
    r"(?:^|\n)\s*<curiosity>\s*(?P<body>.*?)\s*</curiosity>"
    r"(?P<suffix>\s*(?:<frame>\s*.*?\s*</frame>\s*)?\Z)",
    flags=re.IGNORECASE | re.DOTALL,
)
_LINE_PREFIX_RE = re.compile(r"^\s*(?:[-*]\s+|\d+[.)]\s+)?")


@dataclass(frozen=True)
class CuriosityCaptureReport:
    captured: int
    rejected: int
    claims: tuple[str, ...]
    message: str


def capture_enabled() -> bool:
    raw = str(os.environ.get(_FLAG_ENV, "0")).strip().lower()
    return raw in {"1", "true", "yes", "on"}


def strip_curiosity_blocks(text: str) -> str:
    """Remove private curiosity marker blocks from user/peer-visible text."""
    return _BLOCK_RE.sub(lambda m: m.group("suffix") or "", str(text or "")).strip()


def _candidate_lines(text: str) -> list[str]:
    out: list[str] = []
    for match in _BLOCK_RE.finditer(str(text or "")):
        for line in match.group("body").splitlines():
            item = _LINE_PREFIX_RE.sub("", line.strip()).strip()
            if not item or item.startswith("#") or item.startswith("```"):
                continue
            if "|" not in item:
                continue
            out.append(item.strip("`").strip())
    return out


def extract_curiosity_claims(text: str) -> tuple[tuple[str, ...], int]:
    """Return accepted canonical claims plus rejected candidate-line count."""
    claims: list[str] = []
    seen: set[str] = set()
    rejected = 0
    for raw in _candidate_lines(text):
        canonical = normalize_canonical(raw)
        triple = parse_triple(canonical)
        if triple is None or triple.entity_a != "monolith" or not is_atomic(canonical).ok:
            rejected += 1
            continue
        if canonical not in seen:
            seen.add(canonical)
            claims.append(canonical)
    return tuple(claims), rejected


def _run_detectors() -> None:
    try:
        from core.curiosity import detect_pulls
        detect_pulls()
    except Exception:
        pass
    try:
        from core.identity_emergence import detect_emergence_best_effort
        detect_emergence_best_effort(min_new_acus=1)
    except Exception:
        pass


def capture_from_assistant_text(
    text: str,
    *,
    source_event: int | None = None,
    force: bool = False,
    run_detectors: bool = True,
) -> CuriosityCaptureReport:
    """Ingest explicit Monolith self-curiosity claims as self-provenance ACUs."""
    if not force and not capture_enabled():
        return CuriosityCaptureReport(0, 0, (), "disabled")

    claims, rejected = extract_curiosity_claims(text)
    if not claims:
        return CuriosityCaptureReport(0, rejected, (), "no curiosity claims")

    from core.acatalepsy import intake

    captured: list[str] = []
    for claim in claims:
        res = intake.ingest_l1(
            raw_form=claim,
            provenance="self",
            evidence_span=claim,
            source_event=source_event,
        )
        if res.outcome == "rejected":
            rejected += 1
            continue
        captured.append(claim)

    if captured and run_detectors:
        _run_detectors()

    return CuriosityCaptureReport(
        len(captured),
        rejected,
        tuple(captured),
        f"captured {len(captured)} curiosity claim(s)",
    )
