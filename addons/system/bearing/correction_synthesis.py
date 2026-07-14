"""MonoFrame v2 — the two LLM calls + the promotion gate.

  synthesize() — call 3: read {bad_frame, better_frame, stateless control} and
                 name the PROCESS MOVE, the ANCHOR the better frame chose, the
                 bad frame's ANCHOR ERROR, and the APERTURE. Builds a candidate
                 card (never promoted here).
  advise()     — the promotion GATE (not a teacher): a critic that ATTACKS the
                 candidate on five tests. Fails closed.
  gate()       — promote iff source is HUMAN and the advisor passed.

Parsers are pure. Orchestration takes an injected ``generate`` (defaults to the
lockless sync_bridge tap, like stateless_reframe). Model output is a small
labeled-line format, parsed tolerantly — no numbers anywhere (E's design).
"""
from __future__ import annotations

import dataclasses
import re
from collections.abc import Callable
from typing import Any

from . import correction_card as cc

# ── prompts (encode E's framework) ──────────────────────────────────

_ANCHOR_MENU = "explicit_noun | implied_task | live_constraint | user_state | continuity"

_SYNTH_PROMPT = (
    "You are comparing three one-sentence situational frames for the SAME turn:\n"
    "  BAD     — the frame the system actually carried (it erred).\n"
    "  BETTER  — a human-supplied correction (ground truth).\n"
    "  CONTROL — a momentum-free machine re-derivation (a SIGNED CONTROL, not truth).\n\n"
    "Name the PROCESS MOVE that turns BAD into BETTER — a cognitive move, bound to "
    "this example, NOT a general rule. Then classify the ANCHOR the BETTER frame "
    f"chose ({_ANCHOR_MENU}) — the real driver, not the loudest surface noun. State "
    "the BAD frame's ANCHOR ERROR (e.g. mirrored_loud_noun). State the APERTURE the "
    "BETTER frame sets: collapse (narrow to one thing) or diffuse (hold breadth) — "
    "the correction must preserve the right direction, not invert it.\n\n"
    "Reply EXACTLY in this shape, no numbers, no preamble:\n"
    "PROCESS_MOVE: <one sentence>\n"
    f"ANCHOR: <{_ANCHOR_MENU}>\n"
    "ANCHOR_ERROR: <short phrase>\n"
    "APERTURE: <collapse | diffuse>"
)

_ADVISOR_PROMPT = (
    "You are a PROMOTION GATE, not a teacher. Attack this proposed frame-correction "
    "card. It may be stored ONLY if it survives all five tests. Answer each yes/no, "
    "defaulting to NO when unsure (fail closed):\n"
    "HUMAN_GROUNDED: is the better frame a real human correction, not a machine guess?\n"
    "PROCESS_SHAPED: is the process move a cognitive move, not content-specific trivia?\n"
    "NOT_OVERFIT: is it free of over-fitting to this turn's specific nouns/topic?\n"
    "SIGN_PRESERVED: did the correction preserve the collapse/diffuse direction (not invert it)?\n"
    "REAL_ANCHOR: did it choose the real anchor rather than mirror the loud surface noun?\n\n"
    "Reply EXACTLY, no preamble:\n"
    "HUMAN_GROUNDED: <yes|no>\n"
    "PROCESS_SHAPED: <yes|no>\n"
    "NOT_OVERFIT: <yes|no>\n"
    "SIGN_PRESERVED: <yes|no>\n"
    "REAL_ANCHOR: <yes|no>"
)

# ── pure parsers ────────────────────────────────────────────────────

_TRUTHY = {"yes", "true", "y", "1"}


def _field(text: str, label: str) -> str:
    m = re.search(rf"^{re.escape(label)}\s*:\s*(.+)$", text or "", re.IGNORECASE | re.MULTILINE)
    return m.group(1).strip() if m else ""


def parse_synthesis(text: str) -> dict[str, Any]:
    """Parse the labeled synthesis output. Tolerant; unknown anchor -> the
    conservative EXPLICIT_NOUN (assume the loud-noun trap until shown otherwise)."""
    anchor_raw = _field(text, "ANCHOR").lower()
    anchor = cc.AnchorType.EXPLICIT_NOUN
    for a in cc.AnchorType:
        if a.value in anchor_raw:
            anchor = a
            break
    aperture_raw = _field(text, "APERTURE").lower()
    aperture = cc.Aperture.DIFFUSE if "diffuse" in aperture_raw else cc.Aperture.COLLAPSE
    return {
        "process_move": _field(text, "PROCESS_MOVE"),
        "anchor_type": anchor,
        "anchor_error": _field(text, "ANCHOR_ERROR"),
        "aperture": aperture,
    }


def parse_advisor(text: str) -> cc.AdvisorVerdict:
    """Parse the five gate checks. Missing/odd -> False (fail closed)."""
    def _yn(label: str) -> bool:
        return _field(text, label).lower() in _TRUTHY
    return cc.AdvisorVerdict(
        human_grounded=_yn("HUMAN_GROUNDED"),
        process_shaped=_yn("PROCESS_SHAPED"),
        not_overfit=_yn("NOT_OVERFIT"),
        sign_preserved=_yn("SIGN_PRESERVED"),
        real_anchor=_yn("REAL_ANCHOR"),
    )


# ── orchestration (injected generate) ───────────────────────────────


def _default_generate(base_config: dict[str, Any]) -> Callable[[list[dict]], str]:
    from engine.sync_bridge import generate_sync_from_config

    def generate(msgs: list[dict]) -> str:
        return generate_sync_from_config(base_config, msgs)

    return generate


def synthesize(
    *,
    bad_frame: str,
    better_frame: str,
    stateless_control: str,
    source: cc.Source,
    turn_id: str = "",
    base_config: dict[str, Any] | None = None,
    generate: Callable[[list[dict]], str] | None = None,
) -> cc.CorrectionCard:
    """Run call 3 and build a CANDIDATE card (never promoted here)."""
    if generate is None:
        generate = _default_generate(base_config or {})
    user = (
        f"BAD: {bad_frame}\nBETTER: {better_frame}\nCONTROL: {stateless_control}"
    )
    msgs = [
        {"role": "system", "content": _SYNTH_PROMPT},
        {"role": "user", "content": user},
    ]
    try:
        out = str(generate(msgs) or "")
    except Exception:
        out = ""
    parsed = parse_synthesis(out)
    return cc.CorrectionCard(
        bad_frame=bad_frame,
        better_frame=better_frame,
        process_move=parsed["process_move"],
        anchor_type=parsed["anchor_type"],
        anchor_error=parsed["anchor_error"],
        aperture=parsed["aperture"],
        stateless_control=stateless_control,
        source=source,
        promoted=False,
        turn_id=turn_id,
    )


def advise(
    card: cc.CorrectionCard,
    *,
    base_config: dict[str, Any] | None = None,
    generate: Callable[[list[dict]], str] | None = None,
) -> cc.AdvisorVerdict:
    """Run the promotion-gate critic. Fails closed on error."""
    if generate is None:
        generate = _default_generate(base_config or {})
    user = (
        f"BAD: {card.bad_frame}\nBETTER: {card.better_frame}\n"
        f"CONTROL: {card.stateless_control}\nPROCESS_MOVE: {card.process_move}\n"
        f"ANCHOR: {card.anchor_type.value}\nANCHOR_ERROR: {card.anchor_error}\n"
        f"APERTURE: {card.aperture.value}\nSOURCE: {card.source.value}"
    )
    msgs = [
        {"role": "system", "content": _ADVISOR_PROMPT},
        {"role": "user", "content": user},
    ]
    try:
        out = str(generate(msgs) or "")
    except Exception:
        out = ""
    return parse_advisor(out)


def gate(card: cc.CorrectionCard, verdict: cc.AdvisorVerdict) -> cc.CorrectionCard:
    """Promote iff the card is HUMAN-sourced AND the advisor passed. Records the
    verdict either way; a Claude candidate is never promoted."""
    promote = bool(verdict.passed() and card.source is cc.Source.HUMAN)
    return dataclasses.replace(card, promoted=promote, advisor_verdict=verdict)
