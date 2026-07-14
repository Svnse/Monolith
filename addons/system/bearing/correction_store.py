"""MonoFrame v2 — CorrectionCard store, nearest retrieval, scaffold render.

Stores cards to CONFIG_DIR/correction_cards.jsonl (flag MONOLITH_MONOFRAME_V1).
ALL cards are logged (human + Claude candidates), but only TRAINABLE cards
(human-sourced AND advisor-promoted) are eligible for injection. The live frame
scaffold gets ONE nearest human card — an example, never a rule pile.

Nearest-match reuses monoframe.token_divergence (content-word overlap) against
the card's better_frame; invariants from clustering are a later concern. Never
raises into the chat path.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any

from core.paths import CONFIG_DIR

from . import correction_card as cc
from . import monoframe

_FLAG_ENV = "MONOLITH_MONOFRAME_V1"
_TRUTHY = {"1", "true", "yes", "on"}

_STORE = CONFIG_DIR / "correction_cards.jsonl"


def enabled() -> bool:
    return os.environ.get(_FLAG_ENV, "0").strip().lower() in _TRUTHY


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _card_to_dict(card: cc.CorrectionCard) -> dict[str, Any]:
    return {
        "ts": card.ts or _now_iso(),
        "turn_id": card.turn_id,
        "bad_frame": card.bad_frame,
        "better_frame": card.better_frame,
        "process_move": card.process_move,
        "anchor_type": card.anchor_type.value,
        "anchor_error": card.anchor_error,
        "aperture": card.aperture.value,
        "stateless_control": card.stateless_control,
        "source": card.source.value,
        "promoted": bool(card.promoted),
        "trainable": card.is_trainable(),
        "advisor_verdict": _verdict_to_dict(card.advisor_verdict),
        "slots": dict(card.slots or {}),
    }


def _verdict_to_dict(v: cc.AdvisorVerdict | None) -> dict[str, Any] | None:
    """Serialize the 5-attack promotion gate so every promote/reject is
    inspectable (which check failed), per the observability contract."""
    if v is None:
        return None
    return {
        "human_grounded": v.human_grounded,
        "process_shaped": v.process_shaped,
        "not_overfit": v.not_overfit,
        "sign_preserved": v.sign_preserved,
        "real_anchor": v.real_anchor,
        "passed": v.passed(),
    }


def store_card(card: cc.CorrectionCard) -> None:
    """Append one card (any source). No-op if disabled; never raises."""
    if not enabled():
        return
    try:
        _STORE.parent.mkdir(parents=True, exist_ok=True)
        with _STORE.open("a", encoding="utf-8") as f:
            f.write(json.dumps(_card_to_dict(card)) + "\n")
    except Exception:
        pass


def read_cards() -> list[dict[str, Any]]:
    if not _STORE.exists():
        return []
    try:
        with _STORE.open("r", encoding="utf-8") as f:
            lines = f.readlines()
    except Exception:
        return []
    out: list[dict[str, Any]] = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        if isinstance(obj, dict):
            out.append(obj)
    return out


def nearest_human_card(query: str) -> dict[str, Any] | None:
    """Return the single nearest TRAINABLE (human + promoted) card to ``query``,
    by content-word overlap with its better_frame. None if there are none."""
    trainable = [c for c in read_cards() if c.get("trainable")]
    if not trainable:
        return None
    return min(
        trainable,
        key=lambda c: monoframe.token_divergence(query or "", str(c.get("better_frame", ""))),
    )


def render_card_for_scaffold(card: dict[str, Any] | None) -> str:
    """Render ONE nearest human card as a worked example for the frame seam —
    an example, not a rule pile. Empty string when there is no card."""
    if not card:
        return ""
    bad = str(card.get("bad_frame", ""))
    better = str(card.get("better_frame", ""))
    move = str(card.get("process_move", ""))
    return (
        "[FRAME — nearest correction (a human example, not a rule)]\n"
        f"A carried frame anchored wrong: \"{bad}\"\n"
        f"Better, for that situation: \"{better}\"\n"
        f"The move: {move}"
    )
