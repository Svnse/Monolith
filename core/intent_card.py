"""intent_card — Seed β: the Monoline sidecar that ENRICHES the prediction floor.

Reads Monolith's own just-produced answer and emits a STRUCTURED, contentful
commitment: {intent_read, directions:[{move,referent}], referents:[...]}, where
`move` is coerced into core/turn_classifier's intent_tags vocabulary so the
membership settler can check it apples-to-apples.

INDEPENDENCE: the card runs ONLY at predict time (turn N) and NEVER sees E's
reply — the falsifier is frozen before the reply exists. The settler is pure code
(no LLM on the reply). The card emits a falsifiable commitment; code grades it.

ROBUSTNESS: `read_intent` returns None on ANY failure (no runtime, empty / over-
budget DeepSeek return, malformed JSON, bad shape) so predict ALWAYS works on the
pure-code floor. The card is enrichment, never a dependency.

STATUS (2026-06-22): parser + validation + vocab-coercion built and mock-tested.
`_invoke_card` returns None until the .monoline asset is wired to the live
runtime (assets/workshop_seeds/intent-read.monoline + monoline_runtime) — a
deliberate follow-up so an unvalidated live LLM call can't reach prod. Until
then the organ runs on the floor, exactly as designed.
"""
from __future__ import annotations

import json
from typing import Any


def _valid_moves() -> set[str]:
    """The turn_classifier intent_tags vocabulary (the shared move enum)."""
    try:
        from core.turn_classifier import _INTENT_PATTERNS
        return {tag for tag, _ in _INTENT_PATTERNS} | {"chat", "vent"}
    except Exception:
        return set()


def _invoke_card(public_answer: str, last_user_msg: str) -> str | None:
    """Run the intent-read Monoline card and return its raw text output, or None.

    Deliberately returns None until the .monoline seed is wired to the live
    monoline runtime (follow-up). Best-effort: any exception → None → floor."""
    try:
        # Follow-up wiring point: load assets/workshop_seeds/intent-read.monoline,
        # run via core.monoline_runtime with {{answer}} bound to public_answer,
        # provider 'monolith' (the bound model). Until wired, no-op.
        return None
    except Exception:
        return None


def _coerce(raw: str | None) -> dict | None:
    """Parse + validate the card's JSON into {intent_read, directions, referents}.

    Returns None on anything malformed. Drops directions whose `move` isn't in the
    shared vocabulary (keeps membership apples-to-apples). Empty after validation
    → None (nothing to enrich with)."""
    if not raw or not str(raw).strip():
        return None
    txt = str(raw).strip()
    # tolerate a ```json fence
    if txt.startswith("```"):
        txt = txt.strip("`")
        if txt[:4].lower() == "json":
            txt = txt[4:]
    try:
        obj = json.loads(txt)
    except Exception:
        return None
    if not isinstance(obj, dict):
        return None

    valid_moves = _valid_moves()
    directions: list[dict] = []
    for d in obj.get("directions") or []:
        if not isinstance(d, dict):
            continue
        move = str(d.get("move") or "").strip()
        referent = str(d.get("referent") or "").strip()
        if move and (not valid_moves or move in valid_moves):
            directions.append({"move": move, "referent": referent})

    referents = [str(r).strip() for r in (obj.get("referents") or [])
                 if isinstance(r, (str, int, float)) and str(r).strip()]
    intent_read = str(obj.get("intent_read") or "").strip()

    if not (directions or referents or intent_read):
        return None
    return {"intent_read": intent_read, "directions": directions, "referents": referents}


def read_intent(public_answer: str, last_user_msg: str = "") -> dict | None:
    """Contentful enrichment of the prediction floor, or None on any failure."""
    try:
        raw = _invoke_card(public_answer, last_user_msg)
        return _coerce(raw)
    except Exception:
        return None
