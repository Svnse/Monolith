"""MonoFrame v2 — the /frame orchestrator.

process_correction ties one correction through the full pipeline, off the chat
gate (lockless generate seam, like a monoline llm block):

  1. stateless CONTROL — a momentum-free re-derivation (signed control, call 2).
  2. synthesize        — {bad, better, control} -> candidate card (call 3).
  3. advisor gate      — attack the card on 5 tests.
  4. gate + store      — promote iff human + passed; log every card either way.

Model calls go through one injected ``generate`` (defaults to the sync_bridge
tap). Never raises into the caller.
"""
from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Any

from . import correction_card as cc
from . import correction_store, correction_synthesis
from .stateless_reframe import build_reframe_messages


def _default_generate(base_config: dict[str, Any]) -> Callable[[list[dict]], str]:
    from engine.sync_bridge import generate_sync_from_config

    def generate(msgs: list[dict]) -> str:
        return generate_sync_from_config(base_config, msgs)

    return generate


def process_correction(
    turn_id: str,
    *,
    bad_frame: str,
    better_frame: str,
    recent_asks: list[str],
    base_config: dict[str, Any],
    source: cc.Source,
    generate: Callable[[list[dict]], str] | None = None,
) -> cc.CorrectionCard:
    """Run the full correction pipeline and return the (possibly promoted) card."""
    if generate is None:
        generate = _default_generate(base_config)

    # 1. stateless control (signed, not truth)
    try:
        control = str(generate(build_reframe_messages(recent_asks, bearing_block=None)) or "").strip()
    except Exception:
        control = ""

    # 2. synthesize candidate
    card = correction_synthesis.synthesize(
        bad_frame=bad_frame,
        better_frame=better_frame,
        stateless_control=control,
        source=source,
        turn_id=turn_id,
        generate=generate,
    )

    # 3. advisor gate
    verdict = correction_synthesis.advise(card, generate=generate)

    # 4. gate + store (every card logged; only human+passed is trainable)
    gated = correction_synthesis.gate(card, verdict)
    correction_store.store_card(gated)
    return gated


def process_correction_async(
    turn_id: str,
    *,
    bad_frame: str,
    better_frame: str,
    recent_asks: list[str],
    base_config: dict[str, Any],
    source: cc.Source,
    generate: Callable[[list[dict]], str] | None = None,
) -> threading.Thread:
    """Fire-and-forget: run process_correction on a daemon thread so the /frame
    command returns immediately (the synthesis + advisor calls take time)."""

    def _work() -> None:
        try:
            process_correction(
                turn_id,
                bad_frame=bad_frame,
                better_frame=better_frame,
                recent_asks=recent_asks,
                base_config=base_config,
                source=source,
                generate=generate,
            )
        except Exception:
            pass

    t = threading.Thread(target=_work, name=f"monoframe-correct-{turn_id}", daemon=True)
    t.start()
    return t
