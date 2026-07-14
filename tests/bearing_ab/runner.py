"""One-arm worker for the Bearing V0 A/B harness.

`run_arm` is the core unit: one fixture × one arm × one model = one blinded
output file. Called twice per fixture by the orchestrator (once for arm="on",
once for arm="off"). Loads model ONCE per orchestrator process; arm toggle
is in-process per §6.A.

Production interceptor wiring per §6.B (locked 2026-05-21): registers all
plane interceptors + ephemeral_coalescer with stub world_state, then layers
Bearing on/off per the arm. Arm toggle is `MONOLITH_BEARING_V1=1` vs `=0`;
everything else stays as default production wiring.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from . import format as ab_format
from .manifest import RuntimeFingerprint


FIXTURES_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "bearing_ab"


def _extract_response_text(out: Any) -> str:
    """Normalize Llama and OpenAICompatLLM response shapes.

    `llama_cpp.Llama.create_chat_completion(stream=False)` returns a dict:
        {"choices": [{"message": {"content": "..."}}]}

    `engine.llm.OpenAICompatLLM.create_chat_completion(...)` is a generator
    yielding chunks (even with stream=False — it yields ONCE then returns):
        {"choices": [{"delta": {"content": "..."}}]}  or
        {"choices": [{"message": {"content": "..."}}]}

    This normalizer accepts both shapes and returns the accumulated text.
    """
    if isinstance(out, dict):
        try:
            choice = out["choices"][0]
            message = choice.get("message") or {}
            text = message.get("content", "")
            return str(text) if text else ""
        except (KeyError, IndexError, TypeError):
            return ""
    parts: list[str] = []
    try:
        for chunk in out:
            if not isinstance(chunk, dict):
                continue
            choices = chunk.get("choices") or []
            if not choices:
                continue
            c = choices[0]
            if not isinstance(c, dict):
                continue
            delta = c.get("delta")
            if isinstance(delta, dict):
                text = delta.get("content")
                if isinstance(text, str):
                    parts.append(text)
                    continue
            message = c.get("message")
            if isinstance(message, dict):
                text = message.get("content")
                if isinstance(text, str):
                    parts.append(text)
    except Exception:
        pass
    return "".join(parts)


# ── stub host state for headless plane wiring ───────────────────────


class _StubWorldState:
    """Minimal world_state surface used by PlaneLoader + adaptive_budget.

    Production WorldStateStore (core/world_state.py:59) exposes `.state`
    (dict) and `.mark_dirty()` (no-op-ok in headless). That's all the
    interceptors read or call.
    """
    def __init__(self) -> None:
        self.state: dict[str, Any] = {}

    def mark_dirty(self) -> None:
        return None

    def set_last_prompt(self, _txt: str) -> None:
        return None


class _StubLedger:
    """No-op stand-in for core.event_ledger.EventLedger.

    adaptive_budget's set_ledger accepts Any; the only call site (line 367)
    is `_ledger.record(...)`. Adaptive_budget_interceptor is parked
    (bootstrap.py:470-477) so the call shouldn't fire in normal A/B flow,
    but providing a no-op `.record` keeps any incidental telemetry quiet.
    """
    def record(self, *_args: Any, **_kwargs: Any) -> None:
        return None


# ── fixture load ─────────────────────────────────────────────────────


def _load_fixture(fixture_id: str) -> dict[str, Any]:
    path = FIXTURES_DIR / f"{fixture_id}.json"
    if not path.exists():
        raise FileNotFoundError(f"fixture not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or "turns" not in data:
        raise ValueError(f"fixture {fixture_id} malformed (no 'turns')")
    return data


# ── production interceptor wiring (headless) ─────────────────────────


def _wire_production_interceptors(stub_ws: _StubWorldState, stub_ledger: _StubLedger) -> None:
    """Register production direct scaffolds in the same order bootstrap.py uses.

    Bearing and command_feedback are registered separately so their order can
    mirror production: prompt -> monothink -> tool -> bearing -> command
    feedback -> coalescer.

    Pre-condition: clear_interceptors() must have been called by the caller
    so we start from a clean slate. Each arm re-wires from scratch.
    """
    from core.prompt_library import (
        set_world_state as set_prompt_ws,
        prompt_interceptor,
        tool_interceptor,
    )
    from core.monothink import (
        set_monothink_world_state,
        monothink_interceptor,
    )
    from core.adaptive_budget import (
        set_world_state as set_bud_ws,
        set_ledger as set_bud_ledger,
    )
    from core.message_interceptors import register_interceptor

    set_prompt_ws(stub_ws)
    set_monothink_world_state(stub_ws)
    set_bud_ws(stub_ws)
    set_bud_ledger(stub_ledger)

    register_interceptor(prompt_interceptor)        # 0  unified prompt scaffolds
    register_interceptor(monothink_interceptor)     # 0a monothink
    register_interceptor(tool_interceptor)          # 0b one-shot tool prompts
    # bearing_interceptor lands next. Caller adds it AFTER this call returns.
    # command_feedback and ephemeral_coalescer register after bearing.


def _wire_command_feedback() -> None:
    from core.command_feedback import command_feedback_interceptor
    from core.message_interceptors import register_interceptor
    register_interceptor(command_feedback_interceptor)


def _wire_ephemeral_coalescer() -> None:
    from core.ephemeral_coalescer import ephemeral_coalescer_interceptor
    from core.message_interceptors import register_interceptor
    register_interceptor(ephemeral_coalescer_interceptor)


# ── arm-bearing wiring ──────────────────────────────────────────────


def _wire_bearing_for_arm(arm: str) -> None:
    """Set env, then wire Bearing addon if arm=='on'.

    Must be called AFTER _wire_production_interceptors so bearing lands
    between tool prompts and command_feedback, matching production order.
    """
    os.environ["MONOLITH_BEARING_V1"] = "1" if arm == "on" else "0"

    from core.turn_classifier import set_bearing_provider
    set_bearing_provider(None)  # always reset DI handle

    from addons.system.bearing import build_addon
    addon = build_addon()  # returns None if MONOLITH_BEARING_V1=0
    if addon is not None:
        set_bearing_provider(addon.provider)
        from core.message_interceptors import register_interceptor
        register_interceptor(addon.interceptor)


# ── per-arm run ─────────────────────────────────────────────────────


def run_arm(
    fixture_id: str,
    arm: str,
    fingerprint: RuntimeFingerprint,
    model: Any,
) -> Path:
    """Run one fixture × one arm. Returns the path of the rater-readable
    output file.

    `model` is a pre-loaded llama_cpp.Llama instance (orchestrator loads once).
    `fingerprint` is the runtime fingerprint (sampler params + run_id).
    """
    if arm not in ("on", "off"):
        raise ValueError(f"arm must be 'on' or 'off', got {arm!r}")

    fixture = _load_fixture(fixture_id)

    # 1. Process-global state reset.
    from core.message_interceptors import clear_interceptors
    clear_interceptors()

    # 2. Production direct scaffolds, excluding bearing.
    stub_ws = _StubWorldState()
    stub_ledger = _StubLedger()
    _wire_production_interceptors(stub_ws, stub_ledger)

    # 3. Bearing per arm, then command-feedback repair block, then coalescer.
    #    This keeps [BEARING] and [COMMAND_FAILED] direct-inject and outside
    #    the runtime-state budget, matching production order.
    _wire_bearing_for_arm(arm)
    _wire_command_feedback()

    # 4. Ephemeral coalescer (registers last — closest to user message in
    #    the rendered prompt).
    _wire_ephemeral_coalescer()

    # 5. Drive turns.
    from core.llm_config import build_system_prompt
    from core.message_interceptors import apply_interceptors

    sampler = fingerprint.sampler
    sampler_kwargs: dict[str, Any] = {}
    for key in ("top_k", "min_p", "repeat_penalty"):
        if sampler.get(key) is not None:
            sampler_kwargs[key] = sampler[key]
    if fingerprint.seed is not None:
        sampler_kwargs["seed"] = fingerprint.seed

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": build_system_prompt()}
    ]
    responses: list[dict[str, Any]] = []
    for turn in fixture["turns"]:
        if not isinstance(turn, dict):
            continue
        user_text = str(turn.get("user", ""))
        if not user_text.strip():
            continue
        messages.append({"role": "user", "content": user_text})
        msgs_with_ephemerals = apply_interceptors(messages, {})

        out = model.create_chat_completion(
            messages=msgs_with_ephemerals,
            temperature=float(sampler.get("temperature", 0.7)),
            top_p=float(sampler.get("top_p", 0.95)),
            stream=False,
            **sampler_kwargs,
        )
        response_text = _extract_response_text(out)

        responses.append({
            "turn": int(turn.get("turn", len(responses) + 1)),
            "user": user_text,
            "assistant": response_text,
        })
        messages.append({"role": "assistant", "content": response_text})

        # Bearing V0 write-side: process <bearing_update> envelopes per turn
        # so the A/B measures a LIVE substrate, not just block injection.
        # Same hook chat_finalize uses, but with a synthesized turn_id since
        # the runner doesn't have a Layer A turn_trace context.
        if arm == "on":
            try:
                from addons.system.bearing import updater
                synthetic_turn_id = f"{fixture['id']}::turn_{turn.get('turn', len(responses))}"
                updater.process_turn_output(
                    turn_id=synthetic_turn_id,
                    response_text=response_text,
                    model_id=fingerprint.model_name,
                )
            except Exception:
                # Same soft-handle as production: substrate write failure
                # must not break the A/B harness.
                pass

    # 6. Blinded write + decoder update.
    fingerprint_summary = {
        "model_name": fingerprint.model_name,
        "preset_confidence": fingerprint.preset_confidence,
        "quantization": fingerprint.quantization,
        "run_id": fingerprint.run_id,
    }
    out_path = ab_format.write_arm_output(
        fixture_id=fixture_id,
        arm=arm,
        run_id=fingerprint.run_id,
        responses=responses,
        fingerprint_summary=fingerprint_summary,
    )
    ab_format.update_arm_decoder(
        run_id=fingerprint.run_id, fixture_id=fixture_id, arm=arm
    )
    return out_path
