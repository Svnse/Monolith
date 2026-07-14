"""Bearing updater — extract <bearing_update> envelopes, verify, apply.

Pipeline:
  1. Extract <bearing_update>JSON</bearing_update> envelopes from turn output.
  2. Run structural_verifier.verify_structural(old, envelope).
  3a. On pass → apply envelope to build new Bearing → persist via store
     → audit "applied" → reset rejection_streak → clear pending_rejection
     → (optional) run grounding_verifier and emit warn-fault on failure
  3b. On fail → audit "rejected" → set pending_rejection → increment streak
     → if streak >= ESCALATION_THRESHOLD, emit warn-fault and audit "escalated"

Reflect-and-retry mechanism: pending_rejection persists across turns
in bearing.json; the compiler reads it next turn and injects the
[BEARING_UPDATE_REJECTED] block. The model gets one repair attempt
implicitly by virtue of seeing the rejection block in next-turn context.

Escalation threshold N=3 per plan §7. Tunable via env
MONOLITH_BEARING_ESCALATION_N.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterable

from . import audit
from . import schema as bs
from . import store
from .grounding_verifier import GroundingVerdict, verify_grounding
from .structural_verifier import StructuralVerdict, verify_structural
from .tolerant_extract import looks_like_attempt, recover_bearing_json

# Pattern matches <bearing_update> ... </bearing_update> across newlines.
_ENVELOPE_RE = re.compile(
    r"<bearing_update>\s*(.*?)\s*</bearing_update>",
    flags=re.DOTALL | re.IGNORECASE,
)

ESCALATION_THRESHOLD = 3
_ESCALATION_ENV = "MONOLITH_BEARING_ESCALATION_N"
_ESCALATION_FAULT_KIND = "bearing_structural_unrecoverable"
_GROUNDING_FAULT_KIND = "bearing_grounding_failed"


@dataclass(frozen=True)
class UpdaterResult:
    found_envelope: bool = False
    parse_failed: bool = False
    structural_verdict: StructuralVerdict | None = None
    grounding_verdict: GroundingVerdict | None = None
    bearing_changed: bool = False
    streak_after: int = 0
    escalated: bool = False
    applied_bearing: bs.Bearing | None = None


# ── envelope extraction ─────────────────────────────────────────────


def extract_envelope(response_text: str) -> tuple[dict[str, Any] | None, bool]:
    """Return (parsed_envelope, parse_failed).

    parse_failed=True means a <bearing_update> tag was found but the
    JSON inside was malformed. parsed_envelope=None + parse_failed=False
    means no envelope was present.
    """
    if not isinstance(response_text, str) or "<bearing_update" not in response_text.lower():
        return (None, False)
    # Strict path: a clean <bearing_update>...</bearing_update> with valid JSON.
    match = _ENVELOPE_RE.search(response_text)
    if match is not None:
        body = (match.group(1) or "").strip()
        if not body:
            return ({}, False)
        try:
            parsed = json.loads(body)
        except json.JSONDecodeError:
            pass  # malformed JSON → tolerant recovery below
        else:
            # Cleanly parsed: a dict is the envelope; a well-formed non-dict is a
            # real wrong-type failure (preserve strict parse_failed behavior).
            return (parsed, False) if isinstance(parsed, dict) else (None, True)
    # Tolerant recovery (parsing only — verify_structural still runs on the
    # result, so semantics are unchanged). The bound model routinely emits the
    # envelope tangled inside its <think> — dangling/missing close tags, an
    # embedded </think>, markdown fences, trailing commas — which the strict
    # path drops every time (the bearing has never once committed an update).
    # Recover the JSON from those real shapes. A truncated/unrecoverable but
    # genuine attempt becomes a real parse_failed (→ rejection block, the model
    # is told) instead of a missing-close-tag being silently ignored; a bare
    # prose mention of the tag is NOT an envelope and must not be rejected.
    recovered = recover_bearing_json(response_text)
    if isinstance(recovered, dict):
        return (recovered, False)
    if looks_like_attempt(response_text):
        return (None, True)
    return (None, False)


# ── apply envelope to bearing ───────────────────────────────────────


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _escalation_threshold() -> int:
    raw = str(os.environ.get(_ESCALATION_ENV, ESCALATION_THRESHOLD)).strip()
    try:
        n = int(raw)
        return max(1, n)
    except (TypeError, ValueError):
        return ESCALATION_THRESHOLD


def _apply_primitive(old_value: str, op: Any) -> str:
    """Apply a primitive slot change (`{"new": ..., "reason": ...}`) and
    return the new string value. If op is malformed, return old."""
    if isinstance(op, dict) and isinstance(op.get("new"), str):
        return op["new"]
    if isinstance(op, str):
        return op
    return old_value


def _apply_open_tensions(
    old: tuple[bs.Tension, ...], ops: Any
) -> tuple[bs.Tension, ...]:
    if not isinstance(ops, dict):
        return old
    remove_indices: set[int] = set()
    for op_name in ("resolve", "drop"):
        items = ops.get(op_name) or []
        if isinstance(items, list):
            for item in items:
                if isinstance(item, dict):
                    idx = item.get("index")
                    if isinstance(idx, int) and 0 <= idx < len(old):
                        remove_indices.add(idx)
    kept = [t for i, t in enumerate(old) if i not in remove_indices]
    adds = ops.get("add") or []
    if isinstance(adds, list):
        for item in adds:
            if isinstance(item, dict) and isinstance(item.get("text"), str) and item["text"].strip():
                kept.append(bs.Tension(
                    text=item["text"],
                    opened_at_turn=str(item.get("opened_at_turn", "")),
                ))
    return tuple(kept)


def _apply_modal_branches(
    old: tuple[bs.ModalBranch, ...], ops: Any, turn_id: str
) -> tuple[bs.ModalBranch, ...]:
    if not isinstance(ops, dict):
        return old
    transitions: dict[int, dict[str, Any]] = {}
    items = ops.get("transition") or []
    if isinstance(items, list):
        for item in items:
            if isinstance(item, dict):
                idx = item.get("index")
                if isinstance(idx, int) and 0 <= idx < len(old):
                    transitions[idx] = item
    new_branches: list[bs.ModalBranch] = []
    for i, b in enumerate(old):
        if i in transitions:
            t = transitions[i]
            to = t.get("to") if isinstance(t.get("to"), str) else b.status
            new_branches.append(bs.ModalBranch(
                text=b.text,
                status=to,
                reason=b.reason,
                last_touched_turn=turn_id,
            ))
        else:
            new_branches.append(b)
    adds = ops.get("add") or []
    if isinstance(adds, list):
        for item in adds:
            if not isinstance(item, dict):
                continue
            text = item.get("text", "")
            status = item.get("status", "active")
            reason = item.get("reason", "")
            if isinstance(text, str) and text.strip():
                new_branches.append(bs.ModalBranch(
                    text=text,
                    status=status if isinstance(status, str) else "active",
                    reason=reason if isinstance(reason, str) else "",
                    last_touched_turn=turn_id,
                ))
    return tuple(new_branches)


def _apply_referents(
    old: tuple[bs.Referent, ...], ops: Any
) -> tuple[bs.Referent, ...]:
    if not isinstance(ops, dict):
        return old
    new_refs = list(old)
    adds = ops.get("add") or []
    if isinstance(adds, list):
        for item in adds:
            if not isinstance(item, dict):
                continue
            name = item.get("name")
            if not isinstance(name, str) or not name.strip():
                continue
            new_refs.append(bs.Referent(
                name=name,
                kind=item.get("kind", "entity") if isinstance(item.get("kind"), str) else "entity",
                status=item.get("status", "observed") if isinstance(item.get("status"), str) else "observed",
                grounded_at_turn=str(item.get("grounded_at_turn", "")),
            ))
    return tuple(new_refs)


def _apply_user_model(old: bs.UserModel | None, op: Any) -> bs.UserModel | None:
    if not isinstance(op, dict):
        return old
    try:
        conf = float(op.get("confidence", 0.0))
    except (TypeError, ValueError):
        conf = 0.0
    return bs.UserModel(
        intent_read=str(op.get("intent_read", "")),
        register=str(op.get("register", "literal")),
        confidence=conf,
    )


def _apply_stakes(old: bs.Stakes | None, op: Any) -> bs.Stakes | None:
    if not isinstance(op, dict):
        return old
    return bs.Stakes(
        reversibility=str(op.get("reversibility", "unknown")),
        urgency=str(op.get("urgency", "low")),
        cost_if_wrong=str(op.get("cost_if_wrong", "")),
    )


def apply_envelope(
    old: bs.Bearing,
    envelope: dict[str, Any],
    turn_id: str,
    model_id: str,
    turn_n: int = 0,
) -> bs.Bearing:
    """Build a new Bearing by applying envelope ops to old.

    Stamps last_writer_model_id + updated_at_turn from the call args.
    `turn_n` is the readable monotonic turn-count; 0 (feature off / not
    threaded) leaves any prior count intact rather than clobbering it.
    Assumes envelope is structurally verified — does NOT re-check.
    """
    return bs.Bearing(
        schema_version=bs.SCHEMA_VERSION,
        current_frame=_apply_primitive(old.current_frame, envelope.get("current_frame")) if "current_frame" in envelope else old.current_frame,
        active_goal=_apply_primitive(old.active_goal, envelope.get("active_goal")) if "active_goal" in envelope else old.active_goal,
        trajectory=_apply_primitive(old.trajectory, envelope.get("trajectory")) if "trajectory" in envelope else old.trajectory,
        next_move=_apply_primitive(old.next_move, envelope.get("next_move")) if "next_move" in envelope else old.next_move,
        open_tensions=_apply_open_tensions(old.open_tensions, envelope.get("open_tensions")),
        modal_branches=_apply_modal_branches(old.modal_branches, envelope.get("modal_branches"), turn_id),
        referents=_apply_referents(old.referents, envelope.get("referents")),
        user_model=_apply_user_model(old.user_model, envelope.get("user_model")) if "user_model" in envelope else old.user_model,
        stakes=_apply_stakes(old.stakes, envelope.get("stakes")) if "stakes" in envelope else old.stakes,
        last_writer_model_id=model_id or old.last_writer_model_id,
        updated_at_turn=turn_id or old.updated_at_turn,
        updated_at_turn_n=turn_n or old.updated_at_turn_n,
    )


# ── escalation hook ─────────────────────────────────────────────────


def _emit_escalation_fault(turn_id: str, fault_kind: str, detail: str) -> bool:
    """Best-effort call into core.fault_response.emit_fault.

    Returns True if the fault was emitted (row id != -1). Returns False
    silently on any failure — escalation must never break the addon path.
    """
    try:
        from core.fault_response import emit_fault
        row_id = emit_fault(
            turn_id=str(turn_id),
            fault_kind=fault_kind,
            detector_name="bearing_addon",
            evidence=detail,
            metadata=None,
        )
        return row_id != -1
    except Exception:
        return False


# ── frame-commit fastpath ────────────────────────────────────────────


def commit_frame(
    turn_id: str,
    observed_frame: str,
    *,
    model_id: str = "",
    turn_n: int = 0,
) -> bool:
    """Scribe the model's <frame> text into bearing.current_frame ONLY, reusing
    structural verification + apply + audit. Returns True iff committed.

    Minimal by design: verifies current_frame (≤400 chars via D5), applies
    current_frame only, audits; does NOT touch rejection streak / grounding
    (those are <bearing_update>'s domain).

    The envelope is synthesised with `previous` (old current_frame) and a
    literal `trigger` so that D2 (current_frame provenance) passes.
    `apply_envelope` reads only `.new`; the provenance fields are inert at
    apply time but required for structural acceptance.
    """
    try:
        frame = str(observed_frame or "").strip()
        if not frame:
            return False

        old = store.get_bearing()

        # Synthesise a minimal envelope that passes all structural rules:
        #   D1: reason present  ✓
        #   D2: previous + trigger present  ✓
        #   D5: character limit enforced by verifier  ✓
        envelope: dict[str, Any] = {
            "current_frame": {
                "new": frame,
                "previous": old.current_frame,   # "" on empty bearing — still a str
                "reason": "frame heartbeat",
                "trigger": "frame heartbeat",
            }
        }

        sv = verify_structural(old, envelope)
        if not sv.ok:
            audit.append(
                "rejected",
                turn_id=turn_id,
                failed_rules=list(sv.failed_rules),
                detail=sv.detail,
                source="frame_fastpath",
            )
            return False

        new = apply_envelope(old, envelope, turn_id, model_id, turn_n)
        store.set_bearing(new)
        audit.append(
            "applied",
            turn_id=turn_id,
            slots_changed=["current_frame"],
            source="frame_fastpath",
        )
        return True
    except Exception:
        return False


# ── public entry ────────────────────────────────────────────────────


def process_turn_output(
    turn_id: str,
    response_text: str,
    *,
    tool_result_ids: Iterable[str] | None = None,
    connected_peers: Iterable[str] | None = None,
    model_id: str = "",
    turn_n: int = 0,
) -> UpdaterResult:
    """Extract, verify, apply (or reject) a <bearing_update> envelope.

    Idempotent for the structural-pass path: re-running on the same
    response with the same store state produces the same Bearing.

    For the reject path, calling this multiple times for the same turn
    would increment the streak each time — caller is responsible for
    calling once per turn.
    """
    envelope, parse_failed = extract_envelope(response_text)

    if parse_failed:
        # Malformed JSON inside the tag — treat as structural rejection.
        # Capture a truncated snippet of the offending body for forensic.
        # Surfaces in the audit row's "evidence" AND in the next-turn
        # [BEARING_UPDATE_REJECTED] block so the model sees what it
        # actually emitted, not just "parse_error".
        ts = _now_iso()
        import re as _re
        m = _re.search(
            r"<bearing_update>\s*(.*?)\s*</bearing_update>",
            response_text or "",
            flags=_re.DOTALL | _re.IGNORECASE,
        )
        offending = (m.group(1) if m else "")[:400]
        store.set_pending_rejection(
            ["parse_error"],
            turn_id=turn_id,
            ts=ts,
            detail="json decode failed",
            evidence=offending,
        )
        streak = store.increment_rejection_streak()
        audit.append(
            "rejected",
            turn_id=turn_id,
            failed_rules=["parse_error"],
            detail="json decode failed",
            evidence=offending,
        )
        escalated = False
        if streak >= _escalation_threshold():
            escalated = _emit_escalation_fault(
                turn_id, _ESCALATION_FAULT_KIND, "json decode failed"
            )
            audit.append(
                "escalated",
                turn_id=turn_id,
                streak=streak,
                fault_emitted=escalated,
            )
        return UpdaterResult(
            found_envelope=True,
            parse_failed=True,
            streak_after=streak,
            escalated=escalated,
        )

    if envelope is None:
        return UpdaterResult(found_envelope=False)

    old = store.get_bearing()
    sv = verify_structural(old, envelope)

    if not sv.ok:
        ts = _now_iso()
        store.set_pending_rejection(
            list(sv.failed_rules),
            turn_id=turn_id,
            ts=ts,
            detail=sv.detail,
        )
        streak = store.increment_rejection_streak()
        audit.append(
            "rejected",
            turn_id=turn_id,
            failed_rules=list(sv.failed_rules),
            detail=sv.detail,
        )
        escalated = False
        if streak >= _escalation_threshold():
            escalated = _emit_escalation_fault(turn_id, _ESCALATION_FAULT_KIND, sv.detail)
            audit.append(
                "escalated",
                turn_id=turn_id,
                streak=streak,
                fault_emitted=escalated,
            )
        return UpdaterResult(
            found_envelope=True,
            structural_verdict=sv,
            streak_after=streak,
            escalated=escalated,
        )

    # Structural pass → apply.
    if not envelope:
        # No-op update; still treat as a successful turn (clear rejection state).
        store.clear_pending_rejection()
        store.reset_rejection_streak()
        return UpdaterResult(
            found_envelope=True,
            structural_verdict=sv,
            streak_after=0,
            applied_bearing=old,
        )

    new_bearing = apply_envelope(old, envelope, turn_id, model_id, turn_n)
    store.set_bearing(new_bearing)
    store.clear_pending_rejection()
    store.reset_rejection_streak()

    audit.append(
        "applied",
        turn_id=turn_id,
        slots_changed=list(envelope.keys()),
    )

    # Grounding verification (post-commit; never rolls back).
    gv = verify_grounding(
        turn_id=turn_id,
        envelope=envelope,
        applied_bearing=new_bearing,
        tool_result_ids=set(tool_result_ids or ()) or None,
        connected_peers=set(connected_peers or ()) or None,
    )
    if not gv.ok:
        _emit_escalation_fault(turn_id, _GROUNDING_FAULT_KIND, gv.detail)
        audit.append(
            "grounding_failed",
            turn_id=turn_id,
            failed_rules=list(gv.failed_rules),
            detail=gv.detail,
            downgrade_indices=list(gv.downgrade_referent_indices),
        )
        # Downgrade observed referents → unverified for next render.
        if gv.downgrade_referent_indices:
            downgraded = list(new_bearing.referents)
            for idx in gv.downgrade_referent_indices:
                if 0 <= idx < len(downgraded):
                    r = downgraded[idx]
                    downgraded[idx] = bs.Referent(
                        name=r.name, kind=r.kind, status="unverified",
                        grounded_at_turn=r.grounded_at_turn,
                    )
            store.set_bearing(bs.Bearing(
                schema_version=new_bearing.schema_version,
                current_frame=new_bearing.current_frame,
                active_goal=new_bearing.active_goal,
                trajectory=new_bearing.trajectory,
                next_move=new_bearing.next_move,
                open_tensions=new_bearing.open_tensions,
                modal_branches=new_bearing.modal_branches,
                referents=tuple(downgraded),
                user_model=new_bearing.user_model,
                stakes=new_bearing.stakes,
                last_writer_model_id=new_bearing.last_writer_model_id,
                updated_at_turn=new_bearing.updated_at_turn,
                updated_at_turn_n=new_bearing.updated_at_turn_n,
            ))

    return UpdaterResult(
        found_envelope=True,
        structural_verdict=sv,
        grounding_verdict=gv,
        bearing_changed=True,
        streak_after=0,
        applied_bearing=new_bearing,
    )
