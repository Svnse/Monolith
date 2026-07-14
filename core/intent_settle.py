"""intent_settle — Path B settlement by pure-code SET-MEMBERSHIP.

Settles a prior-turn prediction by checking E's verbatim reply against the
FROZEN prediction_set committed at predict time. Lexical/content-overlap is NOT
the discriminator (it is structurally blind to the on-topic redirect — spec
§5/§13a). The discriminator is membership: does the reply's *lead salient
referent* fall inside the staked set?

  - reply's lead salient referent ∉ frozen referents  -> REDIRECT (the keystone:
    fires even with high lexical overlap, even for a long answer, because the
    frozen set is the CAPPED-salient staked referents — a passing echo of a
    staked word can't mask a genuinely new focus).
  - lead ∈ frozen AND move ∈ predicted directions          -> uptake (low).
  - repair markers present                                  -> marker type wins (precision channel).
  - reply too short / no frozen set (v1 row)               -> unresolved (NEVER confirmation).

Pure code, NO LLM on the reply (settlement independence). Settlement reads only
the reply + the frozen set; it never re-reads the answer (the answer was captured
cleanly at predict time), which also sidesteps the raw-history tool-loop problem.

friction_type is a CLOSED enum (SETTLE_TYPES). friction_score in [0,1].
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from core import friction_differ as _fd
from core import friction_store as _fs
from core.intent_extract import _moves_from, salient_referents

# Closed enum = friction_differ's set + the NEW Path-B value `redirect`.
SETTLE_TYPES: tuple[str, ...] = _fd.FRICTION_TYPES + ("redirect",)

_REPLY_REF_CAP = 6
_REDIRECT_SCORE = 0.80
_REASK_SCORE = 0.78
_UPTAKE_SCORE = 0.10
_UNRESOLVED_SCORE = 0.40
_MAX_OBS = 500
_INJECT_FLAG = "MONOLITH_FRICTION_INJECT_V1"
_CRYSTALLIZE_FLAG = "MONOLITH_FRICTION_CRYSTALLIZE_V1"
_TRUTHY = {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class SettleResult:
    friction_score: float
    friction_type: str
    channel_json: dict


def _pick_marker(markers: list[str]) -> str:
    """Highest-precision marker wins (by friction_differ._MARKER_SCORE desc)."""
    return max(markers, key=lambda m: _fd._MARKER_SCORE.get(m, 0.6))


def settle(prediction_set: dict | None, reply: str) -> SettleResult:
    """Score the friction of `reply` against the FROZEN `prediction_set`.

    Pure + deterministic. `prediction_set` is {directions:[{move,referent}],
    referents:[...]} frozen at predict time; None/empty (v1 rows) -> unresolved."""
    markers = _fd._detect_markers(reply or "")
    e_refs = salient_referents(reply or "", _REPLY_REF_CAP)
    e_moves = _moves_from(reply or "")
    e_move = e_moves[0] if e_moves else ""
    interrogative = bool(_fd._INTERROGATIVE_RE.search(reply or ""))

    frozen = prediction_set or {}
    frozen_refs = set(frozen.get("referents") or [])
    directions = frozen.get("directions") or []
    frozen_moves = {str(d.get("move", "")) for d in directions if d.get("move")}

    lead_ref = e_refs[0] if e_refs else ""
    refs_in = [r for r in e_refs if r in frozen_refs]
    refs_out = [r for r in e_refs if r not in frozen_refs]
    move_in = bool(e_move and e_move in frozen_moves)
    lead_in = bool(lead_ref and lead_ref in frozen_refs)

    channel = {
        "markers": markers,
        "e_move": e_move,
        "e_refs": e_refs,
        "lead_ref": lead_ref,
        "refs_in": refs_in,
        "refs_out": refs_out,
        "move_in_directions": move_in,
        "has_frozen_set": bool(frozen_refs or frozen_moves),
        "interrogative": interrogative,
    }

    # 1) repair markers — precision channel, overrides membership
    if markers:
        mtype = _pick_marker(markers)
        return SettleResult(round(_fd._MARKER_SCORE.get(mtype, 0.6), 3), mtype, channel)

    # 2) no frozen set (v1 row) or unextractable reply -> unresolved (never confirm)
    if not (frozen_refs or frozen_moves) or not e_refs:
        return SettleResult(_UNRESOLVED_SCORE, "unresolved", channel)

    # 3) KEYSTONE: the reply's salient content is MOSTLY NEW (more unstaked than
    #    staked referents) -> redirect. Robust to (a) high lexical overlap — a
    #    few echoes of staked words can't mask a dominant new focus — and (b)
    #    long answers, because the frozen set is capped-salient. A count ratio
    #    (not a strict lead) tolerates filler openers ("great, but…").
    if refs_out and len(refs_out) > len(refs_in):
        return SettleResult(_REDIRECT_SCORE, "redirect", channel)

    # 4) the reply echoes the staked content: re-ask vs uptake
    if refs_in:
        if interrogative and not move_in:
            return SettleResult(_REASK_SCORE, "reask", channel)
        if move_in or len(refs_in) >= len(refs_out):
            return SettleResult(_UPTAKE_SCORE, "uptake", channel)

    # echoes present but ambiguous (no move match, weak) -> unresolved
    return SettleResult(_UNRESOLVED_SCORE, "unresolved", channel)


# ── interceptor (settlement independence; once per outer turn) ─────────


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _flag(env: str) -> bool:
    import os
    return os.environ.get(env, "0").strip().lower() in _TRUTHY


def _is_peer_turn(messages: list[dict]) -> bool:
    for i in range(len(messages) - 1, -1, -1):
        msg = messages[i]
        if msg.get("role") == "user" and not msg.get("ephemeral"):
            return "[CHANNEL: connect/" in str(msg.get("content") or "")
    return False


def _last_nonephemeral_user(messages: list[dict]) -> str:
    for i in range(len(messages) - 1, -1, -1):
        msg = messages[i]
        if msg.get("role") == "user" and not msg.get("ephemeral"):
            return str(msg.get("content") or "")
    return ""


def _update_bearing_other_fields(claim: str, register: str, confidence: float, trajectory: str) -> None:
    """Read-modify-write bearing: replace user_model + trajectory, PRESERVE
    current_frame (and every other field). Single writer for these two fields.
    Best-effort; silent if bearing off."""
    try:
        import dataclasses

        from addons.system.bearing import store as bstore
        from addons.system.bearing.schema import UserModel
        cur = bstore.get_bearing()
        new_um = UserModel(intent_read=str(claim)[:300], register=str(register or "literal"),
                           confidence=float(confidence))
        new = dataclasses.replace(cur, user_model=new_um, trajectory=str(trajectory)[:300])
        bstore.set_bearing(new)
    except Exception:
        pass


def friction_settle_interceptor(messages: list[dict], config: dict) -> list[dict]:
    """Settle the one open prediction against E's verbatim reply. Side-effect
    only: returns `messages` unchanged, never raises into the chat path.

    Once per OUTER turn: keyed on TurnClock config['_now_iso']. Only settles a
    prediction CREATED IN A PRIOR outer turn (created_at < current _now_iso), so
    a tool-loop followup never settles this turn's own fresh prediction."""
    try:
        if not _fs.flag_enabled() or _is_peer_turn(messages):
            return messages
        user_msg = _last_nonephemeral_user(messages)
        if not user_msg:
            return messages
        pred = _fs.latest_open("intent")
        if not pred:
            return messages

        now = (config or {}).get("_now_iso") or ""
        created = str(pred.get("created_at") or "")
        # once-per-outer-turn guard: skip a prediction from THIS outer turn
        # (created at/after the current frozen instant — a tool-loop followup).
        if now and created and created >= now:
            return messages

        res = settle(pred.get("prediction_set_json"), user_msg)
        settled_turn_id = str((config or {}).get("_turn_id") or "")
        _fs.settle_prediction(int(pred["id"]), res.friction_score, res.friction_type,
                              res.channel_json, user_msg[:_MAX_OBS], settled_turn_id,
                              now_iso=now or _now())

        # bearing write (single writer) gated on INJECT — observe phase is read-only
        if _flag(_INJECT_FLAG):
            claim = str(pred.get("claim") or "")
            try:
                conf = float(pred.get("confidence") or 0.4)
            except (TypeError, ValueError):
                conf = 0.4
            if claim:
                _update_bearing_other_fields(claim, "literal", conf, f"reading: {claim}")

        # crystallize gated on its OWN flag (not the artifact flag) so an
        # unvalidated settler never writes ACUs during the observe phase
        if _flag(_CRYSTALLIZE_FLAG):
            try:
                from core import friction_organ
                friction_organ.maybe_crystallize(res.friction_type)
            except Exception:
                pass
    except Exception:
        pass
    return messages
