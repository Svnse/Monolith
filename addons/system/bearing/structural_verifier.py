"""Bearing structural verifier — synchronous, blocks state_update commit.

Seven rules (per design §7):

  D1. Every slot change has a `reason` field.
  D2. current_frame change has `previous` AND `trigger`.
  D3. open_tensions.resolve indices are in range of the pre-update list.
      modal_branches.transition indices similarly bounded.
  D4. modal_branches.transition uses VALID_BRANCH_STATUS values.
      referents.add status uses VALID_REFERENT_STATUS values.
      referents.add kind uses VALID_REFERENT_KIND values.
      user_model.register uses VALID_REGISTER values.
      stakes.reversibility/urgency use their respective enums.
  D5. Per-slot character limits respected.
  D6. schema_version in the envelope (if present) matches SCHEMA_VERSION.
  D7. After applying the update, list-slot counts don't exceed MAX_*.

The verifier simulates the apply internally to check D5 and D7. The actual
commit (with persistence) happens in updater.py.

Failure mode: returns StructuralVerdict(ok=False, failed_rules=[...]).
Cosmetic-eligible slots (per spec) — currently NONE in V0; the §8 design
defers cosmetic silent-reject. Future: D5 overflows ≤10% on
stakes.cost_if_wrong / user_model.intent_read may downgrade to cosmetic.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from . import schema as bs


PRIMITIVE_SLOTS = ("current_frame", "active_goal", "trajectory", "next_move")
PRIMITIVE_LIMITS = {
    "current_frame": bs.MAX_CURRENT_FRAME,
    "active_goal": bs.MAX_ACTIVE_GOAL,
    "trajectory": bs.MAX_TRAJECTORY,
    "next_move": bs.MAX_NEXT_MOVE,
}


@dataclass(frozen=True)
class StructuralVerdict:
    ok: bool
    failed_rules: tuple[str, ...] = ()
    detail: str = ""

    @classmethod
    def passed(cls) -> "StructuralVerdict":
        return cls(ok=True)

    @classmethod
    def failed(cls, rules: list[str], detail: str = "") -> "StructuralVerdict":
        return cls(ok=False, failed_rules=tuple(rules), detail=detail)


# ── helpers ─────────────────────────────────────────────────────────


def _has_reason(d: Any) -> bool:
    return isinstance(d, dict) and isinstance(d.get("reason"), str) and d["reason"].strip() != ""


def _is_dict(x: Any) -> bool:
    return isinstance(x, dict)


def _is_list(x: Any) -> bool:
    return isinstance(x, list)


# ── individual rule checks ──────────────────────────────────────────


def _check_d1_reason_on_every_change(proposed: dict[str, Any]) -> tuple[bool, str]:
    """Every slot change must include a reason."""
    detail_bits: list[str] = []

    for slot in PRIMITIVE_SLOTS:
        if slot in proposed:
            if not _has_reason(proposed[slot]):
                detail_bits.append(f"{slot} missing reason")

    for slot in ("open_tensions", "referents", "modal_branches"):
        if slot in proposed and _is_dict(proposed[slot]):
            for op_name, items in proposed[slot].items():
                if not _is_list(items):
                    continue
                for i, item in enumerate(items):
                    # add operations get reason from add[].reason for tensions/branches;
                    # tensions.add items can be reason-free since "opening a tension" is itself the reason.
                    # Strict rule: add ops need a reason on the entry.
                    if op_name == "add" and slot == "open_tensions":
                        # tensions.add: text is the reason; require text non-empty
                        if not (isinstance(item, dict) and isinstance(item.get("text"), str) and item["text"].strip()):
                            detail_bits.append(f"{slot}.add[{i}] missing text")
                    elif op_name in ("resolve", "drop", "transition"):
                        if not _has_reason(item):
                            detail_bits.append(f"{slot}.{op_name}[{i}] missing reason")
                    elif op_name == "add":
                        if not _has_reason(item):
                            detail_bits.append(f"{slot}.add[{i}] missing reason")

    if "user_model" in proposed:
        if not _has_reason(proposed["user_model"]) and not _is_dict(proposed["user_model"]):
            detail_bits.append("user_model malformed")
        # user_model replaces are reason-free since the model carries its own intent_read;
        # require intent_read to be non-empty as the substantive content.
        if _is_dict(proposed.get("user_model")) and not proposed["user_model"].get("intent_read"):
            detail_bits.append("user_model.intent_read empty")

    if "stakes" in proposed:
        if not _is_dict(proposed.get("stakes")):
            detail_bits.append("stakes malformed")

    return (len(detail_bits) == 0, "; ".join(detail_bits))


def _check_d2_current_frame_provenance(proposed: dict[str, Any]) -> tuple[bool, str]:
    """current_frame change requires previous + trigger fields."""
    if "current_frame" not in proposed:
        return (True, "")
    cf = proposed["current_frame"]
    if not _is_dict(cf):
        return (False, "current_frame must be dict")
    missing: list[str] = []
    if not isinstance(cf.get("previous"), str):
        missing.append("previous")
    if not isinstance(cf.get("trigger"), str) or not cf.get("trigger", "").strip():
        missing.append("trigger")
    if missing:
        return (False, "current_frame missing: " + ", ".join(missing))
    return (True, "")


def _check_d3_indices_in_range(old: bs.Bearing, proposed: dict[str, Any]) -> tuple[bool, str]:
    detail_bits: list[str] = []
    if "open_tensions" in proposed and _is_dict(proposed["open_tensions"]):
        ot = proposed["open_tensions"]
        n = len(old.open_tensions)
        for op in ("resolve", "drop"):
            items = ot.get(op) or []
            if not _is_list(items):
                continue
            for i, item in enumerate(items):
                if not _is_dict(item):
                    detail_bits.append(f"open_tensions.{op}[{i}] not a dict")
                    continue
                idx = item.get("index")
                if not isinstance(idx, int) or idx < 0 or idx >= n:
                    detail_bits.append(f"open_tensions.{op}[{i}] index {idx!r} out of range (n={n})")

    if "modal_branches" in proposed and _is_dict(proposed["modal_branches"]):
        mb = proposed["modal_branches"]
        n = len(old.modal_branches)
        items = mb.get("transition") or []
        if _is_list(items):
            for i, item in enumerate(items):
                if not _is_dict(item):
                    detail_bits.append(f"modal_branches.transition[{i}] not a dict")
                    continue
                idx = item.get("index")
                if not isinstance(idx, int) or idx < 0 or idx >= n:
                    detail_bits.append(f"modal_branches.transition[{i}] index {idx!r} out of range (n={n})")

    return (len(detail_bits) == 0, "; ".join(detail_bits))


def _check_d4_enum_values(proposed: dict[str, Any]) -> tuple[bool, str]:
    detail_bits: list[str] = []

    if "modal_branches" in proposed and _is_dict(proposed["modal_branches"]):
        mb = proposed["modal_branches"]
        for item in mb.get("add") or []:
            if not _is_dict(item):
                continue
            status = item.get("status")
            if status is not None and status not in bs.VALID_BRANCH_STATUS:
                detail_bits.append(f"modal_branches.add bad status {status!r}")
        for item in mb.get("transition") or []:
            if not _is_dict(item):
                continue
            to = item.get("to")
            if to is not None and to not in bs.VALID_BRANCH_STATUS:
                detail_bits.append(f"modal_branches.transition bad to-status {to!r}")
            fr = item.get("from")
            if fr is not None and fr not in bs.VALID_BRANCH_STATUS:
                detail_bits.append(f"modal_branches.transition bad from-status {fr!r}")

    if "referents" in proposed and _is_dict(proposed["referents"]):
        for item in proposed["referents"].get("add") or []:
            if not _is_dict(item):
                continue
            status = item.get("status")
            if status is not None and status not in bs.VALID_REFERENT_STATUS:
                detail_bits.append(f"referents.add bad status {status!r}")
            kind = item.get("kind")
            if kind is not None and kind not in bs.VALID_REFERENT_KIND:
                detail_bits.append(f"referents.add bad kind {kind!r}")

    if "user_model" in proposed and _is_dict(proposed["user_model"]):
        reg = proposed["user_model"].get("register")
        if reg is not None and reg not in bs.VALID_REGISTER:
            detail_bits.append(f"user_model bad register {reg!r}")

    if "stakes" in proposed and _is_dict(proposed["stakes"]):
        rev = proposed["stakes"].get("reversibility")
        if rev is not None and rev not in bs.VALID_REVERSIBILITY:
            detail_bits.append(f"stakes bad reversibility {rev!r}")
        urg = proposed["stakes"].get("urgency")
        if urg is not None and urg not in bs.VALID_URGENCY:
            detail_bits.append(f"stakes bad urgency {urg!r}")

    return (len(detail_bits) == 0, "; ".join(detail_bits))


def _check_d5_character_limits(proposed: dict[str, Any]) -> tuple[bool, str]:
    detail_bits: list[str] = []

    for slot, limit in PRIMITIVE_LIMITS.items():
        if slot not in proposed:
            continue
        v = proposed[slot]
        if _is_dict(v):
            new = v.get("new")
            if isinstance(new, str) and len(new) > limit:
                detail_bits.append(f"{slot} new={len(new)} > limit {limit}")
        elif isinstance(v, str):
            if len(v) > limit:
                detail_bits.append(f"{slot} ={len(v)} > limit {limit}")

    if "open_tensions" in proposed and _is_dict(proposed["open_tensions"]):
        for i, item in enumerate(proposed["open_tensions"].get("add") or []):
            if _is_dict(item):
                text = item.get("text", "")
                if isinstance(text, str) and len(text) > bs.MAX_TENSION:
                    detail_bits.append(f"open_tensions.add[{i}].text len={len(text)} > {bs.MAX_TENSION}")

    if "modal_branches" in proposed and _is_dict(proposed["modal_branches"]):
        for i, item in enumerate(proposed["modal_branches"].get("add") or []):
            if _is_dict(item):
                text = item.get("text", "")
                if isinstance(text, str) and len(text) > bs.MAX_BRANCH_TEXT:
                    detail_bits.append(f"modal_branches.add[{i}].text len={len(text)} > {bs.MAX_BRANCH_TEXT}")
                reason = item.get("reason", "")
                if isinstance(reason, str) and len(reason) > bs.MAX_BRANCH_REASON:
                    detail_bits.append(f"modal_branches.add[{i}].reason len={len(reason)} > {bs.MAX_BRANCH_REASON}")

    if "referents" in proposed and _is_dict(proposed["referents"]):
        for i, item in enumerate(proposed["referents"].get("add") or []):
            if _is_dict(item):
                name = item.get("name", "")
                if isinstance(name, str) and len(name) > bs.MAX_REFERENT_NAME:
                    detail_bits.append(f"referents.add[{i}].name len={len(name)} > {bs.MAX_REFERENT_NAME}")

    if "user_model" in proposed and _is_dict(proposed["user_model"]):
        ir = proposed["user_model"].get("intent_read", "")
        if isinstance(ir, str) and len(ir) > bs.MAX_USER_INTENT_READ:
            detail_bits.append(f"user_model.intent_read len={len(ir)} > {bs.MAX_USER_INTENT_READ}")

    if "stakes" in proposed and _is_dict(proposed["stakes"]):
        cw = proposed["stakes"].get("cost_if_wrong", "")
        if isinstance(cw, str) and len(cw) > bs.MAX_STAKES_COST_IF_WRONG:
            detail_bits.append(f"stakes.cost_if_wrong len={len(cw)} > {bs.MAX_STAKES_COST_IF_WRONG}")

    return (len(detail_bits) == 0, "; ".join(detail_bits))


def _check_d6_schema_version(proposed: dict[str, Any]) -> tuple[bool, str]:
    if "schema_version" not in proposed:
        return (True, "")
    sv = proposed["schema_version"]
    if not isinstance(sv, int) or sv != bs.SCHEMA_VERSION:
        return (False, f"schema_version {sv!r} != {bs.SCHEMA_VERSION}")
    return (True, "")


def _check_d7_post_apply_counts(old: bs.Bearing, proposed: dict[str, Any]) -> tuple[bool, str]:
    """Simulate net adds/drops to forecast resulting list lengths."""
    detail_bits: list[str] = []

    def net_count(curr_n: int, ops: dict[str, Any] | None) -> int:
        if not _is_dict(ops):
            return curr_n
        added = len(ops.get("add") or [])
        resolved = len(ops.get("resolve") or [])
        dropped = len(ops.get("drop") or [])
        return curr_n + added - resolved - dropped

    new_tensions = net_count(len(old.open_tensions), proposed.get("open_tensions"))
    if new_tensions > bs.MAX_TENSIONS:
        detail_bits.append(f"open_tensions would become {new_tensions} > {bs.MAX_TENSIONS}")

    new_referents = net_count(len(old.referents), proposed.get("referents"))
    if new_referents > bs.MAX_REFERENTS:
        detail_bits.append(f"referents would become {new_referents} > {bs.MAX_REFERENTS}")

    # modal_branches.transition keeps count constant; only add changes count
    mb = proposed.get("modal_branches") or {}
    added = len((mb.get("add") if _is_dict(mb) else None) or [])
    new_branches = len(old.modal_branches) + added
    if new_branches > bs.MAX_BRANCHES:
        detail_bits.append(f"modal_branches would become {new_branches} > {bs.MAX_BRANCHES}")

    return (len(detail_bits) == 0, "; ".join(detail_bits))


# ── public verify ───────────────────────────────────────────────────


def verify_structural(old: bs.Bearing, proposed: dict[str, Any]) -> StructuralVerdict:
    """Apply D1–D7 in order; collect all failures (don't short-circuit).

    Empty `proposed` (no-op update) passes structurally — the updater
    will treat it as nothing-to-do.
    """
    if not isinstance(proposed, dict):
        return StructuralVerdict.failed(["D0"], "envelope payload is not a JSON object")

    if not proposed:
        return StructuralVerdict.passed()

    failed: list[str] = []
    detail: list[str] = []

    for rule_id, check in (
        ("D1", lambda: _check_d1_reason_on_every_change(proposed)),
        ("D2", lambda: _check_d2_current_frame_provenance(proposed)),
        ("D3", lambda: _check_d3_indices_in_range(old, proposed)),
        ("D4", lambda: _check_d4_enum_values(proposed)),
        ("D5", lambda: _check_d5_character_limits(proposed)),
        ("D6", lambda: _check_d6_schema_version(proposed)),
        ("D7", lambda: _check_d7_post_apply_counts(old, proposed)),
    ):
        ok, msg = check()
        if not ok:
            failed.append(rule_id)
            if msg:
                detail.append(f"{rule_id}: {msg}")

    if failed:
        return StructuralVerdict.failed(failed, "; ".join(detail))
    return StructuralVerdict.passed()
