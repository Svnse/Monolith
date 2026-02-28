from __future__ import annotations

from dataclasses import dataclass

from engine.loop.contracts import Pad, Step


@dataclass
class MissionAdjudication:
    ok: bool
    score: float
    missing_required: list[str]


def adjudicate_mission(pad: Pad, step: Step) -> MissionAdjudication:
    contract = getattr(pad, "mission_contract", {}) if isinstance(getattr(pad, "mission_contract", {}), dict) else {}
    criteria = contract.get("success_criteria") if isinstance(contract.get("success_criteria"), list) else []
    if not criteria:
        return MissionAdjudication(ok=True, score=1.0, missing_required=[])

    done_refs = set(str(r).strip() for r in (getattr(pad, "mission_refs", []) or []) if str(r).strip())
    done_refs.update(str(r).strip() for r in (getattr(step, "contract_refs", []) or []) if str(r).strip())

    required_ids: list[str] = []
    missing: list[str] = []
    total = 0
    met = 0
    has_verification = _has_successful_verification(pad)
    has_functional = _has_functional_behavior(pad)
    for item in criteria:
        if not isinstance(item, dict):
            continue
        cid = str(item.get("id") or "").strip()
        if not cid:
            continue
        text = str(item.get("text") or "").strip().lower()
        cid_lower = cid.lower()
        total += 1
        criterion_met = cid in done_refs
        if ("verification" in cid_lower or "evidence" in cid_lower or "verification" in text or "evidence" in text):
            criterion_met = criterion_met and has_verification
        if ("functional_behavior" in cid_lower or "functional" in text):
            criterion_met = criterion_met and has_functional
        if criterion_met:
            met += 1
        if bool(item.get("required", True)):
            required_ids.append(cid)
            if not criterion_met:
                missing.append(cid)

    score = float(met) / float(total) if total > 0 else 1.0
    ok = len(missing) == 0 and (total == 0 or score >= 0.999)
    return MissionAdjudication(ok=ok, score=score, missing_required=missing)


def _last_write_cycle(pad: Pad) -> int:
    last = -1
    for e in list(getattr(pad, "evidence", []) or []):
        tool = str(getattr(e, "tool", "") or "")
        if tool in {"write_file", "apply_patch", "mkdir", "copy_path", "move_path"}:
            try:
                last = max(last, int(getattr(e, "cycle", -1)))
            except Exception:
                continue
    return last


def _has_successful_verification(pad: Pad) -> bool:
    last_write = _last_write_cycle(pad)
    if last_write < 0:
        return False
    for e in list(getattr(pad, "evidence", []) or []):
        tool = str(getattr(e, "tool", "") or "")
        ok = bool(getattr(e, "ok", False))
        try:
            cycle = int(getattr(e, "cycle", -1))
        except Exception:
            cycle = -1
        if cycle > last_write and ok and tool in {"run_cmd", "run_tests", "read_file"}:
            return True
    return False


def _has_functional_behavior(pad: Pad) -> bool:
    last_write = _last_write_cycle(pad)
    if last_write < 0:
        return False
    for e in list(getattr(pad, "evidence", []) or []):
        tool = str(getattr(e, "tool", "") or "")
        ok = bool(getattr(e, "ok", False))
        try:
            cycle = int(getattr(e, "cycle", -1))
        except Exception:
            cycle = -1
        if cycle > last_write and ok and tool in {"run_cmd", "run_tests", "read_file"}:
            return True
    return False
