"""MonoThink Contrast V2 shadow evidence.

This module is a sibling to MonoFrame, not a merge with it. MonoFrame records
and judges answer-frame commitments. This module records and judges whether a
candidate MonoThink scaffold changes held-out reasoning behavior.

Flag: MONOLITH_MONOTHINK_CONTRAST_V1 (default off). When off, MonoThink's
existing evolution path remains byte-identical.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import mean
from typing import Any, Callable, Iterable

from core.paths import LOG_DIR


_FLAG_ENV = "MONOLITH_MONOTHINK_CONTRAST_V1"
_JUDGE_PROFILE_ENV = "MONOLITH_MONOTHINK_CONTRAST_JUDGE_PROFILE"
_RENAME_GATE_PROFILE_ENV = "MONOLITH_MONOTHINK_RENAME_GATE_PROFILE"
_TRUTHY = {"1", "true", "yes", "on"}

CASE_STORE_PATH = LOG_DIR / "monothink_contrast_cases.jsonl"
VERDICT_STORE_PATH = LOG_DIR / "monothink_contrast_verdicts.jsonl"


def enabled() -> bool:
    return os.environ.get(_FLAG_ENV, "0").strip().lower() in _TRUTHY


def sha(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ContrastCase:
    case_id: str
    tag: str
    turn_id: str
    rating: int
    created_ts: float
    base_scaffold_sha: str
    minimized_input: str
    failed_trace: str
    corrected_shape: str
    separation_predicate: str
    role: str = "canary"
    invariant: bool = False
    builder_lineage: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BuiltCase:
    corrected_shape: str
    separation_predicate: str


@dataclass(frozen=True)
class SeparationVerdict:
    admit: bool
    target_gain: float
    worst_invariant_regression: float
    candidate_sha: str
    per_case: dict[str, dict[str, Any]]
    reason: str
    old_scaffold_sha: str = ""
    tag: str = ""
    judge_lineage: dict[str, Any] = field(default_factory=dict)
    k: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "ts": time.time(),
            "kind": "separation_verdict",
            "tag": self.tag,
            "admit": bool(self.admit),
            "target_gain": float(self.target_gain),
            "worst_invariant_regression": float(self.worst_invariant_regression),
            "old_scaffold_sha": self.old_scaffold_sha,
            "candidate_sha": self.candidate_sha,
            "judge_lineage": dict(self.judge_lineage or {}),
            "k": int(self.k),
            "reason": self.reason,
            "per_case": dict(self.per_case or {}),
        }

    def journal_summary(self) -> dict[str, Any]:
        return {
            "contrast_shadow": True,
            "contrast_would_admit": bool(self.admit),
            "contrast_target_gain": float(self.target_gain),
            "contrast_worst_invariant_regression": float(self.worst_invariant_regression),
            "contrast_old_scaffold_sha": self.old_scaffold_sha,
            "contrast_candidate_sha": self.candidate_sha,
            "contrast_case_count": len(self.per_case or {}),
            "contrast_reason": self.reason,
            "contrast_judge_lineage": dict(self.judge_lineage or {}),
        }


class ContrastStore:
    """Append-only JSONL store for contrast cases."""

    def __init__(self, path: Path | None = None, prompt_ratio: float = 0.5):
        self.path = path or CASE_STORE_PATH
        self.prompt_ratio = float(prompt_ratio)

    def append(self, case: ContrastCase) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(case), ensure_ascii=False) + "\n")

    def all(self) -> list[ContrastCase]:
        if not self.path.exists():
            return []
        out: list[ContrastCase] = []
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    if isinstance(obj, dict):
                        out.append(ContrastCase(**obj))
                except Exception:
                    continue
        return out

    def role_for(self, tag: str, case_id: str) -> str:
        existing = [
            c for c in self.all()
            if c.tag == tag and not c.invariant
        ]
        roles = {c.role for c in existing}
        if not existing:
            return "canary"
        if "prompt" not in roles:
            return "prompt"
        if "canary" not in roles:
            return "canary"
        h = int(sha(case_id)[:8], 16) / 0xFFFFFFFF
        return "prompt" if h < self.prompt_ratio else "canary"

    def record_from_rating(
        self,
        *,
        turn_id: str,
        tag: str,
        rating: int,
        base_scaffold_sha: str,
        minimized_input: str,
        failed_trace: str,
        tag_gloss: str,
        build_case: Callable[[str, str, str], BuiltCase],
        builder_lineage: dict[str, Any] | None = None,
    ) -> ContrastCase:
        built = build_case(failed_trace, tag_gloss, minimized_input)
        case_id = sha(f"{turn_id}:{tag}:{base_scaffold_sha}")[:16]
        case = ContrastCase(
            case_id=case_id,
            tag=tag,
            turn_id=str(turn_id),
            rating=int(rating),
            created_ts=time.time(),
            base_scaffold_sha=str(base_scaffold_sha),
            minimized_input=str(minimized_input),
            failed_trace=str(failed_trace),
            corrected_shape=str(built.corrected_shape),
            separation_predicate=str(built.separation_predicate),
            role=self.role_for(tag, case_id),
            builder_lineage=dict(builder_lineage or {}),
        )
        self.append(case)
        return case


def record_event(event: dict[str, Any], path: Path | None = None) -> None:
    try:
        path = path or VERDICT_STORE_PATH
        path.parent.mkdir(parents=True, exist_ok=True)
        row = dict(event)
        row.setdefault("ts", time.time())
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    except Exception:
        pass


def record_verdict(verdict: SeparationVerdict, path: Path | None = None) -> None:
    record_event(verdict.to_dict(), path=path)


def _dedupe_key(trace: str) -> str:
    normalized = " ".join(str(trace or "").split())
    return sha(normalized[:200] + normalized[-200:])


def select_for_prompt(store: ContrastStore, tag: str, k: int = 3) -> list[ContrastCase]:
    cases = [
        c for c in store.all()
        if c.tag == tag and c.role == "prompt" and not c.invariant
    ]
    cases.sort(key=lambda c: c.created_ts, reverse=True)
    seen: set[str] = set()
    picked: list[ContrastCase] = []
    for case in cases:
        key = _dedupe_key(case.failed_trace)
        if key in seen:
            continue
        seen.add(key)
        picked.append(case)
        if len(picked) >= k:
            break
    return picked


def render_contrast_block(cases: Iterable[ContrastCase]) -> str:
    cases = list(cases)
    if not cases:
        return ""
    parts = [
        "CONTRAST EVIDENCE (concrete instances of the failure this edit must fix).",
        "These are behavioral exemplars, not text to paste. Propose a scaffold change "
        "that makes the corrected move the default reasoning behavior. Adding "
        "vocabulary about the failure is not a fix.",
    ]
    for i, case in enumerate(cases, 1):
        parts.append(
            f"\nCase {i} [{case.tag}]\n"
            f"  FAILED move: {case.failed_trace.strip()}\n"
            f"  CORRECTED move: {case.corrected_shape.strip()}"
        )
    return "\n".join(parts)


def _pass_rate(
    scaffold_text: str,
    case: ContrastCase,
    run_scaffold: Callable[[str, str], str],
    judge: Callable[[str, str], bool],
    k: int,
) -> float:
    hits = 0
    for _ in range(max(1, int(k))):
        trace = run_scaffold(scaffold_text, case.minimized_input)
        if judge(case.separation_predicate, trace):
            hits += 1
    return hits / max(1, int(k))


def evaluate_separation(
    *,
    tag: str,
    old_scaffold: str,
    candidate_scaffold: str,
    store: ContrastStore,
    run_scaffold: Callable[[str, str], str],
    judge: Callable[[str, str], bool],
    k: int = 3,
    delta_improve: float = 0.34,
    delta_regress: float = 0.0,
    baseline_cache: dict[tuple[str, str], float] | None = None,
    judge_lineage: dict[str, Any] | None = None,
) -> SeparationVerdict:
    old_sha = sha(old_scaffold)
    cand_sha = sha(candidate_scaffold)
    cache = baseline_cache if baseline_cache is not None else {}
    all_cases = store.all()
    target = [
        c for c in all_cases
        if c.tag == tag and c.role == "canary" and not c.invariant
    ]
    invariants = [c for c in all_cases if c.invariant]

    if not target:
        return SeparationVerdict(
            admit=False,
            target_gain=0.0,
            worst_invariant_regression=0.0,
            old_scaffold_sha=old_sha,
            candidate_sha=cand_sha,
            per_case={},
            reason="no-canary-case-for-tag",
            tag=tag,
            judge_lineage=dict(judge_lineage or {}),
            k=k,
        )

    def old_pass(case: ContrastCase) -> float:
        key = (old_sha, case.case_id)
        if key not in cache:
            cache[key] = _pass_rate(old_scaffold, case, run_scaffold, judge, k)
        return cache[key]

    per_case: dict[str, dict[str, Any]] = {}
    gains: list[float] = []
    regressions: list[float] = []

    for case in target:
        old_rate = old_pass(case)
        new_rate = _pass_rate(candidate_scaffold, case, run_scaffold, judge, k)
        per_case[case.case_id] = {
            "old": old_rate,
            "cand": new_rate,
            "tag": case.tag,
            "role": case.role,
            "predicate_sha": sha(case.separation_predicate),
        }
        gains.append(new_rate - old_rate)

    for case in invariants:
        old_rate = old_pass(case)
        new_rate = _pass_rate(candidate_scaffold, case, run_scaffold, judge, k)
        per_case[case.case_id] = {
            "old": old_rate,
            "cand": new_rate,
            "tag": case.tag,
            "role": case.role,
            "predicate_sha": sha(case.separation_predicate),
        }
        regressions.append(old_rate - new_rate)

    target_gain = mean(gains) if gains else 0.0
    worst_regression = max(regressions) if regressions else 0.0
    admit = target_gain >= delta_improve and worst_regression <= delta_regress
    reason = (
        f"target_gain={target_gain:+.2f} need>={delta_improve:+.2f}; "
        f"worst_invariant_regression={worst_regression:+.2f} allow<={delta_regress:+.2f}"
    )
    return SeparationVerdict(
        admit=admit,
        target_gain=target_gain,
        worst_invariant_regression=worst_regression,
        old_scaffold_sha=old_sha,
        candidate_sha=cand_sha,
        per_case=per_case,
        reason=reason,
        tag=tag,
        judge_lineage=dict(judge_lineage or {}),
        k=k,
    )


CASE_BUILDER_PROMPT = """You convert a flawed reasoning trace into neutral training evidence.
Return ONLY JSON: {"corrected_shape": "...", "separation_predicate": "..."}

corrected_shape: minimal structural description of the corrected reasoning move.
separation_predicate: one yes/no question about future trace behavior where yes means the corrected move is present.

Failure tag: {tag}
Definition: {gloss}
Original input:
{input}
Trace:
{trace}
"""

JUDGE_PROMPT = """Answer strictly "yes" or "no".
Question: {predicate}
Reasoning trace to evaluate:
{trace}
"""


def resolve_judge_profile() -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """Resolve the pinned decorrelated judge profile."""
    try:
        from core.cloud_profiles import list_profiles
        from core.config import get_config

        cfg = get_config().llm.model_dump()
        profiles = list_profiles(cfg)
    except Exception as exc:
        return None, {"available": False, "reason": f"profile_load_failed:{type(exc).__name__}"}

    wanted = (
        os.environ.get(_JUDGE_PROFILE_ENV, "").strip()
        or os.environ.get(_RENAME_GATE_PROFILE_ENV, "").strip()
    )
    chosen: dict[str, Any] | None = None
    if wanted:
        for prof in profiles:
            if wanted in {str(prof.get("id") or ""), str(prof.get("label") or "")}:
                chosen = prof
                break
        if chosen is None:
            return None, {
                "available": False,
                "reason": "judge_profile_not_found",
                "requested": wanted,
            }
    else:
        for prof in profiles:
            pid = str(prof.get("id") or "")
            if pid == "anthropic|https://api.anthropic.com":
                chosen = prof
                break
        if chosen is None:
            return None, {"available": False, "reason": "no_decorrelated_judge_profile"}

    model = str(chosen.get("last_model") or chosen.get("api_model") or "")
    lineage = {
        "available": bool(model),
        "profile_id": str(chosen.get("id") or ""),
        "label": str(chosen.get("label") or ""),
        "provider": str(chosen.get("api_provider") or ""),
        "api_base": str(chosen.get("api_base") or ""),
        "model": model,
    }
    if not model:
        lineage["reason"] = "judge_profile_missing_model"
        return None, lineage
    return chosen, lineage


def _generate_with_profile(profile: dict[str, Any], messages: list[dict[str, str]], *, max_tokens: int) -> str:
    from engine.llm import make_cloud_llm
    from engine.sync_bridge import generate_sync

    api_base = str(profile.get("api_base") or "").strip()
    api_key = str(profile.get("api_key") or "")
    api_model = str(profile.get("last_model") or profile.get("api_model") or "").strip()
    if not api_base or not api_model:
        raise RuntimeError("missing judge profile api_base/api_model")
    client = make_cloud_llm(api_base, api_key, api_model)
    cfg = {
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": int(max_tokens),
        "api_base": api_base,
        "api_key": api_key,
        "api_model": api_model,
    }
    return generate_sync(client, messages, cfg, thinking_enabled=False)


def default_build_case(
    failed_trace: str,
    tag_gloss: str,
    minimized_input: str,
    *,
    tag: str,
    profile: dict[str, Any],
) -> BuiltCase:
    prompt = CASE_BUILDER_PROMPT.format(
        tag=tag,
        gloss=tag_gloss,
        input=minimized_input,
        trace=failed_trace,
    )
    raw = _generate_with_profile(profile, [{"role": "user", "content": prompt}], max_tokens=800)
    obj = json.loads(raw)
    corrected = str(obj.get("corrected_shape") or "").strip()
    predicate = str(obj.get("separation_predicate") or "").strip()
    if not corrected or not predicate:
        raise ValueError("case_builder_missing_fields")
    return BuiltCase(corrected_shape=corrected, separation_predicate=predicate)


def default_judge(predicate: str, trace: str, *, profile: dict[str, Any]) -> bool:
    prompt = JUDGE_PROMPT.format(predicate=predicate, trace=trace)
    raw = _generate_with_profile(profile, [{"role": "user", "content": prompt}], max_tokens=12)
    return raw.strip().lower().startswith("yes")


def default_run_scaffold(scaffold_text: str, minimized_input: str) -> str:
    from core.config import get_config
    from engine.llm import make_cloud_llm
    from engine.sync_bridge import generate_sync_parts

    cfg = get_config().llm.model_dump()
    api_base = str(cfg.get("api_base") or "").strip()
    api_model = str(cfg.get("api_model") or "").strip()
    if not api_base or not api_model:
        raise RuntimeError("missing active generator api_base/api_model")
    client = make_cloud_llm(api_base, str(cfg.get("api_key") or ""), api_model)
    messages = [
        {
            "role": "user",
            "content": (
                "[MONOTHINK] - reasoning scaffold for this shadow replay; "
                "runtime-injected, not user text.\n"
                f"{scaffold_text}"
            ),
        },
        {"role": "user", "content": minimized_input},
    ]
    call_cfg = dict(cfg)
    call_cfg["temperature"] = 0.3
    call_cfg["top_p"] = 1.0
    call_cfg["max_tokens"] = 2048
    content, reasoning = generate_sync_parts(client, messages, call_cfg, thinking_enabled=True)
    return reasoning or content


def build_case_with_default_profile(
    *,
    store: ContrastStore,
    turn_id: str,
    tag: str,
    rating: int,
    base_scaffold_sha: str,
    minimized_input: str | None,
    failed_trace: str | None,
    tag_gloss: str,
) -> ContrastCase | None:
    if not (minimized_input and str(minimized_input).strip()):
        record_event({"kind": "case_build_skipped", "turn_id": turn_id, "tag": tag, "reason": "missing_replay_input"})
        return None
    if not (failed_trace and str(failed_trace).strip()):
        record_event({"kind": "case_build_skipped", "turn_id": turn_id, "tag": tag, "reason": "missing_failed_trace"})
        return None

    profile, lineage = resolve_judge_profile()
    if profile is None:
        record_event({
            "kind": "case_build_skipped",
            "turn_id": turn_id,
            "tag": tag,
            "reason": "judge_unavailable",
            "judge_lineage": lineage,
        })
        return None

    try:
        return store.record_from_rating(
            turn_id=turn_id,
            tag=tag,
            rating=rating,
            base_scaffold_sha=base_scaffold_sha,
            minimized_input=str(minimized_input),
            failed_trace=str(failed_trace),
            tag_gloss=tag_gloss,
            build_case=lambda trace, gloss, inp: default_build_case(
                trace, gloss, inp, tag=tag, profile=profile,
            ),
            builder_lineage=lineage,
        )
    except Exception as exc:
        record_event({
            "kind": "case_build_failed",
            "turn_id": turn_id,
            "tag": tag,
            "reason": f"{type(exc).__name__}:{exc}",
            "judge_lineage": lineage,
        })
        return None


def evaluate_shadow_default(
    *,
    tag: str,
    old_scaffold: str,
    candidate_scaffold: str,
    store: ContrastStore,
    k: int = 3,
) -> SeparationVerdict:
    profile, lineage = resolve_judge_profile()
    if profile is None:
        return SeparationVerdict(
            admit=False,
            target_gain=0.0,
            worst_invariant_regression=0.0,
            old_scaffold_sha=sha(old_scaffold),
            candidate_sha=sha(candidate_scaffold),
            per_case={},
            reason="judge_unavailable",
            tag=tag,
            judge_lineage=lineage,
            k=k,
        )
    return evaluate_separation(
        tag=tag,
        old_scaffold=old_scaffold,
        candidate_scaffold=candidate_scaffold,
        store=store,
        run_scaffold=default_run_scaffold,
        judge=lambda predicate, trace: default_judge(predicate, trace, profile=profile),
        k=k,
        judge_lineage=lineage,
    )
