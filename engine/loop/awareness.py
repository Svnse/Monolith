from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

from engine.loop.contracts import PreflightResult, RunPolicy


@dataclass
class HardwareProfile:
    """Persistent observations about local inference behavior."""

    latency_ema: float = 0.0
    latency_samples: int = 0
    latency_last: float = 0.0
    latency_max_seen: float = 0.0

    chars_per_token_ema: float = 3.5
    throughput_samples: int = 0

    truncation_total: int = 0
    truncation_hits: int = 0

    model_path_hash: str = ""
    n_ctx: int = 0

    EMA_ALPHA: float = 0.3


@dataclass
class RunProfile:
    """Ephemeral telemetry for a single run."""

    run_id: str = ""
    wall_start: float = 0.0
    preflight_start: float = 0.0
    preflight_end: float = 0.0
    runtime_start: float = 0.0

    call_latencies: list[float] = field(default_factory=list)
    call_output_chars: list[int] = field(default_factory=list)
    call_truncated: list[bool] = field(default_factory=list)

    avg_latency: float = 0.0
    estimated_remaining_cycles: int = 0
    pace_ratio: float = 1.0
    truncation_streak: int = 0
    budget_viable: bool = True

    effective_max_elapsed: float = 0.0
    current_max_tokens: int = 0
    open_todo_count: int = 3
    cycle_count: int = 0

    _pending_call_starts: dict[int, float] = field(default_factory=dict, repr=False)


@dataclass(frozen=True)
class AwarenessVerdict:
    viable: bool
    adjusted_policy: RunPolicy | None
    adjusted_infer: dict[str, Any] | None
    warnings: list[str]
    estimated_max_cycles: int
    estimated_tokens_needed: int
    reason: str


def _hardware_profile_path(config_dir: str | Path) -> Path:
    return Path(config_dir).expanduser() / "hardware_profile.json"


def load_hardware_profile(
    config_dir: str | Path,
    *,
    model_path_hash: str = "",
    n_ctx: int = 0,
) -> HardwareProfile:
    path = _hardware_profile_path(config_dir)
    profile = HardwareProfile()
    if path.exists():
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                profile = HardwareProfile(
                    latency_ema=float(raw.get("latency_ema", 0.0) or 0.0),
                    latency_samples=int(raw.get("latency_samples", 0) or 0),
                    latency_last=float(raw.get("latency_last", 0.0) or 0.0),
                    latency_max_seen=float(raw.get("latency_max_seen", 0.0) or 0.0),
                    chars_per_token_ema=float(raw.get("chars_per_token_ema", 3.5) or 3.5),
                    throughput_samples=int(raw.get("throughput_samples", 0) or 0),
                    truncation_total=int(raw.get("truncation_total", 0) or 0),
                    truncation_hits=int(raw.get("truncation_hits", 0) or 0),
                    model_path_hash=str(raw.get("model_path_hash", "") or ""),
                    n_ctx=int(raw.get("n_ctx", 0) or 0),
                )
        except Exception:
            profile = HardwareProfile()

    incoming_hash = str(model_path_hash or "").strip()
    incoming_ctx = int(n_ctx or 0)
    if incoming_hash and profile.model_path_hash and incoming_hash != profile.model_path_hash:
        profile = HardwareProfile(model_path_hash=incoming_hash, n_ctx=incoming_ctx)
    else:
        if incoming_hash:
            profile.model_path_hash = incoming_hash
        if incoming_ctx > 0:
            profile.n_ctx = incoming_ctx
    return profile


def save_hardware_profile(config_dir: str | Path, profile: HardwareProfile) -> None:
    path = _hardware_profile_path(config_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = asdict(profile)
    payload.pop("EMA_ALPHA", None)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


class RuntimeAwareness:
    ENVELOPE_OVERHEAD = 350
    DEFAULT_LATENCY_SEC = 30.0

    def start_run(self, *, run_id: str, max_elapsed_sec: float, max_tokens: int, wall_start: float | None = None) -> RunProfile:
        now = float(wall_start if wall_start is not None else time.time())
        return RunProfile(
            run_id=str(run_id or ""),
            wall_start=now,
            preflight_start=now,
            effective_max_elapsed=float(max_elapsed_sec),
            current_max_tokens=int(max_tokens),
        )

    def estimate_needed_tokens(self, preflight: PreflightResult | None, user_prompt: str) -> int:
        if preflight is not None:
            complexity = int(preflight.execution_weight or 3)
        else:
            n = len(str(user_prompt or ""))
            if n < 120:
                complexity = 2
            elif n < 450:
                complexity = 3
            elif n < 900:
                complexity = 4
            else:
                complexity = 5
        mapping = {1: 400, 2: 800, 3: 1200, 4: 2000, 5: 3000}
        return int(mapping.get(complexity, 1200))

    def pre_run_gate(
        self,
        *,
        policy: RunPolicy,
        infer_config: dict[str, Any],
        hardware: HardwareProfile,
        preflight: PreflightResult | None,
        user_prompt: str,
    ) -> AwarenessVerdict:
        warnings: list[str] = []
        adjustments_policy: dict[str, Any] = {}
        adjustments_infer: dict[str, Any] = {}

        max_tokens = int(infer_config.get("max_tokens", 0) or 0)
        estimated_latency = float(hardware.latency_ema or self.DEFAULT_LATENCY_SEC)
        estimated_preflight = estimated_latency * 2.0
        estimated_per_cycle = estimated_latency * 1.3
        usable_time = max(0.0, float(policy.max_elapsed_sec) - estimated_preflight)
        max_affordable = int(math.floor(usable_time / max(1.0, estimated_per_cycle)))

        todo_count = len(list(preflight.todo or [])) if preflight is not None else 3
        min_cycles_needed = max(3, int(todo_count) * 2)
        floor_time = float(estimated_preflight) + (float(min_cycles_needed) * float(estimated_per_cycle))
        if floor_time > float(policy.max_elapsed_sec):
            recommended = int(floor_time * 1.2)
            warnings.append(
                f"gate: floor_time~{floor_time:.0f}s (preflight~{estimated_preflight:.0f}s + "
                f"{min_cycles_needed} cycles x ~{estimated_per_cycle:.0f}s) exceeds max_elapsed_sec="
                f"{float(policy.max_elapsed_sec):.0f}. Recommend max_elapsed_sec~{recommended}s."
            )
            adjusted_time = min(900.0, float(floor_time) * 1.2)
            if adjusted_time > float(policy.max_elapsed_sec):
                adjustments_policy["max_elapsed_sec"] = adjusted_time

        usable_content_tokens = max_tokens - self.ENVELOPE_OVERHEAD
        estimated_needed = self.estimate_needed_tokens(preflight, user_prompt)
        if usable_content_tokens < estimated_needed:
            floor_tokens = estimated_needed + self.ENVELOPE_OVERHEAD
            max_sane = (int(hardware.n_ctx) // 3) if int(hardware.n_ctx) > 0 else 8192
            adjusted_tokens = min(max_sane, max(int(floor_tokens), 2048))
            if adjusted_tokens > max_tokens:
                adjustments_infer["max_tokens"] = adjusted_tokens
                warnings.append(
                    f"gate: max_tokens={max_tokens} leaves ~{usable_content_tokens} content tokens; "
                    f"estimated need is ~{estimated_needed}. Adjusting max_tokens to {adjusted_tokens}."
                )

        if int(hardware.truncation_total) >= 5:
            rate = float(hardware.truncation_hits) / max(1.0, float(hardware.truncation_total))
            if rate > 0.3 and "max_tokens" not in adjustments_infer:
                max_sane = (int(hardware.n_ctx) // 3) if int(hardware.n_ctx) > 0 else 8192
                bumped = min(max_sane, int(max_tokens * 1.5))
                if bumped > max_tokens:
                    adjustments_infer["max_tokens"] = bumped
                    warnings.append(
                        f"gate: historical truncation rate is {rate:.0%}; bumping max_tokens to {bumped}."
                    )

        effective_time = float(adjustments_policy.get("max_elapsed_sec", policy.max_elapsed_sec))
        realistic_max_cycles = int(max(0.0, (effective_time - estimated_preflight)) / max(1.0, estimated_per_cycle))
        effective_max_cycles = int(adjustments_policy.get("max_cycles", policy.max_cycles))
        if realistic_max_cycles > 0 and effective_max_cycles > int(realistic_max_cycles * 1.5):
            adjustments_policy["max_cycles"] = realistic_max_cycles
            warnings.append(
                f"gate: max_cycles={effective_max_cycles} is unreachable in {effective_time:.0f}s at "
                f"~{estimated_per_cycle:.0f}s/cycle (+preflight~{estimated_preflight:.0f}s). "
                f"Clamping to {realistic_max_cycles}."
            )

        adjusted_policy = replace(policy, **adjustments_policy) if adjustments_policy else None
        adjusted_infer = dict(adjustments_infer) if adjustments_infer else None
        return AwarenessVerdict(
            viable=True,
            adjusted_policy=adjusted_policy,
            adjusted_infer=adjusted_infer,
            warnings=warnings,
            estimated_max_cycles=max_affordable,
            estimated_tokens_needed=int(estimated_needed),
            reason="ok",
        )

    def observe_event(
        self,
        *,
        profile: RunProfile,
        hardware: HardwareProfile,
        kind: str,
        data: dict[str, Any],
    ) -> list[str]:
        advisories: list[str] = []
        event = str(kind or "").strip()
        if event == "cycle_start":
            if profile.runtime_start <= 0.0:
                profile.runtime_start = time.time()
            profile.cycle_count += 1
            pad = data.get("pad") if isinstance(data.get("pad"), dict) else {}
            todo_open = pad.get("todo_open")
            if isinstance(todo_open, int):
                profile.open_todo_count = max(0, todo_open)
        elif event == "llm_input":
            call_index = int(data.get("call_index", 0) or 0)
            if call_index > 0:
                profile._pending_call_starts[call_index] = time.time()
        elif event == "llm_call":
            call_index = int(data.get("call_index", 0) or 0)
            start_ts = profile._pending_call_starts.pop(call_index, None)
            if start_ts is not None:
                latency = max(0.0, time.time() - start_ts)
                profile.call_latencies.append(latency)
                profile.avg_latency = sum(profile.call_latencies) / max(1, len(profile.call_latencies))

            output_chars = int(data.get("response_chars", 0) or 0)
            profile.call_output_chars.append(output_chars)
            max_tokens = max(1, int(profile.current_max_tokens or 1))

            chars_per_token = float(hardware.chars_per_token_ema or 3.5)
            ceiling = float(max_tokens) * chars_per_token * 0.90
            was_parse_amputated = bool(((data.get("repair_flags") or {}).get("actions_likely_amputated")))
            truncated = bool(output_chars >= ceiling and data.get("ok", True) is not False) or was_parse_amputated
            profile.call_truncated.append(truncated)
            if truncated:
                profile.truncation_streak += 1
                if profile.truncation_streak >= 2:
                    advisories.append(
                        f"truncation: {profile.truncation_streak} consecutive near-ceiling outputs "
                        f"(max_tokens={max_tokens}, response_chars={output_chars})."
                    )
            else:
                profile.truncation_streak = 0

            if profile.runtime_start <= 0.0:
                profile.runtime_start = time.time()
            elapsed = max(0.0, time.time() - profile.runtime_start)
            remaining_time = max(0.0, float(profile.effective_max_elapsed) - elapsed)
            if profile.avg_latency > 0:
                remaining_cycles = int(remaining_time / profile.avg_latency)
                profile.estimated_remaining_cycles = max(0, remaining_cycles)
                needed = max(1, int(max(1, profile.open_todo_count) * 1.5))
                profile.pace_ratio = float(remaining_cycles) / max(1.0, float(needed))
                if profile.pace_ratio < 0.5 and remaining_cycles < 3:
                    advisories.append(
                        f"pace: ~{remaining_cycles} cycles left, ~{needed} needed. "
                        "Likely max_elapsed wall without simplification."
                    )
                profile.budget_viable = remaining_cycles >= 1
        return advisories

    def record(
        self,
        *,
        profile: RunProfile,
        hardware: HardwareProfile,
    ) -> HardwareProfile:
        hw = replace(hardware)
        alpha = float(hw.EMA_ALPHA)
        for latency in profile.call_latencies:
            if hw.latency_samples == 0:
                hw.latency_ema = float(latency)
            else:
                hw.latency_ema = alpha * float(latency) + (1.0 - alpha) * float(hw.latency_ema)
            hw.latency_samples += 1
            hw.latency_last = float(latency)
            hw.latency_max_seen = max(float(hw.latency_max_seen), float(latency))

        hw.truncation_total += len(profile.call_truncated)
        hw.truncation_hits += sum(1 for t in profile.call_truncated if t)

        mt = max(1, int(profile.current_max_tokens or 1))
        for chars in profile.call_output_chars:
            if int(chars) <= 0:
                continue
            ratio = float(chars) / float(mt)
            if hw.throughput_samples == 0:
                hw.chars_per_token_ema = ratio
            else:
                hw.chars_per_token_ema = alpha * ratio + (1.0 - alpha) * float(hw.chars_per_token_ema)
            hw.throughput_samples += 1
        return hw
