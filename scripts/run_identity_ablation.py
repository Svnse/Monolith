from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.identity import load_identity
from core.identity_ablation import (
    build_identity_strip_ablation_prompt,
    get_false_premise_fixtures,
    judge_false_premise_response,
)
from core.llm_config import build_system_prompt, load_config
from core.llm_prompt import MASTER_PROMPT
from core.observed_state import format_observed_state_block
from engine.sync_bridge import generate_sync_from_config


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the Monolith identity-strip false-premise ablation."
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Call the configured OpenAI-compatible backend. Default is dry-run.",
    )
    parser.add_argument("--output", type=Path, help="Optional JSON output path.")
    parser.add_argument("--max-tokens", type=int, default=360)
    parser.add_argument("--temp", type=float, default=0.2)
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="Live-call concurrency. Default 1 preserves sequential behavior.",
    )
    parser.add_argument(
        "--attempts",
        type=int,
        default=1,
        help="Live-call attempts per cell. Retries only empty responses.",
    )
    parser.add_argument(
        "--fixture",
        action="append",
        default=[],
        help="Fixture id to run. Repeat for multiple. Default runs all fixtures.",
    )
    parser.add_argument(
        "--variant",
        choices=("both", "baseline", "identity_stripped"),
        default="both",
        help="Variant subset to run.",
    )
    parser.add_argument(
        "--no-observed-state",
        action="store_true",
        help="Do not inject the observed-state block before the fixture user prompt.",
    )
    parser.add_argument(
        "--thinking",
        choices=("default", "on", "off"),
        default="default",
        help="Override enable_thinking for live calls.",
    )
    args = parser.parse_args()

    identity_text = load_identity()
    baseline_prompt = build_system_prompt({"system_prompt": MASTER_PROMPT})
    stripped_prompt = build_identity_strip_ablation_prompt(
        master_prompt=baseline_prompt,
        identity_text=identity_text,
    )

    base_config = load_config()
    observed_state_block = ""
    if not args.no_observed_state:
        observed_state_block = format_observed_state_block(base_config)
    live_config = {
        "max_tokens": max(1, int(args.max_tokens)),
        "temp": float(args.temp),
    }
    thinking_enabled = None
    if args.thinking == "on":
        thinking_enabled = True
    elif args.thinking == "off":
        thinking_enabled = False

    rows: list[dict[str, Any]] = []
    live_jobs: list[tuple[int, dict[str, Any], str, str, str]] = []
    requested_fixtures = {str(item).strip() for item in args.fixture if str(item).strip()}
    for fixture in get_false_premise_fixtures():
        if requested_fixtures and fixture.fixture_id not in requested_fixtures:
            continue
        variants = (
            ("baseline", baseline_prompt),
            ("identity_stripped", stripped_prompt),
        )
        for variant, system_prompt in variants:
            if args.variant != "both" and variant != args.variant:
                continue
            row: dict[str, Any] = {
                "fixture_id": fixture.fixture_id,
                "premise_kind": fixture.premise_kind,
                "variant": variant,
                "expected_behavior": fixture.expected_behavior,
                "prompt": fixture.prompt,
                "system_prompt_chars": len(system_prompt),
                "observed_state_chars": len(observed_state_block),
            }
            if args.live:
                live_jobs.append(
                    (
                        len(rows),
                        row,
                        system_prompt,
                        fixture.prompt,
                        fixture.premise_kind,
                    )
                )
            else:
                row.update({"ok": None, "judgment": "not_run"})
            rows.append(row)

    if args.live and live_jobs:
        concurrency = max(1, min(int(args.concurrency or 1), len(live_jobs)))
        if concurrency == 1:
            for index, row, system_prompt, fixture_prompt, premise_kind in live_jobs:
                rows[index] = _run_live_row(
                    row=row,
                    base_config=base_config,
                    live_config=live_config,
                    system_prompt=system_prompt,
                    observed_state_block=observed_state_block,
                    fixture_prompt=fixture_prompt,
                        premise_kind=premise_kind,
                        thinking_enabled=thinking_enabled,
                        attempts=max(1, int(args.attempts)),
                    )
        else:
            with ThreadPoolExecutor(max_workers=concurrency) as pool:
                futures = {
                    pool.submit(
                        _run_live_row,
                        row=row,
                        base_config=base_config,
                        live_config=live_config,
                        system_prompt=system_prompt,
                        observed_state_block=observed_state_block,
                        fixture_prompt=fixture_prompt,
                        premise_kind=premise_kind,
                        thinking_enabled=thinking_enabled,
                        attempts=max(1, int(args.attempts)),
                    ): index
                    for index, row, system_prompt, fixture_prompt, premise_kind in live_jobs
                }
                for future in as_completed(futures):
                    rows[futures[future]] = future.result()

    payload = {
        "mode": "live" if args.live else "dry_run",
        "baseline_prompt_chars": len(baseline_prompt),
        "identity_stripped_prompt_chars": len(stripped_prompt),
        "observed_state_chars": len(observed_state_block),
        "rows": rows,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print(_format_matrix(payload))
    return 0


def _run_live_row(
    *,
    row: dict[str, Any],
    base_config: dict[str, Any],
    live_config: dict[str, Any],
    system_prompt: str,
    observed_state_block: str,
    fixture_prompt: str,
    premise_kind: str,
    thinking_enabled: bool | None,
    attempts: int,
) -> dict[str, Any]:
    result = dict(row)
    messages = [{"role": "system", "content": system_prompt}]
    if observed_state_block:
        messages.append({"role": "user", "content": observed_state_block})
    messages.append({"role": "user", "content": fixture_prompt})
    seen: list[dict[str, Any]] = []
    for attempt in range(1, max(1, int(attempts)) + 1):
        try:
            response = generate_sync_from_config(
                base_config,
                messages,
                llm_config=live_config,
                thinking_enabled=thinking_enabled,
            )
            judgment = judge_false_premise_response(
                response,
                prompt_text=fixture_prompt,
                premise_kind=premise_kind,
            )
            attempt_result = {
                "ok": True,
                "attempt": attempt,
                "response": response,
                "judgment": judgment.label,
                "judgment_evidence": list(judgment.evidence),
            }
        except Exception as exc:
            attempt_result = {
                "ok": False,
                "attempt": attempt,
                "error": f"{type(exc).__name__}: {exc}",
                "judgment": "error",
            }
            seen.append(attempt_result)
            result.update(attempt_result)
            result["attempts"] = seen
            return result
        seen.append(attempt_result)
        if attempt_result["judgment"] != "empty":
            result.update(attempt_result)
            result["attempts"] = seen
            return result
    result.update(seen[-1])
    result["attempts"] = seen
    return result


def _format_matrix(payload: dict[str, Any]) -> str:
    lines = [
        f"mode: {payload['mode']}",
        f"baseline_prompt_chars: {payload['baseline_prompt_chars']}",
        f"identity_stripped_prompt_chars: {payload['identity_stripped_prompt_chars']}",
        f"observed_state_chars: {payload.get('observed_state_chars', 0)}",
        "",
        "| fixture | kind | baseline | stripped |",
        "| --- | --- | --- | --- |",
    ]
    grouped: dict[str, dict[str, str]] = {}
    kinds: dict[str, str] = {}
    for row in payload["rows"]:
        fixture_id = str(row["fixture_id"])
        grouped.setdefault(fixture_id, {})[str(row["variant"])] = str(
            row.get("judgment") or row.get("error") or "unknown"
        )
        kinds[fixture_id] = str(row["premise_kind"])
    for fixture_id, variants in grouped.items():
        lines.append(
            "| {fixture} | {kind} | {baseline} | {stripped} |".format(
                fixture=fixture_id,
                kind=kinds.get(fixture_id, ""),
                baseline=variants.get("baseline", ""),
                stripped=variants.get("identity_stripped", ""),
            )
        )
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
