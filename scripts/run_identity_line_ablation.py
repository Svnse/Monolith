from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.identity_ablation import (
    compare_judgment_effect,
    find_identity_line_candidates,
    get_fixture,
    judge_false_premise_response,
    remove_prompt_line,
)
from core.llm_config import build_system_prompt, load_config
from core.llm_prompt import MASTER_PROMPT
from core.observed_state import format_observed_state_block
from engine.sync_bridge import generate_sync_from_config


DEFAULT_FIXTURES = ("direct_runtime_identity", "embedded_local_memory")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run line-by-line prompt attribution for identity ablation."
    )
    parser.add_argument(
        "--rescore-input",
        type=Path,
        help="Re-score an existing JSON artifact without running model calls.",
    )
    parser.add_argument("--live", action="store_true", help="Run live model calls.")
    parser.add_argument("--output", type=Path, help="Optional JSON output path.")
    parser.add_argument("--markdown", type=Path, help="Optional markdown table path.")
    parser.add_argument("--max-tokens", type=int, default=700)
    parser.add_argument("--temp", type=float, default=0.1)
    parser.add_argument(
        "--attempts",
        type=int,
        default=1,
        help="Live-call attempts per cell. Retries only empty responses.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="Live-call concurrency. Default 1 preserves sequential behavior.",
    )
    parser.add_argument(
        "--thinking",
        choices=("default", "on", "off"),
        default="off",
        help="Override enable_thinking for live calls.",
    )
    parser.add_argument(
        "--fixture",
        action="append",
        default=[],
        help="Fixture id to run. Default: direct_runtime_identity and embedded_local_memory.",
    )
    parser.add_argument(
        "--line",
        action="append",
        type=int,
        default=[],
        help="Only test this assembled-prompt line number. Repeat for multiple.",
    )
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=0,
        help="Limit candidate count after filtering. Default 0 means all.",
    )
    parser.add_argument(
        "--no-observed-state",
        action="store_true",
        help="Do not inject the observed-state block before the fixture prompt.",
    )
    args = parser.parse_args()

    if args.rescore_input:
        payload = _rescore_payload(args.rescore_input)
        markdown = _format_markdown(payload)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        if args.markdown:
            args.markdown.parent.mkdir(parents=True, exist_ok=True)
            args.markdown.write_text(markdown, encoding="utf-8")
        print(markdown)
        return 0

    prompt = build_system_prompt({"system_prompt": MASTER_PROMPT})
    candidates = list(find_identity_line_candidates(prompt))
    if args.line:
        requested = {int(line) for line in args.line}
        candidates = [candidate for candidate in candidates if candidate.line_number in requested]
    if args.max_candidates and args.max_candidates > 0:
        candidates = candidates[: int(args.max_candidates)]

    fixture_ids = tuple(args.fixture or DEFAULT_FIXTURES)
    fixtures = [get_fixture(fixture_id) for fixture_id in fixture_ids]

    base_config = load_config()
    observed_state_block = ""
    if not args.no_observed_state:
        observed_state_block = format_observed_state_block(base_config)
    live_config = {"max_tokens": max(1, int(args.max_tokens)), "temp": float(args.temp)}
    thinking_enabled = None
    if args.thinking == "on":
        thinking_enabled = True
    elif args.thinking == "off":
        thinking_enabled = False

    baseline: dict[str, dict[str, Any]] = {}
    candidate_rows: list[dict[str, Any]] = []

    concurrency = max(1, int(args.concurrency or 1))
    attempts_each = max(1, int(args.attempts))

    def _baseline_job(fixture):
        return fixture.fixture_id, _run_or_describe(
            live=bool(args.live),
            system_prompt=prompt,
            observed_state_block=observed_state_block,
            fixture_id=fixture.fixture_id,
            fixture_prompt=fixture.prompt,
            premise_kind=fixture.premise_kind,
            base_config=base_config,
            live_config=live_config,
            thinking_enabled=thinking_enabled,
            attempts=attempts_each,
        )

    if args.live and concurrency > 1 and len(fixtures) > 1:
        with ThreadPoolExecutor(max_workers=min(concurrency, len(fixtures))) as pool:
            for fid, result in pool.map(_baseline_job, fixtures):
                baseline[fid] = result
    else:
        for fixture in fixtures:
            fid, result = _baseline_job(fixture)
            baseline[fid] = result

    candidate_variants = [
        (c_idx, candidate, remove_prompt_line(prompt, candidate.line_number))
        for c_idx, candidate in enumerate(candidates)
    ]
    cell_jobs = [
        (c_idx, variant_prompt, fixture)
        for (c_idx, _, variant_prompt) in candidate_variants
        for fixture in fixtures
    ]

    def _cell_job(job):
        c_idx, variant_prompt, fixture = job
        return c_idx, fixture.fixture_id, _run_or_describe(
            live=bool(args.live),
            system_prompt=variant_prompt,
            observed_state_block=observed_state_block,
            fixture_id=fixture.fixture_id,
            fixture_prompt=fixture.prompt,
            premise_kind=fixture.premise_kind,
            base_config=base_config,
            live_config=live_config,
            thinking_enabled=thinking_enabled,
            attempts=attempts_each,
        )

    cell_results: dict[int, dict[str, dict[str, Any]]] = {
        c_idx: {} for c_idx in range(len(candidates))
    }
    if args.live and concurrency > 1 and len(cell_jobs) > 1:
        with ThreadPoolExecutor(max_workers=min(concurrency, len(cell_jobs))) as pool:
            for c_idx, fid, result in pool.map(_cell_job, cell_jobs):
                cell_results[c_idx][fid] = result
    else:
        for job in cell_jobs:
            c_idx, fid, result = _cell_job(job)
            cell_results[c_idx][fid] = result

    for c_idx, candidate, _ in candidate_variants:
        results = cell_results[c_idx]
        effects = {
            fixture.fixture_id: compare_judgment_effect(
                str(baseline[fixture.fixture_id].get("judgment") or "unknown"),
                str(results.get(fixture.fixture_id, {}).get("judgment") or "unknown"),
            )
            for fixture in fixtures
        }
        candidate_rows.append(
            {
                "line_number": candidate.line_number,
                "text": candidate.text,
                "reason": candidate.reason,
                "effects": effects,
                "results": results,
            }
        )

    payload = {
        "mode": "live" if args.live else "dry_run",
        "prompt_chars": len(prompt),
        "observed_state_chars": len(observed_state_block),
        "fixture_ids": [fixture.fixture_id for fixture in fixtures],
        "candidate_count": len(candidates),
        "baseline": baseline,
        "candidates": candidate_rows,
    }
    markdown = _format_markdown(payload)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    if args.markdown:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(markdown, encoding="utf-8")

    print(markdown)
    return 0


def _run_or_describe(
    *,
    live: bool,
    system_prompt: str,
    observed_state_block: str,
    fixture_id: str,
    fixture_prompt: str,
    premise_kind: str,
    base_config: dict[str, Any],
    live_config: dict[str, Any],
    thinking_enabled: bool | None,
    attempts: int,
) -> dict[str, Any]:
    if not live:
        return {"ok": None, "judgment": "not_run"}

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
        except Exception as exc:
            result = {
                "ok": False,
                "attempt": attempt,
                "error": f"{type(exc).__name__}: {exc}",
                "judgment": "error",
            }
            seen.append(result)
            return dict(result, attempts=seen)

        judgment = judge_false_premise_response(
            response,
            prompt_text=fixture_prompt,
            premise_kind=premise_kind,
        )
        result = {
            "ok": True,
            "attempt": attempt,
            "fixture_id": fixture_id,
            "judgment": judgment.label,
            "judgment_evidence": list(judgment.evidence),
            "response": response,
        }
        seen.append(result)
        if judgment.label != "empty":
            return dict(result, attempts=seen)
    final = dict(seen[-1])
    final["attempts"] = seen
    return final


def _rescore_payload(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    fixture_ids = list(payload.get("fixture_ids") or [])
    fixtures = {fixture_id: get_fixture(fixture_id) for fixture_id in fixture_ids}

    for fixture_id, result in dict(payload.get("baseline") or {}).items():
        fixture = fixtures.get(fixture_id)
        if fixture is None:
            continue
        judgment = judge_false_premise_response(
            str(result.get("response") or ""),
            prompt_text=fixture.prompt,
            premise_kind=fixture.premise_kind,
        )
        result["judgment"] = judgment.label
        result["judgment_evidence"] = list(judgment.evidence)

    for row in list(payload.get("candidates") or []):
        effects: dict[str, str] = {}
        for fixture_id, result in dict(row.get("results") or {}).items():
            fixture = fixtures.get(fixture_id)
            if fixture is None:
                continue
            judgment = judge_false_premise_response(
                str(result.get("response") or ""),
                prompt_text=fixture.prompt,
                premise_kind=fixture.premise_kind,
            )
            result["judgment"] = judgment.label
            result["judgment_evidence"] = list(judgment.evidence)
            effects[fixture_id] = compare_judgment_effect(
                str(payload["baseline"][fixture_id].get("judgment") or "unknown"),
                judgment.label,
            )
        row["effects"] = effects
    payload["mode"] = str(payload.get("mode") or "live") + "_rescored"
    return payload


def _format_markdown(payload: dict[str, Any]) -> str:
    fixture_ids = list(payload["fixture_ids"])
    lines = [
        "# Identity Line Ablation Attribution",
        "",
        f"- mode: {payload['mode']}",
        f"- candidates: {payload['candidate_count']}",
        f"- observed_state_chars: {payload['observed_state_chars']}",
        "",
        "## Baseline",
        "",
        "| fixture | judgment |",
        "| --- | --- |",
    ]
    for fixture_id in fixture_ids:
        judgment = str(payload["baseline"].get(fixture_id, {}).get("judgment", "unknown"))
        lines.append(f"| {_md(fixture_id)} | {_md(judgment)} |")

    header = "| line | prompt line | reason | " + " | ".join(_md(fixture_id) for fixture_id in fixture_ids) + " |"
    separator = "| --- | --- | --- | " + " | ".join("---" for _ in fixture_ids) + " |"
    lines.extend(["", "## Candidate Effects", "", header, separator])
    for row in payload["candidates"]:
        cells = [
            str(row["line_number"]),
            _md(_truncate(str(row["text"]), 110)),
            _md(str(row["reason"])),
        ]
        for fixture_id in fixture_ids:
            result = row["results"].get(fixture_id, {})
            judgment = str(result.get("judgment") or "unknown")
            effect = str(row["effects"].get(fixture_id) or "unknown")
            cells.append(_md(f"{effect} ({judgment})"))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _truncate(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: limit - 3].rstrip() + "..."


def _md(text: str) -> str:
    return str(text).replace("|", "\\|").replace("\n", " ")


if __name__ == "__main__":
    raise SystemExit(main())
