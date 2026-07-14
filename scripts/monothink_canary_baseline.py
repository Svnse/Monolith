"""MonoThink canary baseline collector (Phase 0c of the v2 evolution plan).

Captures a baseline fingerprint of (scaffold × fixture) → response under the
*current* monothink scaffold. Phase 3's background canary compares its own
post-apply runs against this baseline to detect drift.

Run ONCE manually before Phase 3 ships. Output lives at
``LOG_DIR/monothink_canary_baseline.jsonl`` (NOT under prompts/reasoning/ —
keeps the model-mutable directory clean; canary data is operational
telemetry, sharing storage with turn_trace.sqlite3).

Usage::

    python scripts/monothink_canary_baseline.py
    python scripts/monothink_canary_baseline.py --runs 1 --fixtures-glob "single_turn_*"

Per-fixture: takes the FIRST user turn from the fixture's ``turns`` array as
the prompt. Multi-turn fixtures are deliberately reduced to single-turn here
— the canary asks "does the scaffold shape outputs well in isolation",
not "does the model maintain conversational coherence" (that's Bearing's job,
which is why these fixtures already exist).

Predicates captured per run (defined here, refine later as data accumulates):
  * ``response_length`` — raw char count of the model's response.
  * ``parse_ok`` — response is non-empty after strip.
  * ``first_section_present`` — response opens with a markdown heading,
    numbered list, or bulleted list within the first 200 chars.
  * ``has_decision_token`` — response contains at least one of: therefore,
    thus, conclude, decision, primary, must, shall, because.
  * ``elapsed_sec`` — wall-clock seconds for the LLM call.

The point of the baseline is COMPARISON, not absolute thresholds. Phase 3's
canary computes the same predicates after an applied scaffold edit and
compares distributions against this file.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Make the project root importable when run as a plain script
# (python scripts/monothink_canary_baseline.py).
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from core.llm_config import load_config
from core.monothink import read_scaffold
from core.monothink_canary_predicates import (
    extract_first_section,
    has_decision_token,
)
from core.paths import LOG_DIR
from engine.sync_bridge import generate_sync_from_config


_FIXTURES_DIR = _REPO_ROOT / "tests" / "fixtures" / "bearing_ab"
_DEFAULT_OUTPUT = LOG_DIR / "monothink_canary_baseline.jsonl"

# Predicates live in core/monothink_canary_predicates.py — shared with the
# Phase 3d observer so baseline-vs-observer comparisons can't drift.

# Files inside the fixture dir that aren't actually fixtures.
_NON_FIXTURE_NAMES = {"manifest.json", "README.md", "rater_briefing.md", "manifest.md"}


def _extract_first_user_prompt(fixture: dict) -> str | None:
    """Return the first user turn's content. Multi-turn fixtures get
    reduced to single-turn for the canary — see module docstring."""
    turns = fixture.get("turns")
    if isinstance(turns, list):
        for turn in turns:
            if isinstance(turn, dict):
                user = turn.get("user") or turn.get("content")
                if isinstance(user, str) and user.strip():
                    return user.strip()
    # Some fixtures may stash the prompt at the top level.
    prompt = fixture.get("prompt") or fixture.get("input")
    if isinstance(prompt, str) and prompt.strip():
        return prompt.strip()
    return None


def _evaluate_response(text: str) -> dict:
    """Compute canary predicates over a model response.

    Booleans are preserved at this layer for back-compat with any existing
    baseline files. ``extract_first_section`` returns the actual matched
    line (richer signal) but we coerce to ``is not None`` so the JSONL
    schema stays the same as the pre-0d version of this script.
    """
    t = str(text or "")
    return {
        "response_length": len(t),
        "parse_ok": bool(t.strip()),
        "first_section_present": extract_first_section(t) is not None,
        "has_decision_token": has_decision_token(t),
    }


def _build_messages(scaffold: str, user_prompt: str) -> list[dict]:
    """Feed the bare scaffold as the system prompt + the fixture's prompt as
    the user message. Deliberately minimal — no interceptors, no other
    layers — so the canary measures *the scaffold's* effect, not the chat
    plumbing around it. The signal must come from the substrate under test."""
    return [
        {"role": "system", "content": scaffold},
        {"role": "user", "content": user_prompt},
    ]


def _run_one(scaffold: str, prompt: str, base_config: dict) -> dict:
    messages = _build_messages(scaffold, prompt)
    started = time.monotonic()
    try:
        text = generate_sync_from_config(
            base_config,
            messages,
            llm_config={"max_tokens": 8192, "temp": 0.3},
            thinking_enabled=False,
        )
    except Exception as exc:
        return {
            "ok": False,
            "error": f"{type(exc).__name__}: {str(exc)[:200]}",
            "elapsed_sec": round(time.monotonic() - started, 1),
            "response_length": 0,
            "parse_ok": False,
            "first_section_present": False,
            "has_decision_token": False,
        }
    elapsed = round(time.monotonic() - started, 1)
    return {"ok": True, "elapsed_sec": elapsed, **_evaluate_response(text or "")}


def _list_fixtures(fixtures_dir: Path, fixture_glob: str | None) -> list[Path]:
    pattern = fixture_glob or "*.json"
    out: list[Path] = []
    for path in sorted(fixtures_dir.glob(pattern)):
        if path.name in _NON_FIXTURE_NAMES:
            continue
        if path.name.endswith(".annotations.json"):
            continue
        out.append(path)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="MonoThink canary baseline collector")
    parser.add_argument(
        "--runs", type=int, default=3,
        help="Runs per fixture (default 3). Bigger N → tighter distribution, more LLM cost.",
    )
    parser.add_argument(
        "--fixtures-dir", type=Path, default=_FIXTURES_DIR,
        help=f"Fixture directory (default {_FIXTURES_DIR})",
    )
    parser.add_argument(
        "--fixtures-glob", type=str, default=None,
        help="Restrict to a glob pattern, e.g. 'single_turn_*' to skip multi-turn.",
    )
    parser.add_argument(
        "--output", type=Path, default=_DEFAULT_OUTPUT,
        help=f"Output JSONL path (default {_DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="List the work without making LLM calls. Useful for sanity checking before paying for 50+ calls.",
    )
    opts = parser.parse_args(argv)

    scaffold = read_scaffold()
    if not scaffold.strip():
        print("ERROR: monothink scaffold is empty", file=sys.stderr)
        return 1
    base_config = load_config()
    if not base_config.get("api_base") or not base_config.get("api_model"):
        print("ERROR: api_base or api_model not set in llm_config", file=sys.stderr)
        return 1

    fixtures = _list_fixtures(opts.fixtures_dir, opts.fixtures_glob)
    if not fixtures:
        print(f"ERROR: no fixtures matched in {opts.fixtures_dir}", file=sys.stderr)
        return 1

    total_runs = len(fixtures) * opts.runs
    print("== monothink canary baseline ==")
    print(f"scaffold_chars  = {len(scaffold)}")
    print(f"backend         = {base_config.get('api_base')} / {base_config.get('api_model')}")
    print(f"fixtures        = {len(fixtures)} (× {opts.runs} runs = {total_runs} total LLM calls)")
    print(f"output          = {opts.output}")
    if opts.dry_run:
        print("DRY RUN — listing fixtures only:")
        for f in fixtures:
            print(f"  {f.name}")
        return 0

    opts.output.parent.mkdir(parents=True, exist_ok=True)
    done = 0
    skipped = 0
    with opts.output.open("a", encoding="utf-8") as fh:
        for fixture_path in fixtures:
            try:
                data = json.loads(fixture_path.read_text(encoding="utf-8"))
            except Exception as exc:
                print(f"  skip {fixture_path.name}: parse error {exc}")
                skipped += 1
                continue
            prompt = _extract_first_user_prompt(data)
            if not prompt:
                print(f"  skip {fixture_path.name}: no user prompt found")
                skipped += 1
                continue
            for run_idx in range(opts.runs):
                done += 1
                print(f"  [{done}/{total_runs}] {fixture_path.name} run={run_idx}", flush=True)
                result = _run_one(scaffold, prompt, base_config)
                record = {
                    "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "fixture_id": data.get("id", fixture_path.stem),
                    "fixture_file": fixture_path.name,
                    "fixture_shape": data.get("shape", "unknown"),
                    "run_idx": run_idx,
                    "scaffold_chars": len(scaffold),
                    "api_model": base_config.get("api_model"),
                    **result,
                }
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
                fh.flush()

    print(f"== baseline complete: {done} runs written, {skipped} fixtures skipped ==")
    print(f"   {opts.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
