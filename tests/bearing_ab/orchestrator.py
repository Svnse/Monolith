"""Two-arm driver for the Bearing V0 A/B harness.

Loads the model ONCE per orchestrator process, then runs each fixture × 2
arms in-process per §6.A (locked 2026-05-21). For the full gate (27 fixtures
× 2 arms = 54 runs), model load is amortized across all runs.

CLI:
  python -m tests.bearing_ab.orchestrator --manifest tests/fixtures/bearing_ab/manifest.json
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from . import format as ab_format
from .manifest import (
    ManifestError,
    RuntimeFingerprint,
    build_fingerprint,
    load_run_input,
    write_fingerprint,
)
from .runner import run_arm


def _load_model(fingerprint: RuntimeFingerprint) -> Any:
    """Build the LLM client. Branches on fingerprint.backend.

    gguf     -> in-process llama_cpp.Llama against a local GGUF.
    gguf_api -> engine.llm.OpenAICompatLLM (HTTP, OpenAI-compatible endpoint).
    """
    print(
        f"[orchestrator] backend={fingerprint.backend} "
        f"model={fingerprint.model_name}",
        flush=True,
    )
    if fingerprint.backend == "gguf":
        try:
            from llama_cpp import Llama
        except ImportError as exc:
            raise RuntimeError(
                "backend='gguf' requires llama-cpp-python. Install it or "
                "switch backend to 'gguf_api'."
            ) from exc
        print(
            f"[orchestrator]   loading local GGUF: {fingerprint.model_identity}",
            flush=True,
        )
        print(
            f"[orchestrator]   family: {fingerprint.family_name} "
            f"(variant={fingerprint.variant_id or '--'}, "
            f"confidence={fingerprint.preset_confidence})",
            flush=True,
        )
        kwargs: dict[str, Any] = {
            "model_path": fingerprint.model_identity,
            "verbose": False,
        }
        if fingerprint.context_length:
            kwargs["n_ctx"] = int(fingerprint.context_length)
        if fingerprint.seed is not None:
            kwargs["seed"] = int(fingerprint.seed)
        return Llama(**kwargs)

    if fingerprint.backend in {"gguf_api", "openai"}:
        # engine/llm.py imports PySide6 at module level; pulling the class
        # in still works headlessly -- no Qt event loop required.
        from engine.llm import OpenAICompatLLM
        from core.config import get_config
        llm_cfg = get_config().llm
        print(
            f"[orchestrator]   endpoint: {fingerprint.api_base}",
            flush=True,
        )
        print(
            f"[orchestrator]   provider: {fingerprint.api_provider}, "
            f"model: {fingerprint.model_name}",
            flush=True,
        )
        return OpenAICompatLLM(
            base_url=str(llm_cfg.api_base),
            api_key=str(llm_cfg.api_key) or None,
            model=str(llm_cfg.api_model),
        )

    raise RuntimeError(
        f"unknown backend {fingerprint.backend!r}; "
        "expected 'gguf', 'gguf_api', or 'openai'"
    )


def run(manifest_path: Path) -> int:
    try:
        run_input = load_run_input(manifest_path)
    except ManifestError as exc:
        print(f"[orchestrator] manifest error: {exc}", file=sys.stderr)
        return 2

    try:
        fingerprint = build_fingerprint(run_input)
    except ManifestError as exc:
        print(f"[orchestrator] fingerprint error: {exc}", file=sys.stderr)
        return 2

    rdir = ab_format.run_dir(run_input.run_id)
    fp_path = write_fingerprint(fingerprint, rdir)
    print(f"[orchestrator] fingerprint written: {fp_path}", flush=True)
    print(f"[orchestrator]   preset_confidence={fingerprint.preset_confidence}", flush=True)
    if fingerprint.preset_confidence == "fallback":
        print(
            f"[orchestrator]   PROCEEDING ON FALLBACK PRESET -- rationale: "
            f"{run_input.accept_fallback_rationale or '(none provided)'}",
            flush=True,
        )

    model = _load_model(fingerprint)

    for fixture_id in run_input.fixture_ids:
        for arm in ("on", "off"):
            print(f"[orchestrator] running fixture={fixture_id} arm={arm}", flush=True)
            try:
                out_path = run_arm(fixture_id, arm, fingerprint, model)
            except Exception as exc:  # noqa: BLE001
                print(
                    f"[orchestrator] ERROR fixture={fixture_id} arm={arm}: "
                    f"{type(exc).__name__}: {exc}",
                    file=sys.stderr,
                )
                return 1
            print(f"[orchestrator]   -> {out_path}", flush=True)

    print(f"[orchestrator] done. artifacts at: {rdir}", flush=True)
    return 0


def _cli() -> int:
    parser = argparse.ArgumentParser(description="Bearing V0 A/B orchestrator")
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="Path to manifest.json (RunInput)",
    )
    args = parser.parse_args()
    return run(args.manifest)


if __name__ == "__main__":
    raise SystemExit(_cli())
