"""Manifest load + config-derived fingerprint for the A/B harness.

Two layers:
  RunInput        — human-edited JSON (~8 fields); minimal surface E fills before a run.
  RuntimeFingerprint — derived + written by runner (~15 fields); the canonical
                      record of "what ran" for post-hoc analysis. Pinned to disk
                      BEFORE any model invocation so partial runs are reconstructable.

The runner reads the actual model identity from `core.config.LLMConfig`
(matching how chat.py / engine/llm.py work). Two backends supported:

  backend="gguf"     → in-process llama_cpp.Llama against a local GGUF.
                       Fingerprint includes GGUF metadata + sha256 + registry
                       preset (with §6.E preset_confidence halt rule).
  backend="gguf_api" → HTTP to an OpenAI-compatible endpoint (local llama.cpp
                       server, LM Studio, DeepSeek, OpenAI, etc). Fingerprint
                       captures api_base + api_model + sampler from config.
                       preset_confidence rule does not apply — sampler comes
                       from config, which is the operator's tuned choice.

Per spec §8.2 + plan §5. Registry adoption (E informed 2026-05-21) and config-
adoption (E informed 2026-05-21) collapse the human-fillable surface to run-
identity only (run_id, raters, fixtures, seed). Backend and model identity
are taken from `core.config.get_config().llm`.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from core import model_registry
from core.config import get_config, LLMConfig


class ManifestError(Exception):
    """Raised when manifest is malformed, incomplete, or references missing files."""


# ── input layer (human-edited) ──────────────────────────────────────


@dataclass(frozen=True)
class RunInput:
    """Human-edited manifest contents. Loaded from JSON.

    Required: run_id, run_kind.
    Optional: seed, rater_a, rater_b, adjudicator, briefing_path, fixture_ids,
              accept_fallback_preset (gguf-mode override only).

    Model identity is NOT in RunInput — taken from `core.config.LLMConfig`
    at fingerprint-build time. This matches how the rest of Monolith reads
    the active model (chat.py, engine/llm.py).
    """
    run_id: str
    run_kind: str                 # "dryrun" | "gate" | "rerun"
    seed: int | None = None       # None = non-deterministic
    rater_a: str = ""
    rater_b: str = ""
    adjudicator: str = ""
    briefing_path: str = ""
    fixture_ids: tuple[str, ...] = ("multi_turn_design_01",)
    # Override flag — set True only if proceeding on fallback preset_confidence
    # with documented rationale (per spec §10 step 5; gguf mode only).
    accept_fallback_preset: bool = False
    accept_fallback_rationale: str = ""


_VALID_RUN_KINDS = frozenset({"dryrun", "gate", "rerun"})


def load_run_input(path: Path) -> RunInput:
    if not path.exists():
        raise ManifestError(f"manifest not found: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ManifestError(f"manifest is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ManifestError("manifest top-level must be an object")
    for key in ("run_id", "run_kind"):
        if not isinstance(data.get(key), str) or not data[key].strip():
            raise ManifestError(f"manifest missing required field: {key}")
    if data["run_kind"] not in _VALID_RUN_KINDS:
        raise ManifestError(
            f"run_kind {data['run_kind']!r} not in {sorted(_VALID_RUN_KINDS)}"
        )
    fixture_ids = data.get("fixture_ids")
    if fixture_ids is None:
        fixture_ids_tuple: tuple[str, ...] = ("multi_turn_design_01",)
    elif isinstance(fixture_ids, list) and all(isinstance(x, str) for x in fixture_ids):
        fixture_ids_tuple = tuple(fixture_ids)
    else:
        raise ManifestError("fixture_ids must be a list of strings or omitted")
    seed = data.get("seed")
    if seed is not None and not isinstance(seed, int):
        raise ManifestError("seed must be int or null")
    return RunInput(
        run_id=data["run_id"],
        run_kind=data["run_kind"],
        seed=seed,
        rater_a=str(data.get("rater_a", "")),
        rater_b=str(data.get("rater_b", "")),
        adjudicator=str(data.get("adjudicator", "")),
        briefing_path=str(data.get("briefing_path", "")),
        fixture_ids=fixture_ids_tuple,
        accept_fallback_preset=bool(data.get("accept_fallback_preset", False)),
        accept_fallback_rationale=str(data.get("accept_fallback_rationale", "")),
    )


# ── derived layer (runner-written) ──────────────────────────────────


@dataclass(frozen=True)
class RuntimeFingerprint:
    # Backend identity
    backend: str                  # "gguf" | "gguf_api"
    model_name: str
    model_identity: str           # gguf: file path; gguf_api: api_base + api_model
    model_sha256: str             # gguf: file sha256; gguf_api: "n/a"
    gguf_architecture: str        # gguf: from metadata; gguf_api: ""
    context_length: int | None
    # Registry resolution (gguf only; cloud sets these to descriptive defaults)
    family_id: str
    family_name: str
    variant_id: str | None
    preset_confidence: str        # "fallback"|"low"|"medium"|"high"|"config-pinned"
    quantization: str             # gguf: from filename; gguf_api: "n/a"
    sampler: dict[str, Any] = field(default_factory=dict)
    # Cloud-specific (empty for local)
    api_provider: str = ""
    api_base: str = ""
    # Derived from git
    system_prompt_sha: str = ""
    bearing_sha: str = ""
    # From RunInput
    seed: int | None = None
    run_id: str = ""
    run_kind: str = ""
    run_started_utc: str = ""
    # Pass-through metadata
    accept_fallback_preset: bool = False
    accept_fallback_rationale: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _sha256_of_file(path: Path) -> str:
    """Streaming sha256. Large GGUFs are multi-GB; don't slurp."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


_QUANT_TOKENS = ("Q8_0", "Q6_K", "Q5_K_M", "Q5_K_S", "Q5_0", "Q5_1",
                 "Q4_K_M", "Q4_K_S", "Q4_0", "Q4_1", "Q3_K_L", "Q3_K_M",
                 "Q3_K_S", "Q3_K", "Q2_K", "IQ4_XS", "IQ4_NL", "IQ3_M",
                 "IQ3_S", "IQ3_XS", "IQ2_M", "IQ2_S", "IQ2_XS", "F16", "F32", "BF16")


def _quantization_from_filename(name: str) -> str:
    upper = name.upper()
    for token in _QUANT_TOKENS:
        if token in upper:
            return token
    return "unknown"


def _git_sha_of_path(repo_root: Path, rel_path: str) -> str:
    """Return short git sha of last commit touching rel_path; empty on any failure."""
    try:
        out = subprocess.run(
            ["git", "log", "-1", "--format=%h", "--", rel_path],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            timeout=10,
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except Exception:
        pass
    return ""


def _build_local_fingerprint(
    run_input: RunInput, llm_cfg: LLMConfig
) -> RuntimeFingerprint:
    """Build fingerprint for backend='gguf' (in-process llama_cpp)."""
    if not llm_cfg.gguf_path:
        raise ManifestError(
            "backend='gguf' but config.llm.gguf_path is empty. "
            "Set the GGUF path in config or switch backend to 'gguf_api'."
        )
    model_path = Path(str(llm_cfg.gguf_path)).expanduser()
    if not model_path.exists() or not model_path.is_file():
        raise ManifestError(
            f"config.llm.gguf_path does not exist or is not a file: {model_path}"
        )

    metadata = model_registry.read_gguf_metadata(str(model_path))
    if not metadata:
        raise ManifestError(
            f"GGUF metadata unreadable at {model_path} — "
            "file may be corrupted or not a GGUF"
        )

    preset = model_registry.resolve_model_preset(
        str(model_path), metadata=metadata
    )

    if preset.preset_confidence == "fallback" and not run_input.accept_fallback_preset:
        raise ManifestError(
            "preset_confidence resolved to 'fallback' for this model. "
            "Per spec §10 step 5 (locked 2026-05-21), do NOT burn gate inference "
            "on a fallback sampler. Either: (a) find/write a tuned preset and "
            "re-resolve, or (b) set manifest `accept_fallback_preset=true` "
            "with `accept_fallback_rationale` documenting why. "
            f"Model: {model_path.name}"
        )

    model_sha256 = _sha256_of_file(model_path)
    project_root = Path(__file__).resolve().parents[2]
    system_prompt_sha = _git_sha_of_path(project_root, "prompts/system.md")
    bearing_sha = _git_sha_of_path(project_root, "addons/system/bearing")

    sampler = {
        "temperature": preset.sampler.get("temperature", 0.7),
        "top_p": preset.sampler.get("top_p", 0.95),
        "top_k": preset.sampler.get("top_k"),
        "min_p": preset.sampler.get("min_p"),
        "repeat_penalty": preset.sampler.get("repeat_penalty"),
    }

    return RuntimeFingerprint(
        backend="gguf",
        model_name=str(model_registry.metadata_model_name(metadata) or model_path.stem),
        model_identity=str(model_path),
        model_sha256=model_sha256,
        gguf_architecture=str(metadata.get("general.architecture") or ""),
        context_length=preset.context_window,
        family_id=preset.family_id,
        family_name=preset.family_name,
        variant_id=preset.variant_id,
        preset_confidence=preset.preset_confidence,
        quantization=_quantization_from_filename(model_path.name),
        sampler=sampler,
        system_prompt_sha=system_prompt_sha,
        bearing_sha=bearing_sha,
        seed=run_input.seed,
        run_id=run_input.run_id,
        run_kind=run_input.run_kind,
        run_started_utc=datetime.now(timezone.utc).isoformat(),
        accept_fallback_preset=run_input.accept_fallback_preset,
        accept_fallback_rationale=run_input.accept_fallback_rationale,
    )


def _build_cloud_fingerprint(
    run_input: RunInput, llm_cfg: LLMConfig
) -> RuntimeFingerprint:
    """Build fingerprint for backend='gguf_api' (OpenAI-compatible HTTP)."""
    if not llm_cfg.api_model:
        raise ManifestError(
            "backend='gguf_api' but config.llm.api_model is empty. "
            "Set the model name (e.g. 'deepseek-chat') in config."
        )
    if not llm_cfg.api_base:
        raise ManifestError(
            "backend='gguf_api' but config.llm.api_base is empty. "
            "Set the endpoint URL in config."
        )
    project_root = Path(__file__).resolve().parents[2]
    system_prompt_sha = _git_sha_of_path(project_root, "prompts/system.md")
    bearing_sha = _git_sha_of_path(project_root, "addons/system/bearing")

    # Sampler comes from config (operator-pinned for cloud mode; the §6.E
    # preset_confidence halt rule applies to GGUF resolution only).
    sampler: dict[str, Any] = {
        "temperature": llm_cfg.temp,
        "top_p": llm_cfg.top_p,
        "top_k": llm_cfg.top_k if llm_cfg.top_k else None,
        "min_p": llm_cfg.min_p if llm_cfg.min_p > 0 else None,
        "repeat_penalty": (
            llm_cfg.repetition_penalty if llm_cfg.repetition_penalty != 1.0 else None
        ),
    }

    return RuntimeFingerprint(
        backend="gguf_api",
        model_name=str(llm_cfg.api_model),
        model_identity=f"{llm_cfg.api_base}::{llm_cfg.api_model}",
        model_sha256="n/a",
        gguf_architecture="",
        context_length=int(llm_cfg.ctx_limit) if llm_cfg.ctx_limit else None,
        family_id="n/a",
        family_name=f"{llm_cfg.api_provider}:{llm_cfg.api_model}",
        variant_id=None,
        preset_confidence="config-pinned",
        quantization="n/a",
        sampler=sampler,
        api_provider=str(llm_cfg.api_provider),
        api_base=str(llm_cfg.api_base),
        system_prompt_sha=system_prompt_sha,
        bearing_sha=bearing_sha,
        seed=run_input.seed,
        run_id=run_input.run_id,
        run_kind=run_input.run_kind,
        run_started_utc=datetime.now(timezone.utc).isoformat(),
        accept_fallback_preset=run_input.accept_fallback_preset,
        accept_fallback_rationale=run_input.accept_fallback_rationale,
    )


_LOCAL_BACKENDS = frozenset({"gguf"})
_CLOUD_BACKENDS = frozenset({"gguf_api", "openai"})


def build_fingerprint(run_input: RunInput) -> RuntimeFingerprint:
    """Derive RuntimeFingerprint from RunInput + the live config.

    Branches on config.llm.backend:
      gguf                 → in-process llama_cpp.Llama (local GGUF).
      gguf_api | openai    → engine.llm.OpenAICompatLLM (HTTP, OpenAI-compatible).
                             Same HTTP path; different api_base targets
                             (local llama.cpp server vs cloud provider).
    Other backend values (vllm, sglang, ...) are documented in
    docs/specs/turn_trace_spec_v1.md:167 but unsupported here; runner errors
    cleanly rather than silently routing through the wrong adapter.
    """
    llm_cfg = get_config().llm
    backend = str(llm_cfg.backend or "gguf_api").lower()
    if backend in _LOCAL_BACKENDS:
        return _build_local_fingerprint(run_input, llm_cfg)
    if backend in _CLOUD_BACKENDS:
        fp = _build_cloud_fingerprint(run_input, llm_cfg)
        # Preserve the actual backend value (gguf_api vs openai) for
        # post-hoc analysis. _build_cloud_fingerprint stamps "gguf_api"
        # generically; overwrite with the actual config value.
        return RuntimeFingerprint(**{**fp.to_dict(), "backend": backend})
    raise ManifestError(
        f"backend {backend!r} not supported by the A/B harness. "
        f"Supported: {sorted(_LOCAL_BACKENDS | _CLOUD_BACKENDS)}. "
        "If you need {vllm, sglang}, the OpenAI-compatible HTTP adapter "
        "may work; switch config to 'openai' and point api_base accordingly."
    )


def write_fingerprint(fingerprint: RuntimeFingerprint, run_dir: Path) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "fingerprint.json"
    path.write_text(json.dumps(fingerprint.to_dict(), indent=2), encoding="utf-8")
    return path
