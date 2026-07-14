"""Acatalepsy bootstrap — schema migrate + flag-gated workers.

Called from Monolith's main bootstrap.py at startup. Responsibilities:

1. **Always**: run the (idempotent) schema migration.
2. **Flag-gated**: start the **AuditorWorker** (``MONOLITH_ACATALEPSY_AUDITOR_V1``)
   — extracts candidate claims from the canonical_log.
3. **Flag-gated**: start the **VerifierWorker** (``MONOLITH_ACATALEPSY_VERIFIER_V1``)
   — periodically grounds un-verified world-fact/causal ACUs via Tavily. Requires
   a Tavily key (env ``TAVILY_API_KEY`` or ``CONFIG_DIR/tavily.json``).

Both worker starts are independent and degrade gracefully — a missing flag, key,
or model never takes down Monolith startup.

Public API:
  - bootstrap_acatalepsy() -> BootstrapResult
  - shutdown_acatalepsy() -> None  (graceful worker stop)
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

from core.acatalepsy import runtime as _runtime


__all__ = (
    "AUDITOR_FLAG_ENV",
    "VERIFIER_FLAG_ENV",
    "BootstrapResult",
    "bootstrap_acatalepsy",
    "shutdown_acatalepsy",
)


AUDITOR_FLAG_ENV: str = "MONOLITH_ACATALEPSY_AUDITOR_V1"
VERIFIER_FLAG_ENV: str = "MONOLITH_ACATALEPSY_VERIFIER_V1"

# The verifier worker isn't tracked by runtime (which holds the single auditor);
# keep a module-level handle for graceful shutdown.
_verifier_worker = None


@dataclass
class BootstrapResult:
    schema_ok: bool = False
    schema_summary: dict[str, Any] = field(default_factory=dict)
    worker_started: bool = False
    worker_skip_reason: str | None = None
    verifier_started: bool = False
    verifier_skip_reason: str | None = None
    error: str | None = None


def _flag_enabled(name: str) -> bool:
    return os.environ.get(name, "0").strip().lower() in {"1", "true", "yes", "on"}


def bootstrap_acatalepsy() -> BootstrapResult:
    """Migrate schema (always) + start the auditor and verifier workers (behind
    independent flags). Failures are reported in the result, not raised."""
    result = BootstrapResult()

    try:
        from core.acatalepsy.schema import migrate
        result.schema_summary = migrate()
        result.schema_ok = True
    except Exception as exc:
        result.error = f"schema_migrate:{type(exc).__name__}:{exc}"
        return result

    _start_auditor(result)
    _start_verifier(result)
    return result


def _start_auditor(result: BootstrapResult) -> None:
    if not _flag_enabled(AUDITOR_FLAG_ENV):
        result.worker_skip_reason = f"flag {AUDITOR_FLAG_ENV} not set"
        return
    try:
        from core.acatalepsy.llm_sidecar import (
            make_auditor_llm, SidecarUnsupportedBackend, SidecarConfigError,
        )
        llm = make_auditor_llm()
    except (SidecarUnsupportedBackend, SidecarConfigError) as exc:
        result.worker_skip_reason = f"sidecar:{type(exc).__name__}:{exc}"
        return
    except Exception as exc:
        result.worker_skip_reason = f"sidecar_init:{type(exc).__name__}:{exc}"
        return
    try:
        from core.acatalepsy.triggers import AuditorWorker
        worker = AuditorWorker(llm=llm, source="auditor_monolith")
        worker.start()
        _runtime.register_worker(worker)
        result.worker_started = True
    except Exception as exc:
        result.worker_skip_reason = f"worker_start:{type(exc).__name__}:{exc}"


def _start_verifier(result: BootstrapResult) -> None:
    global _verifier_worker
    if not _flag_enabled(VERIFIER_FLAG_ENV):
        result.verifier_skip_reason = f"flag {VERIFIER_FLAG_ENV} not set"
        return
    try:
        from core.acatalepsy.grounding import get_api_key
        if not get_api_key():
            result.verifier_skip_reason = (
                "no Tavily key (env TAVILY_API_KEY or CONFIG_DIR/tavily.json)"
            )
            return
        from core.acatalepsy.verifier import VerifierWorker
        worker = VerifierWorker()  # defaults: tavily search + LLM judge
        worker.start()
        _verifier_worker = worker
        result.verifier_started = True
    except Exception as exc:
        result.verifier_skip_reason = f"verifier_start:{type(exc).__name__}:{exc}"


def shutdown_acatalepsy(timeout: float | None = 5.0) -> None:
    """Graceful shutdown — stop both workers if running. Idempotent."""
    global _verifier_worker
    worker = _runtime.get_active_worker()
    if worker is not None:
        try:
            worker.stop(timeout=timeout)
        except Exception:
            pass
        _runtime.deregister_worker(worker)
    if _verifier_worker is not None:
        try:
            _verifier_worker.stop(timeout=timeout)
        except Exception:
            pass
        _verifier_worker = None
