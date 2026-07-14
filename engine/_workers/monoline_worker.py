"""Child-process worker for MonolineEngine (Kernel Contract v2 §9 / D10).

Runs Monoline flows inside the spawned engine process and streams contract-shaped event dicts
back over the IPC queue; only dicts cross the boundary, so Monoline's core/ui stay isolated in
this process (the collision is contained here, never in the main app). Import-safe in the
parent: every Monoline/Monolith import is lazy, inside _run_monoline.
See docs/reports/GENESIS_CARD_BUILD_LOG.md.
"""
from __future__ import annotations


def run_flow(config: dict, emit) -> None:
    """Run ONE flow, emitting contract events:
        status:running -> result -> status:ready    (success)
        status:running -> error  -> status:error    (failure)
    `emit` puts a plain dict on the from-worker queue (the kernel maps these to sig_status/etc.).
    """
    emit({"event": "status", "status": "running"})
    try:
        result = _run_monoline(config)
    except Exception as exc:
        emit({"event": "error", "message": str(exc)})
        emit({"event": "status", "status": "error"})
        return
    emit({"event": "result",
          "output": str(result.get("output", "") or ""),
          "error": str(result.get("error", "") or "")})
    emit({"event": "status", "status": "ready"})


def main(to_worker, from_worker) -> None:
    """Op-loop, runs entirely in the child process. Reads {"op": ...} dicts; emits events via
    from_worker.put. Ends on shutdown OR queue death (parent gone) — never hangs or propagates."""
    while True:
        try:
            op = to_worker.get()
        except (EOFError, OSError, ValueError):
            break  # queue closed / parent gone
        kind = str((op or {}).get("op", ""))
        if kind == "shutdown":
            break
        if kind == "generate":
            run_flow((op or {}).get("config") or {}, from_worker.put)
        # "stop"/"load"/"unload": flow runs are short; no-op for the V0 slice


def _run_monoline(config: dict) -> dict:
    """Run a Monoline flow in THIS worker process; return {"output", "error"}.

    Lazy Monoline/Monolith imports keep this module import-safe in the parent. NOT YET
    IMPLEMENTED — the real child-side run (path setup + headless run_workflow + engine_func) and
    its integration gate are slice increment 2; unit tests patch this. Fail-loud until then.
    """
    raise NotImplementedError(
        "MonolineEngine child-side flow run not implemented yet "
        "(slice increment 2 + integration gate)")
