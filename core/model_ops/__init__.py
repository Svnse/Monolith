from __future__ import annotations

import ctypes
import json
import os
import re
import socket
import subprocess
import sys
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

from core.model_registry import metadata_context_window, read_gguf_metadata


@dataclass(frozen=True)
class GPUInfo:
    name: str
    total_vram_mb: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class HardwareSnapshot:
    os_name: str
    cpu_count: int
    ram_mb: int | None
    gpus: tuple[GPUInfo, ...] = ()
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["gpus"] = [gpu.to_dict() for gpu in self.gpus]
        return data


@dataclass(frozen=True)
class ServeProfile:
    model_path: str
    runtime: str
    host: str
    port: int
    ctx_size: int
    n_gpu_layers: int
    threads: int
    estimated_model_mb: int | None = None
    score: int = 0
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ServeDiagnosis:
    kind: str
    message: str
    suggested_actions: tuple[str, ...] = ()
    raw_excerpt: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EndpointProbeResult:
    base_url: str
    reachable: bool
    status: str
    models: tuple[str, ...] = ()
    context_hint: int | None = None
    provider_hint: str | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _ram_mb_windows() -> int | None:
    if os.name != "nt":
        return None

    class MEMORYSTATUSEX(ctypes.Structure):
        _fields_ = [
            ("dwLength", ctypes.c_ulong),
            ("dwMemoryLoad", ctypes.c_ulong),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

    stat = MEMORYSTATUSEX()
    stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
    try:
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
            return int(stat.ullTotalPhys // (1024 * 1024))
    except Exception:
        return None
    return None


def _ram_mb_posix() -> int | None:
    if os.name == "nt":
        return None
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        return int((pages * page_size) // (1024 * 1024))
    except Exception:
        return None


def _scan_nvidia_smi(run_cmd: Callable[..., subprocess.CompletedProcess] | None = None) -> tuple[GPUInfo, ...]:
    runner = run_cmd or subprocess.run
    try:
        result = runner(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
    except Exception:
        return ()
    if getattr(result, "returncode", 1) != 0:
        return ()
    gpus: list[GPUInfo] = []
    for line in str(getattr(result, "stdout", "") or "").splitlines():
        parts = [p.strip() for p in line.split(",", 1)]
        if not parts or not parts[0]:
            continue
        vram = None
        if len(parts) > 1:
            try:
                vram = int(float(parts[1]))
            except (TypeError, ValueError):
                vram = None
        gpus.append(GPUInfo(parts[0], vram))
    return tuple(gpus)


def scan_hardware(run_cmd: Callable[..., subprocess.CompletedProcess] | None = None) -> HardwareSnapshot:
    warnings: list[str] = []
    ram_mb = _ram_mb_windows() or _ram_mb_posix()
    if ram_mb is None:
        warnings.append("system RAM could not be detected")
    gpus = _scan_nvidia_smi(run_cmd)
    if not gpus:
        warnings.append("no NVIDIA GPU detected via nvidia-smi")
    return HardwareSnapshot(
        os_name=os.name,
        cpu_count=max(1, os.cpu_count() or 1),
        ram_mb=ram_mb,
        gpus=gpus,
        warnings=tuple(warnings),
    )


def _model_size_mb(path: str | Path | None) -> int | None:
    try:
        p = Path(str(path or ""))
        if p.is_file():
            return int(p.stat().st_size // (1024 * 1024))
    except Exception:
        return None
    return None


def _detect_quant_hint(path: str | Path | None) -> str:
    name = Path(str(path or "")).name.lower()
    match = re.search(r"\b(q\d(?:_[a-z0-9]+)?|f16|bf16|fp16|fp8|awq)\b", name)
    return match.group(1) if match else "unknown"


def _pick_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def compute_serve_profile(
    model_path: str | Path,
    *,
    hardware: HardwareSnapshot | None = None,
    requested_ctx: int | None = None,
    hard_cap: int | None = None,
    host: str = "127.0.0.1",
    port: int | None = None,
    runtime: str = "llama_server",
) -> ServeProfile:
    hw = hardware or scan_hardware()
    path_s = str(model_path)
    metadata_ctx = metadata_context_window(read_gguf_metadata(path_s))
    ctx = int(requested_ctx or metadata_ctx or 8192)
    if metadata_ctx:
        ctx = min(ctx, int(metadata_ctx))
    cap = int(hard_cap or os.getenv("MONOLITH_LOCAL_CTX_HARD_CAP", "32768") or 32768)
    ctx = max(1024, min(ctx, max(1024, cap)))

    model_mb = _model_size_mb(path_s)
    largest_vram = max((gpu.total_vram_mb or 0 for gpu in hw.gpus), default=0)
    warnings: list[str] = []
    score = 50
    if model_mb and largest_vram:
        # Leave about 1.5 GiB for runtime/KV overhead before full offload.
        if model_mb + 1536 <= largest_vram:
            n_gpu_layers = -1
            score += 35
        else:
            n_gpu_layers = 0
            score -= 10
            warnings.append("model is larger than conservative available VRAM; using CPU layers")
    elif largest_vram:
        n_gpu_layers = -1
        score += 15
        warnings.append("model size unknown; GPU offload may still fail")
    else:
        n_gpu_layers = 0
        warnings.append("no GPU VRAM detected; using CPU profile")

    if hw.ram_mb and model_mb and model_mb + 2048 > hw.ram_mb:
        score -= 25
        warnings.append("model may exceed conservative system RAM budget")

    quant = _detect_quant_hint(path_s)
    if quant in {"q2", "q3", "q4", "q4_k_m", "q5", "q5_k_m"}:
        score += 10
    elif quant in {"f16", "bf16", "fp16"}:
        warnings.append("high precision model detected; memory pressure likely")

    return ServeProfile(
        model_path=path_s,
        runtime=runtime,
        host=host,
        port=int(port or _pick_free_port()),
        ctx_size=ctx,
        n_gpu_layers=n_gpu_layers,
        threads=max(1, hw.cpu_count // 2),
        estimated_model_mb=model_mb,
        score=max(0, min(100, score)),
        warnings=tuple(warnings),
    )


def build_llama_server_command(
    profile: ServeProfile,
    *,
    server_path: str | None = None,
    python_exe: str | None = None,
) -> list[str]:
    if server_path:
        return [
            server_path,
            "--model",
            profile.model_path,
            "--host",
            profile.host,
            "--port",
            str(profile.port),
            "--ctx-size",
            str(profile.ctx_size),
            "--n-gpu-layers",
            str(profile.n_gpu_layers),
            "--threads",
            str(profile.threads),
        ]
    return [
        python_exe or sys.executable,
        "-m",
        "llama_cpp.server",
        "--model",
        profile.model_path,
        "--host",
        profile.host,
        "--port",
        str(profile.port),
        "--n_ctx",
        str(profile.ctx_size),
        "--n_gpu_layers",
        str(profile.n_gpu_layers),
        "--n_threads",
        str(profile.threads),
    ]


def rank_local_model_candidates(
    paths: list[str | Path],
    *,
    hardware: HardwareSnapshot | None = None,
    requested_ctx: int | None = None,
) -> list[ServeProfile]:
    hw = hardware or scan_hardware()
    profiles = [
        compute_serve_profile(path, hardware=hw, requested_ctx=requested_ctx)
        for path in paths
        if Path(str(path)).is_file()
    ]
    return sorted(profiles, key=lambda profile: (profile.score, -len(profile.warnings)), reverse=True)


_DIAG_PATTERNS: tuple[tuple[str, tuple[str, ...], str, tuple[str, ...]], ...] = (
    (
        "port_in_use",
        ("address already in use", "only one usage of each socket address", "bind: address"),
        "The requested local server port is already in use.",
        ("Pick a free port", "Stop the process holding that port", "Clear the local API base and retry"),
    ),
    (
        "context_too_high",
        ("n_ctx", "ctx-size", "context size", "kv cache", "kv self"),
        "The requested context appears too high for this model/runtime.",
        ("Lower the context profile", "Use the safe serve profile", "Try a smaller context cap"),
    ),
    (
        "out_of_memory",
        ("out of memory", "cuda error", "failed to allocate", "not enough memory", "bad allocation"),
        "The model server ran out of memory while loading or serving.",
        ("Use a smaller quant", "Reduce context size", "Use fewer GPU layers", "Close other GPU workloads"),
    ),
    (
        "missing_dependency",
        ("no module named llama_cpp", "modulenotfounderror", "llama-server is not recognized", "not recognized as an internal"),
        "The selected llama.cpp server runtime is missing.",
        ("Install llama-cpp-python", "Point MONOLITH_LLAMA_SERVER to llama-server", "Point MONOLITH_LLAMA_PY to the right Python"),
    ),
    (
        "bad_model_path",
        ("no such file", "cannot open model", "failed to load model", "does not exist", "invalid model path"),
        "The model file could not be opened.",
        ("Choose an existing GGUF file", "Check path permissions", "Avoid moved or deleted model paths"),
    ),
    (
        "no_gpu",
        ("no cuda", "cuda driver", "no gpu", "could not find cuda", "hip error"),
        "GPU acceleration is unavailable for this runtime.",
        ("Use a CPU profile", "Install/update GPU drivers", "Use fewer or zero GPU layers"),
    ),
    (
        "unreachable_endpoint",
        ("connection refused", "timed out", "urlerror", "failed to establish", "actively refused"),
        "The configured model endpoint is not reachable.",
        ("Start the server", "Check API base URL", "Probe the endpoint again"),
    ),
)


def diagnose_serve_output(text: str, *, max_excerpt: int = 700) -> ServeDiagnosis | None:
    haystack = str(text or "")
    lowered = haystack.lower()
    if not lowered.strip():
        return None
    for kind, needles, message, actions in _DIAG_PATTERNS:
        if any(needle in lowered for needle in needles):
            return ServeDiagnosis(
                kind=kind,
                message=message,
                suggested_actions=actions,
                raw_excerpt=haystack[-max_excerpt:].strip(),
            )
    return ServeDiagnosis(
        kind="unknown",
        message="The model server failed, but Restore could not classify the error yet.",
        suggested_actions=("Inspect the local server log", "Retry with a safer serve profile"),
        raw_excerpt=haystack[-max_excerpt:].strip(),
    )


def record_serve_fault(
    turn_id: str,
    diagnosis: ServeDiagnosis,
    *,
    profile: ServeProfile | None = None,
    evidence: str | None = None,
) -> None:
    from datetime import datetime, timezone

    from core import turn_trace

    payload: dict[str, Any] = {
        "evidence": evidence or diagnosis.raw_excerpt,
        "meta": {
            "diagnosis": diagnosis.to_dict(),
            "profile": profile.to_dict() if profile else None,
        },
    }
    record = turn_trace.FaultTraceRecord(
        turn_id=str(turn_id),
        parent_turn_id=None,
        seq=0,
        emitted_at=datetime.now(timezone.utc).isoformat(),
        event_kind="FaultDetectedEvent",
        source_kind="policy",
        source_name="model_ops",
        authority_tier="observation",
        fault_kind="model_serve_failed",
        severity="warn",
        payload=payload,
    )
    turn_trace.record_fault(record)


def _models_url(base_url: str) -> str:
    base = str(base_url or "").strip().rstrip("/")
    if base.endswith("/models"):
        return base
    if base.endswith("/v1"):
        return f"{base}/models"
    return f"{base}/v1/models"


def _context_hint_from_item(item: dict[str, Any]) -> int | None:
    meta = item.get("meta") if isinstance(item.get("meta"), dict) else {}
    candidates: list[int] = []
    for source in (meta, item):
        for key in ("n_ctx", "context_length", "max_context_length", "n_ctx_train"):
            try:
                value = int(source.get(key))
            except (TypeError, ValueError):
                continue
            if value > 0:
                candidates.append(value)
    return min(candidates) if candidates else None


def probe_endpoint(
    base_url: str,
    *,
    api_key: str = "",
    timeout: float = 8.0,
    opener: Callable[..., Any] | None = None,
) -> EndpointProbeResult:
    base = str(base_url or "").strip().rstrip("/")
    if not base:
        return EndpointProbeResult(base, False, "missing_base", error="Missing API base URL.")
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    req = urllib.request.Request(_models_url(base), headers=headers, method="GET")
    open_fn = opener or urllib.request.urlopen
    try:
        with open_fn(req, timeout=timeout) as resp:
            body = resp.read()
    except urllib.error.HTTPError as exc:
        return EndpointProbeResult(base, False, "auth_or_http_error", error=f"HTTP {exc.code}: {exc.reason}")
    except urllib.error.URLError as exc:
        return EndpointProbeResult(base, False, "unreachable", error=str(getattr(exc, "reason", exc)))
    except Exception as exc:
        return EndpointProbeResult(base, False, "error", error=str(exc))
    try:
        data = json.loads(body)
    except Exception as exc:
        return EndpointProbeResult(base, False, "bad_json", error=str(exc))
    items = data.get("data", []) if isinstance(data, dict) else []
    models: list[str] = []
    context_hint = None
    provider_hint = None
    for item in items:
        if isinstance(item, dict):
            if item.get("id"):
                models.append(str(item["id"]))
            if context_hint is None:
                context_hint = _context_hint_from_item(item)
            owned_by = item.get("owned_by")
            if provider_hint is None and isinstance(owned_by, str) and owned_by.strip():
                provider_hint = owned_by.strip()
        elif isinstance(item, str):
            models.append(item)
    return EndpointProbeResult(
        base,
        True,
        "ok",
        models=tuple(sorted({m for m in models if m})),
        context_hint=context_hint,
        provider_hint=provider_hint,
    )
