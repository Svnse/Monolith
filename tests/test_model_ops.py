from __future__ import annotations

import io
import urllib.error
from pathlib import Path

from core import turn_trace
from core.model_ops import (
    GPUInfo,
    HardwareSnapshot,
    build_llama_server_command,
    compute_serve_profile,
    diagnose_serve_output,
    probe_endpoint,
    record_serve_fault,
)


def test_diagnose_serve_output_classifies_context_pressure() -> None:
    diagnosis = diagnose_serve_output("llama.cpp error: KV cache failed; n_ctx is too large")
    assert diagnosis is not None
    assert diagnosis.kind == "context_too_high"
    assert "context" in diagnosis.message.lower()


def test_compute_serve_profile_clamps_context_and_scores_gpu_fit(tmp_path: Path) -> None:
    model = tmp_path / "tiny-q4_k_m.gguf"
    model.write_bytes(b"x" * 1024 * 1024)
    hardware = HardwareSnapshot("nt", cpu_count=12, ram_mb=32768, gpus=(GPUInfo("RTX", 8192),))

    profile = compute_serve_profile(model, hardware=hardware, requested_ctx=999999, hard_cap=4096, port=1234)

    assert profile.ctx_size == 4096
    assert profile.n_gpu_layers == -1
    assert profile.threads == 6
    assert profile.score > 50


def test_build_llama_server_command_supports_native_and_python(tmp_path: Path) -> None:
    model = tmp_path / "m.gguf"
    model.write_text("x", encoding="utf-8")
    profile = compute_serve_profile(model, hardware=HardwareSnapshot("nt", 4, None), port=8080)

    native = build_llama_server_command(profile, server_path="llama-server.exe")
    python = build_llama_server_command(profile, python_exe="python.exe")

    assert native[:3] == ["llama-server.exe", "--model", str(model)]
    assert "--ctx-size" in native
    assert python[:3] == ["python.exe", "-m", "llama_cpp.server"]
    assert "--n_ctx" in python


class _Response:
    def __init__(self, payload: bytes):
        self._payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self) -> bytes:
        return self._payload


def test_probe_endpoint_reads_models_and_context() -> None:
    payload = b'{"data":[{"id":"local-model","meta":{"n_ctx":4096},"owned_by":"llama.cpp"}]}'

    result = probe_endpoint("http://127.0.0.1:8080/v1", opener=lambda *_args, **_kwargs: _Response(payload))

    assert result.reachable is True
    assert result.models == ("local-model",)
    assert result.context_hint == 4096
    assert result.provider_hint == "llama.cpp"


def test_probe_endpoint_reports_unreachable() -> None:
    def opener(*_args, **_kwargs):
        raise urllib.error.URLError("refused")

    result = probe_endpoint("http://127.0.0.1:8080/v1", opener=opener)

    assert result.reachable is False
    assert result.status == "unreachable"


def test_record_serve_fault_writes_turn_trace_fault(tmp_path: Path) -> None:
    turn_trace.set_db_path(tmp_path / "turn_trace.sqlite3")
    try:
        diagnosis = diagnose_serve_output("address already in use")
        assert diagnosis is not None
        record_serve_fault("turn-1", diagnosis)
        faults = turn_trace.list_faults_since("1970-01-01T00:00:00+00:00", limit=10)
        assert len(faults) == 1
        assert faults[0].fault_kind == "model_serve_failed"
        assert faults[0].payload["meta"]["diagnosis"]["kind"] == "port_in_use"
    finally:
        turn_trace.set_db_path(None)
