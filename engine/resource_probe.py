"""
engine/resource_probe.py
Lightweight GPU/RAM resource probe.

Runs in a throwaway subprocess so the main process never imports torch
just to ask "how much VRAM is free?".  Safe to call before loading any
model — does not contaminate the main CUDA context.
"""
from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import asdict, dataclass


@dataclass
class ResourceInfo:
    vram_total_mb: int = 0
    vram_free_mb: int = 0
    ram_total_mb: int = 0
    ram_free_mb: int = 0
    cuda_available: bool = False
    device_name: str = ""
    probe_ok: bool = True
    warn: str = ""

    def vram_free_gb(self) -> float:
        return round(self.vram_free_mb / 1024, 2)

    def ram_free_gb(self) -> float:
        return round(self.ram_free_mb / 1024, 2)


_PROBE_SCRIPT = r"""
import json, sys

result = {
    "vram_total_mb": 0, "vram_free_mb": 0,
    "ram_total_mb": 0,  "ram_free_mb": 0,
    "cuda_available": False, "device_name": "",
    "probe_ok": True, "warn": "",
}

try:
    import torch
    result["cuda_available"] = torch.cuda.is_available()
    if result["cuda_available"]:
        free, total = torch.cuda.mem_get_info(0)
        result["vram_free_mb"]  = free  // (1024 * 1024)
        result["vram_total_mb"] = total // (1024 * 1024)
        result["device_name"]   = torch.cuda.get_device_name(0)
except Exception as exc:
    result["warn"] += f"torch: {exc}; "

try:
    import psutil
    vm = psutil.virtual_memory()
    result["ram_total_mb"] = vm.total     // (1024 * 1024)
    result["ram_free_mb"]  = vm.available // (1024 * 1024)
except Exception as exc:
    result["warn"] += f"psutil: {exc}; "

print(json.dumps(result))
"""


def probe_resources(timeout: float = 8.0) -> ResourceInfo:
    """
    Probe GPU VRAM and system RAM without importing torch in the caller's
    process.  Returns ResourceInfo.  Never raises — all failures are
    reflected in the ``warn`` and ``probe_ok`` fields.
    """
    try:
        proc = subprocess.run(
            [sys.executable, "-c", _PROBE_SCRIPT],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if proc.returncode == 0 and proc.stdout.strip():
            data = json.loads(proc.stdout.strip())
            return ResourceInfo(**data)
        warn = (proc.stderr or "").strip()[:300]
        return ResourceInfo(
            probe_ok=False,
            warn=warn or f"probe returned exit={proc.returncode}",
        )
    except Exception as exc:
        return ResourceInfo(probe_ok=False, warn=str(exc)[:300])


def estimate_model_vram_mb(model_path: str) -> int:
    """
    Rough estimate of VRAM needed to load a model from its file size.
    Heuristic: file_bytes * 1.25 / 1e6  (accounts for activations overhead).
    Returns 0 if the file cannot be stat'd.
    """
    try:
        from pathlib import Path
        p = Path(model_path)
        if p.is_file():
            return int(p.stat().st_size * 1.25 / (1024 * 1024))
        # HuggingFace directory — sum safetensors/bin files
        total = sum(
            f.stat().st_size
            for f in p.rglob("*")
            if f.suffix in (".safetensors", ".bin", ".gguf")
        )
        return int(total * 1.25 / (1024 * 1024))
    except Exception:
        return 0
