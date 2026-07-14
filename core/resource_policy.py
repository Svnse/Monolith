"""
core/resource_policy.py

Persistent resource-limit configuration for Monolith engines.

Stored as JSON in CONFIG_DIR/resource_policy.json.  All fields have safe
defaults so the file is optional — missing or corrupt config falls back to
defaults without error.

Fields
------
generation_timeout_sec : int
    Hard wall on how long a single generate() call may run.  The LLMEngine
    fires a QTimer that calls stop_generation() when this expires.
    0 = disabled (no timeout).

circuit_breaker_threshold : int
    Number of consecutive engine errors before the circuit opens.

circuit_breaker_cooldown_sec : int
    Seconds the circuit stays open before allowing new submissions.

vram_min_free_mb : dict[str, int]
    Per engine-key minimum free VRAM required before a GPU command (generate /
    load) is accepted.  Example: {"vision": 512}.  Empty by default (no quota).
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any

from core.paths import CONFIG_DIR

_POLICY_PATH = CONFIG_DIR / "resource_policy.json"

_FIELDS = {
    "generation_timeout_sec",
    "circuit_breaker_threshold",
    "circuit_breaker_cooldown_sec",
    "vram_min_free_mb",
}


@dataclass
class ResourcePolicy:
    generation_timeout_sec: int = 120
    circuit_breaker_threshold: int = 3
    circuit_breaker_cooldown_sec: int = 60
    vram_min_free_mb: dict[str, int] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    @classmethod
    def load(cls) -> "ResourcePolicy":
        """Load from disk; silently fall back to defaults on any error."""
        try:
            raw: Any = json.loads(_POLICY_PATH.read_text(encoding="utf-8"))
            if not isinstance(raw, dict):
                return cls()
            kwargs = {k: v for k, v in raw.items() if k in _FIELDS}
            return cls(**kwargs)
        except Exception:
            return cls()

    def save(self) -> None:
        _POLICY_PATH.parent.mkdir(parents=True, exist_ok=True)
        _POLICY_PATH.write_text(
            json.dumps(asdict(self), indent=2), encoding="utf-8"
        )
