"""
core/adaptive_budget.py — Single-pass adaptive reasoning budget.

Computes a 0-100 complexity score from message signals and maps it to a
discrete budget tier that caps max_tokens and injects guidance text.

The interceptor runs after context_refresh and before the engine dispatches
to the backend. Guidance is injected as a transient tagged user-context
message ([BUDGET GUIDANCE]) inserted immediately before the latest user turn.

Config is persisted to CONFIG_DIR/adaptive_budget.json and hot-reloaded on
file mtime change.
"""
from __future__ import annotations

import json
import math
import os
import re
import time
from dataclasses import asdict, dataclass, field
from typing import Any, NamedTuple

from core.paths import CONFIG_DIR

# ---------------------------------------------------------------------------
# Config (ResourcePolicy pattern)
# ---------------------------------------------------------------------------

_CONFIG_PATH = CONFIG_DIR / "adaptive_budget.json"

_DEFAULT_WEIGHTS: dict[str, float] = {
    "length": 0.15,
    "entropy": 0.12,
    "questions": 0.08,
    "code_blocks": 0.08,
    "multipart": 0.20,
    "conditionals": 0.12,
    "technical": 0.12,
    "depth": 0.07,
    "errors": 0.06,
}

_CONFIG_FIELDS = {
    "enabled", "tier_boundaries", "tier_max_tokens",
    "feature_weights", "error_decay", "log_scores",
}


@dataclass
class AdaptiveBudgetConfig:
    enabled: bool = True
    tier_boundaries: list[int] = field(default_factory=lambda: [25, 50, 75])
    tier_max_tokens: list[int] = field(default_factory=lambda: [512, 1024, 2048, 4096])
    feature_weights: dict[str, float] = field(default_factory=lambda: dict(_DEFAULT_WEIGHTS))
    error_decay: float = 0.85
    log_scores: bool = True

    @classmethod
    def load(cls) -> "AdaptiveBudgetConfig":
        try:
            raw: Any = json.loads(_CONFIG_PATH.read_text(encoding="utf-8"))
            if not isinstance(raw, dict):
                return cls()
            kwargs = {k: v for k, v in raw.items() if k in _CONFIG_FIELDS}
            return cls(**kwargs)
        except Exception:
            return cls()

    def save(self) -> None:
        _CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        _CONFIG_PATH.write_text(
            json.dumps(asdict(self), indent=2), encoding="utf-8"
        )


# ---------------------------------------------------------------------------
# Budget tiers
# ---------------------------------------------------------------------------

class BudgetTier(NamedTuple):
    name: str
    max_tokens: int
    guidance: str
    suggest_chain: bool


_TIERS = [
    BudgetTier(
        "MINIMAL", 512,
        "Respond concisely in a few sentences. No elaboration needed.",
        False,
    ),
    BudgetTier(
        "STANDARD", 1024,
        "Provide a clear, focused response with moderate detail.",
        False,
    ),
    BudgetTier(
        "DETAILED", 2048,
        "Think step by step. Break the problem into parts before acting.",
        True,
    ),
    BudgetTier(
        "EXHAUSTIVE", 4096,
        "Complex request. Reason carefully and thoroughly. Consider edge cases. "
        "Use chain mode for multi-tool tasks.",
        True,
    ),
]


def score_to_tier(score: int, cfg: AdaptiveBudgetConfig) -> BudgetTier:
    boundaries = cfg.tier_boundaries
    for i, boundary in enumerate(boundaries):
        if score <= boundary:
            return _TIERS[min(i, len(_TIERS) - 1)]
    return _TIERS[-1]


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

_CONDITIONAL_WORDS = frozenset({
    "if", "unless", "when", "depends", "however", "but", "although",
    "whether", "otherwise", "alternatively",
})
_MULTIPART_RE = re.compile(
    r"(?:^\s*(?:\d+[\.\)]\s|[-*]\s|[a-z][\.\)]\s))"
    r"|(?:\b(?:first|then|next|also|finally|additionally|furthermore)\b)",
    re.IGNORECASE | re.MULTILINE,
)
_TECHNICAL_RE = re.compile(
    r"(?:[A-Za-z]:\\|/[\w.]+/[\w.]+)"  # file paths
    r"|(?:\.\w{1,5}\b)"                # file extensions
    r"|(?:```)"                         # code fences
    r"|(?:\b(?:function|class|def|import|return|async|await|const|let|var)\b)",
    re.IGNORECASE,
)
_ERROR_RE = re.compile(
    r"\b(?:error|exception|traceback|failed|failure|crash|panic|segfault)\b",
    re.IGNORECASE,
)


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


def compute_complexity_score(
    messages: list[dict],
    config: dict,
    cfg: AdaptiveBudgetConfig,
) -> tuple[int, dict[str, float]]:
    """Return (score 0-100, feature_dict) from the message history."""
    weights = cfg.feature_weights

    # Extract latest user message
    last_user = ""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            content = msg.get("content", "")
            # Skip budget guidance and system reminders
            if "[BUDGET GUIDANCE]" in content or "[SYSTEM REMINDER]" in content:
                continue
            last_user = content
            break

    words = last_user.lower().split()
    word_count = max(len(words), 1)

    # --- Per-message features ---
    f_length = _clamp(len(last_user) / 2000)
    # Entropy only meaningful with enough words; short messages get 0
    f_entropy = _clamp(len(set(words)) / word_count) if len(words) >= 8 else 0.0
    f_questions = _clamp(last_user.count("?") / 3)
    f_code_blocks = _clamp(last_user.count("```") / 2)
    f_multipart = _clamp(len(_MULTIPART_RE.findall(last_user)) / 5)
    f_conditionals = _clamp(
        sum(1 for w in words if w in _CONDITIONAL_WORDS) / 4
    )
    f_technical = _clamp(len(_TECHNICAL_RE.findall(last_user)) / 5)

    # --- Context-window features (decay-weighted) ---
    f_depth = _clamp(len(messages) / 40)

    error_score = 0.0
    recent = messages[-10:] if len(messages) > 10 else messages
    decay = cfg.error_decay
    for age, msg in enumerate(reversed(recent)):
        content = msg.get("content", "")
        if _ERROR_RE.search(content):
            error_score += decay ** age
    f_errors = _clamp(error_score / 3)

    features = {
        "length": f_length,
        "entropy": f_entropy,
        "questions": f_questions,
        "code_blocks": f_code_blocks,
        "multipart": f_multipart,
        "conditionals": f_conditionals,
        "technical": f_technical,
        "depth": f_depth,
        "errors": f_errors,
    }

    raw = sum(features.get(k, 0.0) * weights.get(k, 0.0) for k in weights)
    # Non-linear scaling: amplify mid-range scores so complex prompts
    # aren't stuck in STANDARD tier.  sqrt mapping: 0.36 raw → 60 score.
    scaled = math.sqrt(_clamp(raw)) * 100
    score = int(_clamp(scaled, 0, 100))
    return score, features


# ---------------------------------------------------------------------------
# Model-aware adjustments
# ---------------------------------------------------------------------------

def _adjust_tier_for_model(
    tier: BudgetTier, model_preset: dict | None,
) -> BudgetTier:
    """Adjust tier based on model capabilities from world state."""
    if not model_preset:
        return tier
    caps = model_preset.get("capabilities") or {}
    ctx = model_preset.get("context_window")

    # If model supports thinking, it can handle more — bump max_tokens
    if caps.get("supports_thinking") and tier.max_tokens < 4096:
        return tier._replace(max_tokens=min(tier.max_tokens * 2, 4096))

    # If model has small context, clamp max_tokens harder
    if ctx and ctx < 8192 and tier.max_tokens > 1024:
        return tier._replace(max_tokens=min(tier.max_tokens, 1024))

    return tier


# ---------------------------------------------------------------------------
# Wiring (set from bootstrap.py)
# ---------------------------------------------------------------------------

_ledger = None
_world_state = None


def set_ledger(ledger: Any) -> None:
    global _ledger
    _ledger = ledger


def set_world_state(ws: Any) -> None:
    global _world_state
    _world_state = ws


# ---------------------------------------------------------------------------
# Interceptor
# ---------------------------------------------------------------------------

_BUDGET_TAG = "[BUDGET GUIDANCE]"

# Config cache
_cached_config: AdaptiveBudgetConfig | None = None
_cached_mtime: float = 0.0
_call_count: int = 0
_RELOAD_INTERVAL = 50  # check file mtime every N calls
_last_budget_snapshot: dict[str, Any] = {}


def _get_config() -> AdaptiveBudgetConfig:
    global _cached_config, _cached_mtime, _call_count
    _call_count += 1

    if _cached_config is not None and _call_count % _RELOAD_INTERVAL != 0:
        return _cached_config

    try:
        mtime = os.path.getmtime(_CONFIG_PATH)
    except OSError:
        mtime = 0.0

    if _cached_config is None or mtime != _cached_mtime:
        _cached_config = AdaptiveBudgetConfig.load()
        _cached_mtime = mtime

    return _cached_config


def _get_model_preset() -> dict | None:
    """Read the active LLM's model_preset from world state."""
    if _world_state is None:
        return None
    engines = _world_state.state.get("engines", {})
    for key, meta in engines.items():
        if key.startswith("llm") and isinstance(meta.get("model_preset"), dict):
            return meta["model_preset"]
    return None


def adaptive_budget_interceptor(
    messages: list[dict], config: dict,
) -> list[dict] | None:
    """Message interceptor: score complexity and inject budget guidance."""
    cfg = _get_config()
    if not cfg.enabled:
        return None

    # Don't double-inject
    for msg in messages:
        if _BUDGET_TAG in msg.get("content", ""):
            return None

    score, features = compute_complexity_score(messages, config, cfg)
    tier = score_to_tier(score, cfg)

    # Model-aware adjustment
    model_preset = _get_model_preset()
    tier = _adjust_tier_for_model(tier, model_preset)

    # Cap max_tokens (engine re-reads this after interceptors)
    user_max = config.get("max_tokens", 2048)
    config["max_tokens"] = min(int(user_max), tier.max_tokens)

    guidance = f"{_BUDGET_TAG} {tier.guidance}"
    result = list(messages)
    last_user_idx = -1
    for i in range(len(result) - 1, -1, -1):
        if result[i].get("role") == "user":
            last_user_idx = i
            break
    if last_user_idx < 0:
        return None
    result.insert(
        last_user_idx,
        {
            "role": "user",
            "content": guidance,
            "ephemeral": True,
            "source": "adaptive_budget",
        },
    )

    # Surface score to world state and local snapshot cache
    global _last_budget_snapshot
    _last_budget_snapshot = {
        "score": score,
        "tier": tier.name,
        "max_tokens_cap": config["max_tokens"],
        "features": dict(features),
        "timestamp": time.time(),
    }

    # Surface score to world state
    if _world_state is not None:
        _world_state.state.setdefault("adaptive_budget", {}).update({
            "last_score": score,
            "last_tier": tier.name,
            "last_max_tokens": config["max_tokens"],
        })

    # Log to event ledger
    if cfg.log_scores and _ledger is not None:
        try:
            _ledger.record(
                source="adaptive_budget",
                kind="scoring",
                name="complexity_scored",
                payload={
                    "score": score,
                    "tier": tier.name,
                    "features": features,
                    "max_tokens_cap": config["max_tokens"],
                    "message_count": len(messages),
                    "model_family": (model_preset or {}).get("family_id"),
                },
            )
        except Exception:
            pass  # never break generation for logging

    return result


def get_last_budget_snapshot() -> dict[str, Any]:
    return dict(_last_budget_snapshot)


def evaluate_budget_for_message(
    message: str,
    *,
    message_count: int = 1,
) -> dict[str, Any]:
    cfg = _get_config()
    count = max(1, int(message_count))
    seed_messages = [{"role": "system", "content": "You are Monolith."}]
    for _ in range(max(1, count - 1)):
        seed_messages.append({"role": "assistant", "content": ""})
    seed_messages.append({"role": "user", "content": str(message or "")})
    score, features = compute_complexity_score(seed_messages, {}, cfg)
    tier = score_to_tier(score, cfg)
    return {
        "score": score,
        "tier": tier.name,
        "max_tokens_cap": tier.max_tokens,
        "features": features,
    }
