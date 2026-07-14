"""lag_watch — per-turn JSONL record of system-side classification.

Writes one JSON line per turn to LOG_DIR/lag_watch.jsonl. Each line captures:
  - ts            : ISO timestamp
  - user_preview  : first 80 chars of the latest non-ephemeral user message
  - system_class  : TurnShape produced by core.turn_classifier upstream
  - llm_class     : always null in the system-side architecture. Reserved
                    field — if the LLM-emit pattern ever returns (e.g.
                    someone re-adds `<axes>`/`<intent>` to the prompt),
                    populate this with the parsed emission and the file
                    becomes a drift log (system_class vs llm_class).

While llm_class stays null turn after turn, the architecture is healthy:
the LLM is not classifying anything. The system computes the shape.

Flag: MONOLITH_LAG_WATCH (default ON). Set =0 to disable.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any

from core.paths import LOG_DIR

_FLAG_ENV = "MONOLITH_LAG_WATCH"
_LOG_PATH = LOG_DIR / "lag_watch.jsonl"


def set_world_state(ws: Any) -> None:
    """Kept for bootstrap compatibility; lag_watch no longer reads world_state.

    Previously read `world_state.get_emitted_axes()` to dry-run the gate
    registry. After Phase 6 removed the emit accessors, the wiring stays
    in bootstrap but the function is a no-op.
    """
    return None


def lag_watch_interceptor(messages: list[dict], config: dict) -> list[dict] | None:
    """Read-only: log one JSON line per turn, never mutate messages."""
    if str(os.environ.get(_FLAG_ENV, "1")).lower() not in {"1", "true", "yes", "on"}:
        return None

    system_class: dict | None = None
    if isinstance(config, dict):
        shape = config.get("_turn_shape")
        if shape is not None and hasattr(shape, "to_dict"):
            try:
                system_class = shape.to_dict()
            except Exception:
                system_class = None

    last_user = next(
        (str(m.get("content", ""))[:80].replace("\n", " ")
         for m in reversed(messages)
         if m.get("role") == "user" and not m.get("ephemeral")),
        "",
    )

    record = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "user_preview": last_user,
        "system_class": system_class,
        "llm_class": None,  # reserved — populates if LLM-emit pattern returns
    }
    try:
        _LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with _LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:
        pass
    return None
