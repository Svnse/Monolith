from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from core.world_state import WorldStateStore


@dataclass
class WorldReducer:
    """Deterministic reducer for WorldStateStore."""

    store: WorldStateStore

    def apply(self, action: dict[str, Any]) -> dict[str, Any]:
        kind = str(action.get("type") or "")
        payload = action.get("payload") or {}

        if kind == "engine_status":
            engine = str(payload.get("engine"))
            status = str(payload.get("status"))
            if engine and status:
                self.store.set_engine_status(engine, status)
            return {"ok": True}

        if kind == "engine_meta":
            engine = str(payload.get("engine"))
            meta = payload.get("meta") or {}
            if engine and isinstance(meta, dict):
                self.store.set_engine_meta(engine, **meta)
            return {"ok": True}

        if kind == "task_active":
            engine = str(payload.get("engine"))
            task = payload.get("task")
            if engine:
                self.store.set_active_task(engine, task if isinstance(task, dict) else None)
            return {"ok": True}

        if kind == "task_queue":
            engine = str(payload.get("engine"))
            queue = payload.get("queue") or []
            if engine and isinstance(queue, list):
                self.store.set_queue(engine, [str(x) for x in queue])
            return {"ok": True}

        if kind == "resources":
            if isinstance(payload, dict):
                self.store.set_resources(**payload)
            return {"ok": True}

        if kind == "session_prompt":
            text = str(payload.get("text") or "")
            self.store.set_last_prompt(text)
            return {"ok": True}

        if kind == "session_action":
            text = str(payload.get("action") or "")
            self.store.set_last_action(text)
            return {"ok": True}

        return {"ok": False, "error": f"unknown action type: {kind}"}
