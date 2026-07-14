from __future__ import annotations

import copy
import json
import math
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from core.paths import CONFIG_DIR

WORLD_STATE_FLUSH_INTERVAL_MS = 1000
_ENGINE_KEY_RE = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")
_MAX_TEXT = 4096
_MAX_ITEMS = 200
_MAX_DEPTH = 8
_RESOURCE_KEYS = frozenset({"cpu_pct", "ram_used_mb", "ram_total_mb", "vram_used_mb", "vram_free_mb"})

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sanitize_text(value: Any, max_len: int = _MAX_TEXT) -> str:
    return str(value or "")[:max_len]


def _sanitize_jsonish(value: Any, depth: int = 0):
    if depth > _MAX_DEPTH:
        return _sanitize_text(value)

    if value is None or isinstance(value, (bool, int)):
        return value

    if isinstance(value, float):
        return value if math.isfinite(value) else None

    if isinstance(value, str):
        return value[:_MAX_TEXT]

    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for idx, (key, item) in enumerate(value.items()):
            if idx >= _MAX_ITEMS:
                break
            safe_key = _sanitize_text(key, max_len=128)
            out[safe_key] = _sanitize_jsonish(item, depth + 1)
        return out

    if isinstance(value, (list, tuple)):
        return [_sanitize_jsonish(item, depth + 1) for item in list(value)[:_MAX_ITEMS]]

    return _sanitize_text(value)


@dataclass
class WorldStateStore:
    """Minimal world state store for Monolith.

    Single source of truth for engine/task/resource/session state.
    Persisted as JSON in CONFIG_DIR.
    """

    path: Path = field(default_factory=lambda: CONFIG_DIR / "world_state.json")
    state: dict[str, Any] = field(default_factory=dict)
    _dirty: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.state = self._load() or self._default_state()
        # Engines and tasks are runtime-only -- their keys are UUIDs tied to
        # chat-tab instances that don't survive a restart. A graceful shutdown
        # clears them via guard.unregister_engine, but a force-quit (kill -9,
        # crash, power loss) leaves stale rows on disk that resurface as
        # "ghost" entries in the vitals footer on next launch. Wipe them.
        self._reset_transient_state()

    def _reset_transient_state(self) -> None:
        """Drop runtime-scoped state that can't meaningfully survive a restart.

        Mark dirty rather than saving eagerly: the normal flush path will
        write it. If a second force-quit happens before the flush, the wipe
        re-runs on next launch -- idempotent and safe.
        """
        changed = False
        if self.state.get("engines"):
            self.state["engines"] = {}
            changed = True
        if self.state.get("tasks"):
            self.state["tasks"] = {}
            changed = True
        if changed:
            self.mark_dirty()

    def _default_state(self) -> dict[str, Any]:
        return {
            "updated_at": _now_iso(),
            "engines": {},
            "tasks": {},
            "resources": {
                "cpu_pct": None,
                "ram_used_mb": None,
                "ram_total_mb": None,
                "vram_used_mb": None,
                "vram_free_mb": None,
            },
            "session": {
                "last_user_prompt": "",
                "last_action": "",
                "last_action_at": "",
                "pending_action": None,
            },
            "action_log": [],
        }

    def _load(self) -> dict[str, Any] | None:
        if not self.path.exists():
            return None
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            return None

    def save(self) -> None:
        self.state["updated_at"] = _now_iso()
        payload = json.dumps(self.state, indent=2)
        tmp_path = self.path.with_name(f"{self.path.name}.tmp")
        tmp_path.write_text(payload, encoding="utf-8")
        try:
            os.replace(tmp_path, self.path)
        except Exception:
            try:
                if tmp_path.exists():
                    tmp_path.unlink()
            except Exception:
                pass
            raise
        self._dirty = False

    def mark_dirty(self) -> None:
        self._dirty = True

    def flush(self) -> None:
        if self._dirty:
            self.save()

    # ---- engine state ----
    def set_engine_status(self, engine_key: str, status: str) -> None:
        engine_key = _sanitize_text(engine_key, max_len=64)
        if not _ENGINE_KEY_RE.match(engine_key):
            return
        engine = self.state.setdefault("engines", {}).setdefault(engine_key, {})
        engine["status"] = _sanitize_text(status, max_len=64)
        self.mark_dirty()

    def set_engine_meta(self, engine_key: str, **meta: Any) -> None:
        engine_key = _sanitize_text(engine_key, max_len=64)
        if not _ENGINE_KEY_RE.match(engine_key):
            return
        engine = self.state.setdefault("engines", {}).setdefault(engine_key, {})
        safe_meta = _sanitize_jsonish(meta)
        if isinstance(safe_meta, dict):
            engine.update(safe_meta)
        self.mark_dirty()

    def clear_engine(self, engine_key: str) -> None:
        """Remove all state for an engine key (status, meta, tasks).

        Called when an engine is unregistered (chat tab destroyed). Without
        this, the vitals footer keeps rendering tombstone rows for engines
        that no longer exist, which is what the "two engines stuck in queue"
        symptom looks like.
        """
        engine_key = _sanitize_text(engine_key, max_len=64)
        if not _ENGINE_KEY_RE.match(engine_key):
            return
        engines = self.state.get("engines") or {}
        if engine_key in engines:
            engines.pop(engine_key, None)
            self.mark_dirty()
        tasks = self.state.get("tasks") or {}
        if engine_key in tasks:
            tasks.pop(engine_key, None)
            self.mark_dirty()

    # ---- task state ----
    def set_active_task(self, engine_key: str, task: dict[str, Any] | None) -> None:
        engine_key = _sanitize_text(engine_key, max_len=64)
        if not _ENGINE_KEY_RE.match(engine_key):
            return
        entry = self.state.setdefault("tasks", {}).setdefault(engine_key, {})
        entry["active"] = _sanitize_jsonish(task) if isinstance(task, dict) else None
        self.mark_dirty()

    def set_queue(self, engine_key: str, queue_ids: list[str]) -> None:
        engine_key = _sanitize_text(engine_key, max_len=64)
        if not _ENGINE_KEY_RE.match(engine_key):
            return
        entry = self.state.setdefault("tasks", {}).setdefault(engine_key, {})
        safe_queue = [_sanitize_text(item, max_len=128) for item in list(queue_ids)[:_MAX_ITEMS]]
        entry["queued"] = safe_queue
        entry["queue_len"] = len(safe_queue)
        self.mark_dirty()

    # ---- resource state ----
    def set_resources(self, **resources: Any) -> None:
        res = self.state.setdefault("resources", {})
        for key, value in resources.items():
            if key not in _RESOURCE_KEYS:
                continue
            safe_value = _sanitize_jsonish(value)
            if isinstance(safe_value, (int, float)) or safe_value is None:
                res[key] = safe_value
        self.mark_dirty()

    # ---- session state ----
    def set_last_prompt(self, text: str) -> None:
        session = self.state.setdefault("session", {})
        session["last_user_prompt"] = _sanitize_text(text)
        session["last_action_at"] = _now_iso()
        self.mark_dirty()

    def set_last_action(self, action: str) -> None:
        session = self.state.setdefault("session", {})
        session["last_action"] = _sanitize_text(action)
        session["last_action_at"] = _now_iso()
        self.mark_dirty()

    # ---- plane mode methods (compressed step 2.6) ----
    #
    # Per-plane methods (set_X_*, consume_X_*, get_X_state, clear_X_*) are
    # thin wrappers around four private helpers below. Each plane has its own
    # public method names for self-documenting call sites — `set_conversation_mode`
    # reads better than `set_plane_mode("conversation", ...)` — but the
    # implementation is shared.
    #
    # Plane key conventions:
    #   * baseline_key = f"{plane}_{suffix}"        (e.g. "effort_tier", "conversation_mode")
    #   * once_key     = f"{plane}_once_{suffix}"   (e.g. "effort_once_tier")
    #   * suffix is "tier" for effort (depth), "mode" for everything else
    #
    # Defaults differ by plane semantics:
    #   * always-fires planes (effort, conversation): consume returns the default
    #     ("med" / "default") when nothing is set
    #   * opt-in planes (reasoning, linguency): consume returns None when nothing
    #     is set; get_state surfaces None as "unset"

    def _plane_set_baseline(self, key: str, value: str) -> None:
        self.state[key] = _sanitize_text(value, max_len=32)
        self.mark_dirty()

    def _plane_set_once(self, key: str, value: str | None) -> None:
        if value is None:
            if key in self.state:
                self.state.pop(key, None)
                self.mark_dirty()
            return
        self.state[key] = _sanitize_text(value, max_len=32)
        self.mark_dirty()

    def _plane_consume(self, once_key: str, baseline_key: str, default: str | None) -> str | None:
        """Once-override (consumed) > baseline > default. Default may be None for opt-in."""
        if once_key in self.state:
            once = self.state.pop(once_key, None)
            self.mark_dirty()
            if once:
                return str(once)
        val = self.state.get(baseline_key)
        if val:
            return str(val)
        return default

    def _plane_clear_baseline(self, key: str) -> None:
        if key in self.state:
            self.state.pop(key, None)
            self.mark_dirty()

    # ── effort tier (always-fires, default "med") ──────────────────────
    def set_effort_tier(self, tier: str) -> None:
        """Set the persistent effort tier baseline."""
        self._plane_set_baseline("effort_tier", tier)

    def set_effort_once(self, tier: str | None) -> None:
        """Set a per-turn effort override that consumes on next read."""
        self._plane_set_once("effort_once_tier", tier)

    def consume_effort_tier(self) -> str:
        """Return the effective effort tier and consume the once-override.

        Precedence: once-override (consumed) > persistent baseline > "med".
        """
        return self._plane_consume("effort_once_tier", "effort_tier", "med") or "med"

    def get_effort_state(self) -> dict[str, str | None]:
        """Read-only snapshot of effort state (for UI / status commands)."""
        return {
            "tier": str(self.state.get("effort_tier") or "med"),
            "once": self.state.get("effort_once_tier"),
        }

    # ── conversation mode (always-fires, default "default") ────────────
    def set_conversation_mode(self, mode: str) -> None:
        """Set the persistent conversation mode baseline."""
        self._plane_set_baseline("conversation_mode", mode)

    def set_conversation_once(self, mode: str | None) -> None:
        """Set a per-turn conversation override that consumes on next read."""
        self._plane_set_once("conversation_once_mode", mode)

    def consume_conversation_mode(self) -> str:
        """Return the effective conversation mode and consume the once-override.

        Precedence: once-override (consumed) > persistent baseline > "default".
        """
        return self._plane_consume("conversation_once_mode", "conversation_mode", "default") or "default"

    def get_conversation_state(self) -> dict[str, str | None]:
        """Read-only snapshot of conversation state (for UI / status commands)."""
        return {
            "mode": str(self.state.get("conversation_mode") or "default"),
            "once": self.state.get("conversation_once_mode"),
        }

    # ── reasoning mode (opt-in, default None) ──────────────────────────
    def set_reasoning_mode(self, mode: str) -> None:
        """Set the persistent reasoning mode baseline."""
        self._plane_set_baseline("reasoning_mode", mode)

    def clear_reasoning_mode(self) -> None:
        """Clear the persistent reasoning mode (back to opt-in default: nothing)."""
        self._plane_clear_baseline("reasoning_mode")

    def set_reasoning_once(self, mode: str | None) -> None:
        """Set a per-turn reasoning override that consumes on next read."""
        self._plane_set_once("reasoning_once_mode", mode)

    def consume_reasoning_mode(self) -> str | None:
        """Return the effective reasoning mode (None if unset — opt-in default)."""
        return self._plane_consume("reasoning_once_mode", "reasoning_mode", None)

    def get_reasoning_state(self) -> dict[str, str | None]:
        """Read-only snapshot of reasoning state (for UI / status commands)."""
        return {
            "mode": self.state.get("reasoning_mode"),  # None = unset (opt-in)
            "once": self.state.get("reasoning_once_mode"),
        }

    # ── linguency mode (opt-in, default None) ──────────────────────────
    def set_linguency_mode(self, mode: str) -> None:
        """Set the persistent linguency mode baseline."""
        self._plane_set_baseline("linguency_mode", mode)

    def clear_linguency_mode(self) -> None:
        """Clear the persistent linguency mode (back to opt-in default: nothing)."""
        self._plane_clear_baseline("linguency_mode")

    def set_linguency_once(self, mode: str | None) -> None:
        """Set a per-turn linguency override that consumes on next read."""
        self._plane_set_once("linguency_once_mode", mode)

    def consume_linguency_mode(self) -> str | None:
        """Return the effective linguency mode (None if unset — opt-in default)."""
        return self._plane_consume("linguency_once_mode", "linguency_mode", None)

    def get_linguency_state(self) -> dict[str, str | None]:
        """Read-only snapshot of linguency state (for UI / status commands)."""
        return {
            "mode": self.state.get("linguency_mode"),  # None = unset (opt-in)
            "once": self.state.get("linguency_once_mode"),
        }

    # ── unified prompt state (opt-in, default empty) ────────────────────

    def set_active_prompts(self, prompts: list[str]) -> None:
        """Set the persistent prompt baseline (ordered list of prompt names)."""
        self.state["active_prompts"] = [
            _sanitize_text(p, max_len=32) for p in prompts if isinstance(p, str)
        ]
        self.mark_dirty()

    def set_prompts_once(self, prompts: list[str] | None) -> None:
        """Set a per-turn prompt override (consumed after one read)."""
        if prompts is None:
            if "active_prompts_once" in self.state:
                self.state.pop("active_prompts_once", None)
                self.mark_dirty()
            return
        self.state["active_prompts_once"] = [
            _sanitize_text(p, max_len=32) for p in prompts if isinstance(p, str)
        ]
        self.mark_dirty()

    def consume_active_prompts(self) -> list[str]:
        """Return the effective prompt list and consume the once-override.

        Precedence: once-override (consumed) > persistent baseline > [].
        """
        if "active_prompts_once" in self.state:
            once = self.state.pop("active_prompts_once", None)
            self.mark_dirty()
            if isinstance(once, list) and once:
                return [str(p) for p in once]
        val = self.state.get("active_prompts")
        if isinstance(val, list) and val:
            return [str(p) for p in val]
        return []

    def get_prompt_state(self) -> dict:
        """Read-only snapshot of prompt state (for UI / status commands)."""
        return {
            "prompts": list(self.state.get("active_prompts") or []),
            "once": self.state.get("active_prompts_once"),
        }

    def clear_prompts(self) -> None:
        """Clear all prompt state (back to opt-in default: nothing)."""
        changed = False
        if "active_prompts" in self.state:
            self.state.pop("active_prompts", None)
            changed = True
        if "active_prompts_once" in self.state:
            self.state.pop("active_prompts_once", None)
            changed = True
        if changed:
            self.mark_dirty()

    # ── monothink toggle (separate from prompts) ──────────────────────

    def set_monothink(self, enabled: bool) -> None:
        """Set the persistent monothink toggle."""
        self.state["monothink_enabled"] = bool(enabled)
        self.mark_dirty()

    def set_monothink_once(self, enabled: bool | None) -> None:
        """Set a per-turn monothink override (consumed after one read)."""
        if enabled is None:
            if "monothink_once" in self.state:
                self.state.pop("monothink_once", None)
                self.mark_dirty()
            return
        self.state["monothink_once"] = bool(enabled)
        self.mark_dirty()

    def consume_monothink(self) -> bool:
        """Return whether monothink is active and consume the once-override.

        Precedence: once-override (consumed) > persistent > False.
        """
        if "monothink_once" in self.state:
            once = self.state.pop("monothink_once", None)
            self.mark_dirty()
            if isinstance(once, bool):
                return once
        val = self.state.get("monothink_enabled")
        if isinstance(val, bool):
            return val
        return False

    def get_monothink_state(self) -> dict:
        """Read-only snapshot of monothink state (for UI / status commands)."""
        return {
            "enabled": bool(self.state.get("monothink_enabled", False)),
            "once": self.state.get("monothink_once"),
        }

    def clear_monothink(self) -> None:
        """Clear monothink state (back to disabled)."""
        changed = False
        if "monothink_enabled" in self.state:
            self.state.pop("monothink_enabled", None)
            changed = True
        if "monothink_once" in self.state:
            self.state.pop("monothink_once", None)
            changed = True
        if changed:
            self.mark_dirty()

    # ── active workflow (chat flow selector) ──────────────────────────

    def set_active_workflow(self, workflow_id: str | None) -> None:
        """Set/clear the active chat flow id. Absence == Genesis (the default).
        Flat top-level key + named accessor, mirroring monothink_enabled."""
        wid = _sanitize_text(workflow_id or "", max_len=128)
        if wid:
            self.state["active_workflow_id"] = wid
        else:
            self.state.pop("active_workflow_id", None)  # absence == Genesis
        self.mark_dirty()

    def get_active_workflow(self) -> str:
        """Read the active flow id. '' == Genesis (default chat flow)."""
        return str(self.state.get("active_workflow_id") or "")

    def set_pending_action(self, action: dict | None) -> None:
        session = self.state.setdefault("session", {})
        session["pending_action"] = _sanitize_jsonish(action) if isinstance(action, dict) else None
        session["last_action_at"] = _now_iso()
        self.mark_dirty()

    def get_pending_action(self) -> dict | None:
        session = self.state.setdefault("session", {})
        return session.get("pending_action")

    # ---- pending ask_user question (mirror of pending_action shape) ----
    # The question payload is the dict passed to the chat's _on_ask_user
    # callback by the ask_user executor. Held in session state so
    # CompanionPane.evaluate_state can auto-route to ASK_USER when set, and
    # so the panel survives a brief UI re-layout. Cleared by the chat's
    # _on_ask_user_answered / _on_ask_user_dismissed handlers, and on any
    # session-reset boundary.

    def set_pending_question(self, payload: dict | None) -> None:
        session = self.state.setdefault("session", {})
        session["pending_question"] = _sanitize_jsonish(payload) if isinstance(payload, dict) else None
        self.mark_dirty()

    def get_pending_question(self) -> dict | None:
        session = self.state.setdefault("session", {})
        return session.get("pending_question")

    # ---- approval-event tracking (consumed by axis_gating.approval_granted) ----
    def set_last_approval_event(
        self,
        action_type: str | None,
        granted_by: str = "user",
    ) -> None:
        """Record that an approval was just granted for a high-risk action.

        Wire this from the UI approval handler (where the user clicks
        "Approve" on the pending-action dialog). The next turn's
        ``approval_granted`` gate reads the event and injects
        APPROVAL_GRANTED_TEXT so the model executes without re-asking.

        ``consumed`` defaults to False; the gate flips it to True after
        firing once so the same approval doesn't re-fire on subsequent
        turns. Call ``set_last_approval_event(None)`` to clear explicitly.
        """
        session = self.state.setdefault("session", {})
        if action_type is None:
            session.pop("last_approval_event", None)
        else:
            session["last_approval_event"] = {
                "action_type": _sanitize_text(action_type, max_len=120),
                "granted_by": _sanitize_text(granted_by, max_len=40),
                "ts": _now_iso(),
                "consumed": False,
            }
        self.mark_dirty()

    def get_last_approval_event(self) -> dict | None:
        """Return the most recent approval event, or None.

        Read by ``axis_gating._approval_granted_recently``. The gate
        checks ``consumed`` itself before deciding to fire so this getter
        stays a pure read.
        """
        session = self.state.setdefault("session", {})
        evt = session.get("last_approval_event")
        return copy.deepcopy(evt) if isinstance(evt, dict) else None

    def consume_last_approval_event(self) -> None:
        """Mark the last approval event consumed so it doesn't re-fire.

        Call after the approval_granted gate has injected once. The event
        record stays in world_state for telemetry; only the consumed flag
        changes.
        """
        session = self.state.setdefault("session", {})
        evt = session.get("last_approval_event")
        if isinstance(evt, dict):
            evt["consumed"] = True
            self.mark_dirty()

    # ---- action audit log ----
    def append_action_log(self, entry: dict) -> None:
        """Append an entry to the rolling action audit log (max 200 entries)."""
        log = self.state.setdefault("action_log", [])
        safe_entry = _sanitize_jsonish(entry)
        if not isinstance(safe_entry, dict):
            safe_entry = {"entry": _sanitize_text(safe_entry)}
        log.append({"ts": _now_iso(), **safe_entry})
        if len(log) > 200:
            self.state["action_log"] = log[-200:]
        self.mark_dirty()

    def get_action_log(self) -> list[dict]:
        return list(self.state.get("action_log", []))

    def clear_action_log(self) -> None:
        self.state["action_log"] = []
        self.mark_dirty()

    def snapshot(self) -> dict[str, Any]:
        return copy.deepcopy(self.state)
