"""/prompt — unified prompt scaffold injection.

Replaces the 4-plane system (effort, conversation, reasoning, linguency)
with a single composable prompt library. Users select prompts by name via
`/prompt <name1> <name2> ...`; multiple prompts compose with left-to-right
priority (first-listed = closest to user message = highest authority).

Fully opt-in: no /prompt set = no scaffold injected.

Auto-discovery: scans prompts/*.md at resolve time, excluding reserved
names (system.md, classification.md). New prompts added by dropping a .md
file — zero code changes.

MonoThink is NOT part of this library. It has its own injection path
(core/monothink.py:monothink_interceptor) due to bounded-autonomy
self-evolution. This module does not load, validate, or inject monothink.

World state keys:
  active_prompts       — persistent baseline (JSON list, ordered)
  active_prompts_once  — per-turn override (consumed after one read)

Flag: MONOLITH_PROMPT_V1 (default ON). Set =0 to disable injection.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


_PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"
_RESERVED = frozenset({"system", "classification", "README"})
_FLAG_ENV = "MONOLITH_PROMPT_V1"

_BASELINE_KEY = "active_prompts"
_ONCE_KEY = "active_prompts_once"


class PromptLibrary:
    def __init__(self, prompts_dir: Path | None = None):
        self._dir = prompts_dir or _PROMPTS_DIR
        self._world_state: Any = None

    def set_world_state(self, ws: Any) -> None:
        self._world_state = ws

    # ── discovery ───────────────────────────────────────────────────────

    def valid_prompts(self) -> frozenset[str]:
        """Return the set of selectable prompt names (stems of .md files
        in the prompts directory, minus reserved names)."""
        if not self._dir.is_dir():
            return frozenset()
        return frozenset(
            p.stem for p in self._dir.glob("*.md")
            if p.stem not in _RESERVED
        )

    def load_content(self, name: str) -> str | None:
        """Read the scaffold text for *name*. Returns None if missing or empty."""
        path = self._dir / f"{name}.md"
        if not path.exists():
            return None
        try:
            text = path.read_text(encoding="utf-8").strip()
            return text or None
        except Exception:
            return None

    # ── resolution ──────────────────────────────────────────────────────

    def _flag_enabled(self) -> bool:
        raw = str(os.environ.get(_FLAG_ENV, "1")).strip().lower()
        return raw in {"1", "true", "yes", "on"}

    def resolve(self, config: dict) -> list[str]:
        """Return the ordered list of active prompt names for this turn.

        Resolution layers (highest priority first):
          1. config["force_prompts"]           — testing/automation (JSON list)
          2. world_state once-prompts (consumed)
          3. world_state baseline prompts
          4. Empty list (fully opt-in default)

        Invalid names (not in valid_prompts()) are silently dropped.
        """
        valid = self.valid_prompts()

        # Layer 1: force override
        if isinstance(config, dict):
            forced = config.get("force_prompts")
            if isinstance(forced, list) and forced:
                return [n for n in forced if isinstance(n, str) and n in valid]

        if self._world_state is not None:
            # Layer 2: once-override (consume on read)
            try:
                once_raw = self._world_state.state.get(_ONCE_KEY)
                if once_raw is not None:
                    self._world_state.state.pop(_ONCE_KEY, None)
                    self._world_state.mark_dirty()
                    if isinstance(once_raw, list) and once_raw:
                        return [n for n in once_raw if isinstance(n, str) and n in valid]
                    if isinstance(once_raw, str):
                        try:
                            parsed = json.loads(once_raw)
                            if isinstance(parsed, list):
                                return [n for n in parsed if isinstance(n, str) and n in valid]
                        except (json.JSONDecodeError, TypeError):
                            pass
            except Exception:
                pass

            # Layer 3: persistent baseline
            try:
                baseline_raw = self._world_state.state.get(_BASELINE_KEY)
                if baseline_raw is not None:
                    if isinstance(baseline_raw, list):
                        return [n for n in baseline_raw if isinstance(n, str) and n in valid]
                    if isinstance(baseline_raw, str):
                        try:
                            parsed = json.loads(baseline_raw)
                            if isinstance(parsed, list):
                                return [n for n in parsed if isinstance(n, str) and n in valid]
                        except (json.JSONDecodeError, TypeError):
                            pass
            except Exception:
                pass

        # Layer 4: opt-in default — nothing
        return []

    def peek(self) -> list[str]:
        """Non-consuming read of active prompts for observation surfaces
        (channel_tag, vitals footer). Returns explicitly-set prompts without
        popping the once key."""
        if self._world_state is None:
            return []
        valid = self.valid_prompts()
        try:
            once_raw = self._world_state.state.get(_ONCE_KEY)
            if isinstance(once_raw, list) and once_raw:
                return [n for n in once_raw if isinstance(n, str) and n in valid]
        except Exception:
            pass
        try:
            baseline_raw = self._world_state.state.get(_BASELINE_KEY)
            if isinstance(baseline_raw, list) and baseline_raw:
                return [n for n in baseline_raw if isinstance(n, str) and n in valid]
        except Exception:
            pass
        return []

    # ── interceptor ─────────────────────────────────────────────────────

    def interceptor(self, messages: list[dict], config: dict) -> list[dict] | None:
        """Inject active prompt scaffolds before the latest user turn.

        Returns None when:
          - flag is off
          - no prompts resolved (opt-in default)
          - all resolved prompts have no content
          - prompts already injected (source-field defense)
          - no non-ephemeral user message exists

        Composition: prompts are inserted in forward order at last_user_idx.
        Each insertion pushes prior ones further from the user message, so
        the first-listed prompt ends up closest to user (highest priority).

        Side effect: writes resolved prompt list into config["_resolved_prompts"].
        """
        if not self._flag_enabled():
            return None

        prompts = self.resolve(config)
        if isinstance(config, dict):
            config["_resolved_prompts"] = prompts
        if not prompts:
            return None

        # Source-field dedup: if any prompt is already injected, skip
        for msg in messages:
            src = msg.get("source", "")
            if isinstance(src, str) and src.startswith("prompt:"):
                return None

        # Find last non-ephemeral user message
        last_user_idx = -1
        for i in range(len(messages) - 1, -1, -1):
            msg = messages[i]
            if msg.get("role") == "user" and not msg.get("ephemeral"):
                last_user_idx = i
                break
        if last_user_idx < 0:
            return None

        # Load content for each prompt
        loaded: list[tuple[str, str]] = []
        for name in prompts:
            content = self.load_content(name)
            if content:
                loaded.append((name, content))
        if not loaded:
            return None

        # Insert in forward order: first-listed ends up closest to user
        result = list(messages)
        for name, content in loaded:
            result.insert(
                last_user_idx,
                {
                    "role": "user",
                    "content": content,
                    "ephemeral": True,
                    "source": f"prompt:{name}",
                    "prompt_name": name,
                },
            )
        return result


# ── tool prompts (standalone commands, not composable via /prompt) ───────

_TOOLS_DIR = _PROMPTS_DIR / "tools"


def load_tool_prompt(name: str) -> str | None:
    """Load a tool prompt from prompts/tools/{name}.md.

    Tool prompts are standalone command scaffolds (like /skill-creator)
    that inject for one turn. They live in a separate directory from
    the composable /prompt library and are NOT auto-discovered by
    valid_prompts().
    """
    path = _TOOLS_DIR / f"{name}.md"
    if not path.exists():
        return None
    try:
        text = path.read_text(encoding="utf-8").strip()
        return text or None
    except Exception:
        return None


def tool_interceptor(messages: list[dict], config: dict) -> list[dict] | None:
    """Inject a one-shot tool prompt when _tool_prompt_once is set in config.

    Called from the interceptor chain. The slash command handler sets
    config["_tool_prompt_once"] = "skill-creator" (or similar); this
    interceptor reads it, loads the tool prompt, and injects it.
    """
    name = None
    if isinstance(config, dict):
        name = config.pop("_tool_prompt_once", None)
    if not name or not isinstance(name, str):
        # Check world_state for tool_prompt_once key
        if _library._world_state is not None:
            try:
                name = _library._world_state.state.pop("tool_prompt_once", None)
                if name:
                    _library._world_state.mark_dirty()
            except Exception:
                pass
    if not name:
        return None

    content = load_tool_prompt(name)
    if not content:
        return None

    for msg in messages:
        if msg.get("source") == f"tool:{name}":
            return None

    last_user_idx = -1
    for i in range(len(messages) - 1, -1, -1):
        msg = messages[i]
        if msg.get("role") == "user" and not msg.get("ephemeral"):
            last_user_idx = i
            break
    if last_user_idx < 0:
        return None

    result = list(messages)
    result.insert(
        last_user_idx,
        {
            "role": "user",
            "content": content,
            "ephemeral": True,
            "source": f"tool:{name}",
            "tool_prompt": name,
        },
    )
    return result


# ── module-level singleton ──────────────────────────────────────────────

_library = PromptLibrary()

# Public API
set_world_state = _library.set_world_state
valid_prompts = _library.valid_prompts
load_content = _library.load_content
resolve = _library.resolve
peek = _library.peek
prompt_interceptor = _library.interceptor
