"""@mention routing with dynamic participant registry and loop guard."""

import re
import threading


# Default colors for known CLI agents — overridden by what they pass in chat_join
KNOWN_COLORS = {
    "claude":  "#a78bfa",
    "codex":   "#facc15",
    "kimi":    "#38bdf8",
    "gemini":  "#4285f4",
}

# Cycling palette for unknown agents
_PALETTE = ["#f472b6", "#34d399", "#fb923c", "#818cf8", "#a3e635", "#22d3ee"]


class ParticipantRegistry:
    """Tracks live participants — both external (CLI via MCP) and internal (Loop runs)."""

    def __init__(self):
        self._lock = threading.Lock()
        self._participants: dict[str, dict] = {}   # name → {color, label, kind, ...}
        self._palette_idx = 0

    def join(self, name: str, color: str = "", label: str = "", kind: str = "external") -> dict:
        with self._lock:
            color = color or KNOWN_COLORS.get(name.lower()) or self._next_color()
            label = label or name
            entry = {
                "name":  name,
                "color": color,
                "label": label,
                "kind":  kind,   # "external" | "loop" | "human"
            }
            self._participants[name.lower()] = entry
            return entry

    def leave(self, name: str) -> None:
        with self._lock:
            self._participants.pop(name.lower(), None)

    def get_all(self) -> list[dict]:
        with self._lock:
            return list(self._participants.values())

    def get(self, name: str) -> dict | None:
        with self._lock:
            return self._participants.get(name.lower())

    def names(self) -> set[str]:
        with self._lock:
            return set(self._participants.keys())

    def _next_color(self) -> str:
        color = _PALETTE[self._palette_idx % len(_PALETTE)]
        self._palette_idx += 1
        return color


class Router:
    """Routes @mentions to the right participants. Loop guard limits agent chains."""

    def __init__(self, max_hops: int = 4):
        self.max_hops = max_hops
        self._hop_count = 0
        self._paused = False
        self.guard_emitted = False
        self.registry = ParticipantRegistry()

    # ------------------------------------------------------------------
    # Mention parsing — rebuilt dynamically as participants join/leave
    # ------------------------------------------------------------------

    def parse_mentions(self, text: str) -> list[str]:
        names = self.registry.names()
        if not names:
            return []
        pattern = re.compile(
            r"@(" + "|".join(re.escape(n) for n in sorted(names)) + r"|all|everyone)\b",
            re.IGNORECASE,
        )
        mentioned: set[str] = set()
        for m in pattern.finditer(text):
            token = m.group(1).lower()
            if token in ("all", "everyone"):
                mentioned.update(names)
            else:
                mentioned.add(token)
        return list(mentioned)

    def _is_agent(self, sender: str) -> bool:
        entry = self.registry.get(sender)
        return entry is not None and entry.get("kind") != "human"

    def get_targets(self, sender: str, text: str) -> list[str]:
        if self._paused:
            return []

        mentions = self.parse_mentions(text)

        if not self._is_agent(sender):
            # Human message resets hop counter
            self._hop_count = 0
            self._paused = False
            self.guard_emitted = False
            return mentions   # empty = nobody woken (default: none)
        else:
            # Agent message — only explicit @mentions
            if not mentions:
                return []
            self._hop_count += 1
            if self._hop_count > self.max_hops:
                self._paused = True
                return []
            return [m for m in mentions if m != sender.lower()]

    def continue_routing(self) -> None:
        self._hop_count = 0
        self._paused = False
        self.guard_emitted = False

    @property
    def is_paused(self) -> bool:
        return self._paused
