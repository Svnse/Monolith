"""PlaneLoader — shared base for effort, conversation, reasoning, linguency.

The plane-separation refactor (S11) shipped 4 parallel loader modules in step 2
to prove the abstraction across 4 real planes. This module is the
compression-after-proving pass: one class encodes the loader contract; each
plane file now provides just its PlaneConfig + a thin alias surface preserving
the existing public function names.

Why now: extracting before seeing the variation would have been premature
abstraction — the shape of "what varies between planes" only became legible
after writing four real implementations. Now that variance is observable
(key suffix for effort vs others, opt-in vs always-fires defaults, silent
modes, classifier-driven Layer 4), it can be encoded as PlaneConfig fields.

Plane-specific behavior is captured in PlaneConfig. PlaneLoader is plane-
agnostic — touch the config to change what a plane does, touch this class
only to change machinery shared by all planes.

Naming notes:
  * effort uses "tier" (depth) suffix; other planes use "mode" (categorical).
    PlaneConfig.key_suffix controls this — "tier" for effort, "mode" otherwise.
  * Silent modes (e.g. conversation's "default"): resolve correctly so traces
    record the mode, but load_mode_content returns None — no scaffold injects.
    Matches the philosophy that "default conversation mode" is the absence of
    explicit framing, not a special mode.
  * Classifier-driven Layer 4 (effort only): when the turn classifier produces
    a shape with the named attribute, that becomes the resolved mode if none
    of the higher-priority layers (force / once / baseline) fired.

S11 plane isolation: PlaneConfig.valid_modes is NOT shared across planes.
Each plane file owns its own frozenset. Re-introducing cross-plane validation
collapse here would be the regression this refactor exists to prevent.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class PlaneConfig:
    """Per-plane configuration consumed by PlaneLoader.

    Fields define the plane's identity (plane_name, valid_modes), its
    activation semantics (default_mode = None for opt-in), its loading
    behavior (scaffolds_dir, silent_modes, flag_env), and its key
    conventions (key_suffix). Most fields are required; sensible defaults
    only for the optional behavior knobs.
    """

    plane_name: str
    valid_modes: frozenset[str]
    default_mode: str | None  # None = opt-in plane (no scaffold fires unless set)
    scaffolds_dir: Path
    flag_env: str
    key_suffix: str = "mode"  # "tier" for effort, "mode" elsewhere
    silent_modes: frozenset[str] = field(default_factory=frozenset)
    classifier_attr: str | None = None  # if set, _resolve checks _turn_shape.<attr>

    @property
    def baseline_key(self) -> str:
        return f"{self.plane_name}_{self.key_suffix}"

    @property
    def once_key(self) -> str:
        return f"{self.plane_name}_once_{self.key_suffix}"

    @property
    def resolved_config_key(self) -> str:
        return f"_resolved_{self.plane_name}_{self.key_suffix}"

    @property
    def force_config_key(self) -> str:
        return f"force_{self.plane_name}_{self.key_suffix}"

    @property
    def msg_field_key(self) -> str:
        return f"{self.plane_name}_{self.key_suffix}"


class PlaneLoader:
    """One compressed implementation of the plane-loader contract.

    Per-plane behavior is encoded in PlaneConfig; this class is plane-agnostic.
    Each plane module instantiates exactly one PlaneLoader and aliases its
    methods as module-level functions matching the historical public API.
    """

    def __init__(self, config: PlaneConfig):
        self.config = config
        self._world_state: Any = None

    def set_world_state(self, ws: Any) -> None:
        self._world_state = ws

    def valid_modes(self) -> frozenset[str]:
        return self.config.valid_modes

    def _flag_enabled(self) -> bool:
        raw = str(os.environ.get(self.config.flag_env, "1")).strip().lower()
        return raw in {"1", "true", "yes", "on"}

    def _resolve_mode(self, config: dict) -> str | None:
        """Layered fall-through (highest priority first):

          1. config[force_config_key]                       — testing/automation
          2. once-mode in world_state (consumed if present) — /<plane> once <mode>
          3. baseline mode in world_state                   — /<plane> <mode>
          4. system-side classifier (only if classifier_attr is set)
          5. config.default_mode                            — may be None
        """
        # Layer 1: explicit config override
        if isinstance(config, dict):
            forced = config.get(self.config.force_config_key)
            if isinstance(forced, str) and forced.lower() in self.config.valid_modes:
                return forced.lower()

        if self._world_state is not None:
            # Layer 2: once-override (consume on read)
            try:
                once_raw = self._world_state.state.get(self.config.once_key)
                if isinstance(once_raw, str) and once_raw.lower() in self.config.valid_modes:
                    self._world_state.state.pop(self.config.once_key, None)
                    self._world_state.mark_dirty()
                    return once_raw.lower()
            except Exception:
                pass

            # Layer 3: persistent baseline
            try:
                baseline_raw = self._world_state.state.get(self.config.baseline_key)
                if isinstance(baseline_raw, str) and baseline_raw.lower() in self.config.valid_modes:
                    return baseline_raw.lower()
            except Exception:
                pass

        # Layer 4: system-side classifier (only for planes that declare it — effort)
        if self.config.classifier_attr is not None and isinstance(config, dict):
            shape = config.get("_turn_shape")
            if shape is not None:
                try:
                    inferred = getattr(shape, self.config.classifier_attr, None)
                    if isinstance(inferred, str) and inferred.lower() in self.config.valid_modes:
                        return inferred.lower()
                except Exception:
                    pass

        # Layer 5: hardcoded default (None for opt-in planes)
        return self.config.default_mode

    def peek_mode(self) -> str | None:
        """Non-consuming sibling of _resolve_mode for observation surfaces (tag, footer).

        Returns the explicitly-set mode (once > baseline) without popping the
        once key. Force/classifier/default layers are intentionally skipped —
        observation surfaces should display only what the user or caller
        explicitly chose, not inferred or always-on values.

        Returns None when no explicit value is set. The interceptor's
        _resolve_mode remains the single consumer of the once key.
        """
        if self._world_state is None:
            return None
        try:
            once_raw = self._world_state.state.get(self.config.once_key)
            if isinstance(once_raw, str) and once_raw.lower() in self.config.valid_modes:
                return once_raw.lower()
        except Exception:
            pass
        try:
            baseline_raw = self._world_state.state.get(self.config.baseline_key)
            if isinstance(baseline_raw, str) and baseline_raw.lower() in self.config.valid_modes:
                return baseline_raw.lower()
        except Exception:
            pass
        return None

    def load_mode_content(self, mode: str | None) -> str | None:
        """Read scaffold for *mode*. Returns None for invalid, missing, or silent."""
        if mode is None or mode not in self.config.valid_modes:
            return None
        if mode in self.config.silent_modes:
            return None
        path = self.config.scaffolds_dir / f"{mode}.md"
        if not path.exists():
            return None
        try:
            text = path.read_text(encoding="utf-8").strip()
            return text or None
        except Exception:
            return None

    def interceptor(self, messages: list[dict], config: dict) -> list[dict] | None:
        """Inject the current mode's scaffold before the latest user turn.

        Returns None when:
          - flag is off
          - no mode resolved (opt-in default with nothing set)
          - resolved mode is silent or has no scaffold file
          - block is already present (source-field defense)
          - no non-ephemeral user message exists

        Side effect: writes resolved mode into config[resolved_config_key]
        BEFORE early-returns on missing scaffold so traces record the mode
        even when no scaffold was injected.
        """
        if not self._flag_enabled():
            return None
        mode = self._resolve_mode(config)
        if isinstance(config, dict):
            config[self.config.resolved_config_key] = mode
        if mode is None:
            return None
        content = self.load_mode_content(mode)
        if content is None:
            return None
        for msg in messages:
            if msg.get("source") == self.config.plane_name:
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
                "source": self.config.plane_name,
                self.config.msg_field_key: mode,
            },
        )
        return result

    def contribute_section(self, messages: list[dict], config: dict):
        """Section-contributor variant for the ephemeral_coalescer.

        Side effect: writes resolved mode into config[resolved_config_key]
        regardless of whether the section survives budget.
        """
        from core.ephemeral_coalescer import SectionResult
        if not self._flag_enabled():
            return None
        mode = self._resolve_mode(config)
        if isinstance(config, dict):
            config[self.config.resolved_config_key] = mode
        if mode is None:
            return None
        content = self.load_mode_content(mode)
        if content is None:
            return None
        return SectionResult(name=self.config.plane_name, text=content)
