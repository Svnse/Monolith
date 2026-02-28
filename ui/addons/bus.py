"""
ui/addons/bus.py  —  AddonEventBus
Lightweight Qt pub/sub that lets addons subscribe to engine events and
publish to each other without hard-wiring direct signal connections.

Channel naming convention
──────────────────────────
  "{engine_key}:{event_type}"   e.g. "vision:image", "audio:transcription"
  "system:{event}"              e.g. "system:agent_step"

Typical usage inside an addon factory
──────────────────────────────────────
  def setup(ctx: AddonContext):
      # React to vision generating an image
      ctx.bus.subscribe("vision:image", lambda p: inject_to_agent(p["image"]))

      # Feed mic transcription into the code input field
      ctx.bus.subscribe("audio:transcription",
                        lambda p: ctx.pages["code"].inject_text(p["text"]))

      # Publish from within the addon
      ctx.bus.publish("my_addon:ready", {"status": "ok"})

Bridging multiple engines
──────────────────────────
  # vision output → TTS narration
  ctx.bus.subscribe("vision:image",
      lambda p: ctx.bus.publish("audio:speak",
                                {"text": "Image generated successfully."}))

  # guard wires engine events into the bus automatically (see guard.py)
"""
from __future__ import annotations

from typing import Callable
from PySide6.QtCore import QObject, Signal


class AddonEventBus(QObject):
    """
    Simple channel-based pub/sub built on a single Qt signal.

    Thread safety: publish() must be called from the Qt main thread
    (same rule as all Qt signal emissions).
    """

    # Single broadcast signal — subscribers filter by channel
    sig_event = Signal(str, dict)   # (channel, payload)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._subs: dict[str, list[Callable[[dict], None]]] = {}

    # ── subscribe ─────────────────────────────────────────────────────────

    def subscribe(self, channel: str, fn: Callable[[dict], None]) -> None:
        """
        Register *fn* to be called whenever *channel* is published.
        Multiple subscribers per channel are supported.
        fn receives the payload dict.
        """
        self._subs.setdefault(channel, []).append(fn)

    def unsubscribe(self, channel: str, fn: Callable[[dict], None]) -> None:
        subs = self._subs.get(channel)
        if subs and fn in subs:
            subs.remove(fn)

    # ── publish ───────────────────────────────────────────────────────────

    def publish(self, channel: str, payload: dict | None = None) -> None:
        """
        Broadcast *payload* on *channel*.
        - Emits sig_event(channel, payload) for any Qt-connected listeners.
        - Calls all Python callbacks registered via subscribe().
        """
        data = payload or {}
        self.sig_event.emit(channel, data)
        for fn in list(self._subs.get(channel, [])):
            try:
                fn(data)
            except Exception:
                pass   # subscriber errors must not kill the bus

    # ── engine bridge helpers ─────────────────────────────────────────────

    def wire_engine(self, engine_key: str, engine_obj) -> None:
        """
        Connect an EngineProcess.sig_event to the bus, fanning its raw
        events out as "{engine_key}:{event_type}" channels.

        Called automatically by MonoGuard when an engine is registered.
        """
        if not hasattr(engine_obj, "sig_event"):
            return

        def _on_event(event: dict, ek: str = engine_key) -> None:
            kind = str(event.get("event") or "unknown")
            self.publish(f"{ek}:{kind}", event)

        engine_obj.sig_event.connect(_on_event)
