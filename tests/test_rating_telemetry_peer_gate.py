"""Stage A: [RATING TELEMETRY] is suppressed on CONNECT peer turns.

The observer-effect guard — a peer/trainer LLM (the examiner) must not see
Monolith's own rating telemetry, or it performs to its grades and warms a cold
training session. UI (E) turns keep telemetry.
"""
from __future__ import annotations

from core import rating_telemetry as rt


def test_is_peer_turn_detects_connect_channel() -> None:
    assert rt._is_peer_turn(
        [{"role": "user", "content": "[CHANNEL: connect/Claude, /chat blocking] do X"}]
    ) is True
    assert rt._is_peer_turn(
        [{"role": "user", "content": "[CHANNEL: USER] hey"}]
    ) is False
    assert rt._is_peer_turn([]) is False
    # ephemeral-injected messages are skipped; the real user turn decides
    msgs = [
        {"role": "user", "content": "[CHANNEL: connect/X] q"},
        {"role": "user", "content": "[RUNTIME STATE] ...", "ephemeral": True},
    ]
    assert rt._is_peer_turn(msgs) is True


def test_telemetry_suppressed_on_peer_turn_kept_on_ui(monkeypatch) -> None:
    monkeypatch.setattr(rt, "_flag_enabled", lambda: True)
    monkeypatch.setattr(rt, "render_telemetry_block", lambda **k: "[RATING TELEMETRY] avg 70")
    peer = [{"role": "user", "content": "[CHANNEL: connect/Claude] probe"}]
    ui = [{"role": "user", "content": "[CHANNEL: USER] question"}]
    # peer: both injection paths suppress
    assert rt.contribute_section(peer, {}) is None
    assert rt.rating_telemetry_interceptor(peer, {}) is None
    # UI: still injected
    assert rt.contribute_section(ui, {}) is not None
    assert rt.rating_telemetry_interceptor(ui, {}) is not None
