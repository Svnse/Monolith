"""Tests for stateless_reframe.session_asks — server-session ask extraction.

The agent-server / connect session stores the human/examiner ask as role="agent"
with the text in a `text` field (content is None) and an "[AgentName]" prefix —
which drift.recent_asks (role=="user", reads content) misses entirely. This
extractor handles that shape AND the GUI shape (role=="user", content=str).
"""
from __future__ import annotations

import pytest


def test_extracts_agent_text_shape_and_strips_name_prefix():
    from addons.system.bearing import stateless_reframe as sr
    msgs = [
        {"role": "agent", "content": None,
         "text": "[Claude (examiner)] \n\nIn two lines, what does a WAL do?"},
        {"role": "assistant", "content": None, "text": "A WAL records changes..."},
    ]
    asks = sr.session_asks(msgs)
    assert asks == ["In two lines, what does a WAL do?"]


def test_backcompat_gui_user_content_shape():
    from addons.system.bearing import stateless_reframe as sr
    msgs = [
        {"role": "user", "content": "how does a mutex work"},
        {"role": "assistant", "content": "a mutex is..."},
    ]
    assert sr.session_asks(msgs) == ["how does a mutex work"]


def test_skips_assistant_system_and_empty():
    from addons.system.bearing import stateless_reframe as sr
    msgs = [
        {"role": "system", "text": "system block"},
        {"role": "assistant", "text": "answer"},
        {"role": "agent", "text": "   "},                 # empty -> skipped
        {"role": "agent", "text": "the real ask"},
    ]
    assert sr.session_asks(msgs) == ["the real ask"]


def test_returns_last_k_oldest_first():
    from addons.system.bearing import stateless_reframe as sr
    msgs = [
        {"role": "agent", "text": "ask one"},
        {"role": "agent", "text": "ask two"},
        {"role": "agent", "text": "ask three"},
    ]
    assert sr.session_asks(msgs, k=2) == ["ask two", "ask three"]


def test_strips_channel_prefix_too():
    from addons.system.bearing import stateless_reframe as sr
    msgs = [{"role": "agent", "text": "[CHANNEL: connect] actual question here"}]
    assert sr.session_asks(msgs) == ["actual question here"]


def test_empty_on_no_asks():
    from addons.system.bearing import stateless_reframe as sr
    assert sr.session_asks([{"role": "assistant", "text": "x"}]) == []
    assert sr.session_asks([]) == []
