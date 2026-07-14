from __future__ import annotations

import threading
import time

import pytest

from core import log_mirror as lm


@pytest.fixture(autouse=True)
def _isolated_ring():
    """Each test gets a fresh ring buffer. Default cap 1000."""
    lm.reset_ring(capacity=1000)
    yield
    lm.reset_ring(capacity=1000)


def test_empty_ring_tail_returns_empty() -> None:
    assert lm.get_ring().tail() == []


def test_append_assigns_monotonic_seq() -> None:
    ring = lm.get_ring()
    s1 = ring.append("chat", "trace", "line one")
    s2 = ring.append("chat", "debug", "line two")
    s3 = ring.append("engine", "trace", "line three")
    assert s1 < s2 < s3
    lines = ring.tail()
    assert [l["text"] for l in lines] == ["line one", "line two", "line three"]
    assert [l["seq"] for l in lines] == [s1, s2, s3]


def test_tail_limits_to_n() -> None:
    ring = lm.get_ring()
    for i in range(20):
        ring.append("chat", "trace", f"line {i}")
    last_five = ring.tail(5)
    assert len(last_five) == 5
    assert last_five[0]["text"] == "line 15"
    assert last_five[-1]["text"] == "line 19"


def test_capacity_enforced_old_lines_fall_off() -> None:
    lm.reset_ring(capacity=5)
    ring = lm.get_ring()
    for i in range(10):
        ring.append("chat", "trace", f"line {i}")
    lines = ring.tail()
    assert len(lines) == 5
    assert [l["text"] for l in lines] == [f"line {i}" for i in range(5, 10)]
    # seq survives rollover
    assert ring.latest_seq() == 10
    assert ring.head_seq() == 6


def test_since_returns_newer_lines_only() -> None:
    ring = lm.get_ring()
    for i in range(5):
        ring.append("chat", "trace", f"line {i}")
    seq_after_three = ring.tail()[2]["seq"]
    newer = ring.since(seq_after_three)
    assert len(newer) == 2
    assert [l["text"] for l in newer] == ["line 3", "line 4"]


def test_since_zero_returns_everything_in_buffer() -> None:
    ring = lm.get_ring()
    for i in range(3):
        ring.append("chat", "trace", f"line {i}")
    assert len(ring.since(0)) == 3


def test_since_detects_gap_via_head_seq() -> None:
    lm.reset_ring(capacity=3)
    ring = lm.get_ring()
    for i in range(10):
        ring.append("chat", "trace", f"line {i}")
    # caller's last-seen seq is 5; head_seq is 8 (only lines 8,9,10 remain)
    lines = ring.since(5)
    assert lines[0]["seq"] > 5 + 1  # gap signal: head > requested+1
    assert ring.head_seq() == 8


def test_append_is_threadsafe() -> None:
    ring = lm.get_ring()
    n_threads = 8
    per_thread = 50
    barrier = threading.Barrier(n_threads)

    def _worker(tid: int) -> None:
        barrier.wait()
        for i in range(per_thread):
            ring.append("chat", "trace", f"t{tid}-i{i}")

    threads = [threading.Thread(target=_worker, args=(t,)) for t in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert ring.latest_seq() == n_threads * per_thread
    # No duplicates, no gaps (best-effort check across all observed seqs)
    seqs = [l["seq"] for l in ring.tail(n=n_threads * per_thread)]
    assert seqs == sorted(seqs)
    assert len(seqs) == len(set(seqs))


def test_connect_signals_attaches_handler() -> None:
    class _FakeSignal:
        def __init__(self):
            self._slots = []

        def connect(self, fn):
            self._slots.append(fn)

        def emit(self, text):
            for fn in self._slots:
                fn(text)

    class _FakeEmitter:
        sig_trace = _FakeSignal()

    emitter = _FakeEmitter()
    lm.connect_signals((emitter, "sig_trace", "fake", "trace"))
    emitter.sig_trace.emit("hello")
    emitter.sig_trace.emit("world")
    lines = lm.get_ring().tail()
    assert len(lines) == 2
    assert lines[0]["source"] == "fake"
    assert lines[0]["text"] == "hello"
    assert lines[1]["text"] == "world"


def test_connect_signals_skips_missing_attrs() -> None:
    class _Empty:
        pass

    # Should not raise
    lm.connect_signals((_Empty(), "sig_nonexistent", "fake", "trace"))
    lm.connect_signals((None, "sig_trace", "fake", "trace"))
    assert lm.get_ring().tail() == []


def test_clear_keeps_seq_counter() -> None:
    ring = lm.get_ring()
    ring.append("chat", "trace", "a")
    ring.append("chat", "trace", "b")
    assert ring.latest_seq() == 2
    ring.clear()
    assert ring.tail() == []
    # seq does not reset — preserves "since(seq)" semantics across clears
    ring.append("chat", "trace", "c")
    assert ring.latest_seq() == 3
