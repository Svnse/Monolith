"""Tests for the semantic-drift detector (addons/system/bearing/drift.py).

The detector flags when bearing.current_frame has gone STALE relative to what
the user is actually now asking — the "confident-wrong-on-stale-frame" case
measured at ~40% of outer turns (tools/frame_drift). Pure function; no I/O.

Runnable without pytest:  python tests/test_bearing_drift.py   (from repo root)
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from addons.system.bearing import drift  # noqa: E402


def test_drift_fires_on_aged_low_overlap_frame():
    obs = drift.detect_drift(
        current_frame="Presented claim warrant graph structure with two node classes and rendering truncation",
        recent_asks=["What would you like for yourself if I could grant that one thing?"],
        frame_age=10,
    )
    assert obs.is_drift is True
    assert obs.overlap < 0.18


def test_no_drift_when_ask_matches_frame_topic():
    obs = drift.detect_drift(
        current_frame="Helping E understand the bearing and frame system in the codebase",
        recent_asks=["find anything related to frame in the bearing system"],
        frame_age=10,
    )
    assert obs.is_drift is False
    assert obs.overlap >= 0.18


def test_no_drift_when_frame_is_young():
    obs = drift.detect_drift(
        current_frame="Presented claim warrant graph structure with two node classes",
        recent_asks=["What would you like for yourself if I could grant that one thing?"],
        frame_age=1,
    )
    assert obs.is_drift is False


def test_no_drift_on_empty_frame():
    obs = drift.detect_drift(current_frame="   ", recent_asks=["anything at all here"], frame_age=50)
    assert obs.is_drift is False


def test_no_drift_when_no_recent_asks():
    obs = drift.detect_drift(current_frame="a specific frame about warrant graphs", recent_asks=[], frame_age=50)
    assert obs.is_drift is False


def test_overlap_ignores_stopwords():
    # frame and ask share only stopwords/short words -> ~zero content overlap -> drift (aged).
    obs = drift.detect_drift(
        current_frame="database migration runbook for the production incident",
        recent_asks=["what is the meaning of authenticity in trained values"],
        frame_age=8,
    )
    assert obs.is_drift is True
    assert obs.overlap < 0.18


def test_recent_asks_union_counts_any_match():
    # overlap is computed against the UNION of recent asks; a match in any clears it.
    obs = drift.detect_drift(
        current_frame="database migration runbook production incident remediation",
        recent_asks=[
            "unrelated philosophical question about values",
            "walk me through the database migration runbook for the incident",
        ],
        frame_age=8,
    )
    assert obs.is_drift is False


# ── recent_asks (live message extraction) ───────────────────────────────────
def test_recent_asks_returns_last_k_real_user_asks():
    msgs = [
        {"role": "user", "content": "first question about cats"},
        {"role": "assistant", "content": "..."},
        {"role": "user", "content": "[RUNTIME STATE] ambient", "ephemeral": True, "source": "ephemeral_coalescer"},
        {"role": "user", "content": "Tool results: [tool_result_1] ok"},
        {"role": "user", "content": "[CHANNEL: USER] the real latest question about dogs"},
    ]
    assert drift.recent_asks(msgs, k=2) == ["first question about cats", "the real latest question about dogs"]


def test_recent_asks_skips_tool_and_ephemeral_blocks():
    msgs = [
        {"role": "user", "content": "real ask here please"},
        {"role": "user", "content": "Tool results: stuff"},
        {"role": "user", "content": "block text", "ephemeral": True},
    ]
    assert drift.recent_asks(msgs, k=3) == ["real ask here please"]


# ── drift_observe ledger (flag-gated, observe-only) ──────────────────────────
def _tmp_ledger(name: str):
    import pathlib
    import tempfile
    p = pathlib.Path(tempfile.gettempdir()) / name
    if p.exists():
        p.unlink()
    return p


def test_drift_observe_noop_when_flag_off():
    from addons.system.bearing import drift_observe
    os.environ.pop("MONOLITH_FRAME_DRIFT_V1", None)
    p = _tmp_ledger("drift_test_off.jsonl")
    drift_observe._LEDGER = p
    drift_observe.record("t1", "a specific frame about warrant graphs",
                         [{"role": "user", "content": "an unrelated question about cats"}])
    assert not p.exists()


def test_drift_observe_writes_row_with_overlap_when_flag_on():
    import json
    from addons.system.bearing import drift_observe
    os.environ["MONOLITH_FRAME_DRIFT_V1"] = "1"
    try:
        p = _tmp_ledger("drift_test_on.jsonl")
        drift_observe._LEDGER = p
        drift_observe.record("t1", "database migration runbook production incident",
                             [{"role": "user", "content": "what is authenticity in trained values"}])
        assert p.exists()
        rows = [json.loads(ln) for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]
        assert len(rows) == 1
        assert rows[0]["turn_id"] == "t1"
        assert rows[0]["overlap"] < 0.18
    finally:
        os.environ.pop("MONOLITH_FRAME_DRIFT_V1", None)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"  PASS {fn.__name__}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"  FAIL {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
