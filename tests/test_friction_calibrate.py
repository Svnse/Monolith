"""Tests for tools/friction_calibrate — the offline falsification gate.

Confirms AUC is computed correctly and that run() scores the MEMBERSHIP settler
over prediction_set-bearing rows (the gate that blocks decorative deployment).
"""
from __future__ import annotations

import importlib.util
import json
import os

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CAL = os.path.join(_REPO, "tools", "friction_calibrate.py")
_spec = importlib.util.spec_from_file_location("friction_calibrate", _CAL)
cal = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cal)


def test_compute_auc_perfect_separation():
    # all positives outscore all negatives -> AUC 1.0
    assert cal.compute_auc([0.9, 0.8, 0.2, 0.1], [1, 1, 0, 0]) == 1.0
    # inverted -> 0.0
    assert cal.compute_auc([0.1, 0.2, 0.8, 0.9], [1, 1, 0, 0]) == 0.0


def test_compute_auc_single_class_is_nan():
    auc = cal.compute_auc([0.5, 0.6], [1, 1])
    assert auc != auc  # NaN


def test_run_scores_membership_settler(tmp_path):
    rows = [
        # on-topic redirect: salient focus outside the staked set -> high friction
        {"id": "r1", "prediction_set": {"referents": ["token", "rotation", "session"],
                                        "directions": [{"move": "plan", "referent": "token"}]},
         "user_msg": "what's the database migration schema rollout for downstream services",
         "label_missed": 1, "label_source": "provisional"},
        # clean uptake -> low friction
        {"id": "r2", "prediction_set": {"referents": ["token", "rotation", "session"],
                                        "directions": [{"move": "plan", "referent": "token"}]},
         "user_msg": "yes let's refine the token rotation and session handling",
         "label_missed": 0, "label_source": "provisional"},
        # explicit correction -> high friction
        {"id": "r3", "prediction_set": {"referents": ["token"], "directions": []},
         "user_msg": "no, that's wrong, drop the token idea",
         "label_missed": 1, "label_source": "provisional"},
    ]
    p = tmp_path / "labels.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    res = cal.run(str(p), threshold=0.75, verbose=False)
    assert res["n"] == 3 and res["n_missed"] == 2 and res["n_ok"] == 1
    # the settler separates these cleanly -> AUC 1.0, gate passes
    assert res["auc"] == 1.0 and res["passes"] is True


def test_run_gate_fails_below_threshold(tmp_path):
    # a settler that can't separate (here: two identical-scoring rows, opposite labels)
    rows = [
        {"id": "a", "prediction_set": {"referents": ["x"], "directions": []},
         "user_msg": "ok", "label_missed": 1, "label_source": "provisional"},
        {"id": "b", "prediction_set": {"referents": ["x"], "directions": []},
         "user_msg": "sure", "label_missed": 0, "label_source": "provisional"},
    ]
    p = tmp_path / "labels.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    res = cal.run(str(p), threshold=0.75, verbose=False)
    # both score 'unresolved' (0.4) -> tied -> AUC 0.5 -> gate fails
    assert res["passes"] is False
