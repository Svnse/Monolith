"""friction_calibrate — the offline falsification gate for the settler.

Runs core/intent_settle.settle over a held-out labeled set
(tools/friction_labels.jsonl): each row carries the FROZEN `prediction_set` and
E's verbatim `user_msg`. Reports AUC of friction_score vs the human miss-labels.
If the settler does not track the labels it is DECORATIVE — fix it before
MONOLITH_FRICTION_INJECT_V1 is ever set. External grounding: label provenance =
the human's judgment on held-out data, independent of the live loop.

The holdout MUST be weighted toward LONG-answer on-topic redirects — the
broad-set false-negative hides there; a set full of marked corrections
re-inflates AUC and proves nothing (see docs/roadmaps/FRICTION_RECALIBRATE.md).

Usage:
    python tools/friction_calibrate.py [--labels PATH] [--threshold 0.75]

Re-calibrate: edit tools/friction_labels.jsonl (set label_missed / label_source
to "e"), rerun.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

# allow running from repo root without install
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from core import intent_settle as isettle  # noqa: E402

_DEFAULT_LABELS = os.path.join(_REPO, "tools", "friction_labels.jsonl")
_DEFAULT_THRESHOLD = 0.75


def load_labels(path: str) -> list[dict]:
    rows: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("//"):
                continue
            rows.append(json.loads(line))
    return rows


def compute_auc(scores: list[float], labels: list[int]) -> float:
    """ROC AUC via the rank (Mann-Whitney U) statistic with tie-averaged ranks.

    labels are binary (1 = missed / positive). Returns NaN if a class is empty."""
    pos = [s for s, y in zip(scores, labels) if y == 1]
    neg = [s for s, y in zip(scores, labels) if y == 0]
    n_pos, n_neg = len(pos), len(neg)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    # average ranks over all scores
    paired = sorted(zip(scores, labels), key=lambda t: t[0])
    ranks = [0.0] * len(paired)
    i = 0
    while i < len(paired):
        j = i
        while j + 1 < len(paired) and paired[j + 1][0] == paired[i][0]:
            j += 1
        avg = (i + j) / 2.0 + 1.0  # 1-based average rank
        for k in range(i, j + 1):
            ranks[k] = avg
        i = j + 1
    sum_pos_ranks = sum(r for r, (_, y) in zip(ranks, paired) if y == 1)
    auc = (sum_pos_ranks - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return auc


def run(path: str = _DEFAULT_LABELS, threshold: float = _DEFAULT_THRESHOLD,
        verbose: bool = True) -> dict:
    rows = load_labels(path)
    scores: list[float] = []
    labels: list[int] = []
    detail = []
    for r in rows:
        res = isettle.settle(r.get("prediction_set"), r.get("user_msg", ""))
        y = 1 if int(r.get("label_missed", 0)) > 0 else 0
        scores.append(res.friction_score)
        labels.append(y)
        detail.append((r.get("id", "?"), y, round(res.friction_score, 3), res.friction_type))
    auc = compute_auc(scores, labels)
    n_pos = sum(labels)
    result = {"n": len(rows), "n_missed": n_pos, "n_ok": len(rows) - n_pos,
              "auc": auc, "threshold": threshold,
              "passes": (auc == auc and auc >= threshold)}  # auc==auc filters NaN
    if verbose:
        circular = any(str(r.get("label_source", "")).lower() != "e" for r in rows)
        print(f"friction MEMBERSHIP settler calibration — n={result['n']} "
              f"(missed={result['n_missed']}, ok={result['n_ok']})")
        print(f"  AUC = {auc:.3f}   threshold = {threshold}   "
              f"{'>= threshold' if result['passes'] else 'BELOW threshold'}")
        if circular:
            print("  ** PROVISIONAL / CIRCULAR — labels are not all 'e' and the differ "
                  "was tuned on these rows. This is a CONSISTENCY CHECK, NOT validation.")
            print("  ** Real grade: relabel with label_source='e' on a FRESH holdout the "
                  "differ was never tuned on. See docs/roadmaps/FRICTION_RECALIBRATE.md.")
        print("  id                         label  score  type")
        for cid, y, sc, ty in detail:
            flag = "MISS" if y else "ok  "
            print(f"   {str(cid)[:24]:<24}  {flag}  {sc:>5}  {ty}")
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default=_DEFAULT_LABELS)
    ap.add_argument("--threshold", type=float, default=_DEFAULT_THRESHOLD)
    args = ap.parse_args()
    res = run(args.labels, args.threshold)
    return 0 if res["passes"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
