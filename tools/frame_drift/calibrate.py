"""Calibrate the cheap runtime drift detector against the LLM-judged labels.

Sweeps (age_threshold x overlap_threshold) for addons/system/bearing/drift and
reports precision / recall / F1 vs the 188 semantic judge labels (DRIFT_STALE =
positive). This is the offline observe->compare step that sets the live
detector's defaults before any nudge fires.

Labels are read from the judging workflow output (saved once to judge_labels.json).
"""
from __future__ import annotations

import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))

from addons.system.bearing import drift  # noqa: E402

_PAIRS = os.path.join(_HERE, "drift_pairs.json")
_LABELS = os.path.join(_HERE, "judge_labels.json")
_WF_OUT = os.environ.get("MONOLITH_FRAME_DRIFT_LABEL_SOURCE", "")


def _load_labels() -> dict[int, str]:
    if not os.path.exists(_LABELS):
        if not _WF_OUT or not os.path.isfile(_WF_OUT):
            raise FileNotFoundError(
                "judge_labels.json is missing; place it beside calibrate.py or "
                "set MONOLITH_FRAME_DRIFT_LABEL_SOURCE to the judging-workflow output"
            )
        with open(_WF_OUT, encoding="utf-8") as f:
            wf = json.load(f)
        verdicts = wf["result"]["all_verdicts"]
        with open(_LABELS, "w", encoding="utf-8") as f:
            json.dump(verdicts, f, indent=2)
    with open(_LABELS, encoding="utf-8") as f:
        return {v["seq"]: v["verdict"] for v in json.load(f)}


def _prf(pairs, labels, age_t, ov_t):
    tp = fp = fn = tn = 0
    for p in pairs:
        obs = drift.detect_drift(p["current_frame"], [p["user_ask"]], p.get("frame_run_len", 1),
                                 age_threshold=age_t, overlap_threshold=ov_t)
        gold = labels.get(p["seq"]) == "DRIFT_STALE"
        pred = obs.is_drift
        if pred and gold:
            tp += 1
        elif pred and not gold:
            fp += 1
        elif not pred and gold:
            fn += 1
        else:
            tn += 1
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return tp, fp, fn, tn, prec, rec, f1


def main() -> None:
    with open(_PAIRS, encoding="utf-8") as f:
        pairs = json.load(f)
    labels = _load_labels()
    gold_pos = sum(1 for p in pairs if labels.get(p["seq"]) == "DRIFT_STALE")
    print(f"pairs {len(pairs)} | gold DRIFT_STALE {gold_pos} ({gold_pos/len(pairs)*100:.1f}%)\n")

    print(f"{'age':>3} {'ovl':>5} | {'flag':>4} {'prec':>5} {'rec':>5} {'F1':>5}  (tp fp fn tn)")
    best = None
    for age_t in (2, 3, 4, 6, 8):
        for ov_t in (0.04, 0.06, 0.08, 0.10, 0.12, 0.15, 0.18):
            tp, fp, fn, tn, prec, rec, f1 = _prf(pairs, labels, age_t, ov_t)
            flag = tp + fp
            print(f"{age_t:>3} {ov_t:>5.2f} | {flag:>4} {prec:>5.2f} {rec:>5.2f} {f1:>5.2f}  ({tp} {fp} {fn} {tn})")
            if best is None or f1 > best[0]:
                best = (f1, age_t, ov_t, prec, rec, flag)
        print()

    f1, age_t, ov_t, prec, rec, flag = best
    print(f"BEST F1={f1:.2f} at age>={age_t}, overlap<{ov_t}: "
          f"precision {prec:.2f}, recall {rec:.2f}, flags {flag} ({flag/len(pairs)*100:.0f}%)")


if __name__ == "__main__":
    main()
