"""Does EMBEDDING cosine separate frame-drift better than lexical overlap?

Reuses core.identity_alignment's embed backend (all-MiniLM-L6-v2) to score
similarity(current_frame, user_ask) for the 188 judged pairs, then sweeps
(age_threshold x sim_threshold) vs the labels. If F1/precision beats the lexical
~0.60/0.45 ceiling, the live detector should use embed.
"""
from __future__ import annotations

import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))

from core import identity_alignment as ia  # noqa: E402

_PAIRS = os.path.join(_HERE, "drift_pairs.json")
_LABELS = os.path.join(_HERE, "judge_labels.json")
_SIMS = os.path.join(_HERE, "embed_sims.json")


def main() -> None:
    with open(_PAIRS, encoding="utf-8") as f:
        pairs = json.load(f)
    with open(_LABELS, encoding="utf-8") as f:
        labels = {v["seq"]: v["verdict"] for v in json.load(f)}

    # cache the (slow) embedding sims so threshold sweeps are instant on re-run
    if os.path.exists(_SIMS):
        with open(_SIMS, encoding="utf-8") as f:
            sims = json.load(f)
    else:
        print("encoding 188 pairs with all-MiniLM-L6-v2 (first run loads the model)...")
        sims = [ia.compute_identity_alignment(p["current_frame"], p["user_ask"], backend="embed")
                for p in pairs]
        with open(_SIMS, "w", encoding="utf-8") as f:
            json.dump(sims, f)
        print("done.\n")

    gold_pos = sum(1 for p in pairs if labels.get(p["seq"]) == "DRIFT_STALE")
    print(f"pairs {len(pairs)} | gold DRIFT_STALE {gold_pos} ({gold_pos/len(pairs)*100:.1f}%)")
    print(f"embed cosine[0,1]: min {min(sims):.2f} max {max(sims):.2f} mean {sum(sims)/len(sims):.2f}\n")

    print(f"{'age':>3} {'sim<':>5} | {'flag':>4} {'prec':>5} {'rec':>5} {'F1':>5}  (tp fp fn tn)")
    best = None
    for age_t in (2, 3, 4, 6, 8):
        for st in (0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45):
            tp = fp = fn = tn = 0
            for p, sim in zip(pairs, sims):
                pred = (p.get("frame_run_len", 1) >= age_t) and (sim < st)
                gold = labels.get(p["seq"]) == "DRIFT_STALE"
                if pred and gold:
                    tp += 1
                elif pred and not gold:
                    fp += 1
                elif (not pred) and gold:
                    fn += 1
                else:
                    tn += 1
            prec = tp / (tp + fp) if (tp + fp) else 0.0
            rec = tp / (tp + fn) if (tp + fn) else 0.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
            print(f"{age_t:>3} {st:>5.2f} | {tp+fp:>4} {prec:>5.2f} {rec:>5.2f} {f1:>5.2f}  ({tp} {fp} {fn} {tn})")
            if best is None or f1 > best[0]:
                best = (f1, age_t, st, prec, rec, tp + fp)
        print()

    f1, age_t, st, prec, rec, flag = best
    print(f"BEST F1={f1:.2f} at age>={age_t}, embed_sim<{st}: "
          f"precision {prec:.2f}, recall {rec:.2f}, flags {flag} ({flag/len(pairs)*100:.0f}%)")


if __name__ == "__main__":
    main()
