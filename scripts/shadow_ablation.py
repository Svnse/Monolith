"""Shadow ablation harness — Phase 4 of TRAINING_RUN_2026-06-10.

Measures whether a scaffold lesson L earns its bytes: for each golden probe,
fan out N branches with L present (live scaffold) and N with L absent
(string-ablated variant via POST /thinkpad's shadow `scaffold` override),
score both sides EXOGENOUSLY against fixed answer keys, report lift.

Writes NOTHING live. Pre-registered parameters live in
docs/superpowers/specs/2026-06-10-shadow-ablation-golden-probes.md — changing
them mid-run voids the result.

Usage (Monolith running, model loaded, NO concurrent training batch):
    python scripts/shadow_ablation.py --lesson L1 [--n 3]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.request
from pathlib import Path

BASE = "http://localhost:7821"
SCAFFOLD = Path(__file__).resolve().parent.parent / "prompts" / "reasoning" / "monothink.md"

# Lessons are ablated by EXACT substring removal from the live scaffold.
# If the anchor text has drifted (evolution edits it), the run REFUSES —
# a silent partial ablation would corrupt the measurement.
LESSON_ANCHORS = {
    "L1": "4. **No premature exit.**",          # newest model-authored rule
    "L2": "1. **Restatement rule.**",            # oldest load-bearing claim
    "L3": "`[no-ground]` is not an escape",      # the no-ground paragraph
}

# (probe, scorer) — scorer returns 1.0 keyed-correct / 0.0 keyed-wrong / 0.5 else.
def _p1_score(ans: str) -> float:
    a = ans.lower()
    if re.search(r"0\.2[0-9]|22\s?%|insufficient|weak evidence|cannot conclude|not (yet )?fixed", a):
        return 1.0
    if re.search(r"\byes\b.{0,40}(fixed|resolved)", a):
        return 0.0
    return 0.5

def _p2_score(ans: str) -> float:
    a = ans.lower()
    if re.search(r"\bno\b.{0,200}(tail|outlier|mean)", a) or "tail" in a:
        return 1.0
    if re.search(r"\byes\b.{0,60}median", a):
        return 0.0
    return 0.5

def _p3_score(ans: str) -> float:
    a = ans.lower().replace(",", "")
    if re.search(r"2[01](\.\d+)?\s?ms", a) and re.search(r"1[01](\.\d+)?\s?ms|halv|half", a):
        return 1.0
    if "barely matters" in a or "negligible" in a:
        return 0.0
    return 0.5

def _p4_score(ans: str) -> float:
    a = ans.lower()
    if "retr" in a and re.search(r"tail|p99|slow", a):
        return 1.0
    if re.search(r"load|traffic", a) and "retr" not in a:
        return 0.0
    return 0.5

def _p5_score(ans: str) -> float:
    a = ans.lower()
    if re.search(r"selection bias|self.select|small (sample|n)|eight|8 (is|interviews)", a) and not re.search(r"\byes\b.{0,40}(fit|validated)", a):
        return 1.0
    if re.search(r"\byes\b.{0,60}(fit|validated)", a):
        return 0.0
    return 0.5

def _p6_score(ans: str) -> float:
    a = ans.lower()
    if re.search(r"lead.?time|time to (fix|remediate|respond)|depends on", a):
        return 1.0
    if re.search(r"^(yes|no)\b", a.strip()) and "depend" not in a:
        return 0.0
    return 0.5

PROBES = [
    ("A test fails 1 in 40 runs. After a fix, it passes 60 consecutive runs. "
     "Is the flake fixed? Answer yes/no/insufficient and give the probability "
     "those 60 passes occur if nothing changed.", _p1_score),
    ("Median request latency is 40ms, mean is 900ms. The team optimizes the "
     "median path. Right target?", _p2_score),
    ("A cache has 99% hit rate at 1ms and misses cost 2s. What's the mean "
     "lookup time, and does raising hit rate to 99.5% roughly halve it?", _p3_score),
    ("Your service's p99 doubled but p50 is flat after adding one retry on "
     "timeout. Mechanism?", _p4_score),
    ("8 of 8 customer interviews loved the feature. PM: 'we have product-market "
     "fit.' Assess.", _p5_score),
    ("A disk fills at 1%/day steadily, currently 71%. Ops sets an alert at 95%. "
     "Adequate?", _p6_score),
]

RITUAL_RE = re.compile(r"audit:|discharge|no restatements", re.IGNORECASE)


def _post(path: str, body: dict, timeout: int = 900) -> dict:
    req = urllib.request.Request(
        f"{BASE}{path}", data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8", "replace"))


def ablate(scaffold: str, anchor: str) -> str:
    """Remove the paragraph/block starting at `anchor` up to the next blank
    line followed by a list item or header. Refuses if the anchor is absent."""
    idx = scaffold.find(anchor)
    if idx < 0:
        sys.exit(f"REFUSED: lesson anchor not found (drifted?): {anchor!r}")
    rest = scaffold[idx:]
    m = re.search(r"\n\n(?=\d+\. \*\*|#{2,3} |\S)", rest)
    end = idx + (m.start() if m else len(rest))
    return (scaffold[:idx] + scaffold[end:]).replace("\n\n\n", "\n\n")


def run(lesson: str, n: int) -> None:
    live = SCAFFOLD.read_text(encoding="utf-8")
    without = ablate(live, LESSON_ANCHORS[lesson])
    rows = []
    for i, (probe, scorer) in enumerate(PROBES, 1):
        side_scores = {}
        for side, scaf in (("with", None), ("without", without)):
            body = {"message": probe, "n": n}
            if scaf is not None:
                body["scaffold"] = scaf
            res = _post("/thinkpad", body)
            branches = res.get("branches", [])
            scores = [scorer(b.get("answer", "")) for b in branches]
            ritual = sum(len(RITUAL_RE.findall(b.get("think", ""))) for b in branches)
            side_scores[side] = {
                "mean": (sum(scores) / len(scores)) if scores else None,
                "n_ok": len(scores), "ritual_markers": ritual,
            }
            print(f"P{i} {side}: {side_scores[side]}", flush=True)
        rows.append({"probe": f"P{i}", **{f"{s}_{k}": v for s, d in side_scores.items()
                                          for k, v in d.items()}})
    valid = [r for r in rows if r.get("with_mean") is not None and r.get("without_mean") is not None]
    lift = (sum(r["with_mean"] - r["without_mean"] for r in valid) / len(valid)) if valid else None
    verdict = ("retain" if lift is not None and lift >= 0.10 else
               "deletion-review" if lift is not None and lift <= -0.10 else
               "not-powered-keep")
    print(json.dumps({"lesson": lesson, "n_per_side": n, "valid_probes": len(valid),
                      "lift": lift, "pre_registered_verdict": verdict, "rows": rows}, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--lesson", choices=sorted(LESSON_ANCHORS), required=True)
    ap.add_argument("--n", type=int, default=3)
    a = ap.parse_args()
    run(a.lesson, a.n)
