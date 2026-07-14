"""Classifier-accuracy eval — the upper-half BRANCH falsifier.

The divergence pilot proved the LOWER half: forced frames steer the answer (and
DeepSeek is frame-obedient — it follows a forced frame even into the wrong
answer). That makes the classify pass load-bearing: a wrong frame is executed
confidently, not self-corrected. This harness measures the UPPER half — does the
classifier, reading natural task text, pick the type whose forced-frame branch
passes?

It runs the classify pass on BOTH the cued prompt and the cue-NEUTRAL twin and
reports two accuracies that are NEVER pooled (R3): cued accuracy and neutral
accuracy. The gap is the rec#54 signal — a classifier that scores high cued and
low neutral is keyword-matching (the buried-keyword crutch), not frame-selection.

Requires Monolith running with a model loaded:
    python -m tools.branch_pilot.classify_eval [--out results.json]
"""
from __future__ import annotations

import argparse
import json
import urllib.request

from core import branch_classify as bc
from tools.branch_pilot.probes import PROBES, CORRECT_TYPE, NEUTRAL

BASE = "http://localhost:7821"


def _post(path: str, body: dict, timeout: int = 300) -> dict:
    req = urllib.request.Request(
        f"{BASE}{path}", data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8", "replace"))


def _http_call(prompt: str) -> str | None:
    """Run the classify prompt over the proven /thinkpad HTTP path (n=1), with a
    minimal scaffold override so the monothink scaffold is NOT injected — the
    classify call stays clean. (core.branch_classify._default_call is the in-app
    path; it imports engine.llm/Qt and only works inside the running process.)"""
    body = {"message": prompt, "n": 1,
            "scaffold": "Read the message and follow its instructions exactly. Output only what it asks for."}
    res = _post("/thinkpad", body)
    branches = res.get("branches", [])
    return branches[0].get("answer", "") if branches else None


def run() -> dict:
    rows = []
    for p in PROBES:
        pid = p["id"]
        want = CORRECT_TYPE[pid]
        cued = bc.classify(p["prompt"], call=_http_call)
        neutral = bc.classify(NEUTRAL[pid], call=_http_call)
        row = {
            "id": pid, "want": want,
            "cued": cued, "cued_ok": cued == want,
            "neutral": neutral, "neutral_ok": neutral == want,
        }
        rows.append(row)
        print(f"{pid}: want={want}\n     cued    -> {cued}   {'OK' if row['cued_ok'] else 'MISS'}"
              f"\n     neutral -> {neutral}   {'OK' if row['neutral_ok'] else 'MISS'}", flush=True)

    cued_ok = sum(1 for r in rows if r["cued_ok"])
    neutral_ok = sum(1 for r in rows if r["neutral_ok"])
    n = len(rows)
    summary = {
        "n": n,
        "cued_accuracy": cued_ok / n if n else None,
        "neutral_accuracy": neutral_ok / n if n else None,
        "keyword_dependence_gap": (cued_ok - neutral_ok) / n if n else None,
        "rows": rows,
    }
    print("\n" + json.dumps({k: v for k, v in summary.items() if k != "rows"}, indent=2))
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    result = run()
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        print(f"\nfull results -> {a.out}")
