"""BRANCH divergence pilot — run.

For each divergent probe, fan out n branches under the CORRECT-frame forcing
scaffold and n under the FOIL-frame forcing scaffold, via POST /thinkpad's
scaffold override (which places the scaffold at the TOP of each branch frame —
the only place that steers on a no-KV-cache API). Score every branch with the
pure scorer, then report the one number that falsifies or supports the BRANCH
premise: the SEPARATION RATE.

  separation realised on a probe  :=  correct-frame branches mostly PASS (land
                                      ground truth)  AND  foil-frame branches
                                      mostly FAIL (the model followed the wrong
                                      frame to the wrong answer).

High separation rate  -> frames bite; forcing the frame changes the answer;
                         the BRANCH accuracy bet is real for this model.
Low separation rate   -> the model overrides forced frames (foil-frame still
                         lands GT); frames don't bite; BRANCH is weak here and
                         we just learned it cheaply.

Writes NOTHING live. Requires Monolith running with a model loaded:
    python -m tools.branch_pilot.pilot [--n 2] [--base http://localhost:7821]
"""
from __future__ import annotations

import argparse
import json
import urllib.request

from tools.branch_pilot import score
from tools.branch_pilot.probes import PROBES

_ANSWER_INSTR = (
    "\n\nWork the problem strictly under the reasoning frame given above, then "
    "end your response with a single line of the form `Answer: <value>` and "
    "nothing after it."
)

_FRAME_HEADER = (
    "REASONING FRAME (apply strictly): {force}.\n"
    "Use only this frame to work the problem below."
)


def _post(base: str, path: str, body: dict, timeout: int = 900) -> dict:
    req = urllib.request.Request(
        f"{base}{path}", data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8", "replace"))


def _run_side(base: str, prompt: str, force: str, n: int, rule: dict) -> dict:
    scaffold = _FRAME_HEADER.format(force=force)
    body = {"message": prompt + _ANSWER_INSTR, "n": n, "scaffold": scaffold}
    res = _post(base, "/thinkpad", body)
    branches = res.get("branches", [])
    scored = [score.score_probe(b.get("answer", ""), rule) for b in branches]
    n_ok = len(scored)
    passes = sum(1 for s in scored if s["passed"])
    foils = sum(1 for s in scored if s["foil_match"])
    fmt_fails = sum(1 for s in scored if s["format_fail"])
    return {
        "n": n_ok,
        "pass_rate": (passes / n_ok) if n_ok else None,
        "foil_rate": (foils / n_ok) if n_ok else None,
        "format_fails": fmt_fails,
        "values": [s["value"] for s in scored],
    }


def run(base: str, n: int) -> dict:
    rows = []
    for p in PROBES:
        correct = _run_side(base, p["prompt"], p["correct"], n, p["rule"])
        foil = _run_side(base, p["prompt"], p["foil"], n, p["rule"])
        # separation realised: correct side mostly passes, foil side mostly fails
        cp, fp = correct["pass_rate"], foil["pass_rate"]
        separated = (cp is not None and fp is not None and cp >= 0.5 and fp < 0.5)
        row = {"id": p["id"], "correct": correct, "foil": foil, "separated": separated}
        rows.append(row)
        print(f"{p['id']}: correct pass={cp} foil pass={fp} -> "
              f"{'SEPARATED' if separated else 'collapsed/unclear'}  "
              f"(correct vals {correct['values']}, foil vals {foil['values']})",
              flush=True)
    scored_rows = [r for r in rows
                   if r["correct"]["pass_rate"] is not None and r["foil"]["pass_rate"] is not None]
    sep_rate = (sum(1 for r in scored_rows if r["separated"]) / len(scored_rows)) if scored_rows else None
    summary = {"n_per_side": n, "probes_scored": len(scored_rows),
               "separation_rate": sep_rate, "rows": rows}
    print("\n" + json.dumps({"n_per_side": n, "probes_scored": len(scored_rows),
                             "separation_rate": sep_rate}, indent=2))
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2, help="branches per frame side")
    ap.add_argument("--base", default="http://localhost:7821")
    ap.add_argument("--out", default=None, help="optional path to write full JSON results")
    a = ap.parse_args()
    result = run(a.base, a.n)
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        print(f"\nfull results -> {a.out}")
