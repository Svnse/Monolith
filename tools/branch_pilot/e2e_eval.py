"""End-to-end BRANCH value test — does classify-then-solve beat no-frame?

The pilot proved frames bite; classify_eval proved the router routes. Neither
measures the product: does the FULL BRANCH turn (classify -> frame-at-top ->
solve) produce more correct answers than a plain unframed solve? A correct
classification is wasted if the baseline already gets the answer; a mislabel
that still steers to the right approach is fine. Only answer accuracy counts.

Two arms per probe (cued prompt), scored with the probe's own rule:
  baseline : solve with NO frame.
  branch   : classify -> frame_instruction(type) at top -> solve.

Reports baseline_accuracy, branch_accuracy, and the two diagnostics that say
where value lives: RESCUES (baseline miss -> branch hit) and BREAKS (baseline
hit -> branch miss, e.g. classifier picked a frame that steered wrong).

Requires Monolith running + model loaded:
    python -m tools.branch_pilot.e2e_eval [--out results.json]
"""
from __future__ import annotations

import argparse
import json
import urllib.request

from core import branch_solve as bs
from tools.branch_pilot import score
from tools.branch_pilot.probes import PROBES, CORRECT_TYPE

BASE = "http://localhost:7821"
_ANSWER_INSTR = ("\n\nEnd your response with a single line of the form "
                 "`Answer: <value>` and nothing after it.")


def _post(path: str, body: dict, timeout: int = 300) -> dict:
    req = urllib.request.Request(
        f"{BASE}{path}", data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8", "replace"))


def _thinkpad_answer(message: str, scaffold: str) -> str | None:
    res = _post("/thinkpad", {"message": message, "n": 1, "scaffold": scaffold})
    br = res.get("branches", [])
    return br[0].get("answer", "") if br else None


def _classify_call(prompt: str) -> str | None:
    return _thinkpad_answer(
        prompt, "Read the message and follow its instructions exactly. Output only what it asks for.")


def _solve_call(task: str, frame: str | None) -> str | None:
    scaffold = frame if frame else "Solve the problem in the message."
    return _thinkpad_answer(task + _ANSWER_INSTR, scaffold)


def run() -> dict:
    rows = []
    for p in PROBES:
        pid, rule = p["id"], p["rule"]
        base_ans = _solve_call(p["prompt"], None)
        base = score.score_probe(base_ans or "", rule)
        bt = bs.branch_turn(p["prompt"], classify_call=_classify_call, solve_call=_solve_call)
        br = score.score_probe(bt["answer"] or "", rule)
        row = {
            "id": pid, "want_type": CORRECT_TYPE[pid], "got_type": bt["type"],
            "baseline_pass": base["passed"], "baseline_val": base["value"],
            "branch_pass": br["passed"], "branch_val": br["value"],
            "rescue": (not base["passed"]) and br["passed"],
            "break": base["passed"] and (not br["passed"]),
        }
        rows.append(row)
        tag = "RESCUE" if row["rescue"] else "BREAK" if row["break"] else "="
        print(f"{pid}: baseline {'OK' if base['passed'] else 'miss'} ({base['value']})  "
              f"branch {'OK' if br['passed'] else 'miss'} ({br['value']}) "
              f"[type {bt['type']}]  {tag}", flush=True)

    n = len(rows)
    base_ok = sum(1 for r in rows if r["baseline_pass"])
    branch_ok = sum(1 for r in rows if r["branch_pass"])
    summary = {
        "n": n,
        "baseline_accuracy": base_ok / n if n else None,
        "branch_accuracy": branch_ok / n if n else None,
        "rescues": [r["id"] for r in rows if r["rescue"]],
        "breaks": [r["id"] for r in rows if r["break"]],
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
