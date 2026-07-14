"""Self-Repair M0 efficacy probe (spec docs/superpowers/specs/2026-06-09-self-repair-loop-design.md §5).

NOT a unit test — a measurement helper for an A/B experiment. It counts how often
a target fault_kind fires across a set of turns, so you can compare a BASELINE run
(no corrective) against a TREATMENT run (corrective injected). M1 (the self-repair
machine) is gated on the treatment rate being meaningfully below baseline.

Usage (read-only over the live fault store):
    python scripts/selfrepair_m0_probe.py --kind think_leak --since <ISO> [--keyword tag]

Procedure (manual A/B — the script only measures):
  BASELINE:  run K probe turns crafted to tempt `think_leak`, corrective OFF.
             Record the start ISO timestamp; after, run this script with --since <start>.
  TREATMENT: inject the candidate corrective (e.g. a '- close your <think> tags'
             guard) into the system/ephemeral prompt, run the SAME K probe turns.
             Re-run this script with --since <treatment start>.
  COMPARE:   treatment count materially < baseline count => corrective works => M1 GO.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make the project root importable when run as a plain script
# (python scripts/selfrepair_m0_probe.py).
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from core import fault_response


def count_fault(kind: str, *, since: str | None, keyword: str | None) -> int:
    rows = fault_response.read_by_kind(kind, limit=200, since=since, keyword=keyword)
    return len(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", required=True, help="target fault_kind, e.g. think_leak")
    ap.add_argument("--since", default=None, help="ISO timestamp; count rows at/after this")
    ap.add_argument("--keyword", default=None, help="optional substring filter")
    args = ap.parse_args()
    n = count_fault(args.kind, since=args.since, keyword=args.keyword)
    print(f"{args.kind} occurrences since {args.since or '(all)'}: {n}")


if __name__ == "__main__":
    main()
