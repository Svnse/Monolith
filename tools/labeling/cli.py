"""V1 labeling CLI — blind-first, one keystroke per label.

Run from repo root:
    python -m tools.labeling.cli              # surface_note partition
    python -m tools.labeling.cli --curiosity  # curiosity partition
    python -m tools.labeling.cli --seed 42    # explicit sampling seed

Keys: [h]it  [i]nteresting  [d]ud  [q]uit; after a label, [r] reveals the
hidden verdict fields (rating_value / failure_tags / reason), any other
key continues. All labels append to the partition's JSONL ledger.
"""
from __future__ import annotations

import argparse
import random
import sys

from . import core


def _getch() -> str:
    try:
        import msvcrt
        return msvcrt.getwch().lower()
    except ImportError:  # non-Windows fallback
        return (input("> ").strip()[:1] or " ").lower()


_KEY_TO_LABEL = {"h": "hit", "i": "interesting", "d": "dud"}


def _show_blind(item: dict, pos: int, total: int) -> None:
    b = core.blind_view(item)
    print(f"\n--- {pos}/{total} --- [{b['species']}] {b['recorded_at']}")
    print(b["note"])
    print("[h]it / [i]nteresting / [d]ud / [q]uit > ", end="", flush=True)


def _maybe_reveal(item: dict) -> None:
    print("label saved. [r]eveal verdict / any key to continue > ", end="", flush=True)
    if _getch() == "r":
        print(f"\n  rating_value: {item['rating_value']}")
        print(f"  failure_tags: {item['failure_tags']}")
        print(f"  reason: {item['reason']}")
    else:
        print()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--curiosity", action="store_true",
                    help="label the curiosity-canonicals partition")
    ap.add_argument("--seed", type=int, default=None,
                    help="sampling seed (default: random, stored per ledger row)")
    ap.add_argument("--size", type=int, default=20)
    args = ap.parse_args(argv)

    if args.curiosity:
        items = core.load_curiosity(core.MILESTONES_PATH)
        ledger, namespace = core.LEDGER_CUR, "cur"
    else:
        items = core.load_population(core.DB_PATH)
        ledger, namespace = core.LEDGER_SN, "sn"

    seed = args.seed if args.seed is not None else random.SystemRandom().randrange(10**6)
    done = core.labeled_ids(ledger)
    batch = core.sample_batch(items, labeled_ids=done, seed=seed, size=args.size)
    if not batch:
        print(f"nothing unlabeled in this partition ({len(items)} items, {len(done)} labeled).")
        return 0
    batch_id = core.next_batch_id(ledger, namespace)
    print(f"batch {batch_id}: {len(batch)} items, seed={seed}, "
          f"{len(done)}/{len(items)} already labeled. blind-first; verdicts hidden.")

    for pos, item in enumerate(batch, 1):
        _show_blind(item, pos, len(batch))
        while True:
            key = _getch()
            if key == "q":
                print("\nstopped; ledger keeps what you labeled.")
                _finish(ledger)
                return 0
            if key in _KEY_TO_LABEL:
                break
        print(key)
        core.append_label(ledger, outcome_id=item["outcome_id"],
                          label=_KEY_TO_LABEL[key], species=item["species"],
                          batch_id=batch_id, seed=seed)
        print(core.scoreboard_line(core.tally(ledger),
                                   batch_pos=pos, batch_size=len(batch)))
        _maybe_reveal(item)

    print(f"\nbatch {batch_id} complete.")
    _finish(ledger)
    return 0


def _finish(ledger) -> None:
    # regenerate the curiosity kill-candidate list on every exit (file for
    # E only — kill_pull is never called from this apparatus)
    if ledger == core.LEDGER_CUR:
        n = core.write_kill_candidates(core.LEDGER_CUR, core.KILL_CANDIDATES)
        print(f"kill-candidate list: {core.KILL_CANDIDATES} ({n} canonicals)")


if __name__ == "__main__":
    sys.exit(main())
