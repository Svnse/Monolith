"""Resample proof-vehicle for the grounded verdict.

Generates N independent candidate answers — a *tap* on
``engine.sync_bridge.generate_sync_from_config`` (the existing synchronous generate
seam already used by monothink/planner/expedition), NOT a turn-loop rewire — then
parses each candidate's cited grounds and selects the best-grounded via
``grounded_verdict.select`` (handles resolved through ``recall_handles`` ->
``compute_authority``).

PROVES: the arbiter selects by cited-ground authority. Does NOT prove branching —
these are N independent generations with no shared reasoning prefix. A green
resample-verdict means "the arbiter works, now branching has a proven arbiter to
plug into," not "branching works."

The no-grounding fallback is surfaced (``grounded=False``), never silent: if no
candidate cites a resolvable ground, the verdict still returns one, but the caller
can see it fell back to ungrounded selection (the model-preference leak) and refuse
to trust it. Cheap-to-build does not mean cite-free — the live caller must confirm
candidates actually expose citable grounds (a recall handle / tool span).
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from core import grounded_verdict, recall_handles


@dataclass(frozen=True)
class ResampleVerdict:
    answer: str                  # the winning candidate's text
    grounded: bool               # did the winner cite a resolvable ground?
    winning_cite: str | None     # the ground that won, or None if ungrounded
    authority: int               # the winning ground's Authority (0 if ungrounded)
    n: int                       # candidates that actually generated
    candidates: tuple[str, ...]  # all candidate texts (inspection / model_lab record)


def run_resample_verdict(
    messages: list[dict],
    base_config: dict,
    *,
    n: int = 2,
    generate: Callable[[list[dict]], str] | None = None,
    resolve: Callable[[str], "int | None"] | None = None,
) -> ResampleVerdict | None:
    """Generate ``n`` candidates, parse cites, select the best-grounded.

    ``generate`` defaults to a tap on ``generate_sync_from_config`` (injectable for
    tests); ``resolve`` defaults to ``recall_handles.resolve``. Returns None only
    when no candidate could be generated.
    """
    if n < 1:
        return None
    if generate is None:
        from engine.sync_bridge import generate_sync_from_config

        def generate(msgs: list[dict]) -> str:
            return generate_sync_from_config(base_config, msgs)
    if resolve is None:
        resolve = recall_handles.resolve

    texts: list[str] = []
    for _ in range(n):
        try:
            out = generate(messages)
        except Exception:
            continue  # one failed generation must not sink the verdict
        if out:
            texts.append(str(out))
    if not texts:
        return None

    candidates = [
        grounded_verdict.candidate_from_text(str(i), t) for i, t in enumerate(texts)
    ]
    winner = grounded_verdict.select(candidates, resolve)
    if winner is None:
        return None
    idx = int(winner.candidate.id)
    return ResampleVerdict(
        answer=texts[idx],
        grounded=winner.grounded,
        winning_cite=winner.winning_cite,
        authority=winner.authority,
        n=len(texts),
        candidates=tuple(texts),
    )
