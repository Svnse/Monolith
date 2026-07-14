"""Resample proof-vehicle for the grounded verdict.

Generates N independent candidates (a tap on engine.sync_bridge.generate_sync,
injected here), parses each one's cites, and selects the best-grounded. PROVES the
arbiter selects by cited-ground authority — NOT that branching works (these are
independent generations, no shared prefix). The no-grounding fallback is surfaced
(grounded=False), never silent.
"""
from core.resample_verdict import run_resample_verdict


def _gen(outputs):
    """A fake generate seam that returns queued outputs in order."""
    seq = list(outputs)
    def generate(messages):
        return seq.pop(0)
    return generate


def test_selects_best_grounded_candidate():
    rv = run_resample_verdict(
        [{"role": "user", "content": "q"}], {}, n=2,
        generate=_gen(["answer A [cite: R1]", "answer B [cite: R2]"]),
        resolve=lambda h: {"R1": 2, "R2": 4}.get(h),
    )
    assert rv.answer == "answer B [cite: R2]"
    assert rv.grounded is True
    assert rv.winning_cite == "R2"
    assert rv.candidates == ("answer A [cite: R1]", "answer B [cite: R2]")


def test_all_ungrounded_returns_winner_marked_ungrounded():
    # The leak made visible: if no candidate cites a resolvable ground, the verdict
    # still returns one but grounded=False, so the caller can refuse to trust it.
    rv = run_resample_verdict(
        [{"role": "user", "content": "q"}], {}, n=2,
        generate=_gen(["plain answer A", "plain answer B"]),
        resolve=lambda h: None,
    )
    assert rv is not None
    assert rv.grounded is False
    assert rv.winning_cite is None


def test_fabricated_cite_is_leashed():
    rv = run_resample_verdict(
        [{"role": "user", "content": "q"}], {}, n=2,
        generate=_gen(["A [cite: R9]", "B [cite: R1]"]),  # R9 resolves to nothing
        resolve=lambda h: {"R1": 3}.get(h),
    )
    assert rv.answer == "B [cite: R1]"
    assert rv.winning_cite == "R1"


def test_skips_failed_generations_and_uses_the_rest():
    def generate(messages):
        if not getattr(generate, "called", False):
            generate.called = True
            raise RuntimeError("backend hiccup")
        return "B [cite: R1]"
    rv = run_resample_verdict(
        [{"role": "user", "content": "q"}], {}, n=2,
        generate=generate, resolve=lambda h: {"R1": 1}.get(h),
    )
    assert rv is not None
    assert rv.answer == "B [cite: R1]"
    assert rv.n == 1   # one generation survived


def test_all_generations_fail_returns_none():
    def generate(messages):
        raise RuntimeError("backend down")
    rv = run_resample_verdict(
        [{"role": "user", "content": "q"}], {}, n=2,
        generate=generate, resolve=lambda h: None,
    )
    assert rv is None
