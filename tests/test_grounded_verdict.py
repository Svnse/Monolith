"""Cite-grounded verdict — ranks candidate answers by the authority of the ground
each one cites. A candidate with no resolvable cite is leashed below any grounded
one (self-report doesn't beat evidence), and the verdict names the winning ground.
"""
from core.grounded_verdict import Candidate, select, rank_candidates


def _resolver(table):
    """handle -> authority (AU level), or None if the handle resolves to nothing."""
    return lambda h: table.get(h)


def test_cited_candidate_beats_no_cite():
    # B cites a ground (AU3); A cites nothing -> B wins, A is leashed.
    a = Candidate(id="A", cites=())
    b = Candidate(id="B", cites=("R1",))
    winner = select([a, b], _resolver({"R1": 3}))
    assert winner.candidate.id == "B"
    assert winner.grounded is True
    assert winner.winning_cite == "R1"


def test_higher_authority_ground_wins():
    # Both cite, but A's ground is AU4 (locked) vs B's AU2 (recall) -> A wins.
    a = Candidate(id="A", cites=("R1",))
    b = Candidate(id="B", cites=("R2",))
    winner = select([a, b], _resolver({"R1": 4, "R2": 2}))
    assert winner.candidate.id == "A"
    assert winner.authority == 4
    assert winner.winning_cite == "R1"


def test_unresolvable_cite_is_treated_as_no_cite():
    # A cites a handle that resolves to nothing -> leashed like no-cite; B (grounded) wins.
    a = Candidate(id="A", cites=("R9",))     # R9 not in table
    b = Candidate(id="B", cites=("R1",))
    winner = select([a, b], _resolver({"R1": 1}))
    assert winner.candidate.id == "B"


def test_multi_cite_scores_by_strongest_ground():
    # A leans on its strongest resolvable ground (max), not its weakest.
    a = Candidate(id="A", cites=("R1", "R2"))
    winner = select([a], _resolver({"R1": 1, "R2": 4}))
    assert winner.authority == 4
    assert winner.winning_cite == "R2"


def test_all_ungrounded_still_returns_one_marked_ungrounded():
    a = Candidate(id="A", cites=())
    b = Candidate(id="B", cites=("R9",))     # unresolvable
    winner = select([a, b], _resolver({}))
    assert winner is not None
    assert winner.grounded is False
    assert winner.winning_cite is None
    assert winner.candidate.id == "A"        # stable: first by input order on a tie


def test_authority_tie_breaks_to_input_order():
    a = Candidate(id="A", cites=("R1",))
    b = Candidate(id="B", cites=("R2",))
    winner = select([a, b], _resolver({"R1": 3, "R2": 3}))
    assert winner.candidate.id == "A"


def test_empty_returns_none():
    assert select([], _resolver({})) is None


def test_rank_orders_grounded_above_ungrounded_then_by_authority():
    a = Candidate(id="A", cites=())             # ungrounded
    b = Candidate(id="B", cites=("R1",))        # AU2
    c = Candidate(id="C", cites=("R2",))        # AU4
    ranked = rank_candidates([a, b, c], _resolver({"R1": 2, "R2": 4}))
    assert [r.candidate.id for r in ranked] == ["C", "B", "A"]
