"""Cite parsing — model text -> cited ground handles, feeding grounded_verdict.

The disambiguation the advisor flagged: the recall lane renders `[R1]` as a LABEL
in the prompt; the model references a ground as `[cite: R1]`. Only the latter is a
cite. `[no-ground]` is the explicit honest-absence token.
"""
from core.grounded_verdict import parse_cites, candidate_from_text, select


def test_parse_extracts_cite_handle():
    p = parse_cites("I conclude X. [cite: R3]")
    assert p.handles == ("R3",)
    assert p.no_ground is False


def test_parse_extracts_tool_handle():
    p = parse_cites("from the search result [cite: tool:search]")
    assert p.handles == ("tool:search",)


def test_parse_ignores_bare_recall_label():
    # [R1] is the lane's label; echoing it is NOT a cite (the advisor's disambiguation).
    p = parse_cites("the recalled belief [R1] says X")
    assert p.handles == ()


def test_parse_multiple_cites_in_order():
    p = parse_cites("X [cite: R1] and also Y [cite: R2]")
    assert p.handles == ("R1", "R2")


def test_parse_no_ground_flag():
    p = parse_cites("nothing recalled supports this. [no-ground]")
    assert p.no_ground is True
    assert p.handles == ()


def test_parse_is_case_and_space_tolerant():
    p = parse_cites("[CITE:  R2 ]")
    assert p.handles == ("R2",)


def test_parse_empty_text():
    p = parse_cites("")
    assert p.handles == () and p.no_ground is False


def test_candidate_from_text_feeds_select_end_to_end():
    a = candidate_from_text("A", "answer A, grounded in [cite: R1]")
    b = candidate_from_text("B", "answer B, grounded in [cite: R2]")
    winner = select([a, b], lambda h: {"R1": 2, "R2": 4}.get(h))
    assert winner.candidate.id == "B"          # R2 is the higher-authority ground
    assert winner.winning_cite == "R2"


def test_candidate_with_only_bare_label_is_ungrounded():
    # A cites nothing real (just echoes the label); B cites properly -> B wins.
    a = candidate_from_text("A", "I think X because [R1] looked relevant")
    b = candidate_from_text("B", "X holds [cite: R1]")
    winner = select([a, b], lambda h: {"R1": 3}.get(h))
    assert winner.candidate.id == "B"
