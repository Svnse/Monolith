"""Cite-grounded verdict — selection over candidate answers by the authority of
the ground each one cites.

The arbiter for resampled/branched candidates (the "decide when to collapse, not
when to branch" move). A candidate cites the *handle* of a ground (a recalled
belief or tool-evidence span); the verdict resolves each handle to an Authority
level via the injected resolver (in prod: handle -> ACU-id -> compute_authority)
and ranks by it. A candidate whose cites resolve to nothing is LEASHED below any
grounded candidate — self-report (or a fabricated cite) never out-ranks evidence.
The verdict names its winning ground (`winning_cite`) so selection is itself
citable, not a bare preference.

This module is pure logic: it owns no store and reads no authority itself — the
resolver is injected, so a fabricated cite (handle that resolves to None) is
treated exactly like no cite. That is the laundering guard at resolution time;
the emission-side guard (cite is optional + an explicit no-ground token) lives in
the reasoning scaffold, not here.
"""
from __future__ import annotations

import os
import re
from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class Candidate:
    id: str                      # caller's handle for the candidate (resample idx / branch id)
    cites: tuple[str, ...] = ()  # ground handles it cited (e.g. ("R1", "R3"))


@dataclass(frozen=True)
class ParsedCites:
    handles: tuple[str, ...]   # ground identities cited, e.g. ("R3", "tool:search")
    no_ground: bool            # the explicit honest-absence token was present


# A cite references a ground by its identity: [cite: R3] / [cite: tool:search].
# The recall lane renders [R1] as a LABEL; a bare [R1] is NOT a cite (only [cite: ...]
# counts) — the advisor's disambiguation, so an echoed label can't pass as grounding.
_CITE_RE = re.compile(r"\[cite:\s*([^\]]+?)\s*\]", re.IGNORECASE)
_NO_GROUND_RE = re.compile(r"\[no-ground\]", re.IGNORECASE)


def parse_cites(text: str) -> ParsedCites:
    """Extract cited ground handles + the explicit no-ground token from candidate text."""
    body = text or ""
    handles = tuple(m.group(1).strip() for m in _CITE_RE.finditer(body))
    return ParsedCites(handles=handles, no_ground=bool(_NO_GROUND_RE.search(body)))


def candidate_from_text(id: str, text: str) -> "Candidate":
    """Form a verdict Candidate from a model candidate's text by parsing its cites."""
    return Candidate(id=id, cites=parse_cites(text).handles)


@dataclass(frozen=True)
class Ranked:
    candidate: Candidate
    authority: int            # best resolved ground authority; 0 if ungrounded
    grounded: bool            # True iff >=1 cite resolved to an authority
    winning_cite: str | None  # the handle that supplied `authority`, or None


Resolver = Callable[[str], "int | None"]


def _best_ground(candidate: Candidate, resolve: Resolver) -> tuple[int, str | None]:
    """The candidate's strongest resolvable ground: (authority, handle).

    Max over its cites — a conclusion is as strong as the best ground it leans on.
    Unresolvable cites contribute nothing (the resolution-time laundering guard).
    """
    best_au = 0
    best_handle: str | None = None
    for handle in candidate.cites:
        au = resolve(handle)
        if au is None:
            continue
        if best_handle is None or au > best_au:
            best_au = int(au)
            best_handle = handle
    return best_au, best_handle


def rank_candidates(candidates: list[Candidate], resolve: Resolver) -> list[Ranked]:
    """Rank candidates: grounded above ungrounded, then by ground authority,
    ties broken by input order (stable)."""
    scored: list[tuple[int, Ranked]] = []
    for idx, cand in enumerate(candidates):
        au, handle = _best_ground(cand, resolve)
        grounded = handle is not None
        scored.append((idx, Ranked(candidate=cand, authority=au,
                                    grounded=grounded, winning_cite=handle)))
    # sort key: grounded first (True>False), then authority desc, then input order asc.
    scored.sort(key=lambda t: (not t[1].grounded, -t[1].authority, t[0]))
    return [r for _, r in scored]


def select(candidates: list[Candidate], resolve: Resolver) -> Ranked | None:
    """The winning candidate, or None if there are no candidates."""
    ranked = rank_candidates(candidates, resolve)
    return ranked[0] if ranked else None


# ── V1: single-answer grounded verdict (the per-turn primitive) ──────────────
# B (candidate-selection arbiter) reuses verdict_for_turn as its per-candidate
# primitive; getting the four buckets — especially the fabricated catch — right
# at N=1 is what lets the bigger build stand on proven ground.

_FLAG_ENV = "MONOLITH_GROUNDED_VERDICT_V1"


def grounded_verdict_enabled() -> bool:
    """V1 feature flag (dark by default; activated after a live proof)."""
    raw = str(os.environ.get(_FLAG_ENV, "0")).strip().lower()
    return raw in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class TurnVerdict:
    grounded: bool                  # >=1 cite resolved to an authority
    authority: int                  # best resolved ground authority; 0 if none
    winning_cite: str | None        # the handle that supplied `authority`
    cited: tuple[str, ...]          # every handle the answer cited
    fabricated: tuple[str, ...]     # cited handles that resolved to None — the FAULT
    no_ground: bool                 # the explicit honest-absence [no-ground] token


def verdict_for_turn(text: str, resolve: Resolver) -> TurnVerdict:
    """Grounding verdict over a SINGLE answer: parse its cites, resolve each
    against this turn's recall handles, and split grounded vs fabricated.

    ``fabricated`` = cited a handle that resolves to None (a ground never shown
    this turn) — the laundering signal, the one bucket that is unambiguously a
    fault. It is DISTINCT from ``no_ground`` (the model declined to cite at all):
    a fabricated cite IS a cite, just an unresolvable one. Both leave
    ``grounded`` False, but only fabricated belongs in the fault loop. Resolve
    handling mirrors ``_best_ground`` (None contributes nothing), so the prod
    resolver (``recall_handles.resolve``, which returns None for a missing
    handle) yields fabricated rather than a swallowed or coerced no-ground.
    """
    parsed = parse_cites(text)
    best_au = 0
    winning: str | None = None
    fabricated: list[str] = []
    for handle in parsed.handles:
        au = resolve(handle)
        if au is None:
            fabricated.append(handle)
            continue
        if winning is None or au > best_au:
            best_au = int(au)
            winning = handle
    return TurnVerdict(
        grounded=winning is not None,
        authority=best_au,
        winning_cite=winning,
        cited=tuple(parsed.handles),
        fabricated=tuple(fabricated),
        no_ground=parsed.no_ground,
    )
