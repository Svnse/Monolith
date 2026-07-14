"""MonoFrame v2 — CorrectionCards (pure schema).

A frame correction is an EXAMPLE-bound lesson, not a number and not a standalone
rule. The process_move IS the lesson but stays attached to its example; no
invariant is minted from a single card (invariants emerge later, only once human
cards cluster — handled elsewhere).

Training discipline (E's design):
  - Only HUMAN /frame corrections train. Claude-proposed frames are logged as
    CANDIDATES and never train, even if promoted.
  - The advisor is a promotion GATE, not a teacher: it attacks the proposed card
    on five tests; a card is trainable only once a human-sourced card passes.
  - The stateless re-derivation is attached as a SIGNED CONTROL (a labeled machine
    reference), never as a training signal.

This module is PURE data + the trainability invariant. The advisor call, the
synthesis call, anchor classification, nearest-card retrieval, and scaffold
injection live in sibling modules. See the MonoFrame v2 design.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class AnchorType(Enum):
    """What a frame orients on. The failure mode is mirroring the loud surface
    noun instead of the real driver. Routing avoids injecting more nouns —
    it classifies the STRUCTURAL anchor the better_frame chose."""

    EXPLICIT_NOUN = "explicit_noun"      # the loud surface noun (often the error)
    IMPLIED_TASK = "implied_task"        # the unstated thing actually being done
    LIVE_CONSTRAINT = "live_constraint"  # the binding constraint this turn
    USER_STATE = "user_state"            # where the user actually is
    CONTINUITY = "continuity"            # a genuinely-live thread from prior turns


class Aperture(Enum):
    """The cognitive aperture a frame sets. A correction must PRESERVE the
    correct sign — not flip an over-diffuse frame into a wrong collapse, nor the
    reverse. The advisor rejects sign inversions."""

    COLLAPSE = "collapse"   # narrow to one thing
    DIFFUSE = "diffuse"     # hold breadth


class Source(Enum):
    HUMAN = "human"                      # trains (once promoted)
    CLAUDE_CANDIDATE = "claude_candidate"  # logged only, never trains


@dataclass(frozen=True)
class AdvisorVerdict:
    """The promotion gate's attack result — five tests, all must hold."""

    human_grounded: bool
    process_shaped: bool
    not_overfit: bool
    sign_preserved: bool
    real_anchor: bool

    def passed(self) -> bool:
        return (
            self.human_grounded
            and self.process_shaped
            and self.not_overfit
            and self.sign_preserved
            and self.real_anchor
        )


@dataclass(frozen=True)
class CorrectionCard:
    """One example-bound frame correction."""

    bad_frame: str
    better_frame: str
    process_move: str               # the lesson, bound to this example
    anchor_type: AnchorType         # the anchor the better_frame chose
    anchor_error: str               # what the bad_frame did wrong (e.g. mirrored_loud_noun)
    aperture: Aperture
    stateless_control: str          # signed control — machine reference, not signal
    source: Source
    promoted: bool = False          # set True only after the advisor gate passes
    advisor_verdict: AdvisorVerdict | None = None
    slots: dict = field(default_factory=dict)  # typed-slot completion, keyed by anchor
    turn_id: str = ""
    ts: str = ""

    def is_trainable(self) -> bool:
        """A card trains only if it is HUMAN-sourced AND promoted by the advisor.
        A Claude candidate never trains, even if promoted."""
        return self.source is Source.HUMAN and self.promoted
