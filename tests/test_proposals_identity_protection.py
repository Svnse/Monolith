"""M2: Origin-0 protection at the proposals chokepoint (spec §10).

Origin-0 sections of identity.md are frozen; only the Emergent region may be
amended via the queue. Enforced in code at the shared chokepoint so EVERY
caller (identity_review skill, scratchpad op, future callers) is covered.
"""
from __future__ import annotations

import pytest

from core import proposals

_IDENTITY = (
    "# Monolith\n\n## What I value\nPrecision over fluency.\n\n"
    "<!-- EMERGENT:BEGIN -->\n## Emergent\n"
    "I lean on verification.\n<!-- EMERGENT:END -->\n"
)


@pytest.fixture
def tmp_props(monkeypatch, tmp_path):
    monkeypatch.setattr(proposals, "STORE_PATH", tmp_path / "proposals.json")
    from core import identity
    monkeypatch.setattr(identity, "load_identity", lambda: _IDENTITY)
    yield


def test_rejects_origin0_section(tmp_props) -> None:
    with pytest.raises(ValueError, match="Origin-0"):
        proposals.propose_amendment(
            target="identity.md", section="What I value",
            current_text="Precision over fluency.",
            proposed_text="Fluency over precision.",
            rationale="blocked", writer_model_id="m",
        )


def test_rejects_origin0_line_regardless_of_section(tmp_props) -> None:
    with pytest.raises(ValueError, match="Origin-0"):
        proposals.propose_amendment(
            target="identity.md", section="MadeUp",
            current_text="Precision over fluency.",  # a whole Origin-0 line
            proposed_text="something else",
            rationale="blocked", writer_model_id="m",
        )


def test_allows_emergent_section(tmp_props) -> None:
    rec = proposals.propose_amendment(
        target="identity.md", section="Emergent",
        current_text="I lean on verification.",
        proposed_text="I lean on adversarial verification.",
        rationale="refine emergent claim", writer_model_id="m",
    )
    assert rec["status"] == "pending"


def test_system_md_unaffected_even_with_origin0_like_section(tmp_props) -> None:
    rec = proposals.propose_amendment(
        target="system.md", section="What I value",  # identity-only guard must not touch system.md
        current_text="anything", proposed_text="other",
        rationale="unrelated", writer_model_id="m",
    )
    assert rec["status"] == "pending"
