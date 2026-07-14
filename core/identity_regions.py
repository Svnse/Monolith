"""Identity region split — frozen Origin-0 vs growing Emergent (M2 V0).

The operative identity file (``identity.md``) is two layers:

  * **Origin-0** — the hand-authored seed above the ``EMERGENT:BEGIN`` marker.
    Byte-stable, the diffable gravity well. Never mutated by the evolution loop.
  * **Emergent** — the region from the marker onward, holding E-approved
    self-derived claims. The only region an identity amendment may touch.

Pure and deterministic (no deps, no IO) so the split + Origin-0 protection
behave identically on replay. Enforcement lives here in code, not in the
model-mutable substrate (MonoThink doctrine: the patcher is the enforcer).
"""
from __future__ import annotations

EMERGENT_BEGIN = "EMERGENT:BEGIN"
EMERGENT_END = "EMERGENT:END"


def _begin_line_start(text: str) -> int:
    """Char index of the START of the line containing EMERGENT:BEGIN, or -1."""
    pos = text.find(EMERGENT_BEGIN)
    if pos < 0:
        return -1
    return text.rfind("\n", 0, pos) + 1  # no preceding newline -> -1 + 1 == 0


def split_regions(text: str) -> tuple[str, str]:
    """Split identity text into ``(origin0, emergent)`` at ``EMERGENT:BEGIN``.

    No marker -> ``(text.strip(), "")``. ``origin0`` excludes the marker line
    and all emergent prose; ``emergent`` carries the marker block through EOF.
    """
    s = str(text or "")
    idx = _begin_line_start(s)
    if idx < 0:
        return s.strip(), ""
    return s[:idx].rstrip(), s[idx:].strip()


def locate_snippet(text: str, snippet: str) -> str:
    """Which region a snippet lives in: ``'origin0' | 'emergent' | 'absent'``.

    A snippet present in BOTH regions resolves to ``'origin0'`` — protection is
    conservative; the frozen side wins.
    """
    snip = str(snippet or "").strip()
    if not snip:
        return "absent"
    origin0, emergent = split_regions(text)
    if snip in origin0:
        return "origin0"
    if snip in emergent:
        return "emergent"
    return "absent"


def targets_origin0(text: str, section: str, current_text: str) -> bool:
    """True if an amendment (by ``section`` header or by the ``current_text`` it
    replaces) lands in the frozen Origin-0 region. Section/line-anchored so a
    short value can't false-match a word ("old" in "hold"). The identity_review
    skill uses this to refuse Origin-0 edits before queuing a proposal.
    """
    origin0, _ = split_regions(text)
    o0_sections = {
        ln.lstrip("#").strip()
        for ln in origin0.splitlines() if ln.lstrip().startswith("#")
    }
    o0_lines = {ln.strip() for ln in origin0.splitlines() if ln.strip()}
    sec = str(section or "").strip()
    cur = str(current_text or "").strip()
    if sec and sec in o0_sections:
        return True
    if cur and cur in o0_lines:
        return True
    return False


def apply_emergent_amendment(full_text: str, proposed_full: str) -> tuple[bool, str]:
    """Validate a proposed full-text identity touches ONLY the Emergent region.

    Returns ``(ok, reason)``. Origin-0 must be byte-identical after strip.
    """
    old_o0, _ = split_regions(full_text)
    new_o0, _ = split_regions(proposed_full)
    if old_o0.strip() != new_o0.strip():
        return False, "Origin-0 region is frozen; only the Emergent region may be amended"
    return True, ""
