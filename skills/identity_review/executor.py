"""identity_review — propose-only identity-evolution skill (M2 V0).

Two ops, both explicitly invoked (never autonomous):
  * ``detect`` — deterministic, no LLM. Reports the high-confidentity
    self-derived claims (NOVEL × identity-aligned) the operative identity does
    not yet reflect.
  * ``draft`` — ONE grounded LLM call turning the top emergent candidate(s) into
    a first-person Emergent-region claim, queued via ``proposals`` for E to
    review/apply. Origin-0 is frozen (code-enforced via identity_regions); the
    skill NEVER applies anything (propose-only, explicit human gate).

The dispatch surface only — scoring/detection live in core.identity_emergence,
the queue in core.proposals, region protection in core.identity_regions.
"""
from __future__ import annotations

import re
from typing import Any

from core import identity
from core import identity_emergence as _em
from core import identity_regions as _regions
from core import proposals as _proposals
from core import llm_config

_CLAIM_RE = re.compile(r"^\s*EMERGENT_CLAIM\s*:\s*(.+?)\s*$", re.MULTILINE)
_RATIONALE_RE = re.compile(r"^\s*RATIONALE\s*:\s*(.+)\Z", re.MULTILINE | re.DOTALL)

_RATIONALE_CAP = 800
_PROPOSED_CAP = 2000
_TOP_N = 5


def _call_llm(prompt: str) -> str:
    """Direct, non-tool-dispatched LLM call. Broad-except — drafting must never
    break the caller. Tests monkeypatch this."""
    try:
        from core.llm_config import load_config
        from engine.sync_bridge import generate_sync_from_config
        cfg = load_config()
        text = generate_sync_from_config(
            cfg, [{"role": "user", "content": prompt}],
            llm_config={"max_tokens": 1024, "temp": 0.3}, thinking_enabled=False,
        )
        return str(text or "")
    except Exception:
        return ""


def _parse_draft(raw: str) -> tuple[str, str]:
    cm = _CLAIM_RE.search(raw or "")
    rm = _RATIONALE_RE.search(raw or "")
    claim = cm.group(1).strip() if cm else ""
    rationale = (rm.group(1).strip() if rm else "")[:_RATIONALE_CAP]
    return claim, rationale


def _threshold(cmd: dict) -> float:
    try:
        return float(cmd.get("threshold", _em._DEFAULT_THRESHOLD))
    except (TypeError, ValueError):
        return _em._DEFAULT_THRESHOLD


def _candidates(cmd: dict):
    rep = _em.detect_emergence(
        threshold_confidentity=_threshold(cmd),
        min_new_acus=int(cmd.get("min_new", 1) or 1),
        force=True,
    )
    return rep


def _op_detect(cmd: dict) -> str:
    rep = _candidates(cmd)
    lines = [f"[identity_review: {rep.message}]"]
    for c in rep.candidates[:_TOP_N]:
        lines.append(f"  - {c['canonical']}  (confidentity {c['confidentity']}, {c['provenance']})")
    if not rep.candidates:
        lines.append("  (no emergent candidates above threshold)")
    return "\n".join(lines)


def _build_draft_prompt(rep) -> str:
    bullets = "\n".join(
        f"  - {c['canonical']} (confidentity {c['confidentity']})" for c in rep.candidates[:_TOP_N]
    )
    return (
        "You are drafting one EMERGENT identity claim for Monolith — a self-derived "
        "posture that has accumulated through use and is NOT yet stated in the Origin-0 "
        "seed. Ground it ONLY in the evidence below; do not redeclare Origin-0 values.\n\n"
        f"Emergent candidate claims (self-derived, identity-aligned):\n{bullets}\n\n"
        "Write ONE first-person claim (\"I ...\") capturing the earned posture, plus a "
        "one-paragraph rationale. Output EXACTLY:\n"
        "EMERGENT_CLAIM: <one first-person sentence>\n"
        "RATIONALE: <why this is an earned observation, not an Origin-0 redeclaration>"
    )


def _op_draft(cmd: dict) -> str:
    rep = _candidates(cmd)
    if not rep.candidates:
        return "[identity_review: no emergent candidates above threshold — nothing to propose]"

    raw = _call_llm(_build_draft_prompt(rep))
    claim, rationale = _parse_draft(raw)
    if not claim:
        return "[identity_review: draft produced no EMERGENT_CLAIM — nothing queued]"
    if not rationale:
        rationale = "Earned from accumulated, identity-aligned self-derived claims (see emergence candidates)."

    origin0, emergent = _regions.split_regions(identity.load_identity())
    current_text = emergent if emergent else "(no emergent claims yet)"
    bullet = f"- {claim}"
    proposed_text = (f"{emergent}\n{bullet}" if emergent else bullet)[:_PROPOSED_CAP]

    # Code-enforced Origin-0 protection (defense-in-depth; section is Emergent).
    if _regions.targets_origin0(identity.load_identity(), "Emergent", current_text):
        return "[identity_review: refused — amendment would touch frozen Origin-0]"

    try:
        record = _proposals.propose_amendment(
            target="identity.md",
            section="Emergent",
            current_text=current_text,
            proposed_text=proposed_text,
            rationale=rationale,
            writer_model_id=llm_config.get_current_model_id(),
        )
    except ValueError as exc:
        return f"[identity_review: {exc}]"

    try:
        from core.acatalepsy import canonical_log
        canonical_log.append(
            "identity_amendment_proposed",
            payload={
                "proposal_id": record["id"],
                "claim": claim,
                "candidate_count": len(rep.candidates),
            },
        )
    except Exception:
        pass

    return (
        f"[identity_review: queued as proposal id={record['id']} → "
        f"identity.md:Emergent — E reviews and applies manually]"
    )


def run(cmd: dict, ctx: Any) -> str:
    op = str((cmd or {}).get("op") or "detect").strip().lower()
    if op == "detect":
        return _op_detect(cmd)
    if op == "draft":
        return _op_draft(cmd)
    return f"[identity_review: unknown op {op!r} — use 'detect' or 'draft']"
