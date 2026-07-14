"""Proposals — Monolith-authored substrate amendment queue.

Lets Monolith propose changes to its own identity.md or prompts/system.md.
E reads the queue file, edits the target file if approving, and sets the
proposal status to "approved" or "rejected" in the queue manually.

v1 constraints:
  * Monolith can WRITE proposals. Monolith CANNOT directly modify identity.md
    or system.md.
  * No automatic apply or auto-rollback — explicit human gate.
  * Target file limited to identity.md (at IDENTITY_PATH) or prompts/system.md.

Storage: CONFIG_DIR / "proposals.json"  (next to continuity.json).
Atomic-write pattern: write to tmp file then os.replace (mirrors continuity.py).

Schema (per entry):
    id             : int          # auto-increment, survives 50-cap drops cleanly
    target         : str          # "identity.md" or "system.md"
    section        : str          # section header being amended
    current_text   : str          # verbatim current text being replaced (≤2000)
    proposed_text  : str          # replacement text (≤2000)
    rationale      : str          # one paragraph explaining why (≤800)
    status         : str          # "pending" | "approved" | "rejected" | "superseded"
    created_at     : ISO-8601
    writer_model_id: str          # model that authored this proposal

Bounds:
    50 proposals kept; older ones drop silently (oldest first).

Public API (used by scratchpad ops):
    propose_amendment(target, section, current_text, proposed_text, rationale, writer_model_id) -> dict
    list_proposals(limit=20) -> list[dict]  — newest first
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any

from core.paths import CONFIG_DIR

STORE_PATH = CONFIG_DIR / "proposals.json"

_ALLOWED_TARGETS = {"identity.md", "system.md"}
_PROPOSAL_CAP = 50
_TEXT_LIMIT = 2000
_RATIONALE_LIMIT = 800
_SCHEMA_VERSION = 1


# ── helpers ───────────────────────────────────────────────────────────


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _empty_store() -> dict:
    return {"schema_version": _SCHEMA_VERSION, "proposals": []}


def _load_store() -> dict:
    if not STORE_PATH.exists():
        return _empty_store()
    try:
        with STORE_PATH.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return _empty_store()
    if not isinstance(data, dict):
        return _empty_store()
    data.setdefault("schema_version", _SCHEMA_VERSION)
    data.setdefault("proposals", [])
    if not isinstance(data["proposals"], list):
        data["proposals"] = []
    return data


def _save_store(data: dict) -> None:
    STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = STORE_PATH.with_name(STORE_PATH.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, STORE_PATH)


def _next_id(proposals: list[dict]) -> int:
    if not proposals:
        return 1
    return max(p.get("id", 0) for p in proposals) + 1


# ── public API ────────────────────────────────────────────────────────


def propose_amendment(
    target: str,
    section: str,
    current_text: str,
    proposed_text: str,
    rationale: str,
    writer_model_id: str,
) -> dict:
    """Validate and persist a proposed substrate amendment.

    Validation order (raises ValueError on failure):
    1. Reject if any required field missing or empty after strip.
    2. Reject target if not in allowed set.
    3. Reject if current_text > 2000 chars or proposed_text > 2000 chars.
    4. Reject if rationale > 800 chars.
    5. Reject if current_text == proposed_text (no-op proposal).

    On success: persists the record and returns it.
    """
    # Step 1: check required fields — empty after strip.
    fields: dict[str, Any] = {
        "target": target,
        "section": section,
        "current_text": current_text,
        "proposed_text": proposed_text,
        "rationale": rationale,
    }
    for name, val in fields.items():
        if not str(val or "").strip():
            raise ValueError(f"'{name}' is required and must be non-empty")

    # Normalize to stripped strings for all subsequent checks.
    target_s = str(target).strip()
    section_s = str(section).strip()
    current_s = str(current_text).strip()
    proposed_s = str(proposed_text).strip()
    rationale_s = str(rationale).strip()
    writer_s = str(writer_model_id or "").strip()

    # Step 2: validate target.
    if target_s not in _ALLOWED_TARGETS:
        raise ValueError(
            f"target must be 'identity.md' or 'system.md', got '{target_s}'"
        )


    # Step 3: length caps on current_text and proposed_text.
    if len(current_s) > _TEXT_LIMIT:
        raise ValueError(
            f"current_text exceeds {_TEXT_LIMIT}-char cap (got {len(current_s)})"
        )
    if len(proposed_s) > _TEXT_LIMIT:
        raise ValueError(
            f"proposed_text exceeds {_TEXT_LIMIT}-char cap (got {len(proposed_s)})"
        )

    # Step 4: rationale cap.
    if len(rationale_s) > _RATIONALE_LIMIT:
        raise ValueError(
            f"rationale exceeds {_RATIONALE_LIMIT}-char cap (got {len(rationale_s)})"
        )

    # Step 5: no-op check.
    if current_s == proposed_s:
        raise ValueError("current_text and proposed_text are identical — no-op proposal")

    # Step 6 (M2): Origin-0 is frozen. An identity.md amendment may only touch
    # the Emergent region — reject if it targets an Origin-0 section or replaces
    # a whole Origin-0 line. Enforced at this shared chokepoint so EVERY caller
    # (identity_review skill, scratchpad op, future callers) is covered, not just
    # one — MonoThink doctrine: the patcher is the enforcer. identity.md-only;
    # system.md and E's own direct on-disk edits are unaffected.
    if target_s == "identity.md":
        from core.identity import load_identity
        from core.identity_regions import targets_origin0
        if targets_origin0(load_identity(), section_s, current_s):
            raise ValueError(
                "Origin-0 region is frozen; only the Emergent region of "
                "identity.md may be amended (use the identity_review skill)"
            )

    data = _load_store()
    proposals = data["proposals"]

    record: dict[str, Any] = {
        "id": _next_id(proposals),
        "target": target_s,
        "section": section_s,
        "current_text": current_s,
        "proposed_text": proposed_s,
        "rationale": rationale_s,
        "status": "pending",
        "created_at": _now_iso(),
        "writer_model_id": writer_s,
    }

    proposals.append(record)

    # Cap: keep last 50.
    if len(proposals) > _PROPOSAL_CAP:
        proposals[:] = proposals[-_PROPOSAL_CAP:]

    data["proposals"] = proposals
    _save_store(data)
    return record


def list_proposals(limit: int = 20) -> list[dict]:
    """Return up to `limit` most-recent proposals, newest first."""
    data = _load_store()
    proposals = list(data.get("proposals", []))
    # Newest first — reverse append-order.
    proposals.reverse()
    return proposals[:limit]
