"""Bearing grounding verifier — async, post-commit, never rolls back.

Four rules (per design §7):

  G1. Every referent with status="observed" added this turn cites a
      tool_result_N that exists.
  G2. Every open_tensions.resolve with grounding cites a tool_result_N
      that exists.
  G3. Referents with kind="file" can be re-grounded (file still exists
      at path).
  G4. Referents with kind="peer" match a connected peer in CONNECT's
      /who snapshot.

Runs AFTER the envelope has committed. Failure does NOT roll back —
emits a warn-severity fault via emit_fault (caller's responsibility) and
the caller downgrades affected referents' status from "observed" to
"unverified" on next compiler render.

The split between structural (blocks commit) and grounding (post-commit)
prevents transient grounding hiccups — tool result expired, peer
disconnected, file moved — from blocking otherwise-correct Bearing
transitions. Grounding failures are real signal but post-hoc.

V0 callers may pass empty tool_result_ids / connected_peers; the verifier
treats unknown context conservatively (best-effort, no false positives).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from . import schema as bs


@dataclass(frozen=True)
class GroundingVerdict:
    ok: bool
    failed_rules: tuple[str, ...] = ()
    detail: str = ""
    # Indices into applied_bearing.referents whose status should downgrade
    # to "unverified" on next render. Empty when ok=True or when no
    # referent-targeted failure.
    downgrade_referent_indices: tuple[int, ...] = ()

    @classmethod
    def passed(cls) -> "GroundingVerdict":
        return cls(ok=True)


def _added_referents_in_envelope(envelope: dict[str, Any]) -> list[dict[str, Any]]:
    refs = envelope.get("referents")
    if not isinstance(refs, dict):
        return []
    items = refs.get("add")
    if not isinstance(items, list):
        return []
    return [r for r in items if isinstance(r, dict)]


def _resolves_with_grounding(envelope: dict[str, Any]) -> list[dict[str, Any]]:
    ot = envelope.get("open_tensions")
    if not isinstance(ot, dict):
        return []
    items = ot.get("resolve")
    if not isinstance(items, list):
        return []
    out = []
    for item in items:
        if isinstance(item, dict) and isinstance(item.get("grounding"), str):
            out.append(item)
    return out


def verify_grounding(
    turn_id: str,
    envelope: dict[str, Any],
    applied_bearing: bs.Bearing,
    tool_result_ids: Iterable[str] | None = None,
    connected_peers: Iterable[str] | None = None,
) -> GroundingVerdict:
    """Apply G1–G4. Best-effort: missing context skips the affected check.

    `tool_result_ids` is the set of IDs (e.g. "tool_result_3") visible
    this turn. If None or empty, G1/G2 are SKIPPED — no false positives
    when the caller can't supply the context.

    `connected_peers` is the list of currently-connected peer names. If
    None or empty, G4 is SKIPPED.
    """
    tool_ids: set[str] = set(tool_result_ids or ())
    peers: set[str] = set(connected_peers or ())

    failed: list[str] = []
    detail: list[str] = []
    downgrade_indices: list[int] = []

    # ── G1 — observed referent.add must cite a known tool_result ────
    if tool_ids:
        for ref in _added_referents_in_envelope(envelope):
            if ref.get("status") != "observed":
                continue
            grounding = ref.get("grounding")
            if not isinstance(grounding, str) or grounding not in tool_ids:
                if "G1" not in failed:
                    failed.append("G1")
                detail.append(
                    f"referent.add status=observed grounding {grounding!r} not in tool_result_ids"
                )
                # Find the index in applied_bearing.referents for this name.
                name = ref.get("name")
                if isinstance(name, str):
                    for i, applied in enumerate(applied_bearing.referents):
                        if applied.name == name and applied.status == "observed":
                            downgrade_indices.append(i)
                            break

    # ── G2 — open_tensions.resolve.grounding must cite a known tool_result ─
    if tool_ids:
        for item in _resolves_with_grounding(envelope):
            grounding = item.get("grounding")
            if isinstance(grounding, str) and grounding not in tool_ids:
                if "G2" not in failed:
                    failed.append("G2")
                detail.append(
                    f"open_tensions.resolve grounding {grounding!r} not in tool_result_ids"
                )

    # ── G3 — file referents must point to a path that exists ────────
    for ref in _added_referents_in_envelope(envelope):
        if ref.get("kind") != "file":
            continue
        name = ref.get("name")
        if not isinstance(name, str) or not name.strip():
            continue
        # Treat name as a path; if it doesn't exist, downgrade.
        # Resolve relative to cwd is intentional — caller's responsibility
        # to pass absolute paths in production.
        try:
            exists = Path(name).exists()
        except (OSError, ValueError):
            exists = False
        if not exists:
            if "G3" not in failed:
                failed.append("G3")
            detail.append(f"file referent {name!r} does not exist")
            for i, applied in enumerate(applied_bearing.referents):
                if applied.name == name and applied.kind == "file" and applied.status == "observed":
                    if i not in downgrade_indices:
                        downgrade_indices.append(i)
                    break

    # ── G4 — peer referents must be in connected peers ──────────────
    if peers:
        for ref in _added_referents_in_envelope(envelope):
            if ref.get("kind") != "peer":
                continue
            name = ref.get("name")
            if not isinstance(name, str) or name not in peers:
                if "G4" not in failed:
                    failed.append("G4")
                detail.append(f"peer referent {name!r} not in connected peers")
                if isinstance(name, str):
                    for i, applied in enumerate(applied_bearing.referents):
                        if applied.name == name and applied.kind == "peer" and applied.status == "observed":
                            if i not in downgrade_indices:
                                downgrade_indices.append(i)
                            break

    if failed:
        return GroundingVerdict(
            ok=False,
            failed_rules=tuple(failed),
            detail="; ".join(detail),
            downgrade_referent_indices=tuple(sorted(set(downgrade_indices))),
        )
    return GroundingVerdict.passed()
