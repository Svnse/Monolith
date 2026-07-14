"""Runtime identity projection / mutation from stable high-confidentity ACUs.

Origin-0 stays the frozen seed. This module projects self-derived ACUs into the
prompt-facing identity surface and can rewrite only a generated Emergent block
inside identity.md. That matches the Acatalepsy identity lifecycle shape:
identity seed first, then accumulated stable ACU material begins to describe the
operative identity.
"""
from __future__ import annotations

from dataclasses import dataclass
import os
import re
from typing import Iterable

from core.identity_regions import EMERGENT_BEGIN, EMERGENT_END, apply_emergent_amendment


_FLAG_ENV = "MONOLITH_IDENTITY_ACU_PROJECTION_V1"
_SELF_MUTATE_FLAG_ENV = "MONOLITH_IDENTITY_ACU_SELF_MUTATE_V1"
_USER_ALIASES_ENV = "MONOLITH_USER_ALIASES"
_DEFAULT_LIMIT = 6
_ACU_BEGIN = "ACU_IDENTITY:BEGIN"
_ACU_END = "ACU_IDENTITY:END"
_TOKEN_RE = re.compile(r"[a-z0-9_]+")
_ACU_BLOCK_RE = re.compile(
    rf"\n*\s*{re.escape(_ACU_BEGIN)}.*?{re.escape(_ACU_END)}\s*",
    re.DOTALL,
)
_SELF_TOKENS = {"monolith"}
_USER_SUBJECT_TOKENS = {"user", "human"}
_ANCHOR_STOP_TOKENS = {
    "monolith", "origin", "identity", "acu", "the", "a", "an", "and", "or", "is",
    "am", "are", "be", "being", "i", "me", "my", "this", "that", "what", "how",
}


@dataclass(frozen=True)
class IdentitySyncResult:
    changed: bool
    candidate_count: int
    reason: str


def projection_enabled() -> bool:
    """Whether stable ACUs are projected into runtime identity."""
    raw = str(os.environ.get(_FLAG_ENV, "1")).strip().lower()
    return raw in {"1", "true", "yes", "on", ""}


def self_mutation_enabled() -> bool:
    """Whether the generated ACU identity block is written back to identity.md."""
    raw = str(os.environ.get(_SELF_MUTATE_FLAG_ENV, "0")).strip().lower()
    return raw in {"1", "true", "yes", "on", ""}


def _short(text: object, limit: int = 180) -> str:
    body = " ".join(str(text or "").split())
    if len(body) <= limit:
        return body
    return body[: limit - 3].rstrip() + "..."


def _tokens(text: object) -> set[str]:
    return set(_TOKEN_RE.findall(str(text or "").lower()))


def _canonical_parts(canonical: object) -> list[str]:
    return [part.strip() for part in str(canonical or "").split("|")]


def _user_subject_tokens() -> set[str]:
    aliases = _tokens(os.environ.get(_USER_ALIASES_ENV, "").replace(",", " "))
    return _USER_SUBJECT_TOKENS | aliases


def _about_monolith(canonical: object) -> bool:
    parts = _canonical_parts(canonical)
    if not parts:
        return False
    subject_tokens = _tokens(parts[0])
    all_tokens = _tokens(canonical)
    if subject_tokens & _user_subject_tokens() and not (subject_tokens & _SELF_TOKENS):
        return False
    return bool(all_tokens & _SELF_TOKENS)


def _matches_origin0_anchor(canonical: object, identity_text: str | None) -> bool:
    """Require at least one non-generic Origin-0 overlap beyond "monolith"."""
    if not identity_text:
        return True
    from core.identity_regions import split_regions

    origin0, _ = split_regions(identity_text)
    claim = _tokens(canonical) - _ANCHOR_STOP_TOKENS
    seed = _tokens(origin0) - _ANCHOR_STOP_TOKENS
    return bool(claim & seed)


def _identity_candidate(c: dict, identity_text: str | None) -> bool:
    canonical = c.get("canonical")
    return _about_monolith(canonical) and _matches_origin0_anchor(canonical, identity_text)


def _candidate_lines(candidates: Iterable[dict], *, limit: int) -> list[str]:
    lines: list[str] = []
    for c in list(candidates)[: max(0, int(limit))]:
        canonical = _short(c.get("canonical"))
        if not canonical:
            continue
        try:
            acu_id = int(c.get("id"))
        except (TypeError, ValueError):
            acu_id = 0
        conf = float(c.get("confidentity", 0.0) or 0.0)
        stability = float(c.get("stability", 0.0) or 0.0)
        reinforcement = int(c.get("reinforcement", 1) or 1)
        provenance = str(c.get("provenance") or "self").strip() or "self"
        lines.append(
            f"- [acu:{acu_id} confidentity={conf:.3f} stability={stability:.3f} "
            f"reinforcement={reinforcement} source={provenance}] {canonical}"
        )
    return lines


def emergent_acu_candidates(
    *,
    identity_text: str | None = None,
    threshold_confidentity: float | None = None,
    limit: int = _DEFAULT_LIMIT,
) -> tuple[dict, ...]:
    """Read stable identity candidates without mutating scoring or ledgers."""
    try:
        from core import identity_emergence as emergence
        threshold = (
            emergence._DEFAULT_THRESHOLD
            if threshold_confidentity is None
            else float(threshold_confidentity)
        )
        candidates = tuple(emergence.identity_candidates(
            threshold_confidentity=threshold,
            persist=False,
            limit=max(limit * 4, limit, 1),
            corpus=identity_text,
        ))
        return tuple(c for c in candidates if _identity_candidate(c, identity_text))[:limit]
    except Exception:
        return ()


def format_emergent_acu_block(candidates: Iterable[dict], *, limit: int = _DEFAULT_LIMIT) -> str:
    """Format ACU candidates as prompt-facing Emergent identity material."""
    lines = _candidate_lines(candidates, limit=limit)
    if not lines:
        return ""
    body = [
        _ACU_BEGIN,
        "",
        "## Emergent ACU Identity",
        "",
        "Runtime-projected from stable high-confidentity ACUs. Origin-0 remains frozen.",
        *lines,
        "",
        _ACU_END,
    ]
    return "\n".join(body).strip()


def strip_projected_acu_block(identity_text: str) -> str:
    """Remove any prior runtime-projected ACU block from identity text."""
    return _ACU_BLOCK_RE.sub("\n\n", str(identity_text or "")).strip()


def build_identity_with_acu_block(
    identity_text: str,
    *,
    candidates: Iterable[dict] | None = None,
    threshold_confidentity: float | None = None,
    limit: int = _DEFAULT_LIMIT,
) -> str:
    """Return identity text with the generated ACU block replaced."""
    base = strip_projected_acu_block(identity_text)
    block = format_emergent_acu_block(
        emergent_acu_candidates(
            identity_text=base,
            threshold_confidentity=threshold_confidentity,
            limit=limit,
        ) if candidates is None else candidates,
        limit=limit,
    )
    if not block:
        return base

    if EMERGENT_BEGIN in base:
        if EMERGENT_END in base:
            return base.replace(EMERGENT_END, f"{block}\n\n{EMERGENT_END}", 1)
        return f"{base.rstrip()}\n\n{block}"

    return (
        f"{base.rstrip()}\n\n"
        f"{EMERGENT_BEGIN}\n\n"
        f"{block}\n\n"
        f"{EMERGENT_END}"
    ).strip()


def project_runtime_identity(
    identity_text: str,
    *,
    threshold_confidentity: float | None = None,
    limit: int = _DEFAULT_LIMIT,
) -> str:
    """Return the operative prompt identity: seed plus emergent ACU projection."""
    base = strip_projected_acu_block(identity_text)
    if not projection_enabled():
        return base
    return build_identity_with_acu_block(
        base,
        threshold_confidentity=threshold_confidentity,
        limit=limit,
    )


def sync_identity_file_from_acus(
    *,
    threshold_confidentity: float | None = None,
    limit: int = _DEFAULT_LIMIT,
) -> IdentitySyncResult:
    """Self-mutate identity.md by replacing only the generated Emergent ACU block."""
    if not self_mutation_enabled():
        return IdentitySyncResult(False, 0, "disabled")

    from core import identity

    current = identity.load_identity()
    base = strip_projected_acu_block(current)
    candidates = emergent_acu_candidates(
        identity_text=base,
        threshold_confidentity=threshold_confidentity,
        limit=limit,
    )
    proposed = build_identity_with_acu_block(base, candidates=candidates, limit=limit)
    ok, reason = apply_emergent_amendment(current, proposed)
    if not ok:
        return IdentitySyncResult(False, len(candidates), reason)
    if proposed.strip() == current.strip():
        return IdentitySyncResult(False, len(candidates), "unchanged")

    identity.save_identity(proposed)
    try:
        from core.acatalepsy import canonical_log
        canonical_log.append(
            "identity_acu_self_mutated",
            payload={
                "candidate_count": len(candidates),
                "block_present": bool(candidates),
            },
        )
    except Exception:
        pass
    return IdentitySyncResult(True, len(candidates), "updated")
