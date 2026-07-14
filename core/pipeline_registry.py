"""Pipeline policy registry — declared list of every smart system on the bus.

Every policy that participates in the Turn Pipeline must appear in POLICIES
below. This is the structural fix to the "we keep finding other similar smart
systems" pain — if it isn't here, it isn't on the bus.

Discovery is *declarative*, not by filesystem scan: importing this module
yields the closed set. At bootstrap, `validate_against_filesystem()` asserts
that every module under `core/pipeline_policies/` is registered, so a new
policy file that forgets to register itself fails fast at boot rather than
silently never firing.

Dependencies between policies that subscribe to the same event are declared
via `depends_on`. `topo_sort()` produces a deterministic per-event order at
bootstrap.

Scope: pipeline-tier policies only. Pre-pipeline interceptors
(continuity / effort / context_refresh / adaptive_budget — see
core/message_interceptors.py) emit to Layer A via existing infrastructure
and are NOT part of this registry.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from core.turn_pipeline_events import AuthorityTier


@dataclass(frozen=True)
class PolicyRegistration:
    """Declared metadata for a pipeline policy.

    Fields:
        name: stable identifier (module-local snake_case). Used as the policy's
            source_name on events it emits and as the join key in `depends_on`.
        module_path: dotted import path, e.g. "core.pipeline_policies.output_sanitizer".
        subscribes_to: event-class names (strings, not imports — avoids cycles).
            Empty tuple is legal for policies that only emit (rare).
        depends_on: names of other policies that must run before this one on
            the same event. Empty tuple by default.
        authority_tier: declared authority level. Mutation and dispatch tiers
            require a kill_switch_env_flag.
        kill_switch_env_flag: env var that disables the policy when set to a
            falsey value ("0", "false", "no", "off"). Default-ON semantics.
        retry_budget: meaningful only for dispatch tier; the maximum number of
            self-initiated dispatch actions per turn (e.g., requeues).
    """
    name: str
    module_path: str
    subscribes_to: tuple[str, ...]
    depends_on: tuple[str, ...]
    authority_tier: AuthorityTier
    kill_switch_env_flag: str
    retry_budget: int | None = None

    def __post_init__(self) -> None:
        if not self.name or not self.name.replace("_", "").isalnum():
            raise ValueError(f"invalid policy name: {self.name!r}")
        if not self.module_path.startswith("core.pipeline_policies."):
            raise ValueError(
                f"policy module must live under core.pipeline_policies: {self.module_path!r}"
            )
        if self.authority_tier in (AuthorityTier.MUTATION, AuthorityTier.DISPATCH):
            if not self.kill_switch_env_flag:
                raise ValueError(
                    f"policy {self.name!r} is tier={self.authority_tier.value} "
                    "and must declare a kill_switch_env_flag"
                )
        if self.retry_budget is not None and self.authority_tier != AuthorityTier.DISPATCH:
            raise ValueError(
                f"retry_budget only valid for DISPATCH tier; "
                f"policy {self.name!r} is {self.authority_tier.value}"
            )


# ── the declared registry ───────────────────────────────────────────
#
# Phase 1 shipped empty; Phase 2 declares the five live-turn policies.
# Order is illustrative; topo_sort_per_event handles ordering at boot.
# The validator asserts that every file under core/pipeline_policies/ is
# registered here — adding a policy file without registering it is a
# boot-time error.


def _build_policies() -> tuple[PolicyRegistration, ...]:
    # Import lazily so a missing/broken policy file produces a clean error
    # naming the policy, not a generic ImportError at module load.
    from core.pipeline_policies.output_sanitizer import REGISTRATION as _SAN
    from core.pipeline_policies.tool_failure_classifier import REGISTRATION as _CLS
    from core.pipeline_policies.tool_loop_continuation import REGISTRATION as _CONT
    from core.pipeline_policies.tool_repetition_detector import REGISTRATION as _REPEAT
    from core.pipeline_policies.parse_retry import REGISTRATION as _PARSE
    from core.pipeline_policies.verifier_bridge import REGISTRATION as _VER
    from core.pipeline_policies.subordinate_clause_detector import REGISTRATION as _SUBORD
    from core.pipeline_policies.commitment_detector import REGISTRATION as _COMMIT
    return (_SAN, _CLS, _CONT, _REPEAT, _PARSE, _VER, _SUBORD, _COMMIT)


# Lazy POLICIES construction. Building at module load time creates a circular
# import when a policy module is imported directly (e.g., by its own test):
# the policy module imports PolicyRegistration from this module, which during
# its top-level execution would call _build_policies() and re-import the
# policy module before it has finished defining REGISTRATION. Deferring the
# build to first call breaks the cycle; the module-level __getattr__ keeps
# `from core.pipeline_registry import POLICIES` working as a lazy reference.
_POLICIES_CACHE: tuple[PolicyRegistration, ...] | None = None


def _get_policies() -> tuple[PolicyRegistration, ...]:
    """Return the closed registry tuple, building it on first call."""
    global _POLICIES_CACHE
    if _POLICIES_CACHE is None:
        _POLICIES_CACHE = _build_policies()
    return _POLICIES_CACHE


def __getattr__(name: str):
    """Module-level lazy proxy for `POLICIES` so existing imports keep working."""
    if name == "POLICIES":
        return _get_policies()
    raise AttributeError(
        f"module 'core.pipeline_registry' has no attribute {name!r}"
    )


# ── helpers ─────────────────────────────────────────────────────────


def iter_policies() -> tuple[PolicyRegistration, ...]:
    return _get_policies()


def policies_by_event() -> dict[str, list[PolicyRegistration]]:
    """Return {event_kind_name: [policies subscribed, in declared order]}."""
    out: dict[str, list[PolicyRegistration]] = defaultdict(list)
    for p in _get_policies():
        for ev in p.subscribes_to:
            out[ev].append(p)
    return dict(out)


def topo_sort(policies: list[PolicyRegistration]) -> list[PolicyRegistration]:
    """Topological sort by depends_on. Stable on insertion order for ties.

    Raises ValueError on cycle. Policies not in the input list referenced via
    depends_on are treated as 'already satisfied' (e.g., another event's
    subscriber that this one waits for would not block).
    """
    names_in = {p.name for p in policies}
    by_name = {p.name: p for p in policies}
    indeg: dict[str, int] = {p.name: 0 for p in policies}
    edges: dict[str, list[str]] = {p.name: [] for p in policies}
    for p in policies:
        for dep in p.depends_on:
            if dep in names_in:
                edges[dep].append(p.name)
                indeg[p.name] += 1
    ordered: list[PolicyRegistration] = []
    ready = [p for p in policies if indeg[p.name] == 0]
    while ready:
        cur = ready.pop(0)
        ordered.append(cur)
        for nxt in edges[cur.name]:
            indeg[nxt] -= 1
            if indeg[nxt] == 0:
                ready.append(by_name[nxt])
    if len(ordered) != len(policies):
        remaining = [n for n, d in indeg.items() if d > 0]
        raise ValueError(f"cycle in policy depends_on: {remaining!r}")
    return ordered


def topo_sort_per_event() -> dict[str, list[PolicyRegistration]]:
    """Return per-event ordered subscriber list using topo_sort()."""
    return {ev: topo_sort(ps) for ev, ps in policies_by_event().items()}


# ── validation ──────────────────────────────────────────────────────


def expected_policy_modules() -> set[str]:
    """Module paths declared in POLICIES, as full dotted strings."""
    return {p.module_path for p in _get_policies()}


def discovered_policy_modules(policies_dir: Path) -> set[str]:
    """Scan policies_dir for python files, return their dotted module paths.

    Ignores __init__.py and anything starting with _ (test fixtures, helpers).
    """
    if not policies_dir.is_dir():
        return set()
    out: set[str] = set()
    for entry in policies_dir.iterdir():
        if not entry.is_file() or entry.suffix != ".py":
            continue
        if entry.name.startswith("_"):
            continue
        out.add(f"core.pipeline_policies.{entry.stem}")
    return out


def validate_against_filesystem(policies_dir: Path) -> None:
    """Assert every .py file under policies_dir is in POLICIES, and vice versa.

    Called at bootstrap. Raises RuntimeError with a clear message naming
    exactly which file is missing a registration (or which registration
    points at a missing file). This is the structural guarantee that the
    registry stays the single source of truth.
    """
    on_disk = discovered_policy_modules(policies_dir)
    declared = expected_policy_modules()
    missing_registration = on_disk - declared
    missing_file = declared - on_disk
    problems: list[str] = []
    if missing_registration:
        problems.append(
            "policy file(s) on disk but not in pipeline_registry.POLICIES: "
            + ", ".join(sorted(missing_registration))
        )
    if missing_file:
        problems.append(
            "registry entry has no matching file under core/pipeline_policies/: "
            + ", ".join(sorted(missing_file))
        )
    if problems:
        raise RuntimeError(
            "Turn Pipeline registry mismatch. "
            "Every policy file must be registered in core/pipeline_registry.py. "
            + " | ".join(problems)
        )
