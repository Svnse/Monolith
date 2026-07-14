"""MonoSearch boot wiring — register the adapters and ensure the salience ledger
schema. Called once from the app bootstrap. Idempotent.

Slice 1a registered fault_traces + canonical_log (the two literal day-1 sources).
Slice 1b adds the rest of the backlog: turn_trace, acatalepsy-acus, continuity,
bearing, identity_signals (the `pulling`/`unresolved` source), and identity.
Each adapter is read-only; registration order is cosmetic (the router sorts by
evidence_tier, not registration order).
"""
from __future__ import annotations

from core.monosearch import registry, salience
from core.monosearch.adapters.acu_relations import RelationsAdapter
from core.monosearch.adapters.acus import AcuAdapter
from core.monosearch.adapters.bearing import BearingAdapter
from core.monosearch.adapters.canonical_log import CanonicalLogAdapter
from core.monosearch.adapters.continuity import ContinuityAdapter
from core.monosearch.adapters.faults import FaultAdapter
from core.monosearch.adapters.health import HealthAdapter
from core.monosearch.adapters.identity import IdentityAdapter
from core.monosearch.adapters.identity_signals import IdentitySignalAdapter
from core.monosearch.adapters.investigations import InvestigationAdapter
from core.monosearch.adapters.lag_watch import LagWatchAdapter
from core.monosearch.adapters.mononote import MonoNoteAdapter
from core.monosearch.adapters.outcome_traces import OutcomeTraceAdapter
from core.monosearch.adapters.plan_reminders import PlanReminderAdapter
from core.monosearch.adapters.skills import SkillsAdapter
from core.monosearch.adapters.stage_traces import StageTraceAdapter
from core.monosearch.adapters.tools import ToolsAdapter
from core.monosearch.adapters.turn_trace import TurnTraceAdapter
from core.monosearch.adapters.warrant_graph import WarrantGraphAdapter

_ADAPTERS = (
    ToolsAdapter,
    SkillsAdapter,
    FaultAdapter,
    CanonicalLogAdapter,
    TurnTraceAdapter,
    StageTraceAdapter,
    OutcomeTraceAdapter,
    AcuAdapter,
    RelationsAdapter,
    WarrantGraphAdapter,
    ContinuityAdapter,
    BearingAdapter,
    IdentitySignalAdapter,
    IdentityAdapter,
    PlanReminderAdapter,
    InvestigationAdapter,
    LagWatchAdapter,
    MonoNoteAdapter,
    HealthAdapter,
)


def init_monosearch() -> None:
    for cls in _ADAPTERS:
        registry.register(cls())
    salience.ensure_schema()
