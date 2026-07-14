"""friction_settle — back-compat shim.

The settle beat moved to core/intent_settle.py for Path B (pure-code SET-
MEMBERSHIP settlement against the frozen prediction_set, replacing the v1
reply-friction differ). This module re-exports the interceptor + settle() so the
documented wiring diff (register friction_settle_interceptor in bootstrap) and
any older imports keep resolving.
"""
from __future__ import annotations

from core.intent_settle import friction_settle_interceptor, settle  # noqa: F401
