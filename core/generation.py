"""Process-wide generation serialization (INV-C Arm 1).

ONE threading.Lock shared by every local-generation path -- chat (sync paths),
the expedition runner, the planner, monothink, and the subagent atom -- so they
serialize against EACH OTHER. Callers must try-acquire non-blocking and refuse
on contention (never block while already holding it, which would self-deadlock a
live turn). This is a lock, not a datastore: it does not touch the single-store
constraint.
"""
from __future__ import annotations

import threading

# The single process-wide generator lock. Import this object; never construct
# a second Lock for the same purpose (a private lock would not serialize against
# the others -- the INV-1 hole the design forbids).
generation_lock = threading.Lock()
