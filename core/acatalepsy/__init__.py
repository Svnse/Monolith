"""Acatalepsy v1 substrate — log floor, async auditor, candidate pipeline.

See docs/specs/acatalepsy_v1_spec.md for the design. This package implements
A1 only (substrate + minimal validator). A2 (full triage UI) ships
after A1 has validated candidate quality.

Public surface:
  - canonical_log:        append() + read_since()
  - canonical_log_kinds:  KNOWN_KINDS, is_valid_kind(), assert_valid_kind()
  - schema:               migrate() — idempotent table creation
  - atomicity:            is_atomic() — deterministic gate
  - candidates:           insert_candidate() + queries
  - decisions:            insert_decision() + auth enforcement
  - auditor:              run_audit() — LLM extraction over a log slice
  - triggers:             queue + worker thread
"""
from __future__ import annotations
