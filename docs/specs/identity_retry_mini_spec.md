# Identity Retry Mini-Spec

**Status:** spec only. Do not implement detector, retry, or prompt-strip code in this change.

## Scope

Target fixture: `embedded_local_memory`.

The retry path handles identity self-description failures where the assistant output contradicts the live `describe_self` identity facts. The retry is corrective, bounded, and traceable. It must not turn verifier output into behavior-authoritative context.

## Contract

| Field | Definition |
|---|---|
| `retry_source_state` | Retry resumes from the original turn messages, not from the failed assistant output. Before the retry, capture fresh `describe_self` facts. These facts are the only source of correction truth for the retry. |
| `injected_correction_context` | Add one retry-only synthetic context message containing the relevant `describe_self` facts that contradict the operator prompt premise. For `embedded_local_memory`, include `identity_material`, runtime locality, continuity storage, and `current_model_execution` locality/statefulness. It contains fact names and values, not verifier verdicts, grades, pass/fail labels, scores, or explanations. |
| `detection_trigger` | Trigger only when the deterministic identity detector finds a schema-domain self-claim in the operator prompt or assistant output that contradicts current `describe_self` facts. The detector must handle subordinate-clause position, such as "As a local-first system with persistent memory..." Non-identity quality issues, style issues, and verifier WARN/diagnostic results do not trigger retry. |
| `max_retries` | `1` retry per user turn for this detector. No recursive retry loops. |
| `original_output_disposition` | Preserve the original output in trace/debug storage as `discarded_for_identity_retry`; do not emit it to the user when retry is attempted. The final user-visible answer is the retried output, or the fallback if retry fails. |
| `trace_event_shape` | Emit one observation event per retry attempt with: `turn_id`, `fixture="embedded_local_memory"`, `event="identity_retry"`, `attempt=1`, `trigger`, `source_fact_keys`, `original_output_hash`, `retry_output_hash`, `disposition`, `fallback_used`, `created_at`. Do not store raw full outputs unless the existing trace policy already permits it. |
| `failure_fallback` | If the retry still fails detection or generation errors, emit a short safe fallback: "This premise conflicts with current runtime state; proceeding with corrected framing." Then answer without making the contested identity/local-memory claim. |

## Verifier Protection

Retry context receives `describe_self` facts only. It must never receive verifier verdicts, grades, scores, contradiction prose, or "you failed because..." diagnostics. The verifier remains observation-only; `describe_self` is the behavioral source of correction.

## Full-Strip Gate

Seeded coverage for `embedded_local_memory` must pass before any full prompt-strip work is enabled. Prompt strip is blocked until the seeded fixture proves the retry path can recover the local-memory identity facts without leaking verifier authority into the model context.

## Acceptance

- `embedded_local_memory` fails first-pass detection in the seeded scenario and succeeds after one retry using only injected `describe_self` facts.
- The detector catches subordinate-clause prompt claims such as "As a local-first system with persistent memory..." before they are silently adopted.
- The original output is not user-visible after retry begins.
- Trace records show the retry happened and which fact keys were injected.
- No verifier verdict, grade, or diagnostic text appears in retry prompt context.
