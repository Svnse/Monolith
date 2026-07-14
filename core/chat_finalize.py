"""READY-time finalization for an assistant turn.

Extracted from ui/pages/chat.py (which imports PySide6 at module load) so the
emit-then-verify ordering is unit-testable without Qt fixtures. The caller
extracts ``raw``/``public`` (chat.py uses AssistantStreamNormalizer for that)
and passes them in alongside the pipeline-emit callback.

The pipeline emit must fire regardless of MONOLITH_VERIFIER_V1 — output_sanitizer
and verifier_bridge subscribe to TurnReadyEvent and gate their own work via
independent kill switches. Pre-fix, chat.py:_run_response_verifier short-
circuited at the verifier flag check before the emit, so both downstream
policies silently never received the event when the flag was off.

Bearing write hook (Bearing V0): after emit + before verifier, the raw
turn text is scanned for `<bearing_update>` envelopes; on hit, the
envelope is parsed, structurally verified, and either committed to
bearing.json or queued as a [BEARING_UPDATE_REJECTED] for next turn.
The hook is kill-switch-gated (MONOLITH_BEARING_V1) and soft-fails so
it never breaks the chat path even if the addon errors.

Independence: imports only core.response_verifier + addons.system.bearing
(no engine, no MonoBase/acatalepsy, no Qt). Boundary preserved.
"""
from __future__ import annotations

from typing import Any, Callable

from core.response_verifier import verifier_enabled, verify_response


def _process_bearing_envelope(raw: str, config: dict[str, Any]):
    """Soft-handle parse + verify + commit. Never raises into the chat path.

    Kill switch: MONOLITH_BEARING_V1 (read fresh by the addon's kill_switch
    module on every call). If off, this function still parses but the
    addon's build_addon returns None and no commit happens — same outcome
    as the function not being called.

    Returns an UpdaterResult so the caller can observe the outcome.
    Returns UpdaterResult(found_envelope=False) on early-exit paths.
    Current callers that ignore the return value are unaffected.
    """
    try:
        from addons.system.bearing import kill_switch
        from addons.system.bearing.updater import UpdaterResult
        if not kill_switch.is_enabled():
            return UpdaterResult(found_envelope=False)
        if "<bearing_update" not in raw.lower():
            return UpdaterResult(found_envelope=False)  # cheap pre-check
        from addons.system.bearing import updater
        from core.llm_config import get_current_model_id
        turn_id = ""
        turn_n = 0
        if isinstance(config, dict):
            turn_id = str(config.get("_turn_id") or "")
            try:
                turn_n = int(config.get("_turn_n") or 0)
            except (TypeError, ValueError):
                turn_n = 0
        return updater.process_turn_output(
            turn_id=turn_id or "unknown",
            response_text=raw,
            model_id=get_current_model_id(),
            turn_n=turn_n,
        )
    except Exception:
        # Bearing's write-side must never break the chat finalize path.
        try:
            from addons.system.bearing.updater import UpdaterResult
            return UpdaterResult(found_envelope=False)
        except Exception:
            return None


def _process_observer_boundary(config: dict[str, Any]) -> None:
    """Fire Observer V0 once at assistant-turn boundary. Never raises."""

    try:
        from addons.system import observer
        turn_id = ""
        if isinstance(config, dict):
            turn_id = str(config.get("_turn_id") or "")
        if not turn_id:
            return
        observer.fire_turn_boundary(turn_id=turn_id)
    except Exception:
        return


def _process_curiosity_capture(raw: str, on_debug: Callable[[str], None] | None = None,
                               config: dict | None = None) -> None:
    """Soft-capture the model's private <curiosity> block into ACUs."""
    try:
        from core.curiosity_capture import capture_from_assistant_text
        # Per-occasion id = the outer-turn number, so a self-curiosity claim asserted
        # across DISTINCT turns accrues distinct source_events (the signal that lets a
        # self ACU crystallize to L2 identity memory). 0/missing -> None (fails closed:
        # the claim stays L1 rather than collapsing into a shared null bucket).
        turn_n = int(config.get("_turn_n") or 0) if isinstance(config, dict) else 0
        report = capture_from_assistant_text(raw, source_event=(turn_n or None))
        if report.captured and on_debug is not None:
            on_debug(
                f"[CURIOSITY_CAPTURE] captured={report.captured} "
                f"rejected={report.rejected}"
            )
    except Exception:
        return


def apply_terminal_correction(
    *,
    corrected: str | None,
    public: str,
    get_widget: Callable[[], Any],
    on_debug: Callable[[str], None] | None = None,
) -> bool:
    """Apply a MUTATION-tier output correction to the live assistant bubble.

    The consumption seam's apply step (M1). Extracted here — Qt-free and
    testable — because chat.py is Qt-coupled at module load. The widget is
    obtained lazily via ``get_widget`` (chat.py passes
    ``lambda: self._widget_for_index(idx)``) and must expose
    ``update_main_text(str)``.

    Returns True when the bubble was re-rendered. Returns False when there is
    no correction (``corrected`` is None / unchanged) or no widget — the
    no-widget case is LOGGED via ``on_debug``, never silent, so the failure
    mode can never re-create the write-only disease M1 exists to cure.

    Non-performative: this re-renders the displayed artifact only; nothing is
    injected into the model's context.
    """
    if not corrected or corrected == public:
        return False
    widget = None
    try:
        widget = get_widget()
    except Exception:
        widget = None
    if widget is not None and hasattr(widget, "update_main_text"):
        widget.update_main_text(corrected)
        if on_debug is not None:
            on_debug(
                f"[SANITIZE] re-rendered answer: stripped internal-tag leak "
                f"({len(public)}→{len(corrected)} chars)"
            )
        return True
    if on_debug is not None:
        on_debug(
            "[SANITIZE] correction ready but no widget; display NOT corrected"
        )
    return False


def _record_source_tier_for_turn(
    raw: str,
    public: str,
    tools_used: tuple,
    config: dict,
    on_debug,
) -> None:
    """Source-Tier Gate (Stage 1a): classify the turn's output provenance and
    stamp it on the turn's frame row. Best-effort, flag-gated, NON-PERFORMATIVE
    — nothing is injected into the model's context here; this only persists
    telemetry the runtime consumes later (Gate C). Never raises.
    """
    try:
        from core.source_tier import classify_source_tiers, source_tier_enabled
        if not source_tier_enabled():
            return
        turn_id = config.get("_turn_id") if isinstance(config, dict) else None
        if not turn_id:
            return
        # Real exchange tool usage: the finalize call site passes tools_used=()
        # (the terminal turn has no <tool_call> tag), so fall back to the
        # exchange-level signal chat.py stashes on config before the tool-loop
        # cleanup. Without this, every turn would classify as GENERATION.
        tools = tuple(tools_used) or tuple(config.get("_source_tier_tools", ()) or ())
        result = classify_source_tiers(raw, public, tools)
        from core.turn_trace import record_source_tier
        record_source_tier(turn_id, result.region_tiers["answer"], result.region_tiers)
        if on_debug is not None:
            on_debug(
                f"[SOURCE_TIER] turn={turn_id} tier={result.region_tiers['answer']} "
                f"had_tool={result.had_tool} had_trace={result.had_trace}"
            )
    except Exception:
        # Best-effort: a classification/persist failure must never break a turn.
        pass


def _record_grounded_verdict_for_turn(
    raw: str,
    config: dict,
    on_debug,
) -> None:
    """Grounded-Verdict Gate (V1) — the independent error signal at finalize.

    Parse the answer's ``[cite: Rn]`` handles, resolve each against THIS turn's
    recall handles (still populated — ``recall_handles.reset()`` fires only at the
    NEXT turn's recall render, the single reset call site), and record a per-turn
    grounding verdict. NON-PERFORMATIVE: nothing is injected back into the model
    (it already self-cites). Of the four buckets, ONLY a fabricated cite (a handle
    that resolves to None — a ground never shown this turn) is a fault: it is the
    laundering seam, the one bucket with no innocent reading. grounded / ungrounded
    / honest ``[no-ground]`` stay pure telemetry. Best-effort; never raises.
    """
    try:
        from core.grounded_verdict import grounded_verdict_enabled, verdict_for_turn
        if not grounded_verdict_enabled():
            return
        turn_id = config.get("_turn_id") if isinstance(config, dict) else None
        if not turn_id:
            return
        from core import recall_handles
        verdict = verdict_for_turn(raw, recall_handles.resolve)
        vd = {
            "grounded": verdict.grounded,
            "authority": verdict.authority,
            "winning_cite": verdict.winning_cite,
            "cited": list(verdict.cited),
            "fabricated": list(verdict.fabricated),
            "no_ground": verdict.no_ground,
        }
        from core.turn_trace import record_grounded_verdict
        record_grounded_verdict(turn_id, vd)
        # Born-alive consumer: feed ONLY the fabricated-cite bucket to Self-Check.
        # It is the single bucket that is unambiguously a fault (the model claimed a
        # ground that does not exist). Feeding no-ground would punish honesty;
        # feeding ungrounded would add noise — both are telemetry, not faults.
        if verdict.fabricated:
            from core.fault_response import emit_fault
            emit_fault(
                turn_id, "fabricated_cite", "grounded_verdict",
                "cited unresolvable handle(s): " + ", ".join(verdict.fabricated),
                {"fabricated": list(verdict.fabricated), "cited": list(verdict.cited)},
            )
        if on_debug is not None:
            on_debug(
                f"[GROUNDED_VERDICT] turn={turn_id} grounded={verdict.grounded} "
                f"au={verdict.authority} fabricated={list(verdict.fabricated)} "
                f"no_ground={verdict.no_ground}"
            )
    except Exception:
        # Best-effort: a verdict/persist failure must never break a turn.
        pass


def finalize_assistant_turn(
    *,
    raw: str,
    public: str,
    config: dict[str, Any],
    emit_pipeline_ready: Callable[[str, str, tuple], None],
    record_verdict: Callable[[dict[str, Any]], None],
    on_debug: Callable[[str], None] | None = None,
    tools_used: tuple[str, ...] = (),
) -> None:
    """Emit TurnReadyEvent unconditionally; run verify_response only when enabled.

    Args:
        raw:                  The full assistant turn text (may contain internal tags).
        public:               The public answer text (post-normalization).
        config:               The chat config dict (forwarded to verifier_enabled).
        emit_pipeline_ready:  Called with (raw, public, tools_used) — always called
                              when ``raw`` is non-blank, regardless of verifier flag.
        record_verdict:       Called with the verifier-result payload when the
                              verifier ran. Not called when the verifier is disabled.
        on_debug:             Optional debug-line sink for verifier output.
        tools_used:           Names of tools the model invoked during this turn.
                              Forwarded to both the pipeline emit (for the
                              verifier_bridge policy to consume) and the in-
                              process verifier call (so the tool-evidence
                              check in response_verifier.py is no longer
                              dormant). Default empty preserves prior
                              behavior for callers that haven't wired
                              per-turn tool tracking yet.
    """
    if not raw.strip():
        return

    emit_pipeline_ready(raw, public, tuple(tools_used))

    # Curiosity capture — same raw-output seam as <frame>. The parser accepts
    # only terminal <curiosity> blocks containing Monolith-subject canonical
    # triples; capture_from_assistant_text is flag-gated and best-effort.
    _process_curiosity_capture(raw, on_debug, config)

    # Source-Tier Gate (Stage 1a) — classify + persist the turn's provenance.
    _record_source_tier_for_turn(raw, public, tuple(tools_used), config, on_debug)

    # Grounded-Verdict Gate (V1) — resolve the answer's cites against this turn's
    # recall handles + record the grounding verdict; feed ONLY fabricated-cite to
    # Self-Check (the one bucket that is a true fault). Best-effort, flag-gated.
    _record_grounded_verdict_for_turn(raw, config, on_debug)

    # frame_shift (Phase 1, MONOLITH_FRAME_SHIFT_V1, dark): capture the turn-START
    # current_frame BEFORE any <bearing_update>/commit below mutates it, so the
    # consecutive-frame comparison sees the true previous frame. None when off.
    _fsh_prev_frame = None
    try:
        from addons.system.bearing import frame_shift as _fsh
        if _fsh.enabled():
            from addons.system.bearing import store as _fsh_store
            _fsh_prev_frame = _fsh_store.get_bearing().current_frame or ""
    except Exception:
        _fsh_prev_frame = None

    # Bearing V0 write-side hook: scan raw turn for <bearing_update>
    # envelope, commit if structurally valid. Soft-handled internally.
    _bearing_result = _process_bearing_envelope(raw, config)

    # Observable Frame fastpath (v0) — OBSERVE only, no mutation.
    # Gated solely by MONOLITH_OBSERVABLE_FRAME_V0 (independent of bearing
    # kill switch). Never raises; never changes any existing behavior when off.
    try:
        from addons.system.bearing import frame_observe
        if frame_observe.enabled():
            # Map UpdaterResult -> bu_outcome string (priority order from spec).
            try:
                r = _bearing_result
                if r is not None and getattr(r, "parse_failed", False):
                    _bu_outcome = "parse_failed"
                elif r is None or not getattr(r, "found_envelope", False):
                    _bu_outcome = "none"
                elif getattr(r, "bearing_changed", False):
                    _bu_outcome = "applied"
                elif (
                    getattr(r, "structural_verdict", None) is not None
                    and not getattr(r.structural_verdict, "ok", True)
                ):
                    _bu_outcome = "rejected"
                elif (
                    getattr(r, "found_envelope", False)
                    and getattr(r, "applied_bearing", None) is not None
                    and not getattr(r, "bearing_changed", False)
                ):
                    _bu_outcome = "noop"
                else:
                    _bu_outcome = "none"
            except Exception:
                _bu_outcome = "none"

            try:
                from addons.system.bearing import store as _bearing_store
                _current_frame = _bearing_store.get_bearing().current_frame or ""
            except Exception:
                _current_frame = ""

            _turn_id_obs = str(config.get("_turn_id") or "") if isinstance(config, dict) else ""
            frame_observe.record(
                _turn_id_obs,
                raw,
                bu_outcome=_bu_outcome,
                current_frame=_current_frame,
            )

            # Frame-commit fastpath (MONOLITH_FRAME_COMMIT_V1, dark).
            # Scribe the model's <frame> into bearing.current_frame on disparity,
            # ONLY when a real <bearing_update> has NOT already set current_frame
            # this turn (bu_outcome == "applied" takes precedence).
            try:
                from addons.system.bearing import kill_switch as _ks
                if (
                    _ks.frame_commit_is_enabled()
                    and _ks.is_enabled()
                    and _bu_outcome != "applied"
                ):
                    _obs = frame_observe.observe(raw)
                    _disp = frame_observe.disparity(
                        _obs["observed_frame"],
                        _current_frame,
                        has_frame=_obs["has_frame"],
                    )
                    if _obs["has_frame"] and _disp in ("empty_bearing", "differ"):
                        from addons.system.bearing import updater as _u
                        from core.llm_config import get_current_model_id as _gcmi
                        _fc_turn_id = _turn_id_obs or "unknown"
                        _fc_turn_n = 0
                        if isinstance(config, dict):
                            try:
                                _fc_turn_n = int(config.get("_turn_n") or 0)
                            except (TypeError, ValueError):
                                _fc_turn_n = 0
                        _u.commit_frame(
                            _fc_turn_id,
                            _obs["observed_frame"],
                            model_id=_gcmi(),
                            turn_n=_fc_turn_n,
                        )
            except Exception:
                pass
    except Exception:
        pass

    # frame_shift observe (Phase 1): classify this turn's frame transition
    # (HOLD/SHIFT/AMBIG) against the captured turn-start frame and log it. A
    # SEPARATE rail from frame_drift (frame vs ask). Observe-only; never raises.
    try:
        if _fsh_prev_frame is not None:  # set only when MONOLITH_FRAME_SHIFT_V1 is on
            from addons.system.bearing import frame_shift as _fsh
            from addons.system.bearing import frame_observe as _fsh_fo
            _sobs = _fsh_fo.observe(raw)
            if _sobs.get("has_frame"):
                _sh_turn_n, _sh_turn_id, _sh_sess = 0, "", ""
                if isinstance(config, dict):
                    try:
                        _sh_turn_n = int(config.get("_turn_n") or 0)
                    except (TypeError, ValueError):
                        _sh_turn_n = 0
                    _sh_turn_id = str(config.get("_turn_id") or "")
                    _sh_sess = str(config.get("session_id") or config.get("engine_key") or "")
                _fsh.record(
                    _sh_turn_id or "unknown",
                    _fsh_prev_frame,
                    _sobs["observed_frame"],
                    turn_n=_sh_turn_n,
                    source="frame_heartbeat",
                    session_id=_sh_sess,
                )
    except Exception:
        pass

    # MonoFrame v2 standing recorder — capture the model's committed
    # <frame_selection> into the durable JSONL trace, automatically, every turn
    # (source="auto"). No-op when the flag is off or no block was emitted. The
    # commitment is what the answer can be judged against — not a cognition probe.
    try:
        from addons.system.bearing import frame_selection as _fsel
        if _fsel.enabled():
            from datetime import datetime as _dt, timezone as _tz
            _cfg = config if isinstance(config, dict) else {}
            _sf_before = ""
            try:
                from addons.system.bearing import store as _bs
                _sf_before = _bs.get_bearing().current_frame or ""
            except Exception:
                _sf_before = ""
            _wrote_selection = _fsel.record_from_output(
                raw_output=raw,
                turn_id=str(_cfg.get("_turn_id") or ""),
                session_id=str(_cfg.get("_session_id") or _cfg.get("session_id") or ""),
                timestamp_utc=_dt.now(_tz.utc).isoformat(),
                user_input=str(_cfg.get("_user_input") or ""),
                bearing_before=_sf_before,
                source="auto",
            )
            # Frame Fidelity — exact-once delivery via the job ledger. Every
            # committed frame SYNCHRONOUSLY queues a job (so a missing verdict is a
            # visible job state, not a silent gap); the judge runs async and the
            # ledger tracks queued->running->complete. recover_once retries jobs left
            # incomplete by a crash/restart (once per process, on the first turn).
            # Observational: it records a verdict, never rewrites.
            from addons.system.bearing import frame_fidelity_jobs as _ffjobs
            from addons.system.bearing import frame_fidelity as _ffj
            if _ffjobs.enabled():
                _ffjobs.recover_once(base_config=_cfg)
                _ff_turn_id = str(_cfg.get("_turn_id") or "")
                _ans_digest = _fsel.digest(public)
                if _wrote_selection:
                    _recent = _fsel.read_recent(1)
                    if _recent:
                        _fr = _recent[-1]
                        _jid, _created = _ffjobs.enqueue_job(
                            turn_id=str(_fr.get("turn_id", "")) or _ff_turn_id,
                            frame_record_hash=str(_fr.get("artifact_hash", "")),
                            answer_digest=_ans_digest,
                            judge_version=_ffj.JUDGE_VERSION,
                            answer=public,
                        )
                        if _created:
                            _ffjobs.run_job_async(
                                _jid, frame_record=_fr, answer=public, base_config=_cfg,
                            )
                elif public and public.strip():
                    # Answering turn with NO frame_selection block = recorder failure,
                    # not a skip. Make it a visible failed job (every final answer is
                    # judged or visibly failed for recorder absence).
                    _ffjobs.enqueue_recorder_failure(
                        turn_id=_ff_turn_id, answer_digest=_ans_digest,
                    )
                else:
                    # Legitimate non-answer turn (tool-only / aborted / empty).
                    _ffjobs.enqueue_skipped(
                        turn_id=_ff_turn_id, reason="non_answer_turn",
                        answer_digest=_ans_digest,
                    )
    except Exception:
        pass

    # Observer V0 read-side hook: after turn output is final and Bearing had a
    # chance to commit, compile an advisory IRP-labeled block for next turn.
    _process_observer_boundary(config)

    if not verifier_enabled(config):
        return

    result = verify_response(
        raw_answer=raw,
        public_answer=public,
        tools_used=list(tools_used),
        tool_outcomes=[],
        task_type="chat",
        assumptions=[],
    )
    payload = result.to_payload()
    record_verdict(payload)

    if on_debug is not None:
        on_debug(
            f"[VERIFY] verdict={result.verdict} findings={len(result.findings)} "
            f"duration_ms={payload['duration_ms']}"
        )
        for finding in result.findings[:3]:
            on_debug(
                f"[VERIFY] {finding.severity}: {finding.code} — {finding.message}"
            )
