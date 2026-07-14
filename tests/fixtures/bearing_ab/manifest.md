# Bearing V0 A/B run manifest

**Per spec §8.2, this manifest is filled in BEFORE any A/B run begins.** The 7 fields below pin the runtime so the A/B answers "does Bearing improve trajectory under fixed runtime?" — not cross-model generalization.

After fill-in, this file is committed and referenced from the run's output artifacts. Re-runs with different manifest values are SEPARATE A/B runs; results cross-reference manifests, not the fixture set alone.

---

## Run identifier

- **run_id:** `TBD` (e.g., `bearing_v0_dryrun_2026-05-21` or `bearing_v0_gate_2026-05-22`)
- **run_kind:** `dryrun` | `gate` | `rerun`
- **run_date_utc:** `TBD`

## §8.2 runtime fingerprint (REQUIRED; all 7 fields)

| Field | Value | Notes |
|---|---|---|
| Model name + version | `TBD` | E.g., "Qwen2.5-7B-Instruct" |
| GGUF / checkpoint filename + sha | `TBD` | Filename + sha256 of the actual file loaded |
| Quantization | `TBD` | E.g., Q5_K_M |
| Context length | `TBD` | Tokens (e.g., 8192) |
| Temperature, top_p | `TBD` / `TBD` | Sampler params at run time |
| Seed (if available) | `TBD` | Inference seed, or "n/a" if non-deterministic |
| System prompt version | `TBD` | `prompts/system.md` git sha |
| Bearing version | `TBD` | git sha of HEAD touching `addons/system/bearing/*` |

## Cohort + arm summary

- **Cohort source:** `docs/superpowers/specs/2026-05-21-bearing-ab-fixture-set.md`
- **Cohort version:** locked 2026-05-21
- **Total fixtures:** 27 (6 design + 6 execution + 3 dedicated pressure-test + 12 single-turn)
- **Arms:** 2 (Bearing-on, Bearing-off) → 54 runs total per full gate
- **Dry-run scope:** 1 fixture × 2 arms = 2 runs (multi_turn_design_01 only)

## Rater protocol (per §8.1)

- **Rater A:** `TBD` (blinded)
- **Rater B:** `TBD` (blinded)
- **Adjudicator:** `TBD`
- **Adjudication trigger:** ≥2 point difference on any primary criterion (locked)
- **Briefing artifact:** `TBD` (path to briefing doc; raters must NOT see this manifest before scoring)

## Dispatcher (per §8.3)

- **Chosen surface:** in-process via `ui/pages/chat.py` (LOCKED). The Qt widget at `ui/pages/chat.py:2428` (`chat()` / `generate_text()` / `stream_tokens()`) wraps `_send_message`, which dispatches into `engine/llm.py` via `llama_cpp.Llama` (line 35). The headless runner uses the same underlying engine path — `llama_cpp.Llama` + `core.llm_config.build_system_prompt()` + `core.message_interceptors.apply_interceptors()` — bypassing only the Qt widget event loop. Same surface as chat.py uses; no new infrastructure, no transport variance, no fresh §8.3 lock.
- **Implementation status:** runner script needs to be written (≤150 LoC; mechanical wrapping of the engine path with arm-toggling via `MONOLITH_BEARING_V1`).

## Decision rule version

- **Locked at:** spec §6.5 + §8.4 (2026-05-21)
- **No rule changes from rule-side** after this manifest commits. Data anomalies → follow-up run with NEW manifest + documented rule change.

## Measurement-failure clause status (per §6.5)

- **Adjudication-rate threshold:** ≥30% of scored items triggers measurement-failure verdict.
- **If triggered:** pick exactly one remediation (rubric revision / briefing revision / fixture revision) and re-run with new dated lock.

---

## Runner sketch (per §8.3 dispatcher lock)

A standalone headless runner — `tests/bearing_ab_runner.py` (NOT under `tests/test_*` so pytest doesn't auto-collect) — drives the dry-run and (later) the gate run. Approximate shape:

```python
# Pseudocode — actual implementation pending E's go-ahead
def run_fixture_arm(fixture_id: str, arm: str, manifest: dict) -> Path:
    # arm-pin via env var BEFORE addon construction reads it
    os.environ["MONOLITH_BEARING_V1"] = "1" if arm == "on" else "0"
    # Construct addon (returns None for "off"); wire DI
    from addons.system.bearing import build_addon
    from core.turn_classifier import set_bearing_provider
    from core.message_interceptors import register_interceptor, clear_interceptors
    clear_interceptors()  # clean per-arm
    addon = build_addon()
    if addon is not None:
        set_bearing_provider(addon.provider)
        register_interceptor(addon.interceptor)
    # Load model per manifest
    from llama_cpp import Llama
    llm = Llama(model_path=manifest["gguf_path"], n_ctx=manifest["context_length"], ...)
    # Drive fixture turn-by-turn
    from core.llm_config import build_system_prompt
    from core.message_interceptors import apply_interceptors
    messages = []
    responses = []
    for turn in fixture["turns"]:
        messages.append({"role": "user", "content": turn["user"]})
        msgs_with_interceptors = apply_interceptors(messages, {})
        full_prompt = build_system_prompt() + "\n\n" + format_messages(msgs_with_interceptors)
        out = llm(full_prompt, temperature=manifest["temperature"], top_p=manifest["top_p"], ...)
        response = out["choices"][0]["text"]
        responses.append(response)
        messages.append({"role": "assistant", "content": response})
    # Blinded write
    arm_hash = blake2b(f"{fixture_id}:{arm}:{run_id}".encode()).hexdigest()[:12]
    out_path = ARTIFACTS_DIR / run_id / f"{fixture_id}_{arm_hash}.txt"
    out_path.write_text("\n---\n".join(responses))
    # Decoder kept SEPARATE
    decoder_path = ARTIFACTS_DIR / run_id / "arm_decoder.json"
    decoder = json.loads(decoder_path.read_text()) if decoder_path.exists() else {}
    decoder[arm_hash] = arm
    decoder_path.write_text(json.dumps(decoder, indent=2))
    return out_path
```

Two arm runs MUST be in separate processes (the env var + interceptor registry are process-global; cleaning between arms in-process is error-prone — subprocess-per-arm is cleaner isolation). The runner is the single-arm worker; a thin orchestrator spawns it twice per fixture.

**Status:** sketch only. Real implementation pending E's go-ahead.

---

## Output capture (post-dispatcher-decision)

When dispatcher exists, each fixture × arm run writes:

```
artifacts/bearing_ab/<run_id>/<fixture_id>_<arm_hash>.txt   ← model responses, one per turn
artifacts/bearing_ab/<run_id>/<fixture_id>_<arm_hash>.meta  ← turn timing, token counts
artifacts/bearing_ab/<run_id>/manifest.md                   ← copy of this file
artifacts/bearing_ab/<run_id>/arm_decoder.json              ← maps <arm_hash> → "on" | "off"; kept separate from rater materials
artifacts/bearing_ab/<run_id>/scores/<rater>/<fixture_id>_<arm_hash>.json   ← per-rater scoring
artifacts/bearing_ab/<run_id>/adjudicated_scores.json       ← post-adjudication final
```

Raters receive ONLY the `*.txt` and the rubric — never the arm_decoder until all scoring is complete.

---

## Pre-run sign-off

- [ ] All §8.2 fields filled with real values (no `TBD` remaining)
- [ ] Two raters identified, names recorded above
- [ ] Adjudicator identified, name recorded above
- [ ] Rater briefing doc written + path recorded above
- [ ] Dispatcher decision made (open-infrastructure section resolved)
- [ ] Dry-run cohort defined (default: `multi_turn_design_01`)
- [ ] Sign-off date recorded:  `TBD`
