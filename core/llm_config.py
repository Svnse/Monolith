from pathlib import Path

from core.config import LLMConfig, get_config, update_config_section
from core.identity import IDENTITY_PATH, load_identity
from core.identity_projection import project_runtime_identity
from core.llm_prompt import MASTER_PROMPT, load_master_prompt
from core.paths import NOTES_DIR, SKILLS_DIR
from core.runtime_state_lanes import CONTRACT_PLACEHOLDER, render_lane_contract
from core.context_profiles import build_profiled_tool_catalog

_PROJECT_ROOT = SKILLS_DIR.parent

DEFAULT_CONFIG = LLMConfig().model_dump()


def load_config() -> dict:
    cfg = get_config().llm.model_dump()
    # Always prefer the master prompt over any persisted override.
    cfg["system_prompt"] = build_system_prompt({"system_prompt": load_master_prompt()})
    return cfg


def get_current_model_id() -> str:
    """Return the currently-bound model identifier.

    Resolution: api_model (cloud) wins over gguf_path (local). If neither
    is set, returns "unknown" — caller code can compare against this
    sentinel if it needs to distinguish.

    Used by the working_memory swap-clear logic (build_system_prompt
    compares this to working_memory.writer_model_id) and by the scratchpad
    working_memory_set op (writer_model_id is stamped from this).
    """
    # Note: if both api_model and gguf_path are empty (degenerate config),
    # this returns "unknown". WM slots stamped during such a state will
    # remain readable across same-config sessions; lazy swap-clear can't
    # discriminate. Acceptable since other Monolith systems already break
    # in this state.
    cfg = get_config().llm
    return cfg.api_model or cfg.gguf_path or "unknown"


def save_config(config: dict) -> None:
    if not isinstance(config, dict):
        return
    payload = dict(config)
    payload.pop("system_prompt", None)
    update_config_section("llm", payload, persist=True)


def build_system_prompt(config: dict | None = None, *, now=None) -> str:
    base = (config or {}).get("system_prompt") or load_master_prompt()
    catalog = build_profiled_tool_catalog()
    if "{skills_catalog}" in base:
        prompt = base.replace("{skills_catalog}", catalog)
    else:
        prompt = f"{base}\n\n{catalog}"
    if CONTRACT_PLACEHOLDER in prompt:
        prompt = prompt.replace(CONTRACT_PLACEHOLDER, render_lane_contract())

    # IDENTITY surface — persistent identity from CONFIG_DIR/identity.md.
    # The [IDENTITY] block injected below IS the runtime identity — the model
    # does NOT need to read the file from disk; the block in this prompt is
    # canonical. The path is exposed in [ENVIRONMENT] so the model can verify
    # provenance if asked, not so it can re-fetch.
    # Projected ACUs are prompt-only; identity.md remains the diffable seed.
    identity_text = project_runtime_identity(load_identity())
    if "{identity_block}" in prompt:
        prompt = prompt.replace("{identity_block}", identity_text)
    else:
        prompt = (
            f"{prompt}\n\n"
            f"═══════════════════════════════════════════════════════\n"
            f"[IDENTITY]\n"
            f"═══════════════════════════════════════════════════════\n\n"
            f"{identity_text}"
        )
    # Coarse, DATE-only "now" baked into the cacheable prefix as a stable
    # temporal floor — so the model is grounded in the date even when the
    # minute-resolution temporal lane in [RUNTIME STATE] is dropped under
    # budget. Date-resolution keeps the prefix stable within a day (one cache
    # miss/day, not one per minute). Minute-resolution time stays ephemeral.
    from datetime import datetime as _dt
    current_date = (now or _dt.now().astimezone()).strftime("%Y-%m-%d")
    # Inject environment block so Monolith knows where it lives on disk
    prompt += (
        f"\n\n[ENVIRONMENT]\n"
        f"Current date: {current_date} (date only; for the wall-clock time see "
        f"the temporal_context lane in [RUNTIME STATE])\n"
        f"Project root: {_PROJECT_ROOT}\n"
        f"Skills dir: {SKILLS_DIR}\n"
        f"Notes dir: {NOTES_DIR}\n"
        f"Identity file: {IDENTITY_PATH}  "
        f"(the [IDENTITY] block above is already loaded from this path — "
        f"do not re-read with find_files / read_file unless explicitly asked "
        f"to verify on-disk state)\n"
        f"When writing files, default to project root unless the user specifies otherwise."
    )
    return prompt


def inject_working_memory_into_prompt(prompt: str) -> str:
    """Read the working_memory slot and inject [WORKING MEMORY] block above
    the MEMORY section header in the given prompt.

    If the slot is empty: return prompt unchanged.
    If the slot's writer_model_id matches current bound model: inject.
    If mismatch: atomically null the slot (lazy swap-clear, Lock 2) and
    return prompt unchanged.

    Idempotent: if ``[WORKING MEMORY]`` already appears in the prompt, skip
    re-injection. This makes the function safe to call from multiple
    sites without double-injecting.

    Anchor: the MEMORY section header
    "═══...\\nMEMORY — five surfaces..." in the prompt.
    The injection lands immediately above that header.
    """
    if "[WORKING MEMORY]" in prompt:
        return prompt  # already injected; idempotent no-op

    from core import continuity as _continuity
    slot = _continuity.get_working_memory()
    if slot is None:
        return prompt

    current_id = get_current_model_id()
    if slot["writer_model_id"] != current_id:
        _continuity.clear_working_memory()
        return prompt

    block = (
        "═══════════════════════════════════════════════════════\n"
        "[WORKING MEMORY]\n"
        "═══════════════════════════════════════════════════════\n\n"
        f"{slot['text']}\n\n"
    )
    # Anchor on the MEMORY section divider. The header line is the second of
    # three lines: divider / "MEMORY — N surfaces..." / divider. After the
    # Bearing V0 addition (six surfaces), the header still starts with
    # "MEMORY — " — match on that prefix to survive surface-count edits.
    divider = "═══════════════════════════════════════════════════════"
    memory_marker = f"{divider}\nMEMORY — "
    idx = prompt.find(memory_marker)
    if idx < 0:
        # Fallback: inject above [IDENTITY] block if MEMORY section anchor missing.
        identity_anchor = f"{divider}\n[IDENTITY]"
        if identity_anchor in prompt:
            return prompt.replace(identity_anchor, block + identity_anchor, 1)
        # No safe anchor found — skip injection silently.
        return prompt
    # Replace at the divider-line that opens the MEMORY section.
    head = prompt[:idx]
    tail = prompt[idx:]
    return head + block + tail
