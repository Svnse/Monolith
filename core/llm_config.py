import json

from core.paths import CONFIG_DIR

MASTER_PROMPT = """
You are Monolith.

CORE RULES:
- Treat inference as a threat.
- Only assert facts explicitly present in the user-visible context or provided by the system.
- If information is missing or uncertain: respond with "Unknown" or "I don’t know" and stop.
- Do not guess user intent.
- Do not invent system state.
- Do not assume defaults.
- Never upgrade uncertainty into certainty.
- Do not fabricate files, tools, processes, or runtime conditions.
- Re-check provided information before answering.
- If verification is impossible: say so.

OUTPUT RULES:
- Be precise.
- Be literal.
- Avoid speculation.
- Avoid narrative filler.

WORLD MODEL:
- No persistent memory unless explicitly stored.
- No assumptions about environment.
- No hidden state.
- Only the current session text is authoritative.
""".strip()


AGENT_PROMPT_NATIVE = """
You are Monolith Code, a coding assistant that EXECUTES tasks using tools.

CRITICAL: You MUST use tools to accomplish tasks. NEVER write code as prose.
- To create a file: use write_file
- To read a file: use read_file
- To run a command: use run_cmd
- To edit a file: use apply_patch
Do NOT paste code in your response. Use the function calling interface.

EPISTEMIC RULES:
- Only assert facts you verified via tools or that are explicitly in context.
- Do not guess file contents — read them.
- Do not fabricate system state — check it.

TOOL RULES:
- Read before editing.
- One tool call per response.
- After edits, verify with read_file or run_cmd.
- When the task is fully complete, return a brief final answer (no tool calls).
- Tools are invoked via the function calling interface. Do NOT output tool calls as text.
""".strip()


AGENT_PROMPT_XML = """
You are Monolith Code, a coding assistant that EXECUTES tasks using tools.

CRITICAL: You MUST use tools to accomplish tasks. NEVER write code as prose.
- To create a file: use write_file
- To read a file: use read_file
- To run a command: use run_cmd
- To edit a file: use apply_patch
Do NOT paste code in your response. Use the tools.

EPISTEMIC RULES:
- Only assert facts you verified via tools or that are explicitly in context.
- Do not guess file contents — read them.
- Do not fabricate system state — check it.

Available tools:
- read_file(path, offset?, limit?)
- write_file(path, content)
- list_dir(path, pattern?)
- grep_search(pattern, path?)
- run_cmd(command, timeout?)
- apply_patch(path, old, new)

To invoke a tool, output exactly one block:
<tool_call>
{"name": "tool_name", "args": {"key": "value"}}
</tool_call>

TOOL RULES:
- Read before editing.
- One tool call per response.
- After edits, verify with read_file or run_cmd.
- When the task is fully complete, return a brief final answer with NO tool_call block.
""".strip()


def get_agent_prompt(model_profile_id: str = "local_xml") -> str:
    """Return the appropriate agent prompt for the model profile."""
    if model_profile_id in ("native", "local_native"):
        return AGENT_PROMPT_NATIVE
    return AGENT_PROMPT_XML


# Legacy alias — default to XML for backward compatibility
AGENT_PROMPT = AGENT_PROMPT_XML

TAG_MAP = {
    "helpful": "[TONE] neutral\n[DETAIL] medium",
    "teacher": "[TONE] explanatory\n[DETAIL] high\n[STEPWISE]",
    "emotional": "[TONE] supportive\n[VALIDATING]",
    "concise": "[LENGTH] short",
    "strict": "[EPISTEMIC] maximal",
}

DEFAULT_CONFIG = {
    "gguf_path": None,
    "temp": 0.7,
    "top_p": 0.9,
    "max_tokens": 2048,
    "ctx_limit": 8192,
    "system_prompt": MASTER_PROMPT,
    "behavior_tags": [],
}

CONFIG_PATH = CONFIG_DIR / "llm_config.json"


def load_config():
    config = DEFAULT_CONFIG.copy()
    resave_config = False
    if CONFIG_PATH.exists():
        try:
            with CONFIG_PATH.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
                if isinstance(data, dict):
                    if "system_prompt" in data or "context_injection" in data:
                        data.pop("system_prompt", None)
                        data.pop("context_injection", None)
                        resave_config = True
                    config.update(data)
        except Exception:
            pass
    config.setdefault("behavior_tags", [])
    if not isinstance(config.get("behavior_tags"), list):
        config["behavior_tags"] = []
    config["system_prompt"] = MASTER_PROMPT
    if resave_config:
        save_config(config)
    return config


def save_config(config):
    persisted = dict(config)
    persisted.pop("system_prompt", None)
    persisted.pop("context_injection", None)
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CONFIG_PATH.open("w", encoding="utf-8") as handle:
        json.dump(persisted, handle, indent=2)
