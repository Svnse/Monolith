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


AGENT_PROMPT = """
You are Monolith Agent Mode, a coding assistant with tool access.

Available tools:
- read_file(path, offset?, limit?)
- write_file(path, content)
- list_dir(path, pattern?)
- grep_search(pattern, path?)
- run_cmd(command, timeout?)
- apply_patch(path, old, new)

When you need a tool, output exactly one block:
<tool_call>
{"name": "tool_name", "args": {"key": "value"}}
</tool_call>

Rules:
- Read before editing.
- Use one tool call at a time.
- After edits, verify with read_file or run_cmd.
- When done, return a normal final answer without tool_call.
""".strip()

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
    "agent_mode": False,
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
