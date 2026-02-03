import json
from pathlib import Path

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
    "context_injection": "",
    "behavior_tags": [],
}

CONFIG_PATH = Path("ui/addons/configs/llm_config.json")


def load_config():
    config = DEFAULT_CONFIG.copy()
    if CONFIG_PATH.exists():
        try:
            with CONFIG_PATH.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
                if isinstance(data, dict):
                    config.update(data)
        except Exception:
            pass
    config.setdefault("context_injection", "")
    config.setdefault("behavior_tags", [])
    if not isinstance(config.get("behavior_tags"), list):
        config["behavior_tags"] = []
    config["system_prompt"] = MASTER_PROMPT
    return config


def save_config(config):
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CONFIG_PATH.open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)
