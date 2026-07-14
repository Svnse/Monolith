from pathlib import Path

_PROMPTS_DIR = Path(__file__).parent.parent / "prompts"

_FALLBACK_PROMPT = (
    "You are Monolith, a local AI workstation created by E. "
    "Output plaintext only. Use <tool_call> envelopes for tool calls.\n\n"
    "{skills_catalog}"
)


def load_master_prompt() -> str:
    """Load the system prompt from prompts/system.md, falling back to a minimal default."""
    path = _PROMPTS_DIR / "system.md"
    try:
        return path.read_text(encoding="utf-8").strip()
    except Exception:
        return _FALLBACK_PROMPT


MASTER_PROMPT = load_master_prompt()
