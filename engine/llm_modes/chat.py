from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from core.llm_config import MASTER_PROMPT, load_config


@dataclass
class ChatGenerationSpec:
    prompt: str
    config: dict[str, Any]
    messages: list[dict[str, Any]]
    model_profile_id: str
    temp: float
    top_p: float
    max_tokens: int


class ChatModeStrategy:
    """
    Chat-mode behavior isolated from the shared LLMEngine transport shell.

    This module owns message preparation and completion history updates so new
    modes/pipelines can be introduced without mutating engine/llm.py.
    """

    def build_system_prompt(self, config: dict[str, Any]) -> str:
        base_prompt = MASTER_PROMPT
        tags = config.get("behavior_tags", [])
        cleaned = [tag.strip() for tag in tags if isinstance(tag, str) and tag.strip()]
        if not cleaned:
            return base_prompt
        return f"{base_prompt}\n\n[BEHAVIOR TAGS]\n" + "\n".join(cleaned)

    def prepare_generation(self, engine, payload: dict) -> ChatGenerationSpec:
        prompt = str((payload or {}).get("prompt", ""))
        config = payload.get("config") if isinstance(payload, dict) else None
        if config is None:
            config = load_config()
        if not isinstance(config, dict):
            config = load_config()

        system_prompt = self.build_system_prompt(config)
        temp = float(config.get("temp", 0.7))
        top_p = float(config.get("top_p", 0.9))
        max_tokens = int(config.get("max_tokens", 2048))

        engine._ephemeral_generation = bool(payload.get("ephemeral", False))
        thinking_mode = bool(payload.get("thinking_mode", False))

        if not engine.conversation_history:
            engine.reset_conversation(MASTER_PROMPT)

        system_entry = {"role": "system", "content": system_prompt}
        if engine.conversation_history[0].get("role") != "system":
            engine.conversation_history.insert(0, system_entry)
        else:
            engine.conversation_history[0] = system_entry

        is_update = prompt.startswith("You were interrupted mid-generation.")
        if not engine._ephemeral_generation and not is_update:
            engine.conversation_history.append({"role": "user", "content": prompt})
            engine._pending_user_index = len(engine.conversation_history) - 1
            messages = list(engine.conversation_history)
        else:
            messages = list(engine.conversation_history)
            if not is_update:
                messages.append({"role": "user", "content": prompt})
            engine._pending_user_index = None

        if thinking_mode and not engine._ephemeral_generation:
            messages = list(messages)
            messages.append(
                {
                    "role": "system",
                    "content": "Use private reasoning to think step-by-step, then provide a concise final answer.",
                }
            )

        return ChatGenerationSpec(
            prompt=prompt,
            config=config,
            messages=messages,
            model_profile_id="local_xml",
            temp=temp,
            top_p=top_p,
            max_tokens=max_tokens,
        )

    def on_generation_finished(self, engine, completed: bool, assistant_text: str, loop_history) -> None:
        if completed and not engine._ephemeral_generation:
            engine.conversation_history.append({"role": "assistant", "content": assistant_text})
        engine._pending_user_index = None
        engine._ephemeral_generation = False

    def create_worker_mode(self):
        from engine.llm_modes.chat_worker import ChatWorkerExecutionMode

        return ChatWorkerExecutionMode()
