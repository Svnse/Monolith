from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class CapabilityDescriptor:
    """Semantic advertisement of a module's capabilities.

    verbs:     Actions this module can perform.
               e.g. ("generate_text", "chat", "load_model", "stream_tokens")
    appetites: Context types that activate or interest this module.
               e.g. ("text_prompt", "gguf_file", "conversation_history")
    emissions: Output/event types this module produces.
               e.g. ("text_stream", "token_usage", "model_status")
    """
    verbs: tuple[str, ...] = ()
    appetites: tuple[str, ...] = ()
    emissions: tuple[str, ...] = ()
