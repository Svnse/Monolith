"""Shared helpers for parsing LLM streaming responses.

Extracted from engine/llm.py where GGUFRuntime and GeneratorWorker had
byte-identical copies. These functions are pure — no instance state, no
side effects — and work on any OpenAI-compatible streaming chunk format.
"""
from __future__ import annotations


def coerce_mapping(payload) -> dict:
    """Coerce a dict-like object (Pydantic model, dataclass, etc.) to a plain dict."""
    if isinstance(payload, dict):
        return payload
    for method_name in ("model_dump", "to_dict", "dict"):
        method = getattr(payload, method_name, None)
        if callable(method):
            try:
                data = method(exclude_none=True)
            except TypeError:
                data = method()
            if isinstance(data, dict):
                return data
    return {}


def content_to_text(content) -> str:
    """Extract plain text from a content field (str or list-of-parts)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks: list[str] = []
        for part in content:
            payload = coerce_mapping(part)
            if payload:
                text = payload.get("text")
                if isinstance(text, str):
                    chunks.append(text)
        return "".join(chunks)
    return ""


def extract_chunk_parts(chunk) -> tuple[str, str]:
    """Extract (text, reasoning_text) from a streaming chunk.

    Handles OpenAI, llama.cpp, and various API providers by probing
    multiple field paths in priority order.
    """
    payload = coerce_mapping(chunk)
    choices = payload.get("choices")
    if isinstance(choices, list) and choices:
        choice = coerce_mapping(choices[0])
    else:
        choice = {}
    delta = coerce_mapping(choice.get("delta", {}))

    reasoning_text = content_to_text(delta.get("reasoning_content"))
    if not reasoning_text:
        reasoning_text = content_to_text(delta.get("reasoning"))
    if not reasoning_text:
        reasoning_text = content_to_text(choice.get("reasoning_content"))

    text = content_to_text(delta.get("content"))
    if not text:
        text = content_to_text(delta.get("text"))
    if not text:
        text = content_to_text(delta.get("token"))
    if not text:
        message = coerce_mapping(choice.get("message", {}))
        text = content_to_text(message.get("content"))
    if not text:
        text = content_to_text(choice.get("text"))
    if not text:
        text = content_to_text(choice.get("token"))
    if not text:
        text = content_to_text(payload.get("content"))
    if not text:
        text = content_to_text(payload.get("text"))
    if not text:
        text = content_to_text(payload.get("token"))
    return text, reasoning_text
