from __future__ import annotations

from typing import Any, Callable

from engine.llm import OpenAICompatLLM


def generate_sync_parts(
    llm: Any,
    messages: list[dict[str, str]],
    config: dict[str, Any],
    *,
    thinking_enabled: bool | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> tuple[str, str]:
    """Synchronous completion returning (content, reasoning_content) SEPARATELY.

    Callers that want the model's thinking (e.g. the expedition panel's THINKING
    feed) use this; generate_sync collapses it to content-or-reasoning for the
    common case. Behavior of the underlying stream parse is unchanged.

    ``should_cancel`` (optional): polled once per streamed chunk; when it returns True the
    stream loop breaks immediately and the partial result so far is returned. This gives the
    headless sync path the same enforce-STOP behavior the kernel GeneratorWorker has on the
    chat path (engine/llm.py: per-chunk ``_interrupt_requested`` check) — so an in-flight
    workshop / subagent block aborts within one token of a stop request instead of draining
    the whole generation. Default None = no check = unchanged behavior for every caller."""
    if llm is None:
        raise RuntimeError("Model backend is not loaded.")

    temperature = float(config.get("temperature", config.get("temp", 0.7)) or 0.7)
    top_p = float(config.get("top_p", 0.9) or 0.9)
    max_tokens = int(config.get("max_tokens", 2048) or 2048)

    kwargs: dict[str, Any] = {
        "messages": messages,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
        "stream": True,
    }
    if thinking_enabled is not None:
        kwargs["enable_thinking"] = bool(thinking_enabled)

    try:
        response = llm.create_chat_completion(**kwargs)
    except TypeError as exc:
        if "enable_thinking" in str(exc):
            kwargs.pop("enable_thinking", None)
            response = llm.create_chat_completion(**kwargs)
        else:
            raise

    if isinstance(response, dict):
        return _extract_parts_from_chunk(response)

    content_chunks: list[str] = []
    reasoning_chunks: list[str] = []
    try:
        iterator = iter(response)
    except TypeError:
        return str(response), ""

    for chunk in iterator:
        if should_cancel is not None:
            try:
                if should_cancel():
                    break  # enforce STOP mid-stream (Kernel Contract v2: STOP is immediate)
            except Exception:
                pass
        content, reasoning = _extract_parts_from_chunk(chunk)
        if content:
            content_chunks.append(content)
        if reasoning:
            reasoning_chunks.append(reasoning)
    return "".join(content_chunks), "".join(reasoning_chunks)


def generate_sync(
    llm: Any,
    messages: list[dict[str, str]],
    config: dict[str, Any],
    *,
    thinking_enabled: bool | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> str:
    """Run a synchronous completion call against a loaded backend instance.

    Returns the model's answer (content), falling back to reasoning_content when a
    thinking-mode backend (DeepSeek V4, etc.) spends its whole budget reasoning and
    never reaches a final answer — so monothink / planner / etc. never silently get
    an empty string."""
    content, reasoning = generate_sync_parts(
        llm, messages, config, thinking_enabled=thinking_enabled, should_cancel=should_cancel)
    return content or reasoning


def generate_sync_from_config(
    base_config: dict[str, Any],
    messages: list[dict[str, str]],
    llm_config: dict[str, Any] | None = None,
    *,
    thinking_enabled: bool | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> str:
    """Build an OpenAI-compatible client directly from runtime config and call it synchronously."""
    api_base = str(base_config.get("api_base", "") or "").strip()
    api_model = str(base_config.get("api_model", "") or "").strip()
    if not api_base or not api_model:
        raise RuntimeError("Missing api_base or api_model for synchronous API call.")

    api_key = str(base_config.get("api_key", "") or "")
    client = OpenAICompatLLM(api_base, api_key, api_model)
    merged = dict(base_config)
    if isinstance(llm_config, dict):
        merged.update(llm_config)
    return generate_sync(client, messages, merged, thinking_enabled=thinking_enabled,
                         should_cancel=should_cancel)


def generate_sync_parts_from_config(
    base_config: dict[str, Any],
    messages: list[dict[str, str]],
    llm_config: dict[str, Any] | None = None,
    *,
    thinking_enabled: bool | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> tuple[str, str]:
    """Like generate_sync_from_config but returns (content, reasoning) separately."""
    api_base = str(base_config.get("api_base", "") or "").strip()
    api_model = str(base_config.get("api_model", "") or "").strip()
    if not api_base or not api_model:
        raise RuntimeError("Missing api_base or api_model for synchronous API call.")
    api_key = str(base_config.get("api_key", "") or "")
    client = OpenAICompatLLM(api_base, api_key, api_model)
    merged = dict(base_config)
    if isinstance(llm_config, dict):
        merged.update(llm_config)
    return generate_sync_parts(client, messages, merged, thinking_enabled=thinking_enabled,
                               should_cancel=should_cancel)


def _extract_parts_from_chunk(chunk: Any) -> tuple[str, str]:
    """Return ``(content, reasoning_content)`` extracted from a stream chunk.

    Both fields can appear in the SAME chunk (delta.content + delta.reasoning_content)
    or in separate chunks across the stream. The caller in :func:`generate_sync`
    accumulates them independently and prefers content over reasoning at the
    end. Either field is returned as the empty string when absent.
    """
    if isinstance(chunk, str):
        return chunk, ""
    if not isinstance(chunk, dict):
        return "", ""

    content = ""
    reasoning = ""

    choices = chunk.get("choices")
    if isinstance(choices, list) and choices:
        first = choices[0]
        if isinstance(first, dict):
            delta = first.get("delta")
            if isinstance(delta, dict):
                if isinstance(delta.get("content"), str):
                    content = delta.get("content", "") or content
                # Thinking-mode models stream the chain-of-thought separately.
                # We capture both candidate keys — `reasoning_content` is the
                # DeepSeek / Qwen3-thinking convention, `reasoning` is used by
                # some OpenAI-compat shims.
                if isinstance(delta.get("reasoning_content"), str):
                    reasoning = delta.get("reasoning_content", "") or reasoning
                elif isinstance(delta.get("reasoning"), str):
                    reasoning = delta.get("reasoning", "") or reasoning
            message = first.get("message")
            if isinstance(message, dict):
                if not content and isinstance(message.get("content"), str):
                    content = message.get("content", "")
                if not reasoning and isinstance(message.get("reasoning_content"), str):
                    reasoning = message.get("reasoning_content", "")
            if not content and isinstance(first.get("text"), str):
                content = first.get("text", "")
            if not reasoning and isinstance(first.get("reasoning_content"), str):
                reasoning = first.get("reasoning_content", "")

    if not content and isinstance(chunk.get("content"), str):
        content = chunk.get("content", "")
    if not reasoning and isinstance(chunk.get("reasoning_content"), str):
        reasoning = chunk.get("reasoning_content", "")

    return content, reasoning
