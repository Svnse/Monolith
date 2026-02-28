"""
Optional BAML bridge for engine.loop.

This module is intentionally optional. It allows LoopRuntime to consume a
structured `Step` from a BAML-generated callable when available, while the
default JSON parser path remains the fallback.
"""

from __future__ import annotations

import importlib
import json
import os
from typing import Any, Callable

from engine.loop.contracts import Step


class BamlUnavailableError(RuntimeError):
    pass


def load_callable_from_dotted_path(dotted: str) -> Callable[..., Any]:
    path = str(dotted or "").strip()
    if ":" in path and "." not in path:
        path = path.replace(":", ".", 1)
    if not path or "." not in path:
        raise BamlUnavailableError("invalid BAML callable path")
    mod_name, fn_name = path.rsplit(".", 1)
    try:
        mod = importlib.import_module(mod_name)
    except Exception as exc:
        raise BamlUnavailableError(f"failed to import module '{mod_name}': {exc}") from exc
    fn = getattr(mod, fn_name, None)
    if not callable(fn):
        raise BamlUnavailableError(f"callable '{fn_name}' not found in '{mod_name}'")
    return fn


class BamlStepInferAdapter:
    """
    Wrap a BAML-generated callable as `structured_infer_fn(messages) -> Step`.

    Expected callable signature:
      fn(messages=[{"role": ..., "content": ...}], **kwargs) -> dict/object

    Expected output fields map cleanly onto `engine.loop.contracts.Step`.
    """

    def __init__(self, call_fn: Callable[..., Any], **call_kwargs: Any) -> None:
        self._call = call_fn
        self._call_kwargs = dict(call_kwargs)

    @classmethod
    def from_env(
        cls,
        env_var: str = "MONOLITH_LOOP_BAML_CALL",
        args_env_var: str = "MONOLITH_LOOP_BAML_ARGS_JSON",
        **call_kwargs: Any,
    ) -> "BamlStepInferAdapter":
        dotted = os.environ.get(env_var, "").strip()
        if not dotted:
            raise BamlUnavailableError(f"{env_var} is not set")
        extra_kwargs = cls._parse_kwargs_json(os.environ.get(args_env_var, ""))
        merged_kwargs = dict(extra_kwargs)
        merged_kwargs.update(call_kwargs)
        return cls(load_callable_from_dotted_path(dotted), **merged_kwargs)

    @classmethod
    def from_config(
        cls,
        *,
        call_path: str = "",
        call_kwargs: dict[str, Any] | None = None,
    ) -> "BamlStepInferAdapter":
        path = str(call_path or "").strip()
        kwargs = dict(call_kwargs or {})
        if path:
            return cls(load_callable_from_dotted_path(path), **kwargs)
        return cls.from_env(**kwargs)

    @staticmethod
    def _parse_kwargs_json(raw: str | None) -> dict[str, Any]:
        text = str(raw or "").strip()
        if not text:
            return {}
        try:
            data = json.loads(text)
        except Exception as exc:
            raise BamlUnavailableError(f"invalid MONOLITH_LOOP_BAML_ARGS_JSON: {exc}") from exc
        if not isinstance(data, dict):
            raise BamlUnavailableError("MONOLITH_LOOP_BAML_ARGS_JSON must decode to an object")
        return data

    def __call__(self, messages: list[dict[str, str]]) -> Step:
        raw = self._call(messages=messages, **self._call_kwargs)
        return self._to_step(raw)

    @staticmethod
    def _to_step(raw: Any) -> Step:
        if isinstance(raw, Step):
            return raw
        if hasattr(raw, "model_dump") and callable(raw.model_dump):
            data = raw.model_dump()
        elif hasattr(raw, "dict") and callable(raw.dict):
            data = raw.dict()
        elif hasattr(raw, "__dict__"):
            data = dict(vars(raw))
        elif isinstance(raw, dict):
            data = raw
        else:
            raise BamlUnavailableError(f"unsupported BAML Step output type: {type(raw).__name__}")

        if not isinstance(data, dict):
            raise BamlUnavailableError("BAML output did not normalize to dict")
        if "intent" not in data:
            raise BamlUnavailableError("BAML output missing required field: intent")

        actions = data.get("actions", [])
        if not isinstance(actions, list):
            raise BamlUnavailableError("'actions' must be a list")
        for i, action in enumerate(actions):
            if not isinstance(action, dict) or "tool" not in action:
                raise BamlUnavailableError(f"action[{i}] must include 'tool'")

        return Step(
            intent=str(data.get("intent", "")),
            response=str(data.get("response", "")),
            reasoning=str(data.get("reasoning", "")),
            actions=actions,
            self_check=str(data.get("self_check", "")),
            step_ok=(bool(data.get("step_ok")) if isinstance(data.get("step_ok"), bool) else None),
            todo_update=(str(data.get("todo_update")).strip() if isinstance(data.get("todo_update"), str) else None),
            task_finished=(
                data.get("task_finished")
                if ("task_finished" in data and (isinstance(data.get("task_finished"), bool) or data.get("task_finished") is None))
                else bool(data.get("finish", False))
            ),
            finish_summary=str(data.get("finish_summary", "")),
        )
