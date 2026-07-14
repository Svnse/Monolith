from __future__ import annotations

from dataclasses import dataclass
import io
import json
from pathlib import Path
import re
import struct
from typing import Any


REGISTRY_PATH = Path(__file__).with_name("model_registry.json")


@dataclass(frozen=True)
class ResolvedModelPreset:
    family_id: str
    family_name: str
    mode: str
    variant_id: str | None
    confidence: str
    sampler: dict[str, Any]
    output: dict[str, Any]
    context_window: int | None
    chat_template: str | None
    warnings: tuple[str, ...]
    notes: tuple[str, ...]
    capabilities: dict[str, bool] = None  # type: ignore[assignment]

    @property
    def preset_id(self) -> str:
        parts = [self.family_id]
        if self.variant_id:
            parts.append(self.variant_id)
        parts.append(self.mode)
        return ":".join(parts)


GGUF_MAGIC = b"GGUF"
GGUF_TYPE_UINT8 = 0
GGUF_TYPE_INT8 = 1
GGUF_TYPE_UINT16 = 2
GGUF_TYPE_INT16 = 3
GGUF_TYPE_UINT32 = 4
GGUF_TYPE_INT32 = 5
GGUF_TYPE_FLOAT32 = 6
GGUF_TYPE_BOOL = 7
GGUF_TYPE_STRING = 8
GGUF_TYPE_ARRAY = 9
GGUF_TYPE_UINT64 = 10
GGUF_TYPE_INT64 = 11
GGUF_TYPE_FLOAT64 = 12


def load_model_registry() -> dict[str, Any]:
    try:
        return json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {"schema_version": 1, "generic_fallback": {}, "families": []}


def _normalized(text: str | None) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(text or "").lower()).strip()


def _read_exact(handle: io.BufferedReader, size: int) -> bytes:
    data = handle.read(size)
    if len(data) != size:
        raise EOFError("Unexpected end of GGUF metadata")
    return data


def _read_u32(handle: io.BufferedReader) -> int:
    return struct.unpack("<I", _read_exact(handle, 4))[0]


def _read_u64(handle: io.BufferedReader) -> int:
    return struct.unpack("<Q", _read_exact(handle, 8))[0]


def _read_string(handle: io.BufferedReader) -> str:
    length = _read_u64(handle)
    return _read_exact(handle, length).decode("utf-8", errors="ignore")


def _read_scalar(handle: io.BufferedReader, type_id: int):
    if type_id == GGUF_TYPE_UINT8:
        return struct.unpack("<B", _read_exact(handle, 1))[0]
    if type_id == GGUF_TYPE_INT8:
        return struct.unpack("<b", _read_exact(handle, 1))[0]
    if type_id == GGUF_TYPE_UINT16:
        return struct.unpack("<H", _read_exact(handle, 2))[0]
    if type_id == GGUF_TYPE_INT16:
        return struct.unpack("<h", _read_exact(handle, 2))[0]
    if type_id == GGUF_TYPE_UINT32:
        return struct.unpack("<I", _read_exact(handle, 4))[0]
    if type_id == GGUF_TYPE_INT32:
        return struct.unpack("<i", _read_exact(handle, 4))[0]
    if type_id == GGUF_TYPE_FLOAT32:
        return struct.unpack("<f", _read_exact(handle, 4))[0]
    if type_id == GGUF_TYPE_BOOL:
        return bool(struct.unpack("<?", _read_exact(handle, 1))[0])
    if type_id == GGUF_TYPE_STRING:
        return _read_string(handle)
    if type_id == GGUF_TYPE_UINT64:
        return struct.unpack("<Q", _read_exact(handle, 8))[0]
    if type_id == GGUF_TYPE_INT64:
        return struct.unpack("<q", _read_exact(handle, 8))[0]
    if type_id == GGUF_TYPE_FLOAT64:
        return struct.unpack("<d", _read_exact(handle, 8))[0]
    raise ValueError(f"Unsupported GGUF metadata type: {type_id}")


def _read_value(handle: io.BufferedReader, type_id: int):
    if type_id == GGUF_TYPE_ARRAY:
        element_type = _read_u32(handle)
        length = _read_u64(handle)
        values = []
        for _ in range(length):
            values.append(_read_value(handle, element_type))
        return values
    return _read_scalar(handle, type_id)


def read_gguf_metadata(model_path: str | None) -> dict[str, Any]:
    path = Path(str(model_path or ""))
    if not path.exists() or not path.is_file():
        return {}
    metadata: dict[str, Any] = {}
    try:
        with path.open("rb") as handle:
            if _read_exact(handle, 4) != GGUF_MAGIC:
                return {}
            _version = _read_u32(handle)
            _tensor_count = _read_u64(handle)
            kv_count = _read_u64(handle)
            for _ in range(kv_count):
                key = _read_string(handle)
                value_type = _read_u32(handle)
                metadata[key] = _read_value(handle, value_type)
    except Exception:
        return {}
    return metadata


def metadata_model_name(metadata: dict[str, Any] | None) -> str | None:
    meta = metadata or {}
    for key in ("general.name", "general.basename", "tokenizer.ggml.model", "tokenizer.ggml.name"):
        value = meta.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def metadata_context_window(metadata: dict[str, Any] | None) -> int | None:
    meta = metadata or {}
    if "context_length" in meta:
        try:
            return int(meta["context_length"])
        except (TypeError, ValueError):
            pass
    for key, value in meta.items():
        if key.endswith(".context_length") or key.endswith(".n_ctx_train"):
            try:
                return int(value)
            except (TypeError, ValueError):
                continue
    return None


def _score_patterns(patterns: list[str], *texts: str) -> int:
    score = 0
    for pattern in patterns or []:
        try:
            compiled = re.compile(pattern)
        except re.error:
            continue
        for text in texts:
            if text and compiled.search(text):
                score = max(score, 120)
    return score


def _score_aliases(aliases: list[str], *texts: str) -> int:
    score = 0
    normalized_texts = [_normalized(text) for text in texts if text]
    for alias in aliases or []:
        alias_norm = _normalized(alias)
        if not alias_norm:
            continue
        for text in normalized_texts:
            if alias_norm == text:
                score = max(score, 180)
            elif alias_norm in text or text in alias_norm:
                score = max(score, 90)
    return score


def _pick_mode(entry: dict[str, Any]) -> str:
    modes = entry.get("modes") or {}
    default_mode = str((entry.get("defaults") or {}).get("default_mode") or "").strip()
    if default_mode and default_mode in modes:
        return default_mode
    if "chat" in modes:
        return "chat"
    if modes:
        return next(iter(modes))
    return "chat"


def _resolve_output(mode_entry: dict[str, Any], context_window: int | None, fallback: dict[str, Any]) -> dict[str, Any]:
    output = dict(fallback or {})
    output.update(mode_entry.get("output") or {})
    default_max = output.get("default_max_output_tokens")
    hard_cap = output.get("hard_cap_output_tokens")
    reserved = int(output.get("reserved_context_tokens") or 0)
    if context_window:
        remaining = max(256, int(context_window) - reserved)
        if default_max is None:
            output["default_max_output_tokens"] = remaining
        else:
            output["default_max_output_tokens"] = min(int(default_max), remaining)
        if hard_cap is not None:
            output["hard_cap_output_tokens"] = min(int(hard_cap), remaining)
    return output


def resolve_model_preset(
    model_path: str | None,
    metadata_name: str | None = None,
    metadata: dict[str, Any] | None = None,
    registry: dict[str, Any] | None = None,
) -> ResolvedModelPreset:
    data = registry or load_model_registry()
    fallback_sampler = dict((data.get("generic_fallback") or {}).get("sampler") or {})
    fallback_output = dict((data.get("generic_fallback") or {}).get("output") or {})
    families = data.get("families") or []

    raw_path = str(model_path or "")
    file_name = Path(raw_path).name
    metadata_name = metadata_name or metadata_model_name(metadata)
    metadata_context = metadata_context_window(metadata)
    metadata_architecture = str((metadata or {}).get("general.architecture") or "").strip()
    best_score = -1
    best_family: dict[str, Any] | None = None
    best_variant: dict[str, Any] | None = None

    for family in families:
        match = family.get("match") or {}
        score = 0
        score = max(score, _score_aliases(match.get("family_aliases") or [], file_name, raw_path, metadata_name))
        score = max(score, _score_patterns(match.get("filename_patterns") or [], file_name, raw_path))
        score = max(score, _score_patterns(match.get("metadata_name_patterns") or [], metadata_name))
        score = max(score, _score_patterns(match.get("metadata_architecture_patterns") or [], metadata_architecture))
        family_variant = None
        for variant in family.get("variants") or []:
            variant_match = variant.get("match") or {}
            variant_score = score
            variant_score += _score_patterns(variant_match.get("filename_patterns") or [], file_name, raw_path)
            variant_score += _score_patterns(variant_match.get("metadata_name_patterns") or [], metadata_name)
            variant_score += _score_patterns(variant_match.get("metadata_architecture_patterns") or [], metadata_architecture)
            if variant_score > score and variant_score > best_score:
                best_score = variant_score
                best_family = family
                best_variant = variant
                family_variant = variant
        if family_variant is None and score > best_score:
            best_score = score
            best_family = family
            best_variant = None

    if best_family is None or best_score <= 0:
        return ResolvedModelPreset(
            family_id="generic_fallback",
            family_name="Generic fallback",
            mode="chat",
            variant_id=None,
            confidence="fallback",
            sampler=fallback_sampler,
            output=fallback_output,
            context_window=metadata_context,
            chat_template=None,
            warnings=(),
            notes=(),
            capabilities={},
        )

    mode = _pick_mode(best_family)
    family_defaults = best_family.get("defaults") or {}
    family_mode_entry = dict((best_family.get("modes") or {}).get(mode) or {})
    context_window = metadata_context or family_defaults.get("context_window")
    chat_template = family_defaults.get("chat_template")
    warnings = list(family_mode_entry.get("warnings") or [])
    notes = list(family_mode_entry.get("notes") or [])
    if best_variant:
        overrides = best_variant.get("overrides") or {}
        if overrides.get("context_window") is not None:
            context_window = overrides.get("context_window")
        if overrides.get("chat_template") is not None:
            chat_template = overrides.get("chat_template")
        variant_modes = overrides.get("modes") or {}
        if mode in variant_modes:
            merged_mode_entry = dict(family_mode_entry)
            merged_mode_entry.update(variant_modes[mode] or {})
            family_mode_entry = merged_mode_entry
            warnings = list(family_mode_entry.get("warnings") or warnings)
            notes = list(family_mode_entry.get("notes") or notes)

    sampler = dict(fallback_sampler)
    sampler.update(family_mode_entry.get("sampler") or {})
    output = _resolve_output(family_mode_entry, int(context_window) if context_window else None, fallback_output)
    confidence = str((best_variant or best_family).get("research", {}).get("confidence") or best_family.get("research", {}).get("confidence") or "medium")

    capabilities = dict(best_family.get("capabilities") or {})

    return ResolvedModelPreset(
        family_id=str(best_family.get("id") or "unknown"),
        family_name=str(best_family.get("display_name") or best_family.get("id") or "Unknown"),
        mode=mode,
        variant_id=str(best_variant.get("id")) if best_variant and best_variant.get("id") else None,
        confidence=confidence,
        sampler=sampler,
        output=output,
        context_window=int(context_window) if context_window else None,
        chat_template=str(chat_template) if chat_template else None,
        warnings=tuple(str(item) for item in warnings),
        notes=tuple(str(item) for item in notes),
        capabilities=capabilities,
    )


def apply_model_preset(config: dict[str, Any], preset: ResolvedModelPreset) -> dict[str, Any]:
    updated = dict(config or {})
    sampler = dict(preset.sampler or {})
    output = dict(preset.output or {})
    mapped = {
        "temp": sampler.get("temperature"),
        "top_p": sampler.get("top_p"),
        "top_k": sampler.get("top_k"),
        "min_p": sampler.get("min_p"),
        "repetition_penalty": sampler.get("repeat_penalty"),
        "presence_penalty": sampler.get("presence_penalty"),
        "max_tokens": output.get("default_max_output_tokens"),
        "ctx_limit": preset.context_window,
        "pipeline_preset_id": preset.preset_id,
    }
    for key, value in mapped.items():
        if value is not None:
            updated[key] = value
    return updated


def describe_model_preset(preset: ResolvedModelPreset) -> str:
    label = preset.family_name
    if preset.variant_id:
        label = f"{label} ({preset.variant_id})"
    return f"{label} [{preset.mode}]"
