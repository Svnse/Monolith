"""
Protocol Adapter — Phase 2a of Monolith Agent Contract V2.

Strict boundary between raw LLM provider output and canonical AgentMessage.
Produces ProtocolAdapterResult with status: native | rejected | recovered.

Invariant A: Runtime executes only canonical AgentMessage objects.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal

from engine.agent_runtime import AgentMessage, ToolCall


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

class AdapterStatus(str, Enum):
    NATIVE = "native"
    REJECTED = "rejected"
    RECOVERED = "recovered"


@dataclass
class ProtocolAdapterResult:
    status: AdapterStatus
    canonical_message: AgentMessage | None
    failure_code: str | None = None
    raw_hash: str = ""
    adapter_version: str = "2a.1"
    recovery_detail: str | None = None


# ---------------------------------------------------------------------------
# Model profiles — prompt template + expected tool-call format
# ---------------------------------------------------------------------------

class ToolCallFormat(str, Enum):
    """How the model is expected to emit tool calls."""
    NATIVE_OPENAI = "native_openai"      # model uses native function calling via tool_calls
    XML_BLOCK = "xml_block"              # model emits <tool_call>{"name":...}</tool_call>
    JSON_BLOCK = "json_block"            # model emits ```json\n{"name":...}\n```
    RAW_JSON = "raw_json"               # model emits bare {"name":...} in content


@dataclass
class ModelProfile:
    """Per-model-class configuration for the protocol adapter."""
    profile_id: str
    tool_call_format: ToolCallFormat
    supports_native_tools: bool = False
    strict_mode: bool = False            # if True, recovery is disabled
    max_format_retries: int = 1
    grammar_support: bool = False        # Phase 5: supports grammar-constrained decoding
    description: str = ""


# Built-in profiles
PROFILES: dict[str, ModelProfile] = {
    "native": ModelProfile(
        profile_id="native",
        tool_call_format=ToolCallFormat.NATIVE_OPENAI,
        supports_native_tools=True,
        strict_mode=True,
        max_format_retries=0,
        description="Model with reliable native function calling (GPT-4, Claude, etc.)",
    ),
    "local_xml": ModelProfile(
        profile_id="local_xml",
        tool_call_format=ToolCallFormat.XML_BLOCK,
        supports_native_tools=False,
        strict_mode=False,
        max_format_retries=2,
        grammar_support=True,
        description="Local GGUF model instructed to use <tool_call> XML blocks",
    ),
    "local_json": ModelProfile(
        profile_id="local_json",
        tool_call_format=ToolCallFormat.JSON_BLOCK,
        supports_native_tools=False,
        strict_mode=False,
        max_format_retries=2,
        grammar_support=True,
        description="Local GGUF model instructed to use JSON code blocks",
    ),
    "local_native": ModelProfile(
        profile_id="local_native",
        tool_call_format=ToolCallFormat.NATIVE_OPENAI,
        supports_native_tools=True,
        strict_mode=False,
        max_format_retries=1,
        grammar_support=True,
        description="Local model with native tool calling support (e.g. functionary, hermes)",
    ),
}


def get_profile(profile_id: str) -> ModelProfile:
    return PROFILES.get(profile_id, PROFILES["local_xml"])


# ---------------------------------------------------------------------------
# Grammar-constrained decoding profiles (Phase 5)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GrammarProfile:
    """Grammar constraint applied during decoding to enforce/forbid tool call structure."""
    profile_id: str
    tool_policy: str                      # "required" | "forbidden" | "optional"
    model_profile_id: str
    grammar_type: str                     # "none" | "bnf" | "regex" | "json_schema"
    grammar_spec: str | None = None       # the actual grammar content
    description: str = ""


# BNF grammars for XML tool call format
_XML_REQUIRED_BNF = r'''
root   ::= preamble tool-call
preamble ::= [^\x00]*
tool-call ::= "<tool_call>" ws json-obj ws "</tool_call>"
ws     ::= [ \t\n]*
json-obj ::= "{" ws "\"name\"" ws ":" ws string ws "," ws ("\"args\"" | "\"arguments\"") ws ":" ws "{" [^}]* "}" ws "}"
string ::= "\"" [^"]+ "\""
'''.strip()

_XML_FORBIDDEN_BNF = r'''
root ::= safe-char+
safe-char ::= [^<] | "<" [^t] | "<t" [^o] | "<to" [^o] | "<too" [^l] | "<tool" [^_]
'''.strip()

# BNF grammars for JSON block format
_JSON_REQUIRED_BNF = r'''
root   ::= preamble code-fence
preamble ::= [^\x00]*
code-fence ::= "```json" ws json-obj ws "```"
ws     ::= [ \t\n]*
json-obj ::= "{" ws "\"name\"" ws ":" ws string ws "," ws ("\"args\"" | "\"arguments\"") ws ":" ws "{" [^}]* "}" ws "}"
string ::= "\"" [^"]+ "\""
'''.strip()

_JSON_FORBIDDEN_BNF = r'''
root ::= safe-char+
safe-char ::= [^`] | "`" [^`] | "``" [^`]
'''.strip()


GRAMMAR_PROFILES: dict[tuple[str, str], GrammarProfile] = {
    # local_xml
    ("local_xml", "required"): GrammarProfile(
        profile_id="local_xml_required",
        tool_policy="required",
        model_profile_id="local_xml",
        grammar_type="bnf",
        grammar_spec=_XML_REQUIRED_BNF,
        description="Enforces <tool_call> XML block output",
    ),
    ("local_xml", "forbidden"): GrammarProfile(
        profile_id="local_xml_forbidden",
        tool_policy="forbidden",
        model_profile_id="local_xml",
        grammar_type="bnf",
        grammar_spec=_XML_FORBIDDEN_BNF,
        description="Forbids <tool_call> in output",
    ),
    # local_json
    ("local_json", "required"): GrammarProfile(
        profile_id="local_json_required",
        tool_policy="required",
        model_profile_id="local_json",
        grammar_type="bnf",
        grammar_spec=_JSON_REQUIRED_BNF,
        description="Enforces JSON code block tool call output",
    ),
    ("local_json", "forbidden"): GrammarProfile(
        profile_id="local_json_forbidden",
        tool_policy="forbidden",
        model_profile_id="local_json",
        grammar_type="bnf",
        grammar_spec=_JSON_FORBIDDEN_BNF,
        description="Forbids JSON code block tool calls in output",
    ),
    # local_native — native function calling; grammar enforces via API, not BNF
    ("local_native", "required"): GrammarProfile(
        profile_id="local_native_required",
        tool_policy="required",
        model_profile_id="local_native",
        grammar_type="json_schema",
        grammar_spec=None,
        description="Native function calling enforced by API schema",
    ),
    ("local_native", "forbidden"): GrammarProfile(
        profile_id="local_native_forbidden",
        tool_policy="forbidden",
        model_profile_id="local_native",
        grammar_type="none",
        grammar_spec=None,
        description="Tool calls suppressed by omitting tools from API",
    ),
    # native — strict profile, no grammar needed (API handles it)
    ("native", "required"): GrammarProfile(
        profile_id="native_required",
        tool_policy="required",
        model_profile_id="native",
        grammar_type="none",
        grammar_spec=None,
        description="Native function calling; no grammar needed",
    ),
    ("native", "forbidden"): GrammarProfile(
        profile_id="native_forbidden",
        tool_policy="forbidden",
        model_profile_id="native",
        grammar_type="none",
        grammar_spec=None,
        description="Tool calls suppressed by omitting tools from API",
    ),
}


def get_grammar_profile(
    model_profile_id: str,
    tool_policy: str,
) -> GrammarProfile | None:
    """
    Returns the grammar profile for a given model profile and tool policy.

    Returns None for 'optional' (no constraint applied) or unknown combinations.
    """
    if tool_policy == "optional":
        return None
    return GRAMMAR_PROFILES.get((model_profile_id, tool_policy))


# ---------------------------------------------------------------------------
# Content parsers for recovery path
# ---------------------------------------------------------------------------

# <tool_call>{"name": "...", "args": {...}}</tool_call>
_XML_TOOL_CALL_RE = re.compile(
    r"<tool_call>\s*(\{.*?\})\s*</tool_call>",
    re.DOTALL,
)

# ```json\n{"name": "...", ...}\n```  or  ```\n{"name": "...", ...}\n```
_JSON_BLOCK_RE = re.compile(
    r"```(?:json)?\s*\n?\s*(\{.*?\})\s*\n?\s*```",
    re.DOTALL,
)

# Bare JSON object that looks like a tool call (has "name" key)
_RAW_JSON_RE = re.compile(
    r'(\{\s*"name"\s*:\s*"[^"]+"\s*,\s*"arg(?:ument)?s"\s*:\s*\{.*?\}\s*\})',
    re.DOTALL,
)

# Tags that local models commonly leak from prompt scaffolding
_SCAFFOLD_TAG_RE = re.compile(
    r"</?(response|answer|output|result|thinking|think|scratchpad)>",
    re.IGNORECASE,
)


def _sanitize_content(content: str | None) -> str | None:
    """Strip leaked prompt scaffolding tags from model output."""
    if not content:
        return content
    cleaned = _SCAFFOLD_TAG_RE.sub("", content).strip()
    return cleaned or None


def _parse_tool_json(raw_json: str, call_idx: int) -> ToolCall | None:
    """Parse a single JSON blob into a ToolCall, or None on failure."""
    try:
        parsed = json.loads(raw_json)
    except (json.JSONDecodeError, TypeError):
        return None

    if not isinstance(parsed, dict):
        return None

    name = parsed.get("name")
    if not isinstance(name, str) or not name:
        return None

    # Support both "args" and "arguments"
    arguments = parsed.get("arguments") or parsed.get("args") or {}
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except Exception:
            arguments = {}
    if not isinstance(arguments, dict):
        arguments = {}

    call_id = parsed.get("id") or f"recovered_{call_idx}"
    return ToolCall(id=str(call_id), name=name, arguments=arguments)


def _try_recover_xml(content: str) -> list[ToolCall]:
    """Extract tool calls from <tool_call>...</tool_call> blocks."""
    calls: list[ToolCall] = []
    for idx, match in enumerate(_XML_TOOL_CALL_RE.finditer(content)):
        call = _parse_tool_json(match.group(1), idx)
        if call is not None:
            calls.append(call)
    return calls


def _try_recover_json_block(content: str) -> list[ToolCall]:
    """Extract tool calls from ```json ... ``` code blocks."""
    calls: list[ToolCall] = []
    for idx, match in enumerate(_JSON_BLOCK_RE.finditer(content)):
        call = _parse_tool_json(match.group(1), idx)
        if call is not None:
            calls.append(call)
    return calls


def _try_recover_raw_json(content: str) -> list[ToolCall]:
    """Extract tool calls from bare JSON objects in content."""
    calls: list[ToolCall] = []
    for idx, match in enumerate(_RAW_JSON_RE.finditer(content)):
        call = _parse_tool_json(match.group(1), idx)
        if call is not None:
            calls.append(call)
    return calls


# ---------------------------------------------------------------------------
# Raw hash utility
# ---------------------------------------------------------------------------

def _compute_raw_hash(raw: dict[str, Any]) -> str:
    """Deterministic hash of raw provider response for transcript."""
    try:
        serialized = json.dumps(raw, sort_keys=True, separators=(",", ":"), default=str)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()
    except Exception:
        return hashlib.sha256(repr(raw).encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Normalizer — equivalent to current normalize_openai_response but stricter
# ---------------------------------------------------------------------------

def _normalize_native_openai(raw: dict[str, Any]) -> AgentMessage | None:
    """
    Parse native OpenAI chat-completion format into AgentMessage.
    Returns None if the response is structurally invalid.
    """
    choices = raw.get("choices", []) if isinstance(raw, dict) else []
    if not choices:
        return None

    message = choices[0].get("message", {}) if isinstance(choices[0], dict) else {}
    if not isinstance(message, dict):
        return None

    role = message.get("role", "assistant")
    if role not in {"system", "user", "assistant", "tool"}:
        role = "assistant"

    # Content
    content = message.get("content")
    if isinstance(content, list):
        content = "".join(
            part.get("text", "") for part in content if isinstance(part, dict)
        )
    elif not isinstance(content, str):
        content = None

    # Tool calls
    tool_calls_raw = message.get("tool_calls")
    tool_calls: list[ToolCall] = []
    if isinstance(tool_calls_raw, list):
        for idx, item in enumerate(tool_calls_raw):
            if not isinstance(item, dict):
                continue
            fn = item.get("function", {})
            if not isinstance(fn, dict):
                continue
            name = fn.get("name")
            if not isinstance(name, str) or not name:
                continue
            arguments = fn.get("arguments", {})
            if isinstance(arguments, str):
                try:
                    parsed = json.loads(arguments)
                    arguments = parsed if isinstance(parsed, dict) else {}
                except Exception:
                    arguments = {}
            if not isinstance(arguments, dict):
                arguments = {}
            tool_calls.append(ToolCall(
                id=str(item.get("id") or f"call_{idx}"),
                name=name,
                arguments=arguments,
            ))

    return AgentMessage(
        role=role,
        content=content,
        tool_calls=tool_calls or None,
        tool_call_id=message.get("tool_call_id") if isinstance(message.get("tool_call_id"), str) else None,
        name=message.get("name") if isinstance(message.get("name"), str) else None,
    )


# ---------------------------------------------------------------------------
# ProtocolAdapter — the main boundary
# ---------------------------------------------------------------------------

class ProtocolAdapter:
    """
    Strict boundary between raw LLM provider output and canonical AgentMessage.

    This is the single point where provider-specific output is translated
    into the canonical form that the runtime loop consumes.

    Contract invariant: runtime NEVER touches raw provider output directly.
    """

    VERSION = "2a.1"

    def __init__(self, profile: ModelProfile | None = None) -> None:
        self._profile = profile or PROFILES["local_xml"]

    @property
    def profile(self) -> ModelProfile:
        return self._profile

    def adapt(self, raw_response: dict[str, Any]) -> ProtocolAdapterResult:
        """
        Transform raw provider response into canonical AgentMessage.

        Returns ProtocolAdapterResult with:
          - status=native:    native tool_calls parsed successfully
          - status=recovered: tool calls extracted from content (recovery path)
          - status=rejected:  response is structurally invalid or policy-rejected
        """
        raw_hash = _compute_raw_hash(raw_response)

        # --- Step 1: Try native OpenAI format ---
        native_msg = _normalize_native_openai(raw_response)

        if native_msg is None:
            return ProtocolAdapterResult(
                status=AdapterStatus.REJECTED,
                canonical_message=None,
                failure_code="INVALID_RESPONSE_STRUCTURE",
                raw_hash=raw_hash,
                adapter_version=self.VERSION,
            )

        # Sanitize content to strip leaked prompt scaffolding tags
        native_msg = AgentMessage(
            role=native_msg.role,
            content=_sanitize_content(native_msg.content),
            tool_calls=native_msg.tool_calls,
            tool_call_id=native_msg.tool_call_id,
            name=native_msg.name,
        )

        # If we got native tool calls, accept immediately
        if native_msg.tool_calls:
            return ProtocolAdapterResult(
                status=AdapterStatus.NATIVE,
                canonical_message=native_msg,
                raw_hash=raw_hash,
                adapter_version=self.VERSION,
            )

        # --- Step 2: No native tool calls. Try content recovery. ---

        # If strict mode, no recovery allowed
        if self._profile.strict_mode:
            # Model returned content with no tool calls — this is either a
            # legitimate final answer or a protocol failure. In strict mode
            # we accept it as-is (no tool calls = chat completion).
            return ProtocolAdapterResult(
                status=AdapterStatus.NATIVE,
                canonical_message=native_msg,
                raw_hash=raw_hash,
                adapter_version=self.VERSION,
            )

        # Try recovery based on expected format
        content = native_msg.content or ""
        recovered_calls: list[ToolCall] = []
        recovery_detail = ""

        if self._profile.tool_call_format == ToolCallFormat.XML_BLOCK:
            recovered_calls = _try_recover_xml(content)
            if recovered_calls:
                recovery_detail = "recovered_from_xml_block"

        if not recovered_calls and self._profile.tool_call_format == ToolCallFormat.JSON_BLOCK:
            recovered_calls = _try_recover_json_block(content)
            if recovered_calls:
                recovery_detail = "recovered_from_json_block"

        # Fallback: try all recovery methods in order
        if not recovered_calls:
            recovered_calls = _try_recover_xml(content)
            if recovered_calls:
                recovery_detail = "recovered_from_xml_block_fallback"

        if not recovered_calls:
            recovered_calls = _try_recover_json_block(content)
            if recovered_calls:
                recovery_detail = "recovered_from_json_block_fallback"

        if not recovered_calls:
            recovered_calls = _try_recover_raw_json(content)
            if recovered_calls:
                recovery_detail = "recovered_from_raw_json_fallback"

        if recovered_calls:
            # Strip the tool-call markup and scaffold tags from content
            clean_content = content
            for pattern in (_XML_TOOL_CALL_RE, _JSON_BLOCK_RE):
                clean_content = pattern.sub("", clean_content)
            clean_content = _sanitize_content(clean_content)

            recovered_msg = AgentMessage(
                role=native_msg.role,
                content=clean_content,
                tool_calls=recovered_calls,
                tool_call_id=native_msg.tool_call_id,
                name=native_msg.name,
            )
            return ProtocolAdapterResult(
                status=AdapterStatus.RECOVERED,
                canonical_message=recovered_msg,
                raw_hash=raw_hash,
                adapter_version=self.VERSION,
                recovery_detail=recovery_detail,
            )

        # No tool calls found anywhere — return as-is (chat completion)
        return ProtocolAdapterResult(
            status=AdapterStatus.NATIVE,
            canonical_message=native_msg,
            raw_hash=raw_hash,
            adapter_version=self.VERSION,
        )
