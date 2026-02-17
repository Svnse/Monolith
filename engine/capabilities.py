from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from fnmatch import fnmatch
from pathlib import PurePosixPath
import time
import uuid


class CapabilityScope(str, Enum):
    READ = "READ"
    WRITE = "WRITE"
    EXEC = "EXEC"
    NETWORK = "NETWORK"


TOOL_SCOPE_MAP: dict[str, CapabilityScope] = {
    "read_file": CapabilityScope.READ,
    "list_dir": CapabilityScope.READ,
    "grep_search": CapabilityScope.READ,
    "write_file": CapabilityScope.WRITE,
    "apply_patch": CapabilityScope.WRITE,
    "run_cmd": CapabilityScope.EXEC,
}


@dataclass
class CapabilityToken:
    token_id: str
    scope: CapabilityScope
    path_pattern: str = "**"
    constraints: dict[str, object] = field(default_factory=dict)
    issued_at: float = field(default_factory=time.time)
    expires_at: float | None = None
    branch_id: str | None = None
    revoked: bool = False
    inherited_from: str | None = None

    def is_active(self, now: float | None = None) -> bool:
        now_ts = now if now is not None else time.time()
        if self.revoked:
            return False
        if self.expires_at is None:
            return True
        return now_ts < self.expires_at

    def matches_path(self, path: str | None) -> bool:
        candidate = _normalize_path(path)
        if not candidate:
            candidate = "."
        pattern = _normalize_path(self.path_pattern)
        return fnmatch(candidate, pattern)


@dataclass
class AuthorizationResult:
    ok: bool
    error: str | None = None
    token: CapabilityToken | None = None


class CapabilityManager:
    def __init__(self) -> None:
        self._tokens: dict[str, CapabilityToken] = {}
        self._branch_tokens: dict[str, set[str]] = {}

    def issue_token(
        self,
        *,
        branch_id: str,
        scope: CapabilityScope,
        path_pattern: str = "**",
        constraints: dict[str, object] | None = None,
        ttl_seconds: int | None = None,
        inherited_from: str | None = None,
    ) -> CapabilityToken:
        now = time.time()
        expires_at = None if ttl_seconds is None else now + max(1, int(ttl_seconds))
        normalized_pattern = _normalize_path(path_pattern)
        token = CapabilityToken(
            token_id=f"cap_{uuid.uuid4().hex[:12]}",
            scope=scope,
            path_pattern=normalized_pattern,
            constraints=constraints or {},
            issued_at=now,
            expires_at=expires_at,
            branch_id=branch_id,
            inherited_from=inherited_from,
        )
        self._tokens[token.token_id] = token
        self._branch_tokens.setdefault(branch_id, set()).add(token.token_id)
        return token

    def authorize(self, *, branch_id: str, tool: str, path: str | None) -> AuthorizationResult:
        required_scope = TOOL_SCOPE_MAP.get(tool)
        if required_scope is None:
            return AuthorizationResult(False, f"no capability scope mapping for tool: {tool}")

        token_ids = self._branch_tokens.get(branch_id, set())
        now = time.time()
        for token_id in token_ids:
            token = self._tokens[token_id]
            if not token.is_active(now):
                continue
            if token.scope != required_scope:
                continue
            candidate = _normalize_path(path)
            pattern = _normalize_path(token.path_pattern)
            print(f"[CAPABILITY_AUTH] tool={tool} branch={branch_id} candidate={candidate} pattern={pattern}")
            if not token.matches_path(path):
                continue
            if not self._matches_constraints(token, tool, path):
                continue
            return AuthorizationResult(True, token=token)

        return AuthorizationResult(
            False,
            f"authorization denied: tool '{tool}' requires {required_scope.value} capability for path '{path or '.'}'",
        )

    def fork_branch(self, parent_branch_id: str, child_branch_id: str) -> None:
        inherited = self._branch_tokens.get(parent_branch_id, set())
        for token_id in inherited:
            source = self._tokens[token_id]
            if not source.is_active():
                continue
            ttl_seconds = None
            if source.expires_at is not None:
                ttl_seconds = int(max(1, source.expires_at - time.time()))
            self.issue_token(
                branch_id=child_branch_id,
                scope=source.scope,
                path_pattern=source.path_pattern,
                constraints=dict(source.constraints),
                ttl_seconds=ttl_seconds,
                inherited_from=source.token_id,
            )


    def _matches_constraints(self, token: CapabilityToken, tool: str, path: str | None) -> bool:
        allowed_tools = token.constraints.get("allowed_tools")
        if isinstance(allowed_tools, list) and tool not in {str(item) for item in allowed_tools}:
            return False

        denied_paths = token.constraints.get("denied_path_patterns")
        normalized_path = _normalize_path(path)
        if isinstance(denied_paths, list):
            for pattern in denied_paths:
                if isinstance(pattern, str) and fnmatch(normalized_path, pattern):
                    return False

        return True

    def revoke(self, *, branch_id: str, token_id: str) -> bool:
        if token_id not in self._branch_tokens.get(branch_id, set()):
            return False
        token = self._tokens.get(token_id)
        if token is None:
            return False
        token.revoked = True
        return True

    def update_token(
        self,
        *,
        branch_id: str,
        token_id: str,
        path_pattern: str | None = None,
        ttl_seconds: int | None = None,
        constraints: dict[str, object] | None = None,
    ) -> CapabilityToken | None:
        if token_id not in self._branch_tokens.get(branch_id, set()):
            return None
        token = self._tokens.get(token_id)
        if token is None:
            return None
        if path_pattern is not None:
            token.path_pattern = path_pattern
        if ttl_seconds is not None:
            token.expires_at = time.time() + max(1, int(ttl_seconds))
        if constraints is not None:
            token.constraints = dict(constraints)
        return token

    def active_tokens(self, branch_id: str) -> list[CapabilityToken]:
        token_ids = self._branch_tokens.get(branch_id, set())
        now = time.time()
        return [
            self._tokens[token_id]
            for token_id in token_ids
            if token_id in self._tokens and self._tokens[token_id].is_active(now)
        ]


def extract_tool_path(tool: str, arguments: dict) -> str | None:
    if not isinstance(arguments, dict):
        return None
    if tool in {"read_file", "write_file", "list_dir", "grep_search", "apply_patch"}:
        raw = arguments.get("path", ".")
        if isinstance(raw, str):
            return PurePosixPath(raw).as_posix()
    return "."


def _normalize_path(path: str | None) -> str:
    if path is None:
        return "."
    raw = path.strip()
    if not raw:
        return "."
    return PurePosixPath(raw).as_posix()
