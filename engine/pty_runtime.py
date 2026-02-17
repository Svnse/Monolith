from __future__ import annotations

import base64
import os
import pty
import re
import select
import signal
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class PtySnapshot:
    cwd: str
    env: dict[str, str]
    history: list[str]


@dataclass
class BranchPtySession:
    branch_id: str
    workspace_root: Path
    cwd: str
    env: dict[str, str]
    history: list[str] = field(default_factory=list)
    master_fd: int | None = None
    process: subprocess.Popen | None = None
    suspended: bool = True
    last_used_at: float = field(default_factory=time.monotonic)

    def _spawn(self) -> None:
        if self.process is not None and self.process.poll() is None:
            return
        env = dict(self.env)
        env.setdefault("TERM", "xterm")
        env.setdefault("HISTFILE", "")
        master_fd, slave_fd = pty.openpty()
        proc = subprocess.Popen(
            ["bash", "--noprofile", "--norc", "-i"],
            stdin=slave_fd,
            stdout=slave_fd,
            stderr=slave_fd,
            cwd=self.cwd,
            env=env,
            text=False,
            preexec_fn=os.setsid,
            close_fds=True,
        )
        os.close(slave_fd)
        self.master_fd = master_fd
        self.process = proc
        self.suspended = False
        self._read_until_quiet(0.2)
        self._exec_internal("stty -echo", timeout=2)
        if self.history:
            for entry in self.history:
                self._exec_internal(f"history -s {entry!r}", timeout=2)

    def _terminate(self) -> None:
        if self.process is not None and self.process.poll() is None:
            try:
                os.killpg(self.process.pid, signal.SIGTERM)
            except Exception:
                pass
            try:
                self.process.wait(timeout=1)
            except Exception:
                try:
                    os.killpg(self.process.pid, signal.SIGKILL)
                except Exception:
                    pass
        if self.master_fd is not None:
            try:
                os.close(self.master_fd)
            except Exception:
                pass
        self.master_fd = None
        self.process = None
        self.suspended = True

    def _read_until_quiet(self, quiet_window_s: float) -> str:
        if self.master_fd is None:
            return ""
        end = time.monotonic() + quiet_window_s
        chunks: list[bytes] = []
        while time.monotonic() < end:
            ready, _, _ = select.select([self.master_fd], [], [], 0.05)
            if not ready:
                continue
            data = os.read(self.master_fd, 4096)
            if not data:
                break
            chunks.append(data)
            end = time.monotonic() + quiet_window_s
        return b"".join(chunks).decode("utf-8", errors="replace")

    def _write_line(self, command: str) -> None:
        if self.master_fd is None:
            raise RuntimeError("PTY session not active")
        os.write(self.master_fd, command.encode("utf-8", errors="ignore") + b"\n")

    def _exec_internal(self, command: str, timeout: int) -> tuple[int | None, str]:
        token = uuid.uuid4().hex
        marker = f"__MONOLITH_DONE_{token}__"
        wrapped = f"{command}; __ec=$?; printf '\\n{marker}%s\\n' \"$__ec\""
        self._write_line(wrapped)

        deadline = time.monotonic() + timeout
        buf = ""
        exit_code: int | None = None
        while time.monotonic() < deadline:
            ready, _, _ = select.select([self.master_fd], [], [], 0.1)
            if not ready:
                continue
            data = os.read(self.master_fd, 4096)
            if not data:
                break
            buf += data.decode("utf-8", errors="replace")
            match = re.search(rf"(?:\r?\n){re.escape(marker)}(-?\d+)\r?\n", buf)
            if match:
                exit_code = int(match.group(1))
                buf = buf[: match.start()]
                break
        return exit_code, buf.strip()

    def run_command(self, command: str, timeout: int) -> tuple[int | None, str, str | None]:
        self._spawn()
        self.last_used_at = time.monotonic()
        exit_code, stdout = self._exec_internal(command, timeout=timeout)
        if exit_code is None:
            self._write_line("\x03")
            return None, stdout, f"command timed out after {timeout}s"
        self.history.append(command)
        return exit_code, stdout, None

    def snapshot(self) -> PtySnapshot:
        self._spawn()
        cwd_code, cwd_output = self._exec_internal("pwd", timeout=2)
        if cwd_code == 0 and cwd_output:
            for line in reversed([ln.strip() for ln in cwd_output.splitlines()]):
                if line.startswith("/"):
                    self.cwd = line
                    break
        env_code, env_output = self._exec_internal("env -0 | base64 -w0", timeout=2)
        if env_code == 0 and env_output:
            try:
                raw = base64.b64decode(env_output.encode("utf-8"), validate=False)
                parsed: dict[str, str] = {}
                for entry in raw.split(b"\x00"):
                    if b"=" in entry:
                        k, v = entry.split(b"=", 1)
                        parsed[k.decode("utf-8", errors="ignore")] = v.decode("utf-8", errors="replace")
                if parsed:
                    self.env = parsed
            except Exception:
                pass
        return PtySnapshot(cwd=self.cwd, env=dict(self.env), history=list(self.history))

    def suspend(self) -> None:
        self.snapshot()
        self._terminate()

    def destroy(self) -> None:
        self._terminate()


class BranchPtySessionManager:
    def __init__(self, workspace_root: Path, idle_timeout_seconds: int = 300):
        self.workspace_root = workspace_root
        self.idle_timeout_seconds = max(1, int(idle_timeout_seconds))
        self._sessions: dict[str, BranchPtySession] = {}
        self._lock = threading.RLock()

    def run(self, branch_id: str, command: str, timeout: int) -> tuple[int | None, str, str | None]:
        with self._lock:
            self._suspend_idle_locked()
            session = self._sessions.get(branch_id)
            if session is None:
                session = BranchPtySession(
                    branch_id=branch_id,
                    workspace_root=self.workspace_root,
                    cwd=str(self.workspace_root),
                    env=dict(os.environ),
                )
                self._sessions[branch_id] = session
            return session.run_command(command, timeout)

    def fork_branch(self, parent_branch_id: str, child_branch_id: str) -> None:
        with self._lock:
            parent = self._sessions.get(parent_branch_id)
            if parent is None:
                return
            snapshot = parent.snapshot()
            self.destroy_branch(child_branch_id)
            self._sessions[child_branch_id] = BranchPtySession(
                branch_id=child_branch_id,
                workspace_root=self.workspace_root,
                cwd=snapshot.cwd,
                env=snapshot.env,
                history=snapshot.history,
            )

    def destroy_branch(self, branch_id: str) -> None:
        with self._lock:
            session = self._sessions.pop(branch_id, None)
            if session is not None:
                session.destroy()

    def _suspend_idle_locked(self) -> None:
        now = time.monotonic()
        for session in self._sessions.values():
            if not session.suspended and now - session.last_used_at > self.idle_timeout_seconds:
                session.suspend()

    def destroy_all(self) -> None:
        with self._lock:
            for branch_id in list(self._sessions.keys()):
                self.destroy_branch(branch_id)


_MANAGER: BranchPtySessionManager | None = None


def get_pty_session_manager(workspace_root: Path, idle_timeout_seconds: int = 300) -> BranchPtySessionManager:
    global _MANAGER
    if _MANAGER is None or _MANAGER.workspace_root != workspace_root:
        _MANAGER = BranchPtySessionManager(workspace_root=workspace_root, idle_timeout_seconds=idle_timeout_seconds)
    return _MANAGER
