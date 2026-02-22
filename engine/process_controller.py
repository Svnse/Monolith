"""
ProcessGroupController — Cross-Platform OS Process Enforcement for OFAC v0.2.

The runtime NEVER calls OS process functions directly. All process lifecycle
goes through this controller which guarantees proper cleanup on STOP.

- POSIX:   setsid() + os.killpg(SIGTERM/SIGKILL)
- Windows: CreateJobObject + AssignProcessToJobObject + TerminateJobObject
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import threading
from dataclasses import dataclass, field
from typing import IO, Any


@dataclass
class ProcessHandle:
    """Opaque handle returned by start(). Tracks a managed process."""
    pid: int
    proc: subprocess.Popen
    pgid: int | None = None              # POSIX only
    job_handle: Any = None               # Windows only
    stdout_reader: threading.Thread | None = None
    stderr_reader: threading.Thread | None = None
    stdout_lines: list[str] = field(default_factory=list)
    stderr_lines: list[str] = field(default_factory=list)
    _done_event: threading.Event = field(default_factory=threading.Event)


def _stream_reader(stream: IO[str] | None, output: list[str], done_event: threading.Event) -> None:
    """Threaded reader for stdout/stderr — never blocks the caller."""
    if stream is None:
        return
    try:
        for line in stream:
            output.append(line)
    except (ValueError, OSError):
        pass
    finally:
        done_event.set()


class ProcessGroupController:
    """
    Cross-platform process group controller.

    All tool executions route through this. Supports:
      - start(cmd, cwd) → ProcessHandle
      - poll(handle) → int | None
      - terminate(handle)
      - force_kill(handle)
      - wait(handle, timeout) → int
    """

    def __init__(self) -> None:
        self._active: dict[int, ProcessHandle] = {}
        self._lock = threading.Lock()

    @property
    def active_handles(self) -> list[ProcessHandle]:
        with self._lock:
            return list(self._active.values())

    def start(
        self,
        cmd: str,
        cwd: str,
        *,
        timeout: int = 30,
        env: dict[str, str] | None = None,
        stdin_pipe: bool = False,
    ) -> ProcessHandle:
        """
        Start a subprocess inside a managed process group / job object.

        Uses threaded stream readers so the caller thread can poll for STOP.
        If stdin_pipe=True, proc.stdin will be available for writing.
        """
        kwargs: dict[str, Any] = {
            "shell": True,
            "stdout": subprocess.PIPE,
            "stderr": subprocess.PIPE,
            "stdin": subprocess.PIPE if stdin_pipe else subprocess.DEVNULL,
            "text": True,
            "cwd": cwd,
        }
        if env is not None:
            merged = dict(os.environ)
            merged.update(env)
            kwargs["env"] = merged

        if sys.platform != "win32":
            # POSIX: create new session so we get a process group
            kwargs["preexec_fn"] = os.setsid

        proc = subprocess.Popen(cmd, **kwargs)

        pgid = None
        job_handle = None

        if sys.platform != "win32":
            try:
                pgid = os.getpgid(proc.pid)
            except OSError:
                pgid = proc.pid
        else:
            job_handle = self._create_windows_job(proc)

        handle = ProcessHandle(
            pid=proc.pid,
            proc=proc,
            pgid=pgid,
            job_handle=job_handle,
        )

        # Spin up threaded readers for non-blocking stdout/stderr
        stdout_done = threading.Event()
        stderr_done = threading.Event()

        handle.stdout_reader = threading.Thread(
            target=_stream_reader,
            args=(proc.stdout, handle.stdout_lines, stdout_done),
            daemon=True,
            name=f"proc-stdout-{proc.pid}",
        )
        handle.stderr_reader = threading.Thread(
            target=_stream_reader,
            args=(proc.stderr, handle.stderr_lines, stderr_done),
            daemon=True,
            name=f"proc-stderr-{proc.pid}",
        )
        handle.stdout_reader.start()
        handle.stderr_reader.start()

        with self._lock:
            self._active[proc.pid] = handle

        return handle

    def poll(self, handle: ProcessHandle) -> int | None:
        """Non-blocking poll. Returns exit code or None if still running."""
        return handle.proc.poll()

    def wait(self, handle: ProcessHandle, timeout: float | None = None) -> int:
        """Wait for process to finish. Returns exit code."""
        exit_code = handle.proc.wait(timeout=timeout)
        # Wait for stream readers to drain
        if handle.stdout_reader:
            handle.stdout_reader.join(timeout=2.0)
        if handle.stderr_reader:
            handle.stderr_reader.join(timeout=2.0)
        self._cleanup(handle)
        return exit_code

    def terminate(self, handle: ProcessHandle) -> None:
        """
        Graceful termination. Sends SIGTERM (POSIX) or TerminateJobObject (Windows).
        """
        try:
            if sys.platform != "win32":
                # POSIX: kill entire process group
                if handle.pgid is not None:
                    try:
                        os.killpg(handle.pgid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
                else:
                    handle.proc.terminate()
            else:
                self._terminate_windows_job(handle)
        except OSError:
            pass

    def force_kill(self, handle: ProcessHandle) -> None:
        """
        Forceful kill. SIGKILL (POSIX) or TerminateJobObject (Windows).
        Cleans up tracking state.
        """
        try:
            if sys.platform != "win32":
                if handle.pgid is not None:
                    try:
                        os.killpg(handle.pgid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                else:
                    handle.proc.kill()
            else:
                self._kill_windows_job(handle)
        except OSError:
            pass
        finally:
            self._cleanup(handle)

    def terminate_all(self) -> None:
        """Terminate all active processes."""
        with self._lock:
            handles = list(self._active.values())
        for h in handles:
            self.terminate(h)

    def force_kill_all(self) -> None:
        """Force kill all active processes."""
        with self._lock:
            handles = list(self._active.values())
        for h in handles:
            self.force_kill(h)

    def get_output(self, handle: ProcessHandle) -> tuple[str, str]:
        """Get accumulated stdout and stderr."""
        return "".join(handle.stdout_lines), "".join(handle.stderr_lines)

    # ------------------------------------------------------------------
    # Windows Job Object implementation
    # ------------------------------------------------------------------

    @staticmethod
    def _create_windows_job(proc: subprocess.Popen) -> Any:
        """Create a Windows Job Object and assign the process to it."""
        if sys.platform != "win32":
            return None
        try:
            import ctypes
            from ctypes import wintypes

            kernel32 = ctypes.windll.kernel32

            # CreateJobObjectW(lpJobAttributes, lpName)
            job = kernel32.CreateJobObjectW(None, None)
            if not job:
                return None

            # Set job to kill all processes when job handle is closed
            class JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
                _fields_ = [
                    ("PerProcessUserTimeLimit", ctypes.c_int64),
                    ("PerJobUserTimeLimit", ctypes.c_int64),
                    ("LimitFlags", wintypes.DWORD),
                    ("MinimumWorkingSetSize", ctypes.c_size_t),
                    ("MaximumWorkingSetSize", ctypes.c_size_t),
                    ("ActiveProcessLimit", wintypes.DWORD),
                    ("Affinity", ctypes.POINTER(ctypes.c_ulong)),
                    ("PriorityClass", wintypes.DWORD),
                    ("SchedulingClass", wintypes.DWORD),
                ]

            class IO_COUNTERS(ctypes.Structure):
                _fields_ = [
                    ("ReadOperationCount", ctypes.c_uint64),
                    ("WriteOperationCount", ctypes.c_uint64),
                    ("OtherOperationCount", ctypes.c_uint64),
                    ("ReadTransferCount", ctypes.c_uint64),
                    ("WriteTransferCount", ctypes.c_uint64),
                    ("OtherTransferCount", ctypes.c_uint64),
                ]

            class JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
                _fields_ = [
                    ("BasicLimitInformation", JOBOBJECT_BASIC_LIMIT_INFORMATION),
                    ("IoInfo", IO_COUNTERS),
                    ("ProcessMemoryLimit", ctypes.c_size_t),
                    ("JobMemoryLimit", ctypes.c_size_t),
                    ("PeakProcessMemoryUsed", ctypes.c_size_t),
                    ("PeakJobMemoryUsed", ctypes.c_size_t),
                ]

            JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000
            JobObjectExtendedLimitInformation = 9

            info = JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
            info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE

            kernel32.SetInformationJobObject(
                job,
                JobObjectExtendedLimitInformation,
                ctypes.byref(info),
                ctypes.sizeof(info),
            )

            # Open process handle with ASSIGN rights
            PROCESS_ALL_ACCESS = 0x001F0FFF
            h_process = kernel32.OpenProcess(PROCESS_ALL_ACCESS, False, proc.pid)
            if h_process:
                kernel32.AssignProcessToJobObject(job, h_process)
                kernel32.CloseHandle(h_process)

            return job
        except Exception:
            return None

    @staticmethod
    def _terminate_windows_job(handle: ProcessHandle) -> None:
        """Terminate via Windows Job Object, fallback to proc.terminate()."""
        if handle.job_handle is not None:
            try:
                import ctypes
                kernel32 = ctypes.windll.kernel32
                kernel32.TerminateJobObject(handle.job_handle, 1)
                return
            except Exception:
                pass
        handle.proc.terminate()

    @staticmethod
    def _kill_windows_job(handle: ProcessHandle) -> None:
        """Force kill via Windows Job Object, fallback to proc.kill()."""
        if handle.job_handle is not None:
            try:
                import ctypes
                kernel32 = ctypes.windll.kernel32
                kernel32.TerminateJobObject(handle.job_handle, 9)
                kernel32.CloseHandle(handle.job_handle)
                handle.job_handle = None
                return
            except Exception:
                pass
        handle.proc.kill()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _cleanup(self, handle: ProcessHandle) -> None:
        """Remove handle from active tracking."""
        with self._lock:
            self._active.pop(handle.pid, None)
        # Close Windows job handle if still open
        if handle.job_handle is not None and sys.platform == "win32":
            try:
                import ctypes
                ctypes.windll.kernel32.CloseHandle(handle.job_handle)
            except Exception:
                pass
            handle.job_handle = None
