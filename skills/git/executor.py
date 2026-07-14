"""Git status/diff/log/branch executor (dynamic skill loader path)."""
import subprocess
from pathlib import Path


def _coerce_int(value, default, lo, hi):
    try:
        v = int(value)
    except (TypeError, ValueError):
        return default
    return max(lo, min(hi, v))


def _find_repo_root():
    here = Path(__file__).resolve().parent
    for ancestor in [here] + list(here.parents):
        if (ancestor / ".git").exists():
            return str(ancestor)
    return None


def run(cmd: dict, _ctx) -> str:
    verb = str(cmd.get("verb", "")).strip().lower()
    cwd = str(cmd.get("cwd", "")).strip() or _find_repo_root()

    if verb == "status":
        gitcmd = ["git", "status", "--short", "--branch"]
    elif verb == "diff":
        gitcmd = ["git", "diff"] if bool(cmd.get("full", False)) else ["git", "diff", "--stat"]
    elif verb == "log":
        limit = _coerce_int(cmd.get("limit", 10), 10, 1, 100)
        gitcmd = ["git", "log", "--oneline", "-n", str(limit)]
    elif verb == "branch":
        gitcmd = ["git", "branch", "--show-current"]
    else:
        return f"[git: unknown verb '{verb}' - expected status/diff/log/branch]"

    try:
        proc = subprocess.run(
            gitcmd,
            cwd=cwd,
            timeout=30,
            capture_output=True,
            text=True,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return "[git: timed out after 30s]"
    except FileNotFoundError:
        return "[git: 'git' not found on PATH]"
    except Exception as exc:
        return f"[git: error - {exc}]"

    output = proc.stdout or ""
    if proc.stderr:
        output = (output + ("\n" if output else "") + proc.stderr).rstrip()

    if proc.returncode != 0:
        truncated = output[:1000] if output else ""
        return f"[git: {verb} failed, exit_code={proc.returncode}]\n{truncated}".rstrip()

    if len(output) > 4000:
        output = output[:4000] + "\n... [truncated]"
    if not output.strip():
        if verb == "status":
            return f"[git: {verb}, clean working tree]"
        return f"[git: {verb}, no output]"

    return f"[git: {verb}, exit_code={proc.returncode}]\n{output}"
