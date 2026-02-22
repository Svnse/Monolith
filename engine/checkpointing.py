from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import threading
import time
from typing import Any
from uuid import uuid4

from core.config import DEFAULT_WORKSPACE_ROOT


CHECKPOINT_ROOT = Path(os.getenv("MONOLITH_CHECKPOINT_ROOT", DEFAULT_WORKSPACE_ROOT / ".monolith" / "checkpoints"))
CHECKPOINT_STORAGE_CEILING_BYTES = int(os.getenv("MONOLITH_CHECKPOINT_STORAGE_CEILING_MB", "512")) * 1024 * 1024
WORKSPACE_MANAGED_MARKER = ".monolith_workspace"

# Default ignore patterns (always excluded from manifests)
_ALWAYS_IGNORE = {
    ".git",
    ".monolith",
    "node_modules",
    "__pycache__",
    ".venv",
    "venv",
    ".env",
    ".tox",
    ".mypy_cache",
    ".pytest_cache",
    "dist",
    "build",
    ".eggs",
    "*.egg-info",
}


@dataclass
class CheckpointMetadata:
    checkpoint_id: str
    node_id: str | None
    branch_id: str
    timestamp: str
    message_history: list[dict[str, Any]]
    pending_action: dict[str, Any]
    capabilities: dict[str, Any]
    pty_state_ref: str | None


@dataclass
class FileEntry:
    """Stat-heuristic manifest entry: hash only recomputed when mtime/size change."""
    mtime: float
    size: int
    sha256: str


def _parse_ignore_file(path: Path) -> list[str]:
    """Parse a .gitignore / .monolithignore file into pattern list."""
    patterns: list[str] = []
    if not path.exists():
        return patterns
    try:
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            patterns.append(line)
    except Exception:
        pass
    return patterns


def _should_ignore(rel_path: str, ignore_patterns: list[str]) -> bool:
    """Check if a relative path matches any ignore pattern."""
    from fnmatch import fnmatch

    parts = rel_path.replace("\\", "/").split("/")

    # Check each path component against always-ignore set
    for part in parts:
        if part in _ALWAYS_IGNORE:
            return True
        for pattern in _ALWAYS_IGNORE:
            if "*" in pattern and fnmatch(part, pattern):
                return True

    # Check against user-defined ignore patterns
    for pattern in ignore_patterns:
        # Directory pattern (ends with /)
        if pattern.endswith("/"):
            dir_pat = pattern.rstrip("/")
            for part in parts[:-1]:
                if fnmatch(part, dir_pat):
                    return True
        else:
            # File pattern — match against full relative path or basename
            if fnmatch(parts[-1], pattern) or fnmatch(rel_path, pattern):
                return True

    return False


class CheckpointStore:
    def __init__(self, workspace_root: Path = DEFAULT_WORKSPACE_ROOT, root: Path = CHECKPOINT_ROOT):
        self.workspace_root = workspace_root.resolve()
        self.root = root.resolve()
        self.snapshots_dir = self.root / "snapshots"
        self.blobs_dir = self.root / "blobs"
        self.index_path = self.root / "index.json"
        self._lock = threading.RLock()
        self._ensure_dirs()

        # Stat-cache for fast-stat heuristics: path -> FileEntry
        self._stat_cache: dict[str, FileEntry] = {}

        # Parsed ignore patterns (loaded once)
        self._ignore_patterns = self._load_ignore_patterns()

    def _load_ignore_patterns(self) -> list[str]:
        """Load .gitignore and .monolithignore from workspace root."""
        patterns: list[str] = []
        for name in (".gitignore", ".monolithignore"):
            patterns.extend(_parse_ignore_file(self.workspace_root / name))
        return patterns

    def create_checkpoint(
        self,
        *,
        branch_id: str,
        node_id: str | None,
        message_history: list[dict[str, Any]],
        pending_action: dict[str, Any],
        capabilities: dict[str, Any] | None = None,
        pty_state_ref: str | None = None,
    ) -> dict[str, Any]:
        with self._lock:
            checkpoint_id = f"ckpt-{int(time.time() * 1000)}-{uuid4().hex[:8]}"
            manifest = self._build_workspace_manifest()
            metadata = CheckpointMetadata(
                checkpoint_id=checkpoint_id,
                node_id=node_id,
                branch_id=branch_id,
                timestamp=datetime.now(timezone.utc).isoformat(),
                message_history=message_history,
                pending_action=pending_action,
                capabilities=capabilities or {},
                pty_state_ref=pty_state_ref,
            )

            # Convert manifest to storable form (path -> sha256 only)
            storable_manifest = {p: entry.sha256 for p, entry in manifest.items()}

            payload = {
                "metadata": asdict(metadata),
                "workspace_manifest": storable_manifest,
                "manifest_hash": self._hash_manifest(storable_manifest),
            }
            snapshot_path = self.snapshots_dir / f"{checkpoint_id}.json"
            snapshot_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

            index = self._load_index()
            index.setdefault("checkpoints", {})[checkpoint_id] = {
                "branch_id": branch_id,
                "node_id": node_id,
                "timestamp": metadata.timestamp,
                "path": str(snapshot_path),
            }
            index.setdefault("branch_checkpoints", {}).setdefault(branch_id, []).append(checkpoint_id)
            self._save_index(index)

            self.garbage_collect(index=index)
            return {"checkpoint_id": checkpoint_id, "snapshot_path": str(snapshot_path)}

    def garbage_collect(self, *, pruned_branches: list[str] | None = None, index: dict[str, Any] | None = None) -> dict[str, int]:
        with self._lock:
            idx = index or self._load_index()
            removed_snapshots = self._remove_pruned_branch_snapshots(idx, pruned_branches or [])
            removed_by_ceiling = self._prune_to_storage_ceiling(idx)
            removed_blobs = self._remove_unreferenced_blobs()
            self._save_index(idx)
            return {
                "removed_snapshots": removed_snapshots + removed_by_ceiling,
                "removed_blobs": removed_blobs,
            }

    def mark_branch_pruned(self, branch_id: str) -> dict[str, int]:
        return self.garbage_collect(pruned_branches=[branch_id])

    def restore_checkpoint(self, checkpoint_id: str, workspace_root: Path) -> bool | dict[str, Any]:
        with self._lock:
            index = self._load_index()
            entry = index.get("checkpoints", {}).get(checkpoint_id)
            if not isinstance(entry, dict):
                return False
            snapshot_path = Path(entry.get("path", ""))
            if not snapshot_path.exists():
                return False
            try:
                payload = json.loads(snapshot_path.read_text(encoding="utf-8"))
            except Exception:
                return False

            manifest = payload.get("workspace_manifest", {})
            if not isinstance(manifest, dict):
                return False

            expected_manifest_hash = payload.get("manifest_hash")
            actual_manifest_hash = self._hash_manifest(manifest)
            if expected_manifest_hash and expected_manifest_hash != actual_manifest_hash:
                return False

            target_root = workspace_root.resolve()
            target_root.mkdir(parents=True, exist_ok=True)

            managed_workspace = (target_root / WORKSPACE_MANAGED_MARKER).exists()
            skipped_deletes = False
            if managed_workspace:
                existing_files = [p for p in target_root.rglob("*") if p.is_file() and not self._is_inside_checkpoint_store(p)]
                manifest_paths = {target_root / rel for rel in manifest.keys()}
                for path in existing_files:
                    if path not in manifest_paths:
                        path.unlink(missing_ok=True)
            else:
                skipped_deletes = True

            for rel_path, digest in manifest.items():
                blob_path = self.blobs_dir / str(digest)
                if not blob_path.exists():
                    return False
                if self._hash_file(blob_path) != str(digest):
                    return False
                destination = (target_root / rel_path).resolve()
                if not str(destination).startswith(str(target_root)):
                    return False
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(blob_path, destination)

            rebuilt = self._build_manifest_for_root(target_root)
            result_ok = self._hash_manifest(rebuilt) == actual_manifest_hash
            if not result_ok:
                return False
            if skipped_deletes:
                return {"ok": True, "warning": "workspace not marked as managed; skipped deleting extraneous files"}
            return True

    def load_checkpoint_snapshot(self, checkpoint_id: str) -> dict[str, Any] | None:
        snapshot_path = self.snapshots_dir / f"{checkpoint_id}.json"
        if not snapshot_path.exists():
            return None
        try:
            payload = json.loads(snapshot_path.read_text(encoding="utf-8"))
        except Exception:
            return None
        return payload if isinstance(payload, dict) else None

    def load_manifest(self, checkpoint_id: str) -> dict[str, str] | None:
        snapshot = self.load_checkpoint_snapshot(checkpoint_id)
        if not isinstance(snapshot, dict):
            return None
        manifest = snapshot.get("workspace_manifest")
        if not isinstance(manifest, dict):
            return None
        return {str(path): str(digest) for path, digest in manifest.items()}

    def read_blob_bytes(self, digest: str) -> bytes | None:
        blob_path = self.blobs_dir / str(digest)
        if not blob_path.exists() or not blob_path.is_file():
            return None
        try:
            return blob_path.read_bytes()
        except Exception:
            return None

    def read_blob_text(self, digest: str, max_bytes: int = 400_000) -> str | None:
        raw = self.read_blob_bytes(digest)
        if raw is None or len(raw) > max_bytes:
            return None
        try:
            return raw.decode("utf-8", errors="strict")
        except UnicodeDecodeError:
            return None

    def _ensure_dirs(self) -> None:
        self.snapshots_dir.mkdir(parents=True, exist_ok=True)
        self.blobs_dir.mkdir(parents=True, exist_ok=True)
        if not self.index_path.exists():
            self._save_index({"checkpoints": {}, "branch_checkpoints": {}})

    def _build_workspace_manifest(self) -> dict[str, FileEntry]:
        """
        Build workspace manifest using fast-stat heuristics.

        Only computes SHA-256 when mtime or size has changed since last snapshot.
        Respects .gitignore and .monolithignore patterns.
        """
        manifest: dict[str, FileEntry] = {}

        for path in self.workspace_root.rglob("*"):
            if not path.is_file():
                continue
            if self._is_inside_checkpoint_store(path):
                continue

            rel = path.relative_to(self.workspace_root).as_posix()

            # Ignore check
            if _should_ignore(rel, self._ignore_patterns):
                continue

            try:
                st = path.stat()
            except OSError:
                continue

            cur_mtime = st.st_mtime
            cur_size = st.st_size

            # Fast-stat heuristic: reuse cached hash if mtime+size unchanged
            cached = self._stat_cache.get(rel)
            if cached is not None and cached.mtime == cur_mtime and cached.size == cur_size:
                digest = cached.sha256
            else:
                digest = self._hash_file(path)

            entry = FileEntry(mtime=cur_mtime, size=cur_size, sha256=digest)
            manifest[rel] = entry
            self._stat_cache[rel] = entry

            self._ensure_blob(path, digest)

        return manifest

    def _build_manifest_for_root(self, root: Path) -> dict[str, str]:
        manifest: dict[str, str] = {}
        for path in root.rglob("*"):
            if not path.is_file() or self._is_inside_checkpoint_store(path):
                continue
            rel = path.relative_to(root).as_posix()
            if _should_ignore(rel, self._ignore_patterns):
                continue
            manifest[rel] = self._hash_file(path)
        return manifest

    def _hash_manifest(self, manifest: dict[str, str]) -> str:
        canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def _is_inside_checkpoint_store(self, path: Path) -> bool:
        try:
            path.resolve().relative_to(self.root)
            return True
        except ValueError:
            return False

    def _hash_file(self, path: Path) -> str:
        hasher = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    def _ensure_blob(self, source_path: Path, digest: str) -> None:
        blob_path = self.blobs_dir / digest
        if blob_path.exists():
            return
        shutil.copy2(source_path, blob_path)

    def _load_index(self) -> dict[str, Any]:
        try:
            return json.loads(self.index_path.read_text(encoding="utf-8"))
        except Exception:
            return {"checkpoints": {}, "branch_checkpoints": {}}

    def _save_index(self, index: dict[str, Any]) -> None:
        self.index_path.write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")

    def _remove_pruned_branch_snapshots(self, index: dict[str, Any], pruned_branches: list[str]) -> int:
        removed = 0
        for branch_id in pruned_branches:
            checkpoint_ids = list(index.get("branch_checkpoints", {}).get(branch_id, []))
            for checkpoint_id in checkpoint_ids:
                entry = index.get("checkpoints", {}).pop(checkpoint_id, None)
                if not entry:
                    continue
                snapshot_path = Path(entry["path"])
                if snapshot_path.exists():
                    snapshot_path.unlink()
                removed += 1
            index.get("branch_checkpoints", {}).pop(branch_id, None)
        return removed

    def _prune_to_storage_ceiling(self, index: dict[str, Any]) -> int:
        removed = 0
        while self._store_size_bytes() > CHECKPOINT_STORAGE_CEILING_BYTES:
            checkpoints = index.get("checkpoints", {})
            if not checkpoints:
                break
            oldest_id = min(checkpoints.items(), key=lambda item: item[1].get("timestamp", ""))[0]
            entry = checkpoints.pop(oldest_id)
            branch_id = entry.get("branch_id")
            if branch_id and branch_id in index.get("branch_checkpoints", {}):
                index["branch_checkpoints"][branch_id] = [
                    cid for cid in index["branch_checkpoints"][branch_id] if cid != oldest_id
                ]
                if not index["branch_checkpoints"][branch_id]:
                    index["branch_checkpoints"].pop(branch_id, None)
            snapshot_path = Path(entry["path"])
            if snapshot_path.exists():
                snapshot_path.unlink()
            removed += 1
        return removed

    def _store_size_bytes(self) -> int:
        total = 0
        for path in self.root.rglob("*"):
            if path.is_file():
                total += path.stat().st_size
        return total

    def _remove_unreferenced_blobs(self) -> int:
        referenced: set[str] = set()
        for snapshot_path in self.snapshots_dir.glob("*.json"):
            try:
                payload = json.loads(snapshot_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            manifest = payload.get("workspace_manifest", {})
            if isinstance(manifest, dict):
                referenced.update(str(v) for v in manifest.values())

        removed = 0
        for blob_path in self.blobs_dir.iterdir():
            if blob_path.is_file() and blob_path.name not in referenced:
                blob_path.unlink()
                removed += 1
        return removed


_STORE: CheckpointStore | None = None


def get_checkpoint_store() -> CheckpointStore:
    global _STORE
    if _STORE is None:
        _STORE = CheckpointStore()
    return _STORE
