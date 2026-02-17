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


class CheckpointStore:
    def __init__(self, workspace_root: Path = DEFAULT_WORKSPACE_ROOT, root: Path = CHECKPOINT_ROOT):
        self.workspace_root = workspace_root.resolve()
        self.root = root.resolve()
        self.snapshots_dir = self.root / "snapshots"
        self.blobs_dir = self.root / "blobs"
        self.index_path = self.root / "index.json"
        self._lock = threading.Lock()
        self._ensure_dirs()

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

            payload = {
                "metadata": asdict(metadata),
                "workspace_manifest": manifest,
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

    def _ensure_dirs(self) -> None:
        self.snapshots_dir.mkdir(parents=True, exist_ok=True)
        self.blobs_dir.mkdir(parents=True, exist_ok=True)
        if not self.index_path.exists():
            self._save_index({"checkpoints": {}, "branch_checkpoints": {}})

    def _build_workspace_manifest(self) -> dict[str, str]:
        manifest: dict[str, str] = {}
        for path in self.workspace_root.rglob("*"):
            if not path.is_file():
                continue
            if self._is_inside_checkpoint_store(path):
                continue
            digest = self._hash_file(path)
            manifest[path.relative_to(self.workspace_root).as_posix()] = digest
            self._ensure_blob(path, digest)
        return manifest

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
