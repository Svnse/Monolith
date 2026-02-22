"""
Manifest-Based Mutation Oracle — OFAC v0.2 Section 2.

Tool stdout is IGNORED for determining if files changed. Only filesystem
stat + SHA-256 are authoritative.

Usage:
    oracle = ManifestOracle(workspace_root)
    before = oracle.snapshot()
    # ... execute tool ...
    after = oracle.snapshot()
    diff = oracle.diff(before, after)
    # diff == {"added": [...], "modified": [...], "deleted": [...],
    #          "manifest_hash_before": "...", "manifest_hash_after": "..."}
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core.event_ledger import canonical_json, canonical_hash


# Reuse ignore logic from checkpointing
from engine.checkpointing import _should_ignore, _parse_ignore_file, _ALWAYS_IGNORE


@dataclass(frozen=True)
class FileStatEntry:
    """Single file's stat snapshot."""
    rel_path: str
    mtime: float
    size: int
    sha256: str


class ManifestSnapshot:
    """
    Immutable point-in-time snapshot of workspace file state.

    Indexed by relative POSIX path.
    """

    def __init__(self, entries: dict[str, FileStatEntry], manifest_hash: str) -> None:
        self._entries = entries
        self._manifest_hash = manifest_hash

    @property
    def entries(self) -> dict[str, FileStatEntry]:
        return self._entries

    @property
    def manifest_hash(self) -> str:
        return self._manifest_hash

    def paths(self) -> set[str]:
        return set(self._entries.keys())


@dataclass(frozen=True)
class MutationRecord:
    """
    OFAC Ledger entry — authoritative filesystem mutation record.

    Emitted after every EXECUTE phase. Deterministically hashable.
    """
    added: list[str]
    modified: list[str]
    deleted: list[str]
    manifest_hash_before: str
    manifest_hash_after: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "mutation_record",
            "added": sorted(self.added),
            "modified": sorted(self.modified),
            "deleted": sorted(self.deleted),
            "manifest_hash_before": self.manifest_hash_before,
            "manifest_hash_after": self.manifest_hash_after,
        }

    def to_canonical_hash(self) -> str:
        """Deterministic hash of this mutation record."""
        return canonical_hash(self.to_dict())

    @property
    def has_mutations(self) -> bool:
        return bool(self.added or self.modified or self.deleted)


class ManifestOracle:
    """
    Filesystem mutation oracle for OFAC v0.2.

    Takes fast-stat snapshots before/after tool execution and produces
    authoritative MutationRecords by diffing manifests.
    """

    def __init__(self, workspace_root: Path) -> None:
        self._workspace_root = workspace_root.resolve()
        self._ignore_patterns = self._load_ignore_patterns()

        # Stat cache for fast re-snapshot (mtime+size → reuse sha256)
        self._stat_cache: dict[str, FileStatEntry] = {}

    def _load_ignore_patterns(self) -> list[str]:
        patterns: list[str] = []
        for name in (".gitignore", ".monolithignore"):
            patterns.extend(_parse_ignore_file(self._workspace_root / name))
        return patterns

    def snapshot(self) -> ManifestSnapshot:
        """
        Capture current workspace state using fast-stat heuristics.

        Only computes SHA-256 for files whose mtime or size changed
        since the last snapshot.
        """
        entries: dict[str, FileStatEntry] = {}

        for path in self._workspace_root.rglob("*"):
            if not path.is_file():
                continue

            rel = path.relative_to(self._workspace_root).as_posix()
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
                sha = cached.sha256
            else:
                sha = self._hash_file(path)

            entry = FileStatEntry(
                rel_path=rel,
                mtime=cur_mtime,
                size=cur_size,
                sha256=sha,
            )
            entries[rel] = entry
            self._stat_cache[rel] = entry

        # Compute manifest hash (canonical, over path→sha256 mapping)
        hash_map = {rel: e.sha256 for rel, e in sorted(entries.items())}
        manifest_hash = canonical_hash(hash_map)

        return ManifestSnapshot(entries, manifest_hash)

    def diff(self, before: ManifestSnapshot, after: ManifestSnapshot) -> MutationRecord:
        """
        Compute authoritative filesystem diff between two snapshots.

        Returns a MutationRecord with added/modified/deleted lists.
        """
        before_paths = before.paths()
        after_paths = after.paths()

        added = sorted(after_paths - before_paths)
        deleted = sorted(before_paths - after_paths)
        modified = []

        # Check for content changes in files present in both
        for rel in sorted(before_paths & after_paths):
            b_entry = before.entries[rel]
            a_entry = after.entries[rel]
            if b_entry.sha256 != a_entry.sha256:
                modified.append(rel)

        return MutationRecord(
            added=added,
            modified=modified,
            deleted=deleted,
            manifest_hash_before=before.manifest_hash,
            manifest_hash_after=after.manifest_hash,
        )

    @staticmethod
    def _hash_file(path: Path) -> str:
        hasher = hashlib.sha256()
        try:
            with path.open("rb") as f:
                for chunk in iter(lambda: f.read(1024 * 1024), b""):
                    hasher.update(chunk)
        except OSError:
            return hashlib.sha256(b"<unreadable>").hexdigest()
        return hasher.hexdigest()
