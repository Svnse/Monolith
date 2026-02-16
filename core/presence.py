from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import uuid


_IGNORED_KEYS = {"name", "presence", "layout", "geometry"}


@dataclass(frozen=True)
class PresenceSnapshot:
    version: int
    timestamp: str
    trigger: str
    config_hash: str
    diff: dict


@dataclass
class PresenceMeta:
    presence_id: str
    created_at: str
    current_version: int
    drift_score: float
    drift_threshold: float


class PresenceEngine:
    def __init__(self) -> None:
        pass

    def compute_hash(self, data: dict) -> str:
        behavioral = self._behavioral_data(data)
        payload = json.dumps(behavioral, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def compute_diff(self, old_data: dict, new_data: dict) -> dict:
        old_behavioral = self._behavioral_data(old_data)
        new_behavioral = self._behavioral_data(new_data)
        diff: dict[str, dict] = {}
        self._compare_nodes(old_behavioral, new_behavioral, "", diff)
        return diff

    def compute_drift(self, diff: dict, total_keys: int) -> float:
        drift = len(diff) / max(total_keys, 1)
        return max(0.0, min(1.0, float(drift)))

    def count_leaves(self, data: dict) -> int:
        return self._count_leaves(data)

    def record_snapshot(
        self,
        operator_data: dict,
        trigger: str,
        previous_data: dict | None = None,
    ) -> PresenceSnapshot:
        config_hash = self.compute_hash(operator_data)
        timestamp = self._utc_now_iso()

        if previous_data is None:
            diff = {}
            version = 1
        else:
            diff = self.compute_diff(previous_data, operator_data)
            presence = operator_data.get("presence", {})
            current_version = presence.get("current_version", 0) if isinstance(presence, dict) else 0
            version = int(current_version) + 1

        return PresenceSnapshot(
            version=version,
            timestamp=timestamp,
            trigger=trigger,
            config_hash=config_hash,
            diff=diff,
        )

    def ensure_presence(self, operator_data: dict) -> dict:
        presence = operator_data.get("presence")
        if isinstance(presence, dict):
            return operator_data

        created_at = self._utc_now_iso()
        meta = PresenceMeta(
            presence_id=str(uuid.uuid4()),
            created_at=created_at,
            current_version=1,
            drift_score=0.0,
            drift_threshold=0.5,
        )
        genesis = self.record_snapshot(operator_data, trigger="created", previous_data=None)
        operator_data["presence"] = {
            **asdict(meta),
            "lineage": [asdict(genesis)],
        }
        return operator_data

    def update_presence(
        self,
        operator_data: dict,
        previous_data: dict,
        trigger: str,
    ) -> tuple[dict, bool]:
        payload = self.ensure_presence(operator_data)
        snapshot = self.record_snapshot(payload, trigger=trigger, previous_data=previous_data)

        presence = payload.setdefault("presence", {})
        lineage = presence.setdefault("lineage", [])
        lineage.append(asdict(snapshot))

        behavioral = self._behavioral_data(payload)
        total_keys = self.count_leaves(behavioral)
        drift_score = self.compute_drift(snapshot.diff, total_keys)

        presence["current_version"] = snapshot.version
        presence["drift_score"] = drift_score
        threshold = float(presence.get("drift_threshold", 0.5))
        drift_exceeded = drift_score >= threshold
        return payload, drift_exceeded

    def get_lineage(self, operator_data: dict) -> list[dict]:
        presence = operator_data.get("presence")
        if not isinstance(presence, dict):
            return []
        lineage = presence.get("lineage")
        return lineage if isinstance(lineage, list) else []

    def revert_to_version(self, operator_data: dict, version: int) -> dict | None:
        presence = operator_data.get("presence")
        if not isinstance(presence, dict):
            return None
        lineage = presence.get("lineage")
        if not isinstance(lineage, list):
            return None

        target_exists = any(isinstance(s, dict) and s.get("version") == version for s in lineage)
        if not target_exists:
            return None

        reverted = deepcopy(operator_data)
        for snapshot in reversed(lineage):
            if not isinstance(snapshot, dict):
                continue
            snapshot_version = snapshot.get("version")
            if not isinstance(snapshot_version, int):
                continue
            if snapshot_version <= version:
                break
            diff = snapshot.get("diff", {})
            if isinstance(diff, dict):
                self._apply_reverse_diff(reverted, diff)

        return reverted

    def _behavioral_data(self, data: dict) -> dict:
        if not isinstance(data, dict):
            return {}
        return {k: deepcopy(v) for k, v in data.items() if k not in _IGNORED_KEYS}

    def _compare_nodes(self, old_node, new_node, path: str, out: dict) -> None:
        if isinstance(old_node, dict) and isinstance(new_node, dict):
            keys = set(old_node.keys()) | set(new_node.keys())
            for key in sorted(keys):
                new_path = f"{path}.{key}" if path else str(key)
                self._compare_nodes(old_node.get(key), new_node.get(key), new_path, out)
            return

        if isinstance(old_node, list) and isinstance(new_node, list):
            if len(old_node) != len(new_node):
                length_path = f"{path}.length" if path else "length"
                out[length_path] = {"old": len(old_node), "new": len(new_node)}
            for idx, (old_item, new_item) in enumerate(zip(old_node, new_node)):
                new_path = f"{path}.{idx}" if path else str(idx)
                self._compare_nodes(old_item, new_item, new_path, out)
            return

        if old_node != new_node:
            out[path or "value"] = {"old": old_node, "new": new_node}

    def _count_leaves(self, node) -> int:
        if isinstance(node, dict):
            return sum(self._count_leaves(v) for v in node.values())
        if isinstance(node, list):
            return sum(self._count_leaves(v) for v in node)
        return 1

    def _apply_reverse_diff(self, operator_data: dict, diff: dict) -> None:
        for dotpath in sorted(diff.keys(), key=lambda item: item.count("."), reverse=True):
            change = diff.get(dotpath)
            if not isinstance(change, dict):
                continue
            old_value = deepcopy(change.get("old"))
            if dotpath.endswith(".length"):
                list_path = dotpath[:-7]
                self._set_list_length(operator_data, list_path, old_value)
                continue
            self._set_dotpath_value(operator_data, dotpath, old_value)

    def _set_list_length(self, data: dict, dotpath: str, length) -> None:
        if not isinstance(length, int) or length < 0:
            return
        target = self._get_dotpath_value(data, dotpath)
        if not isinstance(target, list):
            return
        while len(target) > length:
            target.pop()
        while len(target) < length:
            target.append(None)

    def _get_dotpath_value(self, data: dict, dotpath: str):
        node = data
        if not dotpath:
            return node
        for token in dotpath.split("."):
            if isinstance(node, dict):
                if token not in node:
                    return None
                node = node[token]
            elif isinstance(node, list):
                try:
                    idx = int(token)
                except ValueError:
                    return None
                if idx < 0 or idx >= len(node):
                    return None
                node = node[idx]
            else:
                return None
        return node

    def _set_dotpath_value(self, data: dict, dotpath: str, value) -> None:
        tokens = dotpath.split(".")
        node = data
        for token in tokens[:-1]:
            if isinstance(node, dict):
                node = node.get(token)
            elif isinstance(node, list):
                try:
                    idx = int(token)
                except ValueError:
                    return
                if idx < 0 or idx >= len(node):
                    return
                node = node[idx]
            else:
                return
            if node is None:
                return

        leaf = tokens[-1]
        if isinstance(node, dict):
            node[leaf] = value
        elif isinstance(node, list):
            try:
                idx = int(leaf)
            except ValueError:
                return
            if 0 <= idx < len(node):
                node[idx] = value

    def _utc_now_iso(self) -> str:
        return datetime.now(timezone.utc).isoformat()
