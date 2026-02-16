import json
import re
from pathlib import Path

from core.paths import CONFIG_DIR
from core.presence import PresenceEngine


class OperatorManager:
    def __init__(self):
        self._operators_dir = CONFIG_DIR / "operators"
        self._presence = PresenceEngine()

    def _ensure_dir(self) -> Path:
        self._operators_dir.mkdir(parents=True, exist_ok=True)
        return self._operators_dir

    def _slugify(self, name: str) -> str:
        value = re.sub(r"[^a-z0-9]+", "-", (name or "").strip().lower())
        value = re.sub(r"-+", "-", value).strip("-")
        return value or "operator"

    def _path_for_name(self, name: str) -> Path:
        return self._ensure_dir() / f"{self._slugify(name)}.json"

    def list_operators(self) -> list[dict]:
        operators = []
        for path in self._ensure_dir().glob("*.json"):
            try:
                with path.open("r", encoding="utf-8") as handle:
                    data = json.load(handle)
            except Exception:
                continue
            if isinstance(data, dict) and isinstance(data.get("name"), str):
                # Accept both new format (has "modules") and legacy (has "config")
                if isinstance(data.get("modules"), list) or isinstance(data.get("config"), dict):
                    operators.append({"name": data["name"], "path": path})
        operators.sort(key=lambda item: item["name"].lower())
        return operators

    def load_operator(self, name: str) -> dict:
        path = self._path_for_name(name)
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        if not isinstance(data, dict):
            raise ValueError("Operator payload must be a JSON object")
        return data

    def save_operator(
        self,
        name: str,
        data: dict,
        previous_data: dict | None = None,
        trigger: str = "saved",
    ) -> tuple[Path, bool]:
        """Save operator. Returns (path, drift_exceeded).

        If previous_data is provided, computes diff and updates lineage.
        If not provided, treats as genesis or manual save (no diff).
        """
        payload = dict(data or {})
        payload["name"] = name
        payload.setdefault("layout", {})
        payload.setdefault("geometry", {})
        # Strip system_prompt from config (legacy format) or module configs (new format)
        if isinstance(payload.get("config"), dict):
            payload["config"].pop("system_prompt", None)
        for mod in payload.get("modules", []):
            if isinstance(mod.get("config"), dict):
                mod["config"].pop("system_prompt", None)

        drift_exceeded = False
        if previous_data is not None:
            payload, drift_exceeded = self._presence.update_presence(payload, previous_data, trigger)
        else:
            payload = self._presence.ensure_presence(payload)

        path = self._path_for_name(name)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        return path, drift_exceeded


    def get_lineage(self, name: str) -> list[dict]:
        try:
            data = self.load_operator(name)
        except Exception:
            return []
        return self._presence.get_lineage(data)

    def get_presence_info(self, name: str) -> dict | None:
        try:
            data = self.load_operator(name)
        except Exception:
            return None
        presence = data.get("presence")
        if not isinstance(presence, dict):
            return None
        return {
            "presence_id": presence.get("presence_id", ""),
            "current_version": presence.get("current_version", 0),
            "drift_score": presence.get("drift_score", 0.0),
            "drift_threshold": presence.get("drift_threshold", 0.5),
            "lineage_length": len(presence.get("lineage", [])),
            "created_at": presence.get("created_at", ""),
        }

    def delete_operator(self, name: str) -> bool:
        path = self._path_for_name(name)
        if not path.exists():
            return False
        try:
            path.unlink()
        except OSError:
            return False
        return True
