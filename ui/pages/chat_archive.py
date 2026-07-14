from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from ui.pages.chat_session import ChatSessionManager


@dataclass(frozen=True)
class ArchiveItem:
    updated_at: str
    title: str
    message_count: int
    assistant_tokens: int
    path: Path
    tooltip: str

    @property
    def date_label(self) -> str:
        return self.updated_at.split("T")[0] if self.updated_at else "Unknown date"

    @property
    def subtext(self) -> str:
        return (
            f"{self.date_label} \u2022 {self.message_count} msgs \u2022 "
            f"{self.assistant_tokens} assistant tokens"
        )


class ChatArchiveManager:
    def __init__(
        self,
        archive_dir: Path,
        sessions: ChatSessionManager,
        max_list_items: int = 500,
    ) -> None:
        self._archive_dir = Path(archive_dir)
        self._sessions = sessions
        self._max_list_items = max(1, int(max_list_items))

    @property
    def archive_dir(self) -> Path:
        return self._archive_dir

    def save_session(self, session: dict) -> Path:
        messages = session.get("messages", [])
        tree = session.get("tree")
        if isinstance(tree, dict) and "nodes" in tree:
            from core import branch_tree
            messages = branch_tree.project_to_messages(tree)
        now = self._sessions.now_iso()
        created_at = session.get("created_at") or now
        updated_at = now
        title = session.get("title") or self._sessions.derive_title(messages)
        summary = self._sessions.build_summary(messages, title)
        message_payload = []
        for idx, msg in enumerate(messages, start=1):
            entry = {
                "i": idx,
                "time": msg.get("time") or now,
                "role": msg.get("role", "user"),
                "text": msg.get("text", ""),
            }
            if msg.get("task_id"):
                entry["task_id"] = msg["task_id"]
            if msg.get("node_id"):
                entry["node_id"] = msg["node_id"]
            message_payload.append(entry)
        meta = {
            "title": title,
            "created_at": created_at,
            "updated_at": updated_at,
            "message_count": len(message_payload),
            "assistant_tokens": int(session.get("assistant_tokens", 0)),
            "summary": summary,
        }
        payload = {"meta": meta, "messages": message_payload}
        tree = session.get("tree")
        if isinstance(tree, dict) and "nodes" in tree:
            from core import branch_tree
            payload["tree"] = branch_tree.serialize(tree)
        archive_path = session.get("archive_path")
        if not archive_path:
            slug = self._sessions.slugify(title)
            stamp = now.replace(":", "-").replace(".", "-")
            archive_path = self._archive_dir / f"{slug}_{stamp}.json"
        else:
            archive_path = Path(archive_path)
        archive_path.parent.mkdir(parents=True, exist_ok=True)
        with archive_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        session["archive_path"] = str(archive_path)
        session["created_at"] = created_at
        session["updated_at"] = updated_at
        session["summary"] = summary
        return archive_path

    def load_session(self, archive_path: Path) -> dict:
        with archive_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        meta = data.get("meta", {})
        messages = []
        for msg in data.get("messages", []):
            role = msg.get("role", "user")
            text = msg.get("text", "")
            time = msg.get("time", meta.get("updated_at", self._sessions.now_iso()))
            messages.append({"i": msg.get("i"), "time": time, "role": role, "text": text})
        for msg, raw in zip(messages, data.get("messages", [])):
            for key in ("task_id", "node_id"):
                if raw.get(key):
                    msg[key] = raw[key]
        session = self._sessions.create_session(
            messages=messages,
            created_at=meta.get("created_at"),
            updated_at=meta.get("updated_at"),
            archive_path=str(archive_path),
            summary=meta.get("summary", []),
            title=meta.get("title"),
            assistant_tokens=int(meta.get("assistant_tokens", meta.get("token_count", 0))),
        )
        from ui.pages import session_tree
        if session_tree.active() and isinstance(data.get("tree"), dict):
            from core import branch_tree
            try:
                session["tree"] = branch_tree.deserialize(data["tree"])
                session["messages"] = branch_tree.project_to_messages(session["tree"])
            except ValueError:
                session.pop("tree", None)                          # corrupt block → legacy fallback
        return session

    def list_archives(self, limit: int | None = None) -> list[ArchiveItem]:
        effective_limit = self._max_list_items if limit is None else max(1, int(limit))
        items: list[ArchiveItem] = []
        def _mtime(path: Path) -> float:
            try:
                return path.stat().st_mtime
            except OSError:
                return 0.0

        candidates = sorted(self._archive_dir.glob("*.json"), key=_mtime, reverse=True)
        for path in candidates[:effective_limit]:
            try:
                with path.open("r", encoding="utf-8") as handle:
                    data = json.load(handle)
            except Exception:
                continue
            meta = data.get("meta", {})
            title = meta.get("title", path.stem)
            summary = meta.get("summary", [])
            tooltip = "\n".join(summary) if summary else title
            updated_at = meta.get("updated_at", "")
            message_count = meta.get("message_count", len(data.get("messages", [])))
            assistant_tokens = int(meta.get("assistant_tokens", meta.get("token_count", 0)))
            items.append(
                ArchiveItem(
                    updated_at=updated_at,
                    title=title,
                    message_count=message_count,
                    assistant_tokens=assistant_tokens,
                    path=path,
                    tooltip=tooltip,
                )
            )
        items.sort(key=lambda item: item.updated_at, reverse=True)
        return items

    def delete_archive(self, archive_path: Path) -> bool:
        try:
            archive_path.unlink()
            return True
        except OSError:
            return False
