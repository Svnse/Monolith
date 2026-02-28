"""JSONL message persistence for Relay with observer callbacks."""

import json
import threading
import time
from pathlib import Path


class MessageStore:
    def __init__(self, path: str):
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._todos_path = self._path.parent / "relay_todos.json"
        self._messages: list[dict] = []
        self._todos: dict[int, str] = {}   # msg_id → "todo" | "done"
        self._lock = threading.Lock()
        self._callbacks: list = []
        self._load()
        self._load_todos()

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------

    def _load(self) -> None:
        if not self._path.exists():
            return
        with open(self._path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    msg = json.loads(line)
                    msg["id"] = i
                    self._messages.append(msg)
                except json.JSONDecodeError:
                    continue

    # ------------------------------------------------------------------
    # Messages
    # ------------------------------------------------------------------

    def on_message(self, callback) -> None:
        """Register callback(msg) fired on every new message."""
        self._callbacks.append(callback)

    def add(
        self,
        sender: str,
        text: str,
        msg_type: str = "chat",
        attachments: list | None = None,
        reply_to: int | None = None,
    ) -> dict:
        with self._lock:
            msg: dict = {
                "id":          len(self._messages),
                "sender":      sender,
                "text":        text,
                "type":        msg_type,
                "timestamp":   time.time(),
                "time":        time.strftime("%H:%M:%S"),
                "attachments": attachments or [],
            }
            if reply_to is not None:
                msg["reply_to"] = reply_to
            self._messages.append(msg)
            with open(self._path, "a", encoding="utf-8") as f:
                f.write(json.dumps(msg, ensure_ascii=False) + "\n")

        for cb in self._callbacks:
            try:
                cb(msg)
            except Exception:
                pass
        return msg

    def get_by_id(self, msg_id: int) -> dict | None:
        with self._lock:
            if 0 <= msg_id < len(self._messages):
                return self._messages[msg_id]
            return None

    def get_recent(self, count: int = 50) -> list[dict]:
        with self._lock:
            return list(self._messages[-count:])

    def get_since(self, since_id: int) -> list[dict]:
        with self._lock:
            return [m for m in self._messages if m["id"] > since_id]

    def clear(self) -> None:
        with self._lock:
            self._messages.clear()
            self._path.write_text("", encoding="utf-8")

    @property
    def last_id(self) -> int:
        with self._lock:
            return self._messages[-1]["id"] if self._messages else -1

    # ------------------------------------------------------------------
    # Todos
    # ------------------------------------------------------------------

    def _load_todos(self) -> None:
        if self._todos_path.exists():
            try:
                raw = json.loads(self._todos_path.read_text("utf-8"))
                self._todos = {int(k): v for k, v in raw.items()}
            except Exception:
                self._todos = {}

    def _save_todos(self) -> None:
        self._todos_path.write_text(
            json.dumps({str(k): v for k, v in self._todos.items()}, indent=2),
            "utf-8",
        )

    def add_todo(self, msg_id: int) -> bool:
        with self._lock:
            if msg_id < 0 or msg_id >= len(self._messages):
                return False
            self._todos[msg_id] = "todo"
            self._save_todos()
        return True

    def complete_todo(self, msg_id: int) -> bool:
        with self._lock:
            if msg_id not in self._todos:
                return False
            self._todos[msg_id] = "done"
            self._save_todos()
        return True

    def reopen_todo(self, msg_id: int) -> bool:
        with self._lock:
            if msg_id not in self._todos:
                return False
            self._todos[msg_id] = "todo"
            self._save_todos()
        return True

    def remove_todo(self, msg_id: int) -> bool:
        with self._lock:
            if msg_id not in self._todos:
                return False
            del self._todos[msg_id]
            self._save_todos()
        return True

    def get_todo_status(self, msg_id: int) -> str | None:
        return self._todos.get(msg_id)

    def get_todos(self) -> dict[int, str]:
        return dict(self._todos)

    def get_todo_messages(self, status: str | None = None) -> list[dict]:
        with self._lock:
            ids = (
                {k for k, v in self._todos.items() if v == status}
                if status
                else set(self._todos)
            )
            return [m for m in self._messages if m["id"] in ids]
