from __future__ import annotations

import json
from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QApplication,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QMessageBox,
    QVBoxLayout,
    QWidget,
)

import core.style as _s
from core.history_search import search_archives
from ui.pages.chat_archive import ChatArchiveManager
from ui.pages.chat_session import ChatSessionManager


# Rough characters-per-token estimate for English chat text. Real tokenizers
# vary (~3.5–4.5 chars/tok); 4 is a safe middle that doesn't require shipping
# tiktoken. The value is used only for the "used/ctx" hint in the archive
# list — not for any hard budget enforcement.
_CHARS_PER_TOKEN: int = 4


def _estimate_total_tokens(messages: list[dict]) -> int:
    """Sum tokens across ALL roles in the chat, not just assistant output.

    Counts user / assistant / tool_call / tool_result text. Without this, the
    archive list showed only the assistant's streamed token count, which made
    chats with heavy tool-use look artificially small relative to ctx.
    """
    total_chars = 0
    for msg in messages or []:
        if not isinstance(msg, dict):
            continue
        text = str(msg.get("text") or "")
        total_chars += len(text)
    return total_chars // _CHARS_PER_TOKEN


# ── slash-command parsing ─────────────────────────────────────────────────


# Help string shown in the placeholder + reported on /help
_COMMANDS_HELP = (
    "Search or /command  •  /new  /save  /clear [all|chat|logs]  /help"
)


def _parse_command(text: str) -> tuple[str, list[str]] | None:
    """Parse a search input as a slash command. Returns (verb, args) on
    a leading-slash match, or None if the text is a normal search query."""
    raw = (text or "").strip()
    if not raw.startswith("/"):
        return None
    parts = raw[1:].split()
    if not parts:
        return None
    return parts[0].lower(), parts[1:]


class ArchiveBrowserPanel(QWidget):
    sig_new_requested = Signal()
    sig_clear_requested = Signal(str)          # arg: scope = "chat" | "all" | "logs"
    sig_session_loaded = Signal(object)
    sig_current_archive_deleted = Signal()
    sig_summarize_requested = Signal(str)      # arg: full chat text to summarize
    sig_save_requested = Signal()
    sig_status_message = Signal(str)

    def __init__(self, archive_dir: Path, archive_manager: ChatArchiveManager | None = None, parent=None):
        super().__init__(parent)
        self._archive = archive_manager or ChatArchiveManager(archive_dir, ChatSessionManager(""))
        self._current_archive_path: str | None = None
        self._controller = None

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(8)

        # ── Search bar / command line ────────────────────────────────────
        # Replaces the previous NEW/SAVE/LOAD/DELETE/CLEAR button row. The
        # search QLineEdit doubles as a command line: typed text filters
        # the list as before, but lines starting with "/" are parsed on
        # Enter and dispatched to the appropriate handler.
        self.archive_search = QLineEdit()
        self.archive_search.setObjectName("archive_search")
        self.archive_search.setPlaceholderText(_COMMANDS_HELP)
        self.archive_search.setClearButtonEnabled(True)
        self.archive_search.textChanged.connect(self._on_search_text_changed)
        self.archive_search.returnPressed.connect(self._on_search_submit)
        root.addWidget(self.archive_search)

        # ── Archive list ─────────────────────────────────────────────────
        # Selection styling overrides the default Windows-style focus
        # rectangle that produced the visible "ugly white box" around
        # the clicked row.
        self.archive_list = QListWidget()
        self.archive_list.setObjectName("archive_list")
        self.archive_list.setFrameShape(QListWidget.NoFrame)
        self.archive_list.setStyleSheet(
            f"QListWidget {{ background: transparent; border: none; outline: 0; }}"
            f"QListWidget::item {{ padding: 6px 8px; border-radius: 3px; "
            f"color: {_s.FG_TEXT}; }}"
            f"QListWidget::item:hover {{ background: {_s.BG_INPUT}; }}"
            f"QListWidget::item:selected {{ background: {_s.ACCENT_PRIMARY}; "
            f"color: {_s.BG_MAIN}; }}"
            f"QListWidget::item:focus {{ outline: none; border: none; }}"
        )
        self.archive_list.itemDoubleClicked.connect(lambda _item: self._load_selected_archive())
        self.archive_list.setContextMenuPolicy(Qt.CustomContextMenu)
        self.archive_list.customContextMenuRequested.connect(self._show_context_menu)
        root.addWidget(self.archive_list, 1)

        self._session_provider = None

    # ── controller / session wiring ───────────────────────────────────

    def bind_controller(self, controller) -> None:
        self._controller = controller
        if controller is None:
            self._session_provider = None
            return
        self._session_provider = getattr(controller, "current_session_data", None)

    def set_session_provider(self, provider) -> None:
        self._session_provider = provider

    def set_current_archive_path(self, archive_path: str | None) -> None:
        self._current_archive_path = str(archive_path) if archive_path else None

    # ── refresh / search ──────────────────────────────────────────────

    def _ctx_limit(self) -> int:
        """Pull the live ctx_limit from the controller's state if available.
        Falls back to 0 (unknown) which suppresses the '/ctx' suffix."""
        ctrl = self._controller
        if ctrl is None:
            return 0
        state = getattr(ctrl, "state", None)
        if state is None:
            return 0
        try:
            return int(getattr(state, "ctx_limit", 0) or 0)
        except Exception:
            return 0

    def _archive_total_tokens(self, archive_path: Path) -> int:
        """Read the archive file, sum tokens across all messages. Falls back
        to 0 on any IO error so a corrupt file doesn't crash the list."""
        try:
            with archive_path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
        except Exception:
            return 0
        return _estimate_total_tokens(data.get("messages", []))

    def refresh(self) -> None:
        query = self.archive_search.text().strip()
        self.archive_list.clear()

        # Slash command in the box → suppress search filtering; the command
        # fires on Enter via _on_search_submit. Showing every archive while
        # the user is typing "/clear all" is fine — they'll dispatch shortly.
        is_command = query.startswith("/")
        q = "" if is_command else query.lower()

        matched_paths = None
        snippet_by_path: dict[str, str] = {}
        if q:
            results = search_archives(query, self._archive.archive_dir, max_results=500)
            matched_paths = {str(r.path) for r in results}
            for result in results:
                key = str(result.path)
                if key not in snippet_by_path:
                    snippet_by_path[key] = result.snippet

        selected_item = None
        for item in self._archive.list_archives():
            path_str = str(item.path)
            if q and q not in item.title.lower() and (matched_paths is None or path_str not in matched_paths):
                continue
            # Subtext shows the chat's lifetime token total (sum across all
            # messages — user, assistant, tool_call, tool_result). The earlier
            # "<total>/<ctx_limit> tok" form mixed a *cumulative* number with
            # a *per-turn ceiling*, which was misleading: chats easily exceed
            # the ctx limit over many turns even when no single turn does.
            total_tokens = self._archive_total_tokens(item.path)
            date_label = item.updated_at.split("T")[0] if item.updated_at else "Unknown date"
            subtext = f"{date_label}  •  {item.message_count} msgs  •  {total_tokens} tok"

            list_item = QListWidgetItem(f"{item.title}\n{subtext}")
            list_item.setData(Qt.UserRole, path_str)
            list_item.setToolTip(snippet_by_path.get(path_str) or item.tooltip)
            self.archive_list.addItem(list_item)
            if self._current_archive_path and self._current_archive_path == path_str:
                selected_item = list_item

        if selected_item is not None:
            self.archive_list.setCurrentItem(selected_item)

    def set_query(self, query: str) -> None:
        self.archive_search.setText(str(query or ""))

    def _on_search_text_changed(self, _text: str) -> None:
        # When the user is typing a /command, recolor the border to flag it.
        text = self.archive_search.text()
        if text.startswith("/"):
            self.archive_search.setStyleSheet(
                f"QLineEdit#archive_search {{ border: 1px solid {_s.ACCENT_PRIMARY};"
                f" color: {_s.ACCENT_PRIMARY}; }}"
            )
        else:
            self.archive_search.setStyleSheet("")
        self.refresh()

    def _on_search_submit(self) -> None:
        """Enter pressed in the search box. If the text is a /command,
        dispatch it; otherwise leave the existing search filter alone."""
        parsed = _parse_command(self.archive_search.text())
        if parsed is None:
            return
        verb, args = parsed
        self._dispatch_command(verb, args)
        # Clear the search line + reset its style after a command runs.
        self.archive_search.clear()
        self.archive_search.setStyleSheet("")

    # ── slash command dispatch ────────────────────────────────────────

    def _dispatch_command(self, verb: str, args: list[str]) -> None:
        if verb == "new":
            self.sig_new_requested.emit()
            self._status("started new chat")
        elif verb == "save":
            # Auto-save runs per turn; keep the verb for explicit
            # checkpoints / user intent.
            self._save_session()
            self.sig_save_requested.emit()
            self._status("saved")
        elif verb == "clear":
            scope = (args[0].lower() if args else "chat").strip()
            if scope not in ("chat", "all", "logs"):
                self._status(f"unknown clear scope: {scope}  (chat|all|logs)")
                return
            if scope == "all":
                self._clear_all_archives()
            else:
                self.sig_clear_requested.emit(scope)
            self._status(f"clear {scope}")
        elif verb == "help":
            QMessageBox.information(self, "Archive commands", _COMMANDS_HELP)
        else:
            self._status(f"unknown command: /{verb}")

    def _clear_all_archives(self) -> None:
        """/clear all — wipe every archive file in the directory. Confirms
        first because this is irreversible."""
        archives = list(self._archive.archive_dir.glob("*.json"))
        if not archives:
            return
        confirm = QMessageBox.question(
            self,
            "Clear all archives",
            f"Delete all {len(archives)} archived chats? This cannot be undone.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if confirm != QMessageBox.Yes:
            return
        deleted = 0
        for path in archives:
            try:
                path.unlink()
                deleted += 1
            except OSError:
                pass
        # The currently-open session's archive may have been one of these.
        if self._current_archive_path:
            try:
                if not Path(self._current_archive_path).exists():
                    self._current_archive_path = None
                    self.sig_current_archive_deleted.emit()
            except Exception:
                pass
        self._status(f"deleted {deleted} archive(s)")
        self.refresh()

    def _status(self, message: str) -> None:
        # Surfaced via signal so PageChat can pipe to the trace / status bar
        # if it wires sig_status_message. Otherwise harmless no-op listeners.
        self.sig_status_message.emit(str(message))

    # ── context menu (right-click on item) ─────────────────────────────

    def _show_context_menu(self, point) -> None:
        item = self.archive_list.itemAt(point)
        if item is None:
            return
        # Force selection so Copy/Summarize/Delete operate on the clicked
        # row rather than whatever was last selected.
        self.archive_list.setCurrentItem(item)

        menu = QMenu(self)
        menu.setStyleSheet(
            f"QMenu {{ background: {_s.BG_PANEL}; color: {_s.FG_TEXT}; "
            f"border: 1px solid {_s.BORDER_LIGHT}; }}"
            f"QMenu::item {{ padding: 6px 18px; }}"
            f"QMenu::item:selected {{ background: {_s.ACCENT_PRIMARY}; "
            f"color: {_s.BG_MAIN}; }}"
        )

        act_open = QAction("Open", menu)
        act_open.triggered.connect(self._load_selected_archive)
        menu.addAction(act_open)

        act_copy = QAction("Copy", menu)
        act_copy.triggered.connect(self._copy_selected_archive)
        menu.addAction(act_copy)

        act_summarize = QAction("Summarize", menu)
        act_summarize.triggered.connect(self._summarize_selected_archive)
        menu.addAction(act_summarize)

        menu.addSeparator()

        act_delete = QAction("Delete", menu)
        act_delete.triggered.connect(self._delete_selected_archive)
        menu.addAction(act_delete)

        menu.exec(self.archive_list.viewport().mapToGlobal(point))

    def _selected_archive_path(self) -> Path | None:
        item = self.archive_list.currentItem()
        if item is None:
            return None
        return Path(str(item.data(Qt.UserRole)))

    def _read_archive_messages(self, archive_path: Path) -> list[dict]:
        try:
            with archive_path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
        except Exception:
            return []
        return data.get("messages") or []

    def _format_messages_for_copy(self, messages: list[dict]) -> str:
        """Render archive messages as readable plain text for clipboard /
        summarize. One blank line between turns, role tag on its own line."""
        chunks: list[str] = []
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            role = str(msg.get("role") or "unknown").upper()
            text = str(msg.get("text") or "").strip()
            if not text:
                continue
            chunks.append(f"[{role}]\n{text}")
        return "\n\n".join(chunks)

    def _copy_selected_archive(self) -> None:
        archive_path = self._selected_archive_path()
        if archive_path is None:
            return
        messages = self._read_archive_messages(archive_path)
        text = self._format_messages_for_copy(messages)
        if not text:
            self._status("nothing to copy (empty chat)")
            return
        QApplication.clipboard().setText(text)
        self._status(f"copied {len(messages)} message(s) to clipboard")

    def _summarize_selected_archive(self) -> None:
        """Emit the chat content with a 'summarize this' prefix so the chat
        controller can drop it into the current input. Keeps the LLM call
        on the existing per-turn path — no separate engine wiring needed."""
        archive_path = self._selected_archive_path()
        if archive_path is None:
            return
        messages = self._read_archive_messages(archive_path)
        text = self._format_messages_for_copy(messages)
        if not text:
            self._status("nothing to summarize (empty chat)")
            return
        prompt = (
            "Summarize the following archived chat. Capture the user's intent,"
            " what was decided, what was tried, and any unresolved items.\n\n"
            "--- archived chat ---\n"
            f"{text}\n"
            "--- end ---"
        )
        self.sig_summarize_requested.emit(prompt)
        self._status("summarize prompt sent to chat input")

    # ── public ops ────────────────────────────────────────────────────

    def _current_session(self):
        if callable(self._session_provider):
            return self._session_provider()
        return None

    def _save_session(self) -> None:
        session = self._current_session()
        if not isinstance(session, dict):
            return
        self._archive.save_session(session)
        self._current_archive_path = session.get("archive_path")
        self.refresh()

    def _load_selected_archive(self) -> None:
        archive_path = self._selected_archive_path()
        if archive_path is None:
            return
        try:
            session = self._archive.load_session(archive_path)
        except Exception:
            QMessageBox.warning(self, "Load Failed", "Could not read archive file.")
            return
        self._current_archive_path = str(archive_path)
        self.sig_session_loaded.emit(session)

    def _delete_selected_archive(self) -> None:
        archive_path = self._selected_archive_path()
        if archive_path is None:
            return
        try:
            archive_path.unlink()
        except OSError:
            return
        if self._current_archive_path == str(archive_path):
            self._current_archive_path = None
            self.sig_current_archive_deleted.emit()
        self.refresh()
