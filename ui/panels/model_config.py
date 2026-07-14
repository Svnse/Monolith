from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlparse

import urllib.error
import urllib.request

from PySide6.QtCore import QDateTime, QThread, Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

import core.style as _s
from core.llm_config import DEFAULT_CONFIG, load_config, save_config
from core.model_registry import (
    apply_model_preset,
    describe_model_preset,
    metadata_context_window,
    read_gguf_metadata,
    resolve_model_preset,
)
from core.model_ops import build_llama_server_command, compute_serve_profile, diagnose_serve_output
from core.paths import LOG_DIR, ensure_safe_local_path
from core.state import SystemStatus
from core import cloud_profiles as _cloud_profiles
from ui.components.atoms import MonoButton


def _normalize_base_url(base_url: str) -> str:
    return (base_url or "").strip().rstrip("/")


def _models_url(base_url: str) -> str:
    base = _normalize_base_url(base_url)
    if base.endswith("/models"):
        return base
    if base.endswith("/v1"):
        return f"{base}/models"
    return f"{base}/v1/models"


def _extract_ctx_length_from_model_item(item: dict) -> int | None:
    """Return a conservative effective context length from a /models item.

    We intentionally prefer the smallest positive value among known context keys.
    Some backends report both training context (e.g. n_ctx_train=32768) and
    runtime context (e.g. n_ctx=9216). Picking the minimum avoids overestimating.
    """
    if not isinstance(item, dict):
        return None
    meta = item.get("meta", {}) if isinstance(item.get("meta"), dict) else {}
    candidates: list[int] = []
    for source in (meta, item):
        for key in ("n_ctx", "context_length", "max_context_length", "n_ctx_train"):
            raw = source.get(key)
            if raw is None:
                continue
            try:
                value = int(raw)
            except (TypeError, ValueError):
                continue
            if value > 0:
                candidates.append(value)
    if not candidates:
        return None
    return min(candidates)


class ModelListLoader(QThread):
    finished = Signal(object)
    error = Signal(str)

    def __init__(self, base_url: str, api_key: str | None):
        super().__init__()
        self.base_url = base_url
        self.api_key = api_key or ""

    def run(self):
        try:
            if not self.base_url:
                raise RuntimeError("Missing API base URL.")
            url = _models_url(self.base_url)
            headers = {}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            req = urllib.request.Request(url, headers=headers, method="GET")
            try:
                with urllib.request.urlopen(req, timeout=15) as resp:
                    body = resp.read()
            except urllib.error.HTTPError as exc:
                body = exc.read().decode("utf-8", errors="ignore")
                raise RuntimeError(f"HTTP {exc.code}: {body or exc.reason}") from exc
            except urllib.error.URLError as exc:
                reason = getattr(exc, "reason", "") or str(exc)
                raise RuntimeError(f"URLError: {reason}") from exc
            data = json.loads(body)
            items = data.get("data", []) if isinstance(data, dict) else []
            models = []
            ctx_length = None
            for item in items:
                if isinstance(item, dict) and "id" in item:
                    models.append(str(item["id"]))
                    if ctx_length is None:
                        ctx_length = _extract_ctx_length_from_model_item(item)
                elif isinstance(item, str):
                    models.append(item)
            self.finished.emit({"models": sorted({m for m in models if m}), "ctx_length": ctx_length})
        except Exception as exc:
            self.error.emit(str(exc).strip() or repr(exc))


def _combo_style() -> str:
    return f"""
    QComboBox {{
        background: {_s.BG_SURFACE_1};
        color: {_s.FG_TEXT};
        border: 1px solid {_s.BORDER_SUBTLE};
        padding: 2px 8px;
        font-size: 10px;
        font-weight: bold;
        border-radius: 6px;
    }}
    QComboBox:hover {{ border: 1px solid {_s.ACCENT_PRIMARY}; }}
    QComboBox::drop-down {{ border: none; width: 18px; }}
    QComboBox::down-arrow {{ image: none; border: none; }}
    QComboBox QAbstractItemView {{
        background: {_s.BG_SURFACE_1};
        color: {_s.FG_TEXT};
        border: 1px solid {_s.BORDER_SUBTLE};
        selection-background-color: {_s.BG_BUTTON_HOVER};
        selection-color: {_s.ACCENT_PRIMARY};
    }}
    """


def _section_title(text: str) -> QLabel:
    label = QLabel(text)
    label.setStyleSheet(
        f"color: {_s.FG_DIM}; font-size: 9px; font-family: Consolas; font-weight: bold; letter-spacing: 1px;"
    )
    return label


def _surface_frame(level: int = 1) -> QFrame:
    frame = QFrame()
    frame.setStyleSheet(
        f"background: {_s.BG_SURFACE_1 if level <= 1 else _s.BG_SURFACE_2}; border: none; border-radius: 8px;"
    )
    return frame


def _text_link(label: str, accent: bool = False) -> QPushButton:
    btn = QPushButton(label)
    btn.setCursor(Qt.PointingHandCursor)
    color = _s.ACCENT_PRIMARY if accent else _s.FG_DIM
    btn.setStyleSheet(
        f"""
        QPushButton {{
            background: transparent;
            border: none;
            color: {color};
            font-size: 10px;
            font-family: Consolas;
            padding: 0;
        }}
        QPushButton:hover {{
            color: {_s.FG_TEXT if not accent else _s.ACCENT_PRIMARY};
        }}
        """
    )
    return btn


def _fmt_limit(value: int) -> str:
    if abs(int(value)) >= 10000:
        return f"{int(round(int(value) / 1000))}k"
    return str(int(value))


class ModelConfigPanel(QWidget):
    sig_model_payload_changed = Signal(object)
    sig_load_requested = Signal()
    sig_unload_requested = Signal()
    sig_ctx_limit_changed = Signal(int)
    sig_trace_requested = Signal(str, str, bool)

    def __init__(self, state, ui_bridge, config: dict | None = None, parent=None):
        super().__init__(parent)
        self.state = state
        self.ui_bridge = ui_bridge
        self.config = config if config is not None else load_config()
        self._controller = None
        self._model_fetcher: ModelListLoader | None = None
        self._local_server_proc: subprocess.Popen | None = None
        self._local_server_base = ""
        self._local_server_model = ""
        self._local_server_log_fh = None
        self._local_server_log = None
        self._pending_local_load = False
        self._stop_server_on_unload = False
        self._local_fetch_attempts = 0
        self._local_fetch_timer: QTimer | None = None
        self._model_ctx_cap: int | None = None
        self._is_model_loaded = False
        self._last_status: SystemStatus | None = None
        self._config_dirty = False
        self._endpoint_expanded = False
        # Debounced auto-save for connection-shaped fields (api_*, gguf_path,
        # backend, api_provider). Fires 600ms after the last edit so a single
        # paste of a long key/url doesn't write per-keystroke.
        self._autosave_timer = QTimer(self)
        self._autosave_timer.setSingleShot(True)
        self._autosave_timer.setInterval(600)
        self._autosave_timer.timeout.connect(self._autosave_connection_fields)
        self._AUTOSAVE_KEYS = frozenset({
            "api_base", "api_model", "api_key",
            "gguf_path", "backend", "api_provider",
        })

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setFrameShape(QFrame.NoFrame)
        root.addWidget(self._scroll, 1)

        self._body = QWidget()
        self._scroll.setWidget(self._body)
        body = QVBoxLayout(self._body)
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(12)

        self._model_card = self._build_model_section()
        body.addWidget(self._model_card)

        self._endpoint_row = self._build_endpoint_row()
        body.addWidget(self._endpoint_row)

        self._endpoint_edit = self._build_endpoint_edit_section()
        self._endpoint_edit.setVisible(False)
        body.addWidget(self._endpoint_edit)

        body.addStretch()

        self.engine_combo.currentIndexChanged.connect(self._on_engine_changed)
        self.api_base_input.textChanged.connect(lambda v: self._update_config_value("api_base", v))
        self.api_base_input.editingFinished.connect(self.emit_model_payload)
        self.api_model_input.textChanged.connect(lambda v: self._update_config_value("api_model", v))
        self.api_model_input.editingFinished.connect(self.emit_model_payload)
        self.api_model_combo.currentIndexChanged.connect(self._on_model_combo_changed)
        self.btn_fetch_models.clicked.connect(self.fetch_models)
        self.api_key_input.textChanged.connect(lambda v: self._update_config_value("api_key", v))
        self.api_key_input.editingFinished.connect(self.emit_model_payload)

        self.apply_model_config()
        self.update_load_button_text()
        self.set_config_dirty(False)
        if hasattr(self.ui_bridge, "sig_config_changed"):
            self.ui_bridge.sig_config_changed.connect(self.on_external_config_changed)
        if hasattr(self.ui_bridge, "sig_theme_changed"):
            self.ui_bridge.sig_theme_changed.connect(self._on_theme_changed)
        # Style the endpoint-edit inputs so they sit on the surrounding
        # surface color instead of the global BG_INPUT, which on dark themes
        # is markedly darker than the parent surface frame and reads as a
        # "sunken blue island".
        self._apply_input_theme()
        self.destroyed.connect(lambda *_args: self.stop_local_server())

    def bind_controller(self, controller) -> None:
        self._controller = controller

    def _input_stylesheet(self) -> str:
        # Inputs blend fully with the parent panel — no fill, just an
        # outline border. Matches the same treatment as the surrounding
        # blended surface frames, so nothing reads as a "blue island".
        return (
            f"QLineEdit {{"
            f"  background: transparent;"
            f"  color: {_s.FG_TEXT};"
            f"  border: 1px solid {_s.BORDER_SUBTLE};"
            f"  border-radius: 4px;"
            f"  padding: 4px 8px;"
            f"  selection-background-color: {_s.BG_BUTTON_HOVER};"
            f"}}"
            f"QLineEdit:focus {{ border-color: {_s.ACCENT_PRIMARY}; }}"
            f"QLineEdit:read-only {{ color: {_s.FG_DIM}; }}"
        )

    def _blended_frame_stylesheet(self, object_name: str) -> str:
        # Surface frames blend fully with the parent panel — no fill, no
        # border. The bordered "QFrame { ... }" form cascades to QLabel
        # descendants (QLabel inherits QFrame), which produced the outline
        # box around every section title. Scoping the selector by
        # objectName isolates the styling to the frame itself.
        return (
            f"QFrame#{object_name} {{"
            f"  background: transparent;"
            f"  border: none;"
            f"}}"
        )

    def _accent_link_stylesheet(self) -> str:
        return (
            f"QPushButton {{"
            f"  background: transparent;"
            f"  border: none;"
            f"  color: {_s.ACCENT_PRIMARY};"
            f"  font-size: 10px;"
            f"  font-family: Consolas;"
            f"  padding: 0;"
            f"}}"
            f"QPushButton:hover {{ color: {_s.FG_TEXT}; }}"
        )

    def _apply_input_theme(self) -> None:
        ss = self._input_stylesheet()
        for inp in (
            getattr(self, "path_display", None),
            getattr(self, "api_base_input", None),
            getattr(self, "api_model_input", None),
            getattr(self, "api_key_input", None),
        ):
            if inp is not None:
                inp.setStyleSheet(ss)

        for frame, name in (
            (getattr(self, "_model_card", None), "mc_model_card"),
            (getattr(self, "_endpoint_row", None), "mc_endpoint_row"),
            (getattr(self, "_endpoint_edit", None), "mc_endpoint_edit"),
        ):
            if frame is None:
                continue
            frame.setObjectName(name)
            frame.setStyleSheet(self._blended_frame_stylesheet(name))

        link = getattr(self, "_endpoint_toggle", None)
        if link is not None:
            link.setStyleSheet(self._accent_link_stylesheet())

    def _on_theme_changed(self, _theme_name: str = "") -> None:
        # Live tokens have already been refreshed by the global theme
        # handler; we just need to re-emit per-widget stylesheets so the
        # new theme's colors land on every f-stringed widget.
        self._apply_input_theme()

    def _build_model_section(self) -> QFrame:
        card = _surface_frame()
        layout = QVBoxLayout(card)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(4)

        name_row = QHBoxLayout()
        name_row.setContentsMargins(0, 0, 0, 0)
        name_row.setSpacing(8)

        self._model_name = QLabel("No model")
        self._model_name.setStyleSheet(
            f"color: {_s.FG_TEXT}; font-size: 12px; font-weight: bold;"
        )
        name_row.addWidget(self._model_name)

        self._model_status = QLabel("")
        self._model_status.setStyleSheet(
            f"color: {_s.FG_ACCENT}; font-size: 9px; font-family: Consolas;"
        )
        name_row.addWidget(self._model_status)
        name_row.addStretch()

        self._profile_btn = MonoButton("☰")
        self._profile_btn.setFixedWidth(30)
        self._profile_btn.setToolTip("Cloud profiles")
        self._profile_btn.clicked.connect(self._open_profile_menu)
        name_row.addWidget(self._profile_btn)

        layout.addLayout(name_row)

        self._model_meta = QLabel("")
        self._model_meta.setStyleSheet(
            f"color: {_s.FG_DIM}; font-size: 10px; font-family: Consolas;"
        )
        layout.addWidget(self._model_meta)
        return card

    def _build_endpoint_row(self) -> QFrame:
        row = _surface_frame()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(8)

        self._endpoint_label = QLabel("No endpoint")
        self._endpoint_label.setStyleSheet(
            f"color: {_s.FG_SECONDARY}; font-size: 10px; font-family: Consolas;"
        )
        layout.addWidget(self._endpoint_label)
        layout.addStretch()

        self._endpoint_toggle = _text_link("Edit", accent=True)
        self._endpoint_toggle.clicked.connect(self.toggle_endpoint_edit)
        layout.addWidget(self._endpoint_toggle)
        return row

    def _build_endpoint_edit_section(self) -> QFrame:
        frame = _surface_frame(2)
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        row_engine = QHBoxLayout()
        row_engine.setContentsMargins(0, 0, 0, 0)
        row_engine.setSpacing(8)
        lbl_engine = QLabel("ENGINE")
        lbl_engine.setStyleSheet(
            f"color: {_s.FG_DIM}; font-size: 9px; font-family: Consolas; font-weight: bold;"
        )
        row_engine.addWidget(lbl_engine)
        self.engine_combo = QComboBox()
        self.engine_combo.addItem("GGUF (API)", "gguf_api")
        self.engine_combo.addItem("GGUF (llama.cpp)", "gguf")
        self.engine_combo.addItem("Model (API)", "openai")
        # vLLM / SGLang dropped from the menu — both speak OpenAI-compatible
        # protocol, so "Model (API)" + a custom API BASE covers them. Removing
        # the menu items doesn't break loading those backends programmatically
        # (api_provider="vllm" / "sglang" still routes correctly on the engine
        # side); it just removes the dedicated menu picks.
        self.engine_combo.setFixedHeight(28)
        self.engine_combo.setStyleSheet(_combo_style())
        row_engine.addWidget(self.engine_combo, 1)
        layout.addLayout(row_engine)

        self.path_display = QLineEdit()
        self.path_display.setReadOnly(True)
        self.path_display.setPlaceholderText("No model selected")
        self.path_display.setObjectName("path_display")
        self.btn_browse = MonoButton("...")
        self.btn_browse.setFixedWidth(34)
        self.btn_browse.clicked.connect(self.pick_file)
        self.gguf_row = QWidget()
        gguf_layout = QHBoxLayout(self.gguf_row)
        gguf_layout.setContentsMargins(0, 0, 0, 0)
        gguf_layout.setSpacing(6)
        gguf_layout.addWidget(self.path_display, 1)
        gguf_layout.addWidget(self.btn_browse)
        layout.addWidget(self.gguf_row)

        self.remote_panel = QWidget()
        remote = QVBoxLayout(self.remote_panel)
        remote.setContentsMargins(0, 0, 0, 0)
        remote.setSpacing(8)

        self.lbl_base = _section_title("API BASE")
        remote.addWidget(self.lbl_base)
        self.api_base_input = QLineEdit()
        self.api_base_input.setPlaceholderText("http://localhost:8000/v1")
        remote.addWidget(self.api_base_input)

        self.lbl_model = _section_title("MODEL")
        remote.addWidget(self.lbl_model)
        self.api_model_input = QLineEdit()
        self.api_model_input.setPlaceholderText("model-id")
        remote.addWidget(self.api_model_input)

        self.api_model_combo = QComboBox()
        self.api_model_combo.setVisible(False)
        self.api_model_combo.setFixedHeight(28)
        self.api_model_combo.setStyleSheet(_combo_style())
        remote.addWidget(self.api_model_combo)

        self.btn_fetch_models = MonoButton("FETCH MODELS")
        self.btn_fetch_models.setFixedHeight(26)
        remote.addWidget(self.btn_fetch_models)

        self.lbl_key = _section_title("API KEY")
        remote.addWidget(self.lbl_key)
        self.api_key_input = QLineEdit()
        self.api_key_input.setPlaceholderText("(optional)")
        self.api_key_input.setEchoMode(QLineEdit.Password)
        remote.addWidget(self.api_key_input)
        layout.addWidget(self.remote_panel)

        self.btn_load = MonoButton("LOAD MODEL")
        self.btn_load.clicked.connect(self.toggle_load)
        layout.addWidget(self.btn_load)
        return frame

    def describe_active_model(self) -> str:
        backend = self.config.get("backend", "gguf_api")
        if backend in ("gguf", "gguf_api"):
            api_base = self.config.get("api_base") or ""
            api_model = self.config.get("api_model") or ""
            if backend == "gguf_api" and api_base and api_model:
                return f"local:{api_model}@{api_base}"
            path = self.config.get("gguf_path")
            return Path(path).name if path else "none"
        provider = self.config.get("api_provider", "openai")
        model = self.config.get("api_model") or "unknown"
        base = self.config.get("api_base") or "no base"
        return f"{provider}:{model}@{base}"

    def build_model_payload(self) -> dict:
        backend = self.config.get("backend", "gguf_api")
        payload_backend = "openai" if backend == "gguf_api" and self.config.get("api_base") and self.config.get("api_model") else backend
        return {
            "backend": payload_backend,
            "api_provider": self.config.get("api_provider", "openai"),
            "api_base": self.config.get("api_base", ""),
            "api_model": self.config.get("api_model", ""),
            "api_key": self.config.get("api_key", ""),
            "gguf_path": self.config.get("gguf_path"),
            "path": self.config.get("gguf_path"),
        }

    def emit_model_payload(self) -> None:
        self.sig_model_payload_changed.emit(self.build_model_payload())

    def _emit_trace(self, msg: str, tag: str = "INFO", error: bool = False) -> None:
        self.sig_trace_requested.emit(str(msg), str(tag), bool(error))

    def _sync_path_display(self) -> None:
        backend = self.config.get("backend", "gguf_api")
        summary = self.describe_active_model()
        if backend == "gguf":
            gguf_path = self.config.get("gguf_path")
            if gguf_path:
                self.path_display.setText(str(gguf_path))
                self.path_display.setToolTip(str(gguf_path))
            else:
                self.path_display.clear()
                self.path_display.setToolTip("")
        else:
            self.path_display.setText(summary)
            self.path_display.setToolTip(summary)

        provider = "GGUF (API)" if backend == "gguf_api" else backend.replace("_", " ").upper()
        if backend not in ("gguf", "gguf_api"):
            provider = str(self.config.get("api_provider", "openai")).upper()
        model_name = (
            Path(str(self.config.get("gguf_path") or "")).name
            if backend in ("gguf", "gguf_api")
            else str(self.config.get("api_model") or "No model")
        )
        if backend == "gguf_api" and self.config.get("api_model"):
            model_name = str(self.config.get("api_model"))
        self._model_name.setText(model_name or "No model")
        self._model_status.setText("● Loaded" if self._is_model_loaded else "")
        meta = provider
        if self._model_ctx_cap:
            meta = f"{meta} · {_fmt_limit(self._model_ctx_cap)} ctx"
        self._model_meta.setText(meta)
        base = str(self.config.get("api_base") or "").strip()
        self._endpoint_label.setText(base or summary or "No endpoint")

    def apply_backend_visibility(self) -> None:
        backend = self.config.get("backend", "gguf_api")
        is_gguf = backend == "gguf"
        is_gguf_api = backend == "gguf_api"
        self.gguf_row.setVisible(is_gguf or is_gguf_api)
        self.remote_panel.setVisible(is_gguf_api or not is_gguf)
        self.btn_browse.setEnabled(is_gguf or is_gguf_api)
        self.btn_fetch_models.setVisible(not (is_gguf or is_gguf_api))
        self.btn_fetch_models.setEnabled(not (is_gguf or is_gguf_api))
        self.api_key_input.setVisible(not (is_gguf or is_gguf_api))
        self.lbl_key.setVisible(not (is_gguf or is_gguf_api))
        self.api_model_input.setReadOnly(is_gguf_api)
        self.api_model_input.setPlaceholderText("auto" if is_gguf_api else "model-id")
        self.lbl_base.setText("LOCAL API BASE" if is_gguf_api else "API BASE")
        self.lbl_model.setText("MODEL ID" if is_gguf_api else "MODEL")
        self.api_base_input.setPlaceholderText("auto (port will be chosen)" if is_gguf_api else "http://localhost:8000/v1")
        # Hide + clear the fetched-models combo on every backend switch.
        # The combo only makes sense AFTER the user clicks FETCH MODELS on
        # a cloud backend. Without this, switching from "OpenAI-compatible"
        # (with a fetched list like deepseek-v4-pro / deepseek-v4-flash) to
        # GGUF (API) leaves the stale cloud list visible — wrong context.
        # The input QLineEdit always stays visible so the per-engine saved
        # api_model from _load_engine_inputs is still selectable/editable.
        self.api_model_combo.blockSignals(True)
        self.api_model_combo.clear()
        self.api_model_combo.blockSignals(False)
        self.api_model_combo.setVisible(False)
        self.api_model_input.setVisible(True)

    def apply_model_config(self) -> None:
        backend = self.config.get("backend", "gguf")
        provider = self.config.get("api_provider", "openai")
        target = "gguf_api" if backend == "gguf_api" else ("gguf" if backend == "gguf" else provider)
        idx = self.engine_combo.findData(target)
        if idx >= 0:
            self.engine_combo.blockSignals(True)
            self.engine_combo.setCurrentIndex(idx)
            self.engine_combo.blockSignals(False)
        self.api_base_input.blockSignals(True)
        self.api_model_input.blockSignals(True)
        self.api_key_input.blockSignals(True)
        self.api_base_input.setText(str(self.config.get("api_base", "") or ""))
        self.api_model_input.setText(str(self.config.get("api_model", "") or ""))
        self.api_key_input.setText(str(self.config.get("api_key", "") or ""))
        self.api_base_input.blockSignals(False)
        self.api_model_input.blockSignals(False)
        self.api_key_input.blockSignals(False)
        self.api_model_combo.setVisible(False)
        self.api_model_input.setVisible(True)
        self.apply_backend_visibility()
        self._sync_path_display()

    def _pick_free_port(self) -> int:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", 0))
            return int(sock.getsockname()[1])

    def _coerce_local_base(self, raw: str) -> tuple[str, str, int] | None:
        value = (raw or "").strip()
        if not value:
            port = self._pick_free_port()
            return f"http://127.0.0.1:{port}/v1", "127.0.0.1", port
        if value.isdigit():
            port = int(value)
            return f"http://127.0.0.1:{port}/v1", "127.0.0.1", port
        if "://" not in value:
            value = f"http://{value}"
        parsed = urlparse(value)
        host = parsed.hostname or "127.0.0.1"
        port = parsed.port or self._pick_free_port()
        scheme = parsed.scheme or "http"
        path = parsed.path or ""
        if not path or path == "/":
            path = "/v1"
        elif "/v1" not in path:
            path = path.rstrip("/") + "/v1"
        return f"{scheme}://{host}:{port}{path}", host, port

    def _ensure_local_api_base(self) -> tuple[str, str, int] | None:
        raw = self.api_base_input.text().strip() or self.config.get("api_base", "")
        result = self._coerce_local_base(raw)
        if result is None:
            return None
        base, host, port = result
        self.config["api_base"] = base
        self.api_base_input.setText(base)
        self.set_config_dirty(True)
        self._sync_path_display()
        return base, host, port

    def _resolve_native_llama_server(self) -> str | None:
        env = os.getenv("MONOLITH_LLAMA_SERVER", "").strip()
        if env and Path(env).is_file():
            return env
        candidates = [
            Path.home() / "llama.cpp" / "build" / "bin" / "Release" / "llama-server.exe",
            Path.home() / "llama.cpp" / "build" / "bin" / "llama-server.exe",
        ]
        for path in candidates:
            if path.is_file():
                return str(path)
        import shutil
        return shutil.which("llama-server")

    def _resolve_local_server_python(self) -> str:
        env_py = os.getenv("MONOLITH_LLAMA_PY", "").strip()
        if env_py and Path(env_py).exists():
            return env_py
        return sys.executable

    def _build_server_cmd(self, gguf_path: str, host: str, port: int) -> list[str]:
        profile = compute_serve_profile(
            gguf_path,
            requested_ctx=int(self.config.get("ctx_limit", DEFAULT_CONFIG["ctx_limit"]) or DEFAULT_CONFIG["ctx_limit"]),
            host=host,
            port=port,
        )
        self.config["serve_profile"] = profile.to_dict()
        try:
            save_config(self.config)
        except Exception:
            pass
        for warning in profile.warnings:
            self._emit_trace(f"Serve profile: {warning}", "MODEL", True)
        native = self._resolve_native_llama_server()
        if native:
            self._emit_trace(f"Using native llama-server: {native}", "MODEL")
            return build_llama_server_command(profile, server_path=native)
        python_exe = self._resolve_local_server_python()
        self._emit_trace(f"Using python llama_cpp.server: {python_exe}", "MODEL")
        return build_llama_server_command(profile, python_exe=python_exe)

    def _resolve_local_ctx_size(self, gguf_path: str) -> int:
        """Pick a safe local-server context size for the selected GGUF.

        Order of operations:
        1. Start from configured ctx_limit.
        2. Clamp to GGUF metadata context when available.
        3. Clamp to a hard safety cap to avoid stale huge values from prior models.
        """
        configured = max(1024, int(self.config.get("ctx_limit", DEFAULT_CONFIG["ctx_limit"]) or DEFAULT_CONFIG["ctx_limit"]))
        metadata_ctx = metadata_context_window(read_gguf_metadata(gguf_path))
        if metadata_ctx:
            configured = min(configured, int(metadata_ctx))

        hard_cap = int(os.getenv("MONOLITH_LOCAL_CTX_HARD_CAP", "32768") or "32768")
        hard_cap = max(1024, hard_cap)
        resolved = min(configured, hard_cap)

        self._emit_trace(
            (
                f"Local ctx selection: configured={int(self.config.get('ctx_limit', DEFAULT_CONFIG['ctx_limit']))} "
                f"metadata={metadata_ctx or 'unknown'} hard_cap={hard_cap} -> using={resolved}"
            ),
            "CTX",
        )
        return resolved

    def start_local_server(self, gguf_path: str, base: str, host: str, port: int) -> bool:
        if self._local_server_proc and self._local_server_proc.poll() is None:
            if self._local_server_model == gguf_path and self._local_server_base == base:
                return True
            self.stop_local_server()
        try:
            self._emit_trace(f"Local model path: {gguf_path}", "MODEL")
            self._emit_trace(f"Starting local server on {host}:{port} ...", "MODEL")
            cmd = self._build_server_cmd(gguf_path, host, port)
            if "--ctx-size" in cmd:
                idx = cmd.index("--ctx-size")
                if idx + 1 < len(cmd):
                    self._emit_trace(f"Requested server context: {cmd[idx + 1]} tokens", "CTX")
            elif "--n_ctx" in cmd:
                idx = cmd.index("--n_ctx")
                if idx + 1 < len(cmd):
                    self._emit_trace(f"Requested server context: {cmd[idx + 1]} tokens", "CTX")
            creationflags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
            if os.name == "nt":
                creationflags |= subprocess.BELOW_NORMAL_PRIORITY_CLASS
            self._local_server_log = ensure_safe_local_path(LOG_DIR / "llama_server.log")
            self._local_server_log_fh = open(self._local_server_log, "w", encoding="utf-8")
            self._local_server_proc = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=self._local_server_log_fh,
                creationflags=creationflags,
            )
        except Exception as exc:
            self._emit_trace(f"Failed to start local server: {exc}", "MODEL", True)
            self._emit_serve_diagnosis(str(exc))
            return False
        self._local_server_model = gguf_path
        self._local_server_base = base
        return True

    def stop_local_server(self) -> None:
        proc = self._local_server_proc
        self._local_server_proc = None
        self._local_server_model = ""
        self._local_server_base = ""
        if proc and proc.poll() is None:
            try:
                proc.terminate()
            except Exception:
                pass
        if self._local_server_log_fh is not None:
            try:
                self._local_server_log_fh.close()
            except Exception:
                pass
            self._local_server_log_fh = None

    def _local_server_error_output(self) -> str:
        if self._local_server_log and Path(self._local_server_log).exists():
            try:
                return Path(self._local_server_log).read_text(encoding="utf-8", errors="ignore")[-1000:].strip()
            except Exception:
                return ""
        return ""

    def _emit_serve_diagnosis(self, text: str) -> None:
        diagnosis = diagnose_serve_output(text)
        if diagnosis is None:
            return
        self._emit_trace(f"Diagnosis: {diagnosis.message}", "MODEL", True)
        if diagnosis.suggested_actions:
            self._emit_trace("Try: " + "; ".join(diagnosis.suggested_actions[:3]), "MODEL", True)

    def _start_local_server_and_load(self) -> None:
        gguf_path = self.config.get("gguf_path")
        if not gguf_path:
            self._emit_trace("No GGUF selected.", "MODEL", True)
            self.btn_load.setEnabled(True)
            self.update_load_button_text()
            return
        self._apply_registry_preset(str(gguf_path))
        base_info = self._ensure_local_api_base()
        if not base_info:
            self._emit_trace("Invalid local API base.", "MODEL", True)
            self.btn_load.setEnabled(True)
            self.update_load_button_text()
            return
        base, host, port = base_info
        if not self.start_local_server(str(gguf_path), base, host, port):
            self.btn_load.setEnabled(True)
            self.update_load_button_text()
            return
        self._pending_local_load = True
        self._local_fetch_attempts = 0
        if self._local_fetch_timer is None:
            self._local_fetch_timer = QTimer(self)
            self._local_fetch_timer.setInterval(500)
            self._local_fetch_timer.timeout.connect(self._attempt_local_model_fetch)
        self._local_fetch_timer.start()
        self._attempt_local_model_fetch()

    def _attempt_local_model_fetch(self) -> None:
        if not self._pending_local_load:
            if self._local_fetch_timer is not None:
                self._local_fetch_timer.stop()
            return
        if self._local_server_proc and self._local_server_proc.poll() is not None:
            self._pending_local_load = False
            if self._local_fetch_timer is not None:
                self._local_fetch_timer.stop()
            msg = "Local server exited before responding."
            err = self._local_server_error_output()
            if err:
                msg = f"{msg} {err}"
            self._emit_trace(msg, "MODEL", True)
            self._emit_serve_diagnosis(err or msg)
            self.btn_load.setEnabled(True)
            self.update_load_button_text()
            return
        if self._model_fetcher and self._model_fetcher.isRunning():
            return
        self._local_fetch_attempts += 1
        if self._local_fetch_attempts > 180:
            self._pending_local_load = False
            if self._local_fetch_timer is not None:
                self._local_fetch_timer.stop()
            self._emit_trace("Local server did not respond in time. Check model path and server deps.", "MODEL", True)
            self.btn_load.setEnabled(True)
            self.update_load_button_text()
            return
        if self._local_fetch_attempts == 1:
            self._emit_trace("Waiting for local server to respond...", "MODEL")
        elif self._local_fetch_attempts % 20 == 0:
            pid = self._local_server_proc.pid if self._local_server_proc else "?"
            self._emit_trace(f"Still waiting for local server (pid={pid})...", "MODEL")
        self._model_fetcher = ModelListLoader(str(self.config.get("api_base", "")), None)
        self._model_fetcher.finished.connect(self._on_models_loaded)
        self._model_fetcher.error.connect(self._on_models_error)
        self._model_fetcher.start()

    def apply_config_to_controls(self) -> None:
        self.apply_model_config()

    def _on_engine_changed(self, _index: int) -> None:
        target = self.engine_combo.currentData() or "gguf_api"
        # Per-engine input persistence: save the current (api_base, api_model,
        # api_key) under the OUTGOING engine key, then load whatever was last
        # saved under the INCOMING engine key (empty if first time). Without
        # this, switching openai-compat -> gguf_api kept the prior backend's
        # api_model (e.g. "deepseek-v4-pro") visible/active under the new
        # backend where it makes no sense.
        previous_target = self._current_engine_key()
        self._save_engine_inputs(previous_target)
        if target in ("gguf", "gguf_api"):
            self.config["backend"] = "gguf" if target == "gguf" else "gguf_api"
            self.config["api_provider"] = "openai"
        else:
            self.config["backend"] = "openai"
            self.config["api_provider"] = str(target)
            if self._local_server_proc:
                self.stop_local_server()
        if self.config.get("backend") != "gguf_api" and self._local_server_proc:
            self.stop_local_server()
        self._load_engine_inputs(target)
        self.apply_backend_visibility()
        self._sync_path_display()
        self.set_config_dirty(True)
        self.emit_model_payload()

    def _current_engine_key(self) -> str:
        """The engine combo key matching the current config — gguf_api, gguf,
        or the api_provider string (openai / vllm / sglang / etc.).
        """
        backend = self.config.get("backend", "gguf_api")
        if backend in ("gguf", "gguf_api"):
            return backend
        return str(self.config.get("api_provider") or "openai")

    def _save_engine_inputs(self, engine_key: str) -> None:
        """Stash the current api_base / api_model / api_key under
        config['per_engine_inputs'][engine_key]. Called before changing
        engines so the user can switch back and find their inputs intact.
        gguf_path stays at the top level — it's only meaningful for the gguf
        backends and they share it deliberately.
        """
        if not engine_key:
            return
        bucket_root = self.config.setdefault("per_engine_inputs", {})
        if not isinstance(bucket_root, dict):
            bucket_root = {}
            self.config["per_engine_inputs"] = bucket_root
        bucket_root[str(engine_key)] = {
            "api_base": str(self.config.get("api_base", "") or ""),
            "api_model": str(self.config.get("api_model", "") or ""),
            "api_key": str(self.config.get("api_key", "") or ""),
        }

    def _load_engine_inputs(self, engine_key: str) -> None:
        """Restore api_base / api_model / api_key from the per-engine bucket
        for `engine_key` (empty defaults on first visit). Syncs both
        self.config and the QLineEdit widgets so the UI reflects the new
        backend's saved state — without this the user sees the previous
        backend's deepseek model id under a gguf_api engine.
        """
        bucket_root = self.config.get("per_engine_inputs") or {}
        bucket = bucket_root.get(str(engine_key)) if isinstance(bucket_root, dict) else None
        if not isinstance(bucket, dict):
            bucket = {}
        base = str(bucket.get("api_base", "") or "")
        model = str(bucket.get("api_model", "") or "")
        key = str(bucket.get("api_key", "") or "")
        self.config["api_base"] = base
        self.config["api_model"] = model
        self.config["api_key"] = key
        # Block signals so the textChanged -> _update_config_value loop
        # doesn't re-fire and re-stash these values under the new engine
        # key before the user has typed anything.
        for widget, value in (
            (self.api_base_input, base),
            (self.api_model_input, model),
            (self.api_key_input, key),
        ):
            widget.blockSignals(True)
            widget.setText(value)
            widget.blockSignals(False)

    def fetch_models(self) -> None:
        base = self.config.get("api_base", "")
        if not base:
            self._emit_trace("Missing API base; set API BASE first.", "MODEL", True)
            return
        if self._model_fetcher and self._model_fetcher.isRunning():
            return
        self.btn_fetch_models.setEnabled(False)
        self._emit_trace("Fetching models from API base…", "MODEL")
        self._model_fetcher = ModelListLoader(str(base), self.config.get("api_key"))
        self._model_fetcher.finished.connect(self._on_models_loaded)
        self._model_fetcher.error.connect(self._on_models_error)
        self._model_fetcher.start()

    def _on_models_loaded(self, payload: object) -> None:
        models = []
        ctx_length = None
        if isinstance(payload, dict):
            models = payload.get("models") or []
            ctx_length = payload.get("ctx_length")
        elif isinstance(payload, list):
            models = payload
        self.btn_fetch_models.setEnabled(True)
        if not models:
            self._emit_trace("No models returned by server.", "MODEL", True)
            self.api_model_combo.setVisible(False)
            self.api_model_input.setVisible(True)
            return
        if self._pending_local_load:
            self._pending_local_load = False
            if self._local_fetch_timer is not None:
                self._local_fetch_timer.stop()
            model_id = str(models[0])
            self.config["api_model"] = model_id
            self.api_model_input.setText(model_id)
            if ctx_length:
                self._model_ctx_cap = int(ctx_length)
                self.config["ctx_limit"] = int(ctx_length)
                self.config["max_tokens"] = int(ctx_length)
                self.sig_ctx_limit_changed.emit(int(ctx_length))
                self._emit_trace(f"Context: {int(ctx_length):,} tokens (auto-configured to model capacity)", "CTX")
            self.set_config_dirty(True)
            self.emit_model_payload()
            self._emit_trace(f"Local server ready: model={model_id}", "MODEL")
            self.sig_load_requested.emit()
            self._sync_path_display()
            return
        self.api_model_combo.blockSignals(True)
        self.api_model_combo.clear()
        self.api_model_combo.addItems([str(model) for model in models])
        self.api_model_combo.blockSignals(False)
        self.api_model_combo.setVisible(True)
        # Combo replaces the bare text input once a model list is loaded —
        # they previously rendered side-by-side showing the same value.
        self.api_model_input.setVisible(False)
        current = self.api_model_input.text().strip()
        if current and current in models:
            self.api_model_combo.setCurrentText(current)
        else:
            self.api_model_combo.setCurrentIndex(0)
            self.api_model_input.setText(self.api_model_combo.currentText())
            self._update_config_value("api_model", self.api_model_combo.currentText())
            self.emit_model_payload()
        if ctx_length and not self._model_ctx_cap:
            self._model_ctx_cap = int(ctx_length)
            self.config["ctx_limit"] = int(ctx_length)
            self.config["max_tokens"] = int(ctx_length)
            self.sig_ctx_limit_changed.emit(int(ctx_length))
            self._emit_trace(f"Context: {int(ctx_length):,} tokens (auto-configured to model capacity)", "CTX")
        self._emit_trace(f"Loaded {len(models)} model(s).", "MODEL")

    def _on_models_error(self, message: str) -> None:
        if self._pending_local_load:
            if self._local_fetch_attempts in (1, 2, 3):
                base = self.config.get("api_base", "")
                self._emit_trace(f"Model fetch failed: {message or 'no response'} (base={base})", "MODEL", True)
            return
        self.btn_fetch_models.setEnabled(True)
        self._emit_trace(f"Model fetch failed: {message}", "MODEL", True)

    def auto_probe_server(self) -> None:
        """Probe the configured API base on startup to auto-detect model limits."""
        base = self.config.get("api_base", "")
        if not base or self._pending_local_load:
            return
        if self._model_fetcher and self._model_fetcher.isRunning():
            return
        self._model_fetcher = ModelListLoader(str(base), self.config.get("api_key"))
        self._model_fetcher.finished.connect(self._on_models_loaded)
        self._model_fetcher.error.connect(lambda _msg: None)  # silent on startup
        self._model_fetcher.start()

    def _on_model_combo_changed(self, _index: int) -> None:
        if not self.api_model_combo.isVisible():
            return
        value = self.api_model_combo.currentText().strip()
        if value:
            self.api_model_input.setText(value)
            self._update_config_value("api_model", value)
            self.emit_model_payload()

    def set_config_dirty(self, dirty: bool = True) -> None:
        self._config_dirty = bool(dirty)
        return

    def on_external_config_changed(self, payload: dict) -> None:
        if not isinstance(payload, dict):
            return
        llm = payload.get("llm")
        if not isinstance(llm, dict) or self._config_dirty:
            return
        self.config.update(llm)
        self.apply_config_to_controls()
        self.set_config_dirty(False)

    def save_current_config(self) -> None:
        save_config(self.config)
        self._last_saved = QDateTime.currentDateTime()
        self.set_config_dirty(False)

    def _update_config_value(self, key, value) -> None:
        self.config[key] = value
        self.set_config_dirty(True)
        if key in {"api_base", "api_model", "api_key", "gguf_path", "backend", "api_provider"}:
            self._sync_path_display()
        # Auto-persist connection fields. Without this, the user would have
        # to remember to click SAVE SETTINGS after pasting an API key/base
        # and the values would vanish next session.
        if key in self._AUTOSAVE_KEYS:
            self._autosave_timer.start()

    def _autosave_connection_fields(self) -> None:
        """Persist the current self.config to ~/.config/.../config.yaml.

        Called by the debounced timer kicked off in _update_config_value when
        a connection-shaped field changes. Stays cheap because save_config()
        is itself a deep-merge into config.yaml, not a full rewrite.
        """
        try:
            save_config(self.config)
            self._last_saved = QDateTime.currentDateTime()
        except Exception:
            # Never break the UI for a save failure; the explicit SAVE SETTINGS
            # button stays as the user-visible recovery path.
            pass

    def apply_default_limits(self) -> None:
        return

    def handle_model_capabilities(self, payload: dict) -> None:
        self._is_model_loaded = True
        self.update_load_button_text()
        self._record_active_profile_success()
        model_ctx_length = payload.get("model_ctx_length")
        if model_ctx_length is None:
            self._sync_path_display()
            return
        self._model_ctx_cap = int(model_ctx_length)
        self.config["ctx_limit"] = int(model_ctx_length)
        self.config["max_tokens"] = int(model_ctx_length)
        self.sig_ctx_limit_changed.emit(int(model_ctx_length))
        self.set_config_dirty(True)
        self._emit_trace(f"Context: {int(model_ctx_length):,} tokens (auto-configured to model capacity)", "CTX")
        self._sync_path_display()

    def _on_ctx_limit_changed(self, value) -> None:
        self._update_config_value("ctx_limit", int(value))
        self.sig_ctx_limit_changed.emit(int(value))

    def toggle_endpoint_edit(self) -> None:
        self._endpoint_expanded = not self._endpoint_expanded
        self._endpoint_edit.setVisible(self._endpoint_expanded)
        self._endpoint_toggle.setText("Hide" if self._endpoint_expanded else "Edit")

    def reset_config(self) -> None:
        self._model_ctx_cap = None
        for key, value in DEFAULT_CONFIG.items():
            self.config[key] = value
        self.config["backend"] = "gguf_api"
        self.apply_config_to_controls()
        self.sig_ctx_limit_changed.emit(int(DEFAULT_CONFIG["ctx_limit"]))
        self.emit_model_payload()
        self.set_config_dirty(True)

    def pick_file(self) -> None:
        if self.config.get("backend", "gguf_api") not in ("gguf", "gguf_api"):
            return
        path, _ = QFileDialog.getOpenFileName(self, "Select GGUF", "", "GGUF (*.gguf)")
        if path:
            self.config["gguf_path"] = path
            self.config["api_model"] = ""
            self.emit_model_payload()
            self._sync_path_display()
            self.set_config_dirty(True)

    def toggle_load(self) -> None:
        self.btn_load.setEnabled(False)
        self.btn_load.setText("PROCESSING...")
        if self._is_model_loaded:
            if self.config.get("backend", "gguf_api") == "gguf_api" and self._local_server_proc:
                self._stop_server_on_unload = True
            self.sig_unload_requested.emit()
            return
        backend = self.config.get("backend", "gguf_api")
        if backend == "gguf_api":
            self._start_local_server_and_load()
            return
        # Preset resolution is backend-routed: gguf reads from the file path
        # (the file IS the model); openai-compat reads from the api_model id
        # (no local file). The old branch always fell through to gguf_path,
        # which mis-resolved remote loads — e.g. loading deepseek-v4-pro over
        # openai-compat resolved against a stale Qwen3 gguf_path and tagged
        # the turn with a Qwen omnicoder preset.
        if backend == "gguf" and self.config.get("gguf_path"):
            self._apply_registry_preset(str(self.config.get("gguf_path")))
        elif self.config.get("api_model"):
            # Non-gguf backends (openai-compat, vllm, sglang, etc.) — the
            # remote model id is the resolution hint.
            self._apply_registry_preset(str(self.config.get("api_model")))
        self.emit_model_payload()
        self.sig_load_requested.emit()

    def _open_profile_menu(self) -> None:
        from ui.panels.cloud_profile_menu import build_profile_menu
        menu = build_profile_menu(
            self.config,
            on_switch=self._switch_profile,
            on_save=self._save_current_as_profile,
            on_delete=self._delete_profile,
            parent=self,
        )
        menu.exec(self._profile_btn.mapToGlobal(self._profile_btn.rect().bottomLeft()))

    def _switch_profile(self, profile_id: str) -> None:
        self.config = _cloud_profiles.activate(self.config, profile_id)
        self.apply_config_to_controls()
        self.set_config_dirty(True)
        self.save_current_config()       # non-lossy persist (active + mirrored api_*)
        self.emit_model_payload()        # engine picks up new api_* BEFORE load
        self.sig_load_requested.emit()   # load the newly-active cloud model

    def _save_current_as_profile(self) -> None:
        from PySide6.QtWidgets import QInputDialog
        # a non-colliding default so accepting it ADDS a new profile; type an existing name to update.
        suggested = _cloud_profiles.suggest_label(self.config)
        name, ok = QInputDialog.getText(self, "Save profile", "Profile name:", text=suggested)
        if not ok or not name.strip():
            return
        self.config = _cloud_profiles.upsert_from_current(self.config, name.strip())
        self.save_current_config()

    def _delete_profile(self, profile_id: str) -> None:
        self.config = _cloud_profiles.delete(self.config, profile_id)
        self.save_current_config()

    def _record_active_profile_success(self) -> None:
        # Only write back when a cloud profile is active AND the live backend is
        # cloud — a local GGUF load must not stamp the cloud profile's last_model.
        if not _cloud_profiles.is_cloud_active(self.config):
            return
        active = _cloud_profiles.active_id(self.config)
        from datetime import datetime, timezone
        self.config = _cloud_profiles.record_success(
            self.config, active,
            str(self.config.get("api_model") or ""),
            datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        )
        try:
            self.save_current_config()
        except Exception:
            pass

    def update_load_button_text(self) -> None:
        self.btn_load.setText("UNLOAD MODEL" if self._is_model_loaded else "LOAD MODEL")
        self._sync_path_display()

    def _apply_registry_preset(self, gguf_path: str) -> None:
        preset = resolve_model_preset(str(gguf_path))
        self.config = apply_model_preset(self.config, preset)
        self.apply_config_to_controls()
        self.emit_model_payload()
        self._emit_trace(
            (
                f"Preset resolved: {describe_model_preset(preset)} "
                f"id={preset.preset_id} confidence={preset.confidence} "
                f"ctx={preset.context_window or 'auto'} "
                f"temp={preset.sampler.get('temperature')} "
                f"top_p={preset.sampler.get('top_p')} "
                f"top_k={preset.sampler.get('top_k')} "
                f"max_out={preset.output.get('default_max_output_tokens')}"
            ),
            "MODEL",
        )
        if preset.warnings:
            self._emit_trace(f"Preset warnings: {'; '.join(preset.warnings)}", "MODEL", True)
        self._surface_preset_to_world_state(preset)

    def _surface_preset_to_world_state(self, preset) -> None:
        """Push resolved model preset metadata into WorldState for interceptors."""
        ws = getattr(self.state, "world_state", None)
        if ws is None:
            return
        # Find the engine key — use the first LLM engine in world state, or "llm"
        engines = ws.state.get("engines", {})
        engine_key = next(
            (k for k in engines if k.startswith("llm")),
            "llm",
        )
        caps = preset.capabilities or {}
        ws.set_engine_meta(
            engine_key,
            model_preset={
                "family_id": preset.family_id,
                "family_name": preset.family_name,
                "preset_id": preset.preset_id,
                "variant_id": preset.variant_id,
                "confidence": preset.confidence,
                "context_window": preset.context_window,
                "capabilities": caps,
            },
        )

    def set_engine_status(self, status: SystemStatus, is_model_loaded: bool) -> None:
        self._is_model_loaded = bool(is_model_loaded)
        self._last_status = status
        processing = status in (SystemStatus.LOADING, SystemStatus.RUNNING, SystemStatus.UNLOADING)
        self.btn_load.setEnabled(not processing)
        if processing:
            self.btn_load.setText("PROCESSING...")
        else:
            self.update_load_button_text()
        if status == SystemStatus.ERROR:
            self._is_model_loaded = False
            self.update_load_button_text()

    def handle_ready_after_unload(self) -> None:
        self._is_model_loaded = False
        self._model_ctx_cap = None
        self.update_load_button_text()
        if self._stop_server_on_unload:
            self._stop_server_on_unload = False
            self.stop_local_server()

    def apply_operator_config(self, config: dict) -> None:
        if not isinstance(config, dict):
            return
        self.config.update(config)
        self.apply_config_to_controls()
        self.sig_ctx_limit_changed.emit(int(self.config.get("ctx_limit", DEFAULT_CONFIG["ctx_limit"])))
        self.emit_model_payload()
        self.set_config_dirty(True)
