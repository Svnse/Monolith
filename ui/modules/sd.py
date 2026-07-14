from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from PySide6.QtCore import QTimer, Qt, QSize
from PySide6.QtGui import QIcon, QPixmap
from PySide6.QtWidgets import (
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QProgressBar,
    QScrollArea,
    QTextEdit,
    QVBoxLayout,
    QWidget,
    QComboBox,
)

import core.style as _s
from core.paths import MONOLITH_ROOT
from core.config import get_config, update_config_section
from core.state import SystemStatus
from ui.components.atoms import MonoGroupBox, MonoButton, MonoDragSpin, CollapsibleSection

try:
    from PIL.ImageQt import ImageQt
except Exception:  # pragma: no cover - runtime guard
    ImageQt = None


from core.vision_models import (
    ModelEntry as _ModelEntry_Shared,
    detect_backend,
    scan_model_root,
    DEFAULT_MODEL_ROOT,
)

VISION_ARTIFACT_ROOT = MONOLITH_ROOT / "artifacts" / "vision"
INDEX_PATH = VISION_ARTIFACT_ROOT / "index.jsonl"


def _now_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _safe_int(text: str, default: int) -> int:
    try:
        return int(text.strip())
    except Exception:
        return default


def _pil_to_pixmap(image: object) -> QPixmap | None:
    if image is None or ImageQt is None:
        return None
    try:
        qimg = ImageQt(image.convert("RGBA"))
        return QPixmap.fromImage(qimg)
    except Exception:
        return None


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _artifact_day_dir() -> Path:
    now = datetime.now()
    return VISION_ARTIFACT_ROOT / f"{now.year:04d}" / f"{now.month:02d}" / f"{now.day:02d}"


@dataclass
class _VisionResult:
    batch_index: int
    image: object
    pixmap: QPixmap | None
    received_at: str


@dataclass
class _ModelEntry:
    path: str
    label: str
    backend: str


class SDModule(QWidget):
    """
    Vision v2 parity-first module.

    Keeps the original addon entrypoint/class name for compatibility with
    existing builtin registration (`sd_factory`).
    """

    _CFG_DEFAULTS: dict[str, Any] = {
        "model_path": "",
        "model_root": str(DEFAULT_MODEL_ROOT),
        "prompt": "",
        "negative_prompt": "",
        "width": 512,
        "height": 512,
        "steps": 25,
        "guidance_scale": 7.5,
        "seed": -1,
        "scheduler": "dpm++",
        "batch_size": 1,
        "lora_path": "",
        "lora_scale": 0.8,
        "save_dir": "",
    }

    def __init__(self, bridge=None, guard=None, ui_bridge=None, vision_artifact_bridge=None):
        super().__init__()
        self.bridge = bridge
        self.guard = guard
        self._ui_bridge = ui_bridge
        # Optional reference to the skill-side VisionArtifactBridge so the
        # SDModule can dedupe its auto-save against skill-triggered gens
        # (the bridge already saves those via its own pending-set path).
        # None when running standalone / outside the bootstrap-wired addon.
        self._vision_artifact_bridge = vision_artifact_bridge

        self._cfg_timer = QTimer(self)
        self._cfg_timer.setInterval(700)
        self._cfg_timer.setSingleShot(True)
        self._cfg_timer.timeout.connect(self._save_config)

        self._status_flash_timer = QTimer(self)
        self._status_flash_timer.setInterval(1500)
        self._status_flash_timer.setSingleShot(True)
        self._status_flash_timer.timeout.connect(self._clear_status_detail)

        self._engine_status: SystemStatus | None = None
        self._results: list[_VisionResult] = []
        self._last_resource: dict[str, int] = {"vram_used_mb": 0, "vram_free_mb": 0}
        self._backend_label = "unknown"
        self._active_request_meta: dict[str, Any] = {}
        self._selected_result_idx: int | None = None
        self._has_direct_vision_signals = False
        self._model_entries: list[_ModelEntry] = []
        self._status_is_error: bool = False

        self.config = self._load_config()
        self._build_ui()
        self._apply_config_to_ui()
        self._wire_signals()
        self._refresh_button_states()
        self._refresh_widget_styles()
        if self._ui_bridge is not None and hasattr(self._ui_bridge, "sig_theme_changed"):
            self._ui_bridge.sig_theme_changed.connect(self._on_theme_changed)

    # ---------------- UI ----------------
    def _build_ui(self) -> None:
        # Single-column scroll layout. Replaces the previous two-column
        # left/right split which was designed for a top-level window — at
        # the companion-pane width (~360-600px) the right column was
        # cramped to the point of being unusable. Top-to-bottom workflow:
        # MODEL → PROMPT → PARAMS → GENERATE → RESULTS → TRACE.
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        outer.addWidget(scroll, 1)

        body = QWidget()
        left_col = QVBoxLayout(body)
        left_col.setContentsMargins(10, 10, 10, 10)
        left_col.setSpacing(8)
        scroll.setWidget(body)

        # ── MODEL ───────────────────────────────────────────────────────
        # Single visible row for the active model path, single toggle for
        # LOAD/UNLOAD (one button whose text + accent flip based on
        # engine_status — see _refresh_button_states). Discovery controls
        # (library root, scan, picker) live in a collapsible below.
        grp_model = MonoGroupBox("MODEL")

        model_path_row = QHBoxLayout()
        self.inp_model_path = QLineEdit()
        self.inp_model_path.setPlaceholderText("No model selected")
        self.inp_model_path.setReadOnly(True)
        self.btn_browse_model = MonoButton("File")
        self.btn_browse_model_dir = MonoButton("Folder")
        model_path_row.addWidget(self.inp_model_path, stretch=1)
        model_path_row.addWidget(self.btn_browse_model)
        model_path_row.addWidget(self.btn_browse_model_dir)
        grp_model.add_layout(model_path_row)

        # Single LOAD toggle. Replaces the previous Load+Unload pair —
        # button text morphs based on engine status (LOAD / LOADING… /
        # UNLOAD / RETRY LOAD). User clicks once, the handler decides.
        self.btn_load_toggle = MonoButton("● LOAD", accent=True)
        self.btn_load_toggle.setFixedHeight(28)
        grp_model.add_widget(self.btn_load_toggle)

        # LIBRARY — discovery controls, collapsed by default.
        library_section = CollapsibleSection("LIBRARY")
        library_inner = QVBoxLayout()
        library_inner.setSpacing(6)
        library_inner.setContentsMargins(0, 4, 0, 4)
        library_root_row = QHBoxLayout()
        self.inp_model_root = QLineEdit()
        self.inp_model_root.setPlaceholderText("Library root")
        self.btn_browse_model_root = MonoButton("Root")
        self.btn_scan_models = MonoButton("Scan")
        library_root_row.addWidget(self.inp_model_root, stretch=1)
        library_root_row.addWidget(self.btn_browse_model_root)
        library_root_row.addWidget(self.btn_scan_models)
        library_inner.addLayout(library_root_row)

        library_pick_row = QHBoxLayout()
        self.cmb_models = QComboBox()
        self.cmb_models.setPlaceholderText("Scanned models")
        self.btn_use_scanned_model = MonoButton("Use")
        library_pick_row.addWidget(self.cmb_models, stretch=1)
        library_pick_row.addWidget(self.btn_use_scanned_model)
        library_inner.addLayout(library_pick_row)
        library_section.set_content_layout(library_inner)
        grp_model.add_widget(library_section)

        # Compact status grid: Status, Backend, VRAM, last detail.
        status_grid = QGridLayout()
        status_grid.setSpacing(4)
        lbl_status_key = QLabel("Status")
        self.lbl_status = QLabel("IDLE")
        lbl_backend_key = QLabel("Backend")
        self.lbl_backend = QLabel("unknown")
        self.lbl_vram = QLabel("VRAM: --")
        self.lbl_status_detail = QLabel("")
        self.lbl_status_detail.setWordWrap(True)
        status_grid.addWidget(lbl_status_key, 0, 0)
        status_grid.addWidget(self.lbl_status, 0, 1)
        status_grid.addWidget(lbl_backend_key, 1, 0)
        status_grid.addWidget(self.lbl_backend, 1, 1)
        status_grid.addWidget(self.lbl_vram, 2, 0, 1, 2)
        status_grid.addWidget(self.lbl_status_detail, 3, 0, 1, 2)
        grp_model.add_layout(status_grid)

        left_col.addWidget(grp_model)

        # Single GENERATE toggle. Same morph pattern as btn_load_toggle —
        # text flips between GENERATE and STOP based on engine status.
        # Allocated up here so _wire_signals() can reach it; the widget
        # itself is added to its group further down in _build_ui.
        self.btn_gen_toggle = MonoButton("● GENERATE", accent=True)
        self.btn_gen_toggle.setFixedHeight(34)  # big, primary action

        # ---- Generation params ----
        grp_gen = MonoGroupBox("GENERATION")

        # Placeholders carry the labels; no separate QLabel widgets needed.
        self.txt_prompt = QTextEdit()
        self.txt_prompt.setPlaceholderText("Prompt - describe the image to generate")
        self.txt_prompt.setFixedHeight(80)
        grp_gen.add_widget(self.txt_prompt)

        self.txt_negative = QTextEdit()
        self.txt_negative.setPlaceholderText("Negative prompt (optional)")
        self.txt_negative.setFixedHeight(55)
        grp_gen.add_widget(self.txt_negative)

        form = QFormLayout()
        form.setSpacing(8)
        form.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)

        self.spn_width = MonoDragSpin(minimum=64, maximum=4096, step=64)
        self.spn_height = MonoDragSpin(minimum=64, maximum=4096, step=64)
        size_row = QHBoxLayout()
        size_row.addWidget(self.spn_width)
        size_row.addWidget(QLabel("x"))
        size_row.addWidget(self.spn_height)
        size_box = QWidget()
        size_box.setLayout(size_row)
        form.addRow("Size", size_box)

        self.spn_steps = MonoDragSpin(minimum=1, maximum=200, step=1)
        form.addRow("Steps", self.spn_steps)

        self.spn_cfg = MonoDragSpin(minimum=0.0, maximum=30.0, step=0.1, decimals=1)
        form.addRow("Guidance", self.spn_cfg)

        self.inp_seed = QLineEdit()
        self.inp_seed.setPlaceholderText("-1 for random")
        form.addRow("Seed", self.inp_seed)

        self.cmb_scheduler = QComboBox()
        self.cmb_scheduler.addItems(["dpm++", "euler", "ddim", "lcm"])
        form.addRow("Scheduler", self.cmb_scheduler)

        self.spn_batch = MonoDragSpin(minimum=1, maximum=16, step=1)
        form.addRow("Batch Size", self.spn_batch)

        self.inp_lora_path = QLineEdit()
        self.inp_lora_path.setReadOnly(True)
        lora_path_row = QHBoxLayout()
        lora_path_row.addWidget(self.inp_lora_path, stretch=1)
        self.btn_browse_lora = MonoButton("Browse")
        self.btn_clear_lora = MonoButton("Clear")
        lora_path_row.addWidget(self.btn_browse_lora)
        lora_path_row.addWidget(self.btn_clear_lora)
        lora_box = QWidget()
        lora_box.setLayout(lora_path_row)
        form.addRow("LoRA", lora_box)

        self.spn_lora_scale = MonoDragSpin(minimum=0.0, maximum=2.0, step=0.05, decimals=2)
        form.addRow("LoRA Scale", self.spn_lora_scale)

        grp_gen.add_layout(form)
        left_col.addWidget(grp_gen)

        # ── GENERATE ────────────────────────────────────────────────────
        # Big morph button outside any group box — primary action surface.
        # Progress bar above the button shows step progression while running.
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setFixedHeight(8)
        self.lbl_progress = QLabel("0 / 0")
        prog_row = QHBoxLayout()
        prog_row.addWidget(self.progress, stretch=1)
        prog_row.addWidget(self.lbl_progress)
        left_col.addLayout(prog_row)
        left_col.addWidget(self.btn_gen_toggle)

        # ── RESULTS ─────────────────────────────────────────────────────
        # Preview + thumb gallery + auto-save status. Images are saved
        # automatically on arrival (see _add_result → _save_result_artifact);
        # the EXPORT buttons copy to a user-chosen folder for sharing.
        grp_preview = MonoGroupBox("RESULTS")
        self.lbl_preview = QLabel("No image yet")
        self.lbl_preview.setAlignment(Qt.AlignCenter)
        self.lbl_preview.setMinimumHeight(260)
        grp_preview.add_widget(self.lbl_preview)

        self.lst_results = QListWidget()
        self.lst_results.setViewMode(QListWidget.IconMode)
        self.lst_results.setMovement(QListWidget.Static)
        self.lst_results.setResizeMode(QListWidget.Adjust)
        self.lst_results.setIconSize(QSize(80, 80))
        self.lst_results.setSpacing(6)
        self.lst_results.setFixedHeight(100)
        self.lst_results.setFrameShape(QFrame.NoFrame)
        grp_preview.add_widget(self.lst_results)

        self.lbl_save_status = QLabel("")
        self.lbl_save_status.setStyleSheet(f"color: {_s.FG_DIM}; font-size: 9px;")
        self.lbl_save_status.setWordWrap(True)
        grp_preview.add_widget(self.lbl_save_status)

        save_row = QHBoxLayout()
        self.btn_save_selected = MonoButton("Export Selected")
        self.btn_save_all = MonoButton("Export All")
        save_row.addWidget(self.btn_save_selected)
        save_row.addWidget(self.btn_save_all)
        save_row.addStretch()
        grp_preview.add_layout(save_row)
        left_col.addWidget(grp_preview)

        # ── TRACE (collapsed) ───────────────────────────────────────────
        # Engine subprocess log. Debug surface — not load-bearing for the
        # typical generation flow, so hide by default.
        trace_section = CollapsibleSection("TRACE")
        trace_inner = QVBoxLayout()
        trace_inner.setContentsMargins(0, 4, 0, 0)
        self.txt_trace = QTextEdit()
        self.txt_trace.setReadOnly(True)
        self.txt_trace.setMaximumHeight(140)
        trace_inner.addWidget(self.txt_trace)
        trace_section.set_content_layout(trace_inner)
        left_col.addWidget(trace_section)

        left_col.addStretch()

    def _wire_signals(self) -> None:
        self.btn_browse_model.clicked.connect(self._browse_model_file)
        self.btn_browse_model_dir.clicked.connect(self._browse_model_dir)
        self.btn_browse_model_root.clicked.connect(self._browse_model_root)
        self.btn_scan_models.clicked.connect(self._scan_models_clicked)
        self.btn_use_scanned_model.clicked.connect(self._use_scanned_model)
        self.btn_browse_lora.clicked.connect(self._browse_lora)
        self.btn_clear_lora.clicked.connect(self._clear_lora)
        # Single toggle buttons — _on_*_toggle decides load vs unload /
        # generate vs stop based on the current _engine_status.
        self.btn_load_toggle.clicked.connect(self._on_load_toggle_clicked)
        self.btn_gen_toggle.clicked.connect(self._on_gen_toggle_clicked)
        self.btn_save_selected.clicked.connect(self._save_selected)
        self.btn_save_all.clicked.connect(self._save_all)
        self.lst_results.currentRowChanged.connect(self._select_result_row)

        for sig in (
            self.txt_prompt.textChanged,
            self.txt_negative.textChanged,
            self.spn_width.valueChanged,
            self.spn_height.valueChanged,
            self.spn_steps.valueChanged,
            self.spn_cfg.valueChanged,
            self.inp_seed.textChanged,
            self.cmb_scheduler.currentTextChanged,
            self.spn_batch.valueChanged,
            self.inp_lora_path.textChanged,
            self.spn_lora_scale.valueChanged,
            self.inp_model_root.textChanged,
        ):
            sig.connect(self._queue_save_config)
        self.cmb_models.activated.connect(lambda *_: self._use_scanned_model())

        if self.guard is not None:
            self.guard.sig_status.connect(self._on_guard_status)
            self.guard.sig_trace.connect(self._on_guard_trace)
            if hasattr(self.guard, "sig_engine_event"):
                self.guard.sig_engine_event.connect(self._on_guard_engine_event)
            if hasattr(self.guard, "sig_finished"):
                self.guard.sig_finished.connect(self._on_guard_finished)
            # Legacy flattened fallback path (no batch index)
            if hasattr(self.guard, "sig_image"):
                self.guard.sig_image.connect(self._on_guard_image_fallback)

        vision_engine = None
        if self.guard is not None and hasattr(self.guard, "engines"):
            vision_engine = self.guard.engines.get("vision")
        if vision_engine is not None:
            if hasattr(vision_engine, "sig_image"):
                try:
                    vision_engine.sig_image.connect(self._on_vision_image)
                    self._has_direct_vision_signals = True
                except Exception:
                    pass
            if hasattr(vision_engine, "sig_progress"):
                try:
                    vision_engine.sig_progress.connect(self._on_vision_progress)
                except Exception:
                    pass
            if hasattr(vision_engine, "sig_resource"):
                try:
                    vision_engine.sig_resource.connect(self._on_vision_resource)
                except Exception:
                    pass

    # ---------------- config ----------------
    def _load_config(self) -> dict[str, Any]:
        cfg = get_config().vision.model_dump()
        return {**self._CFG_DEFAULTS, **cfg}

    def _queue_save_config(self, *_args) -> None:
        self._cfg_timer.start()

    def _save_config(self) -> None:
        cfg = self._collect_ui_config()
        try:
            update_config_section("vision", cfg, persist=True)
            self.config = cfg
        except Exception as exc:
            self._set_status_detail(f"Config save failed: {exc}", error=True)

    def _apply_config_to_ui(self) -> None:
        cfg = self.config
        self.inp_model_path.setText(str(cfg.get("model_path") or ""))
        self.inp_model_root.setText(str(cfg.get("model_root") or str(DEFAULT_MODEL_ROOT)))
        self.txt_prompt.setPlainText(str(cfg.get("prompt") or ""))
        self.txt_negative.setPlainText(str(cfg.get("negative_prompt") or ""))
        self.spn_width.setValue(int(cfg.get("width", 512)))
        self.spn_height.setValue(int(cfg.get("height", 512)))
        self.spn_steps.setValue(int(cfg.get("steps", 25)))
        self.spn_cfg.setValue(float(cfg.get("guidance_scale", 7.5)))
        self.inp_seed.setText(str(cfg.get("seed", -1)))
        scheduler = str(cfg.get("scheduler", "dpm++"))
        if self.cmb_scheduler.findText(scheduler) >= 0:
            self.cmb_scheduler.setCurrentText(scheduler)
        self.spn_batch.setValue(int(cfg.get("batch_size", 1)))
        self.inp_lora_path.setText(str(cfg.get("lora_path") or ""))
        self.spn_lora_scale.setValue(float(cfg.get("lora_scale", 0.8)))
        self._refresh_model_picker()

    def _collect_ui_config(self) -> dict[str, Any]:
        return {
            "model_path": self.inp_model_path.text().strip(),
            "model_root": self.inp_model_root.text().strip(),
            "prompt": self.txt_prompt.toPlainText().strip(),
            "negative_prompt": self.txt_negative.toPlainText().strip(),
            "width": int(self.spn_width.value()),
            "height": int(self.spn_height.value()),
            "steps": int(self.spn_steps.value()),
            "guidance_scale": float(self.spn_cfg.value()),
            "seed": _safe_int(self.inp_seed.text(), -1),
            "scheduler": self.cmb_scheduler.currentText(),
            "batch_size": int(self.spn_batch.value()),
            "lora_path": self.inp_lora_path.text().strip(),
            "lora_scale": float(self.spn_lora_scale.value()),
            "save_dir": str(self.config.get("save_dir") or ""),
        }

    # ---------------- bridge actions ----------------
    def _submit(self, command: str, payload: dict | None = None) -> None:
        if self.bridge is None:
            raise RuntimeError("Vision bridge unavailable")
        task = self.bridge.wrap("sd", command, "vision", payload=(payload or {}))
        self.bridge.submit(task)

    def _load_model_clicked(self) -> None:
        model_path = self.inp_model_path.text().strip()
        if not model_path:
            self._set_status_detail("Select a model path first", error=True)
            self._log("WARNING", "[vision] load requested without model path")
            return
        self._queue_save_config()
        try:
            self._submit("set_path", {"path": model_path})
            self._submit("load")
            self._set_status_detail("Load requested")
            self._log("INFO", f"[vision] load requested: {model_path}")
        except Exception as exc:
            self._set_status_detail(f"Load failed: {exc}", error=True)
            self._log("ERROR", f"[vision] load failed: {exc}")

    def _unload_model_clicked(self) -> None:
        try:
            self._submit("unload")
            self._set_status_detail("Unload requested")
            self._log("INFO", "[vision] unload requested")
        except Exception as exc:
            self._set_status_detail(f"Unload failed: {exc}", error=True)
            self._log("ERROR", f"[vision] unload failed: {exc}")

    def _generate_clicked(self) -> None:
        prompt = self.txt_prompt.toPlainText().strip()
        model_path = self.inp_model_path.text().strip()
        if not model_path:
            self._set_status_detail("No model selected", error=True)
            self._log("WARNING", "[vision] generate blocked: no model")
            return
        if not prompt:
            self._set_status_detail("Prompt is empty", error=True)
            self._log("WARNING", "[vision] generate blocked: empty prompt")
            return

        cfg = self._collect_ui_config()
        self.config = cfg
        self._save_config()

        self._active_request_meta = {
            "requested_at": _now_iso(),
            "config": cfg,
            "backend": self._backend_label,
        }
        self._results.clear()
        self._selected_result_idx = None
        self.lst_results.clear()
        self._set_preview_pixmap(None)
        self._set_progress(0, max(1, int(cfg.get("steps", 1))))

        try:
            # Load is cheap to queue and ensures model path changes are applied.
            self._submit("set_path", {"path": model_path})
            self._submit("load")
            self._submit("generate", cfg)
            self._set_status_detail("Generation requested")
            self._log("INFO", f"[vision] generate requested ({cfg.get('width')}x{cfg.get('height')}, steps={cfg.get('steps')})")
        except Exception as exc:
            self._set_status_detail(f"Generate failed: {exc}", error=True)

        self._refresh_button_states()

    def _stop_clicked(self) -> None:
        try:
            if self.bridge is not None and hasattr(self.bridge, "stop"):
                self.bridge.stop("vision")
                self._set_status_detail("Stop requested")
                self._log("INFO", "[vision] stop requested")
            else:
                self._set_status_detail("Stop unavailable", error=True)
                self._log("WARNING", "[vision] stop unavailable")
        except Exception as exc:
            self._set_status_detail(f"Stop failed: {exc}", error=True)
            self._log("ERROR", f"[vision] stop failed: {exc}")

    # ---------------- dialogs ----------------
    def _browse_model_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Vision Model File",
            str(Path(self.inp_model_path.text()).parent) if self.inp_model_path.text() else "",
            "Model Files (*.safetensors *.ckpt);;All Files (*)",
        )
        if path:
            self.inp_model_path.setText(path)
            self._queue_save_config()

    def _browse_model_root(self) -> None:
        start_dir = self.inp_model_root.text().strip() or str(DEFAULT_MODEL_ROOT)
        path = QFileDialog.getExistingDirectory(self, "Select Vision Model Root", start_dir)
        if path:
            self.inp_model_root.setText(path)
            self._queue_save_config()
            self._refresh_model_picker()

    def _scan_models_clicked(self) -> None:
        self._refresh_model_picker(force_status=True)

    def _use_scanned_model(self) -> None:
        idx = self.cmb_models.currentIndex()
        if idx < 0:
            return
        path = self.cmb_models.currentData(Qt.UserRole)
        if path:
            self.inp_model_path.setText(str(path))
            self._queue_save_config()
            self._set_status_detail("Model path set from library")

    def _browse_model_dir(self) -> None:
        path = QFileDialog.getExistingDirectory(
            self,
            "Select Diffusers Model Directory",
            str(Path(self.inp_model_path.text()).parent) if self.inp_model_path.text() else "",
        )
        if path:
            self.inp_model_path.setText(path)
            self._queue_save_config()

    def _browse_lora(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select LoRA Weights",
            str(Path(self.inp_lora_path.text()).parent) if self.inp_lora_path.text() else "",
            "LoRA Files (*.safetensors *.ckpt);;All Files (*)",
        )
        if path:
            self.inp_lora_path.setText(path)
            self._queue_save_config()

    def _clear_lora(self) -> None:
        self.inp_lora_path.clear()
        self._queue_save_config()

    # ---------------- guard / engine events ----------------
    def _on_guard_status(self, engine_key: str, status: SystemStatus) -> None:
        if engine_key != "vision":
            return
        self._engine_status = status
        if status == SystemStatus.LOADING:
            self.lbl_status.setText("LOADING")
        elif status == SystemStatus.RUNNING:
            self.lbl_status.setText("RUNNING")
        elif status == SystemStatus.UNLOADING:
            self.lbl_status.setText("UNLOADING")
        elif status == SystemStatus.ERROR:
            self.lbl_status.setText("ERROR")
        elif status == SystemStatus.READY:
            self.lbl_status.setText("READY")
        self._refresh_button_states()

    def _on_guard_trace(self, engine_key: str, message: str) -> None:
        if engine_key != "vision":
            return
        msg = str(message or "")
        self._append_trace(msg)
        self._parse_trace_updates(msg)
        if "error" in msg.lower():
            self._set_status_detail(msg, error=True)

    def _on_guard_finished(self, engine_key: str, *_args) -> None:
        if engine_key != "vision":
            return
        self._refresh_button_states()

    def _on_guard_image_fallback(self, image: object) -> None:
        # Legacy flattened path loses batch index, so append sequentially only
        # if no direct VisionProcess batch signal has delivered results yet.
        if self._has_direct_vision_signals:
            return
        if self._results:
            return
        self._add_result(image, 0)

    def _on_guard_engine_event(self, engine_key: str, event: dict) -> None:
        if engine_key != "vision":
            return
        kind = str(event.get("event") or "")
        if kind == "progress":
            self._on_vision_progress(int(event.get("step") or 0), int(event.get("total") or 0))
        elif kind == "resource":
            self._on_vision_resource({
                "vram_used_mb": int(event.get("vram_used_mb") or 0),
                "vram_free_mb": int(event.get("vram_free_mb") or 0),
            })
        elif kind == "result" and (not self._has_direct_vision_signals) and not self._results:
            self._add_result(event.get("image"), int(event.get("batch_index") or 0))

    def _on_vision_image(self, image: object, batch_index: int, gen_id: int = 0) -> None:
        # gen_id is supplied by VisionProcess.sig_image (extended to a
        # 3-arg signal back when VisionArtifactBridge was added). When this
        # gen_id is in the bridge's pending set, the bridge already owns
        # auto-save for it; we skip the SDModule's auto-save to avoid
        # writing the same PNG twice.
        self._add_result(image, int(batch_index), int(gen_id or 0))

    def _on_vision_progress(self, current: int, total: int, gen_id: int = 0) -> None:
        self._set_progress(current, total)

    def _on_vision_resource(self, info: dict) -> None:
        self._last_resource = {
            "vram_used_mb": int(info.get("vram_used_mb") or 0),
            "vram_free_mb": int(info.get("vram_free_mb") or 0),
        }
        used = self._last_resource["vram_used_mb"]
        free = self._last_resource["vram_free_mb"]
        self.lbl_vram.setText(f"VRAM: used {used} MB | free {free} MB")

    # ---------------- result handling ----------------
    def _add_result(self, image: object, batch_index: int, gen_id: int = 0) -> None:
        pixmap = _pil_to_pixmap(image)
        result = _VisionResult(
            batch_index=batch_index,
            image=image,
            pixmap=pixmap,
            received_at=_now_iso(),
        )
        self._results.append(result)
        self._results.sort(key=lambda r: r.batch_index)
        self._rebuild_result_list()
        if self._selected_result_idx is None:
            self._select_result_row(0)
        self._refresh_button_states()
        self._set_status_detail(f"Received image {batch_index + 1}/{len(self._results)}")

        # Auto-save every USER-triggered result on arrival. Skill-triggered
        # generations are saved by VisionArtifactBridge under the same
        # artifacts/vision/YYYY/MM/DD/ tree — dedupe by checking the
        # bridge's pending set so we don't write the same PNG twice. If
        # the bridge isn't wired (standalone / test runs), we save
        # everything; cheap and predictable.
        is_bridge_owned = False
        if self._vision_artifact_bridge is not None and gen_id:
            try:
                is_bridge_owned = bool(self._vision_artifact_bridge.is_pending(gen_id))
            except Exception:
                is_bridge_owned = False

        if is_bridge_owned:
            self.lbl_save_status.setText("Saved by skill bridge")
            return

        try:
            saved_path = self._save_result_artifact(result)
            if saved_path is not None:
                self.lbl_save_status.setText(f"Saved: {saved_path.name}")
        except Exception as exc:
            # Auto-save failure must not break the result display.
            self.lbl_save_status.setText(f"Auto-save failed: {exc}")

    def _rebuild_result_list(self) -> None:
        current_batch = None
        if self._selected_result_idx is not None and 0 <= self._selected_result_idx < len(self._results):
            current_batch = self._results[self._selected_result_idx].batch_index
        self.lst_results.blockSignals(True)
        self.lst_results.clear()
        for idx, res in enumerate(self._results):
            item = QListWidgetItem(f"#{res.batch_index}")
            if res.pixmap is not None:
                thumb = res.pixmap.scaled(
                    120, 120, Qt.KeepAspectRatio, Qt.SmoothTransformation
                )
                item.setIcon(QIcon(thumb))
            item.setData(Qt.UserRole, idx)
            self.lst_results.addItem(item)
        self.lst_results.blockSignals(False)
        if current_batch is not None:
            for row, res in enumerate(self._results):
                if res.batch_index == current_batch:
                    self.lst_results.setCurrentRow(row)
                    return

    def _select_result_row(self, row: int) -> None:
        if row < 0 or row >= len(self._results):
            self._selected_result_idx = None
            if self._results:
                self._set_preview_pixmap(self._results[0].pixmap)
            else:
                self._set_preview_pixmap(None)
            return
        self._selected_result_idx = row
        self._set_preview_pixmap(self._results[row].pixmap)

    def _set_preview_pixmap(self, pixmap: QPixmap | None) -> None:
        if pixmap is None:
            self.lbl_preview.setText("No image yet")
            self.lbl_preview.setPixmap(QPixmap())
            return
        scaled = pixmap.scaled(
            self.lbl_preview.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        self.lbl_preview.setText("")
        self.lbl_preview.setPixmap(scaled)

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        if self._selected_result_idx is not None and 0 <= self._selected_result_idx < len(self._results):
            self._set_preview_pixmap(self._results[self._selected_result_idx].pixmap)

    # ---------------- saving / artifacts ----------------
    def _save_selected(self) -> None:
        if self._selected_result_idx is None:
            self._set_status_detail("No result selected", error=True)
            self._log("WARNING", "[vision] save selected blocked: no result")
            return
        if not (0 <= self._selected_result_idx < len(self._results)):
            self._set_status_detail("Invalid result selection", error=True)
            return
        saved = self._save_result_artifact(self._results[self._selected_result_idx])
        if saved is not None:
            self._set_status_detail(f"Saved: {saved.name}")
            self._log("INFO", f"[vision] saved {saved.name}")

    def _save_all(self) -> None:
        if not self._results:
            self._set_status_detail("No results to save", error=True)
            self._log("WARNING", "[vision] save all blocked: no results")
            return
        count = 0
        for res in self._results:
            if self._save_result_artifact(res) is not None:
                count += 1
        self._set_status_detail(f"Saved {count} artifact(s)")
        self._log("INFO", f"[vision] saved {count} artifact(s)")

    def _save_result_artifact(self, result: _VisionResult) -> Path | None:
        img = result.image
        if img is None:
            return None
        day_dir = _artifact_day_dir()
        _ensure_dir(day_dir)
        _ensure_dir(INDEX_PATH.parent)

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        artifact_id = f"vision_{ts}_{uuid.uuid4().hex[:8]}_b{result.batch_index}"
        png_path = day_dir / f"{artifact_id}.png"
        meta_path = day_dir / f"{artifact_id}.json"

        try:
            img.save(png_path)
        except Exception as exc:
            self._set_status_detail(f"Save image failed: {exc}", error=True)
            return None

        cfg = dict(self._active_request_meta.get("config") or self._collect_ui_config())
        sidecar = {
            "artifact_id": artifact_id,
            "modality": "image",
            "kind": "generated",
            "producer": "vision",
            "backend": self._backend_label,
            "created_at": _now_iso(),
            "batch_index": result.batch_index,
            "path": str(png_path),
            "metadata_path": str(meta_path),
            "prompt": str(cfg.get("prompt") or ""),
            "negative_prompt": str(cfg.get("negative_prompt") or ""),
            "scheduler": str(cfg.get("scheduler") or ""),
            "seed_requested": cfg.get("seed"),
            "width": int(cfg.get("width", 0)),
            "height": int(cfg.get("height", 0)),
            "steps": int(cfg.get("steps", 0)),
            "guidance_scale": float(cfg.get("guidance_scale", 0.0)),
            "batch_size": int(cfg.get("batch_size", 1)),
            "lora_path": str(cfg.get("lora_path") or ""),
            "lora_scale": float(cfg.get("lora_scale", 0.0)),
            "model_path": str(cfg.get("model_path") or ""),
            "vram": dict(self._last_resource),
            "routing_hints": {},
        }
        try:
            meta_path.write_text(json.dumps(sidecar, indent=2), encoding="utf-8")
            with INDEX_PATH.open("a", encoding="utf-8") as f:
                f.write(json.dumps(sidecar) + "\n")
        except Exception as exc:
            self._set_status_detail(f"Save metadata failed: {exc}", error=True)
            return png_path
        return png_path

    # ---------------- helpers ----------------
    def _append_trace(self, message: str) -> None:
        self.txt_trace.append(message.rstrip())

    def _log(self, severity: str, message: str) -> None:
        if self._ui_bridge is None:
            return
        self._ui_bridge.sig_monitor_log.emit(severity, message)

    def _refresh_model_picker(self, force_status: bool = False) -> None:
        root_text = (self.inp_model_root.text().strip() or str(DEFAULT_MODEL_ROOT)).strip()
        root = Path(root_text)
        self._model_entries = self._scan_model_root(root)

        previous_path = self.inp_model_path.text().strip()
        self.cmb_models.blockSignals(True)
        self.cmb_models.clear()
        if not root.exists():
            self.cmb_models.addItem("(model root not found)")
            self.cmb_models.setEnabled(False)
            self.btn_use_scanned_model.setEnabled(False)
            if force_status:
                self._set_status_detail(f"Model root not found: {root}", error=True)
            self.cmb_models.blockSignals(False)
            return

        if not self._model_entries:
            self.cmb_models.addItem("(no models found)")
            self.cmb_models.setEnabled(False)
            self.btn_use_scanned_model.setEnabled(False)
            if force_status:
                self._set_status_detail(f"No models found in {root}")
            self.cmb_models.blockSignals(False)
            return

        selected_idx = 0
        for i, entry in enumerate(self._model_entries):
            self.cmb_models.addItem(entry.label, entry.path)
            if previous_path and Path(entry.path) == Path(previous_path):
                selected_idx = i

        self.cmb_models.setCurrentIndex(selected_idx)
        self.cmb_models.setEnabled(True)
        self.btn_use_scanned_model.setEnabled(True)
        self.cmb_models.blockSignals(False)
        if force_status:
            self._set_status_detail(f"Scanned {len(self._model_entries)} model(s)")

    def _scan_model_root(self, root: Path) -> list[_ModelEntry]:
        shared = scan_model_root(root)
        return [_ModelEntry(path=e.path, label=e.label, backend=e.backend) for e in shared]

    def _detect_backend_for_path(self, path: Path) -> str:
        return detect_backend(path)

    def _parse_trace_updates(self, message: str) -> None:
        match = re.search(r"backend=([a-zA-Z0-9_\\-]+)", message)
        if match:
            self._backend_label = match.group(1)
            self.lbl_backend.setText(self._backend_label)

    def _set_progress(self, current: int, total: int) -> None:
        current = max(0, int(current))
        total = max(1, int(total))
        pct = int((current / total) * 100) if total else 0
        self.progress.setValue(max(0, min(100, pct)))
        self.lbl_progress.setText(f"{current} / {total}")

    def _refresh_button_states(self) -> None:
        """Update the morph state of btn_load_toggle + btn_gen_toggle based
        on the current engine status. Driven by _on_guard_status whenever
        the vision engine reports a transition.

        btn_load_toggle:
          UNLOADED / IDLE  → "● LOAD" (accent, enabled)
          LOADING          → "○ LOADING…" (disabled)
          UNLOADING        → "○ UNLOADING…" (disabled)
          READY            → "● UNLOAD" (warn-ish, enabled)
          RUNNING          → disabled (can't unload mid-generation)
          ERROR            → "⚠ RETRY LOAD" (warn, enabled)

        btn_gen_toggle:
          RUNNING          → "⏹ STOP" (warn, enabled if bridge)
          READY            → "● GENERATE" (accent, enabled)
          anything else    → "● GENERATE" (disabled)
        """
        status = self._engine_status
        has_results = bool(self._results)

        # Load toggle text / enabled state
        if status == SystemStatus.LOADING:
            self.btn_load_toggle.setText("○ LOADING…")
            self.btn_load_toggle.setEnabled(False)
        elif status == SystemStatus.UNLOADING:
            self.btn_load_toggle.setText("○ UNLOADING…")
            self.btn_load_toggle.setEnabled(False)
        elif status == SystemStatus.READY:
            self.btn_load_toggle.setText("● UNLOAD")
            self.btn_load_toggle.setEnabled(True)
        elif status == SystemStatus.RUNNING:
            self.btn_load_toggle.setText("● UNLOAD")
            self.btn_load_toggle.setEnabled(False)
        elif status == SystemStatus.ERROR:
            self.btn_load_toggle.setText("⚠ RETRY LOAD")
            self.btn_load_toggle.setEnabled(True)
        else:
            self.btn_load_toggle.setText("● LOAD")
            self.btn_load_toggle.setEnabled(True)

        # Generate / stop toggle. Stop needs the bridge to be wired so
        # the kernel-routed stop command actually reaches the engine.
        if status == SystemStatus.RUNNING:
            self.btn_gen_toggle.setText("⏹ STOP")
            self.btn_gen_toggle.setEnabled(bool(self.bridge))
        elif status == SystemStatus.READY:
            self.btn_gen_toggle.setText("● GENERATE")
            self.btn_gen_toggle.setEnabled(True)
        else:
            self.btn_gen_toggle.setText("● GENERATE")
            self.btn_gen_toggle.setEnabled(False)

        self.btn_save_all.setEnabled(has_results)
        self.btn_save_selected.setEnabled(has_results and self._selected_result_idx is not None)

    def _on_load_toggle_clicked(self) -> None:
        """Route the single LOAD button to load or unload based on status.
        READY → unload; anything else (UNLOADED/ERROR/etc.) → load."""
        if self._engine_status == SystemStatus.READY:
            self._unload_model_clicked()
        else:
            self._load_model_clicked()

    def _on_gen_toggle_clicked(self) -> None:
        """Route the single GENERATE button to generate or stop based on
        status. RUNNING → stop via bridge.stop("vision") (kernel path);
        anything else → submit a new generate."""
        if self._engine_status == SystemStatus.RUNNING:
            self._stop_clicked()
        else:
            self._generate_clicked()

    def _set_status_detail(self, text: str, error: bool = False) -> None:
        self._status_is_error = error
        self.lbl_status_detail.setText(text[:180])
        if error:
            self.lbl_status_detail.setStyleSheet(f"color: {_s.FG_ERROR}; font-size: 10px;")
        else:
            self.lbl_status_detail.setStyleSheet(f"color: {_s.FG_DIM}; font-size: 10px;")
        self._status_flash_timer.start()

    def _clear_status_detail(self) -> None:
        # Keep errors visible until another state update if generation is not running.
        if self._engine_status == SystemStatus.RUNNING:
            return
        self._status_is_error = False
        self.lbl_status_detail.setText("")
        self.lbl_status_detail.setStyleSheet(f"color: {_s.FG_DIM}; font-size: 10px;")

    # ---------------- theme ----------------
    def _refresh_widget_styles(self) -> None:
        from core.style import refresh_styles
        refresh_styles()

        input_ss = (
            f"background: {_s.BG_INPUT}; color: {_s.FG_TEXT}; "
            f"border: 1px solid {_s.BORDER_DARK}; border-radius: 3px; "
            f"padding: 4px 6px; font-size: 12px;"
        )
        for w in (self.inp_model_root, self.inp_model_path, self.inp_seed, self.inp_lora_path):
            w.setStyleSheet(input_ss)
        for w in (self.txt_prompt, self.txt_negative, self.txt_trace):
            w.setStyleSheet(input_ss)

        cmb_ss = (
            f"QComboBox {{ background: {_s.BG_INPUT}; color: {_s.FG_TEXT}; "
            f"border: 1px solid {_s.BORDER_DARK}; border-radius: 3px; padding: 4px 6px; }}"
            f"QComboBox::drop-down {{ border: none; }}"
            f"QComboBox QAbstractItemView {{ background: {_s.BG_INPUT}; color: {_s.FG_TEXT}; "
            f"border: 1px solid {_s.BORDER_DARK}; }}"
        )
        for w in (self.cmb_models, self.cmb_scheduler):
            w.setStyleSheet(cmb_ss)

        for w in (self.spn_width, self.spn_height, self.spn_steps,
                  self.spn_cfg, self.spn_batch, self.spn_lora_scale):
            w.refresh_style()

        prog_ss = (
            f"QProgressBar {{ background: {_s.BG_INPUT}; border: 1px solid {_s.BORDER_DARK}; "
            f"border-radius: 3px; text-align: center; color: {_s.FG_DIM}; }}"
            f"QProgressBar::chunk {{ background: {_s.ACCENT_PRIMARY}; border-radius: 2px; }}"
        )
        self.progress.setStyleSheet(prog_ss)

        lbl_ss = f"color: {_s.FG_DIM}; font-size: 10px;"
        self.lbl_progress.setStyleSheet(lbl_ss)
        if self._status_is_error:
            self.lbl_status_detail.setStyleSheet(f"color: {_s.FG_ERROR}; font-size: 10px;")
        else:
            self.lbl_status_detail.setStyleSheet(lbl_ss)

    def _on_theme_changed(self, _theme_name: str = "") -> None:
        self._refresh_widget_styles()
