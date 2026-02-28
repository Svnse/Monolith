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
    QTextEdit,
    QVBoxLayout,
    QWidget,
    QComboBox,
)

import core.style as _s
from core.paths import CONFIG_DIR, MONOLITH_ROOT
from core.state import SystemStatus
from ui.components.atoms import MonoGroupBox, MonoButton, MonoDragSpin

try:
    from PIL.ImageQt import ImageQt
except Exception:  # pragma: no cover - runtime guard
    ImageQt = None


CONFIG_PATH = CONFIG_DIR / "vision_config.json"
VISION_ARTIFACT_ROOT = MONOLITH_ROOT / "artifacts" / "vision"
INDEX_PATH = VISION_ARTIFACT_ROOT / "index.jsonl"
DEFAULT_MODEL_ROOT = MONOLITH_ROOT / "models" / "vision"


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

    def __init__(self, bridge=None, guard=None, ui_bridge=None):
        super().__init__()
        self.bridge = bridge
        self.guard = guard
        self._ui_bridge = ui_bridge

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
        root = QVBoxLayout(self)
        root.setContentsMargins(20, 20, 20, 20)
        root.setSpacing(10)

        top = QHBoxLayout()
        top.setSpacing(10)
        root.addLayout(top, stretch=1)

        left_col = QVBoxLayout()
        left_col.setSpacing(10)
        top.addLayout(left_col, stretch=2)

        right_col = QVBoxLayout()
        right_col.setSpacing(10)
        top.addLayout(right_col, stretch=3)

        # ---- Model / runtime controls ----
        grp_model = MonoGroupBox("VISION MODEL")

        model_root_row = QHBoxLayout()
        self.inp_model_root = QLineEdit()
        self.inp_model_root.setPlaceholderText("Vision model root")
        self.btn_browse_model_root = MonoButton("Set Root")
        self.btn_scan_models = MonoButton("Scan")
        model_root_row.addWidget(self.inp_model_root, stretch=1)
        model_root_row.addWidget(self.btn_browse_model_root)
        model_root_row.addWidget(self.btn_scan_models)
        grp_model.add_layout(model_root_row)

        picker_row = QHBoxLayout()
        self.cmb_models = QComboBox()
        self.cmb_models.setPlaceholderText("Scanned models")
        self.btn_use_scanned_model = MonoButton("Use Selected")
        picker_row.addWidget(self.cmb_models, stretch=1)
        picker_row.addWidget(self.btn_use_scanned_model)
        grp_model.add_layout(picker_row)

        model_path_row = QHBoxLayout()
        self.inp_model_path = QLineEdit()
        self.inp_model_path.setPlaceholderText("Select a model file (.safetensors/.ckpt) or diffusers folder")
        self.inp_model_path.setReadOnly(True)
        self.btn_browse_model = MonoButton("Browse File")
        self.btn_browse_model_dir = MonoButton("Browse Folder")
        model_path_row.addWidget(self.inp_model_path, stretch=1)
        model_path_row.addWidget(self.btn_browse_model)
        model_path_row.addWidget(self.btn_browse_model_dir)
        grp_model.add_layout(model_path_row)

        runtime_btns = QHBoxLayout()
        self.btn_load = MonoButton("Load")
        self.btn_unload = MonoButton("Unload")
        self.btn_generate = MonoButton("Generate", accent=True)
        self.btn_stop = MonoButton("Stop")
        runtime_btns.addWidget(self.btn_load)
        runtime_btns.addWidget(self.btn_unload)
        runtime_btns.addWidget(self.btn_generate)
        runtime_btns.addWidget(self.btn_stop)
        grp_model.add_layout(runtime_btns)

        status_grid = QGridLayout()
        status_grid.setSpacing(6)
        lbl_status_key = QLabel("Status")
        self.lbl_status = QLabel("IDLE")
        lbl_backend_key = QLabel("Backend")
        self.lbl_backend = QLabel("unknown")
        self.lbl_vram = QLabel("VRAM: --")
        self.lbl_status_detail = QLabel("")
        status_grid.addWidget(lbl_status_key, 0, 0)
        status_grid.addWidget(self.lbl_status, 0, 1)
        status_grid.addWidget(lbl_backend_key, 1, 0)
        status_grid.addWidget(self.lbl_backend, 1, 1)
        status_grid.addWidget(self.lbl_vram, 2, 0, 1, 2)
        status_grid.addWidget(self.lbl_status_detail, 3, 0, 1, 2)
        grp_model.add_layout(status_grid)

        left_col.addWidget(grp_model)

        # ---- Generation params ----
        grp_gen = MonoGroupBox("GENERATION")

        lbl_prompt = QLabel("Prompt")
        self.txt_prompt = QTextEdit()
        self.txt_prompt.setPlaceholderText("Describe the image to generate...")
        self.txt_prompt.setFixedHeight(80)
        grp_gen.add_widget(lbl_prompt)
        grp_gen.add_widget(self.txt_prompt)

        lbl_neg = QLabel("Negative Prompt")
        self.txt_negative = QTextEdit()
        self.txt_negative.setPlaceholderText("Negative prompt (optional)")
        self.txt_negative.setFixedHeight(55)
        grp_gen.add_widget(lbl_neg)
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

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.lbl_progress = QLabel("0 / 0")
        prog_row = QHBoxLayout()
        prog_row.addWidget(self.progress, stretch=1)
        prog_row.addWidget(self.lbl_progress)
        grp_gen.add_layout(prog_row)

        left_col.addWidget(grp_gen, stretch=1)

        # ---- Preview + gallery ----
        grp_preview = MonoGroupBox("PREVIEW")

        self.lbl_preview = QLabel("No image yet")
        self.lbl_preview.setAlignment(Qt.AlignCenter)
        self.lbl_preview.setMinimumSize(420, 320)
        self.lbl_preview.setFrameShape(QFrame.StyledPanel)
        grp_preview.add_widget(self.lbl_preview)

        self.lst_results = QListWidget()
        self.lst_results.setViewMode(QListWidget.IconMode)
        self.lst_results.setMovement(QListWidget.Static)
        self.lst_results.setResizeMode(QListWidget.Adjust)
        self.lst_results.setIconSize(QSize(120, 120))
        self.lst_results.setSpacing(8)
        self.lst_results.setMinimumHeight(170)
        lbl_batch = QLabel("Current Batch")
        grp_preview.add_widget(lbl_batch)
        grp_preview.add_widget(self.lst_results)

        save_row = QHBoxLayout()
        self.btn_save_selected = MonoButton("Save Selected")
        self.btn_save_all = MonoButton("Save All")
        save_row.addWidget(self.btn_save_selected)
        save_row.addWidget(self.btn_save_all)
        save_row.addStretch()
        grp_preview.add_layout(save_row)

        right_col.addWidget(grp_preview, stretch=1)

        # ---- Trace pane ----
        grp_trace = MonoGroupBox("VISION TRACE")
        self.txt_trace = QTextEdit()
        self.txt_trace.setReadOnly(True)
        self.txt_trace.setMaximumHeight(150)
        grp_trace.add_widget(self.txt_trace)
        right_col.addWidget(grp_trace)

    def _wire_signals(self) -> None:
        self.btn_browse_model.clicked.connect(self._browse_model_file)
        self.btn_browse_model_dir.clicked.connect(self._browse_model_dir)
        self.btn_browse_model_root.clicked.connect(self._browse_model_root)
        self.btn_scan_models.clicked.connect(self._scan_models_clicked)
        self.btn_use_scanned_model.clicked.connect(self._use_scanned_model)
        self.btn_browse_lora.clicked.connect(self._browse_lora)
        self.btn_clear_lora.clicked.connect(self._clear_lora)
        self.btn_load.clicked.connect(self._load_model_clicked)
        self.btn_unload.clicked.connect(self._unload_model_clicked)
        self.btn_generate.clicked.connect(self._generate_clicked)
        self.btn_stop.clicked.connect(self._stop_clicked)
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
        try:
            if CONFIG_PATH.exists():
                return {**self._CFG_DEFAULTS, **json.loads(CONFIG_PATH.read_text(encoding="utf-8"))}
        except Exception:
            pass
        return dict(self._CFG_DEFAULTS)

    def _queue_save_config(self, *_args) -> None:
        self._cfg_timer.start()

    def _save_config(self) -> None:
        cfg = self._collect_ui_config()
        try:
            _ensure_dir(CONFIG_PATH.parent)
            CONFIG_PATH.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
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
            return
        self._queue_save_config()
        try:
            self._submit("set_path", {"path": model_path})
            self._submit("load")
            self._set_status_detail("Load requested")
        except Exception as exc:
            self._set_status_detail(f"Load failed: {exc}", error=True)

    def _unload_model_clicked(self) -> None:
        try:
            self._submit("unload")
            self._set_status_detail("Unload requested")
        except Exception as exc:
            self._set_status_detail(f"Unload failed: {exc}", error=True)

    def _generate_clicked(self) -> None:
        prompt = self.txt_prompt.toPlainText().strip()
        model_path = self.inp_model_path.text().strip()
        if not model_path:
            self._set_status_detail("No model selected", error=True)
            return
        if not prompt:
            self._set_status_detail("Prompt is empty", error=True)
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
        except Exception as exc:
            self._set_status_detail(f"Generate failed: {exc}", error=True)

        self._refresh_button_states()

    def _stop_clicked(self) -> None:
        try:
            if self.bridge is not None and hasattr(self.bridge, "stop"):
                self.bridge.stop("vision")
                self._set_status_detail("Stop requested")
            else:
                self._set_status_detail("Stop unavailable", error=True)
        except Exception as exc:
            self._set_status_detail(f"Stop failed: {exc}", error=True)

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

    def _on_vision_image(self, image: object, batch_index: int) -> None:
        self._add_result(image, int(batch_index))

    def _on_vision_progress(self, current: int, total: int) -> None:
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
    def _add_result(self, image: object, batch_index: int) -> None:
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
            return
        if not (0 <= self._selected_result_idx < len(self._results)):
            self._set_status_detail("Invalid result selection", error=True)
            return
        saved = self._save_result_artifact(self._results[self._selected_result_idx])
        if saved is not None:
            self._set_status_detail(f"Saved: {saved.name}")

    def _save_all(self) -> None:
        if not self._results:
            self._set_status_detail("No results to save", error=True)
            return
        count = 0
        for res in self._results:
            if self._save_result_artifact(res) is not None:
                count += 1
        self._set_status_detail(f"Saved {count} artifact(s)")

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
        if not root.exists() or not root.is_dir():
            return []

        entries: list[_ModelEntry] = []
        seen: set[str] = set()

        # Diffusers directories
        try:
            for idx_file in root.rglob("model_index.json"):
                model_dir = idx_file.parent
                key = str(model_dir.resolve())
                if key in seen:
                    continue
                seen.add(key)
                backend = self._detect_backend_for_path(model_dir)
                rel = model_dir.relative_to(root)
                label = f"[{backend}] {rel.as_posix()}/"
                entries.append(_ModelEntry(path=str(model_dir), label=label, backend=backend))
        except Exception:
            pass

        # Single-file checkpoints
        patterns = ("*.safetensors", "*.ckpt")
        for pattern in patterns:
            try:
                for file_path in root.rglob(pattern):
                    key = str(file_path.resolve())
                    if key in seen:
                        continue
                    seen.add(key)
                    backend = self._detect_backend_for_path(file_path)
                    rel = file_path.relative_to(root)
                    label = f"[{backend}] {rel.as_posix()}"
                    entries.append(_ModelEntry(path=str(file_path), label=label, backend=backend))
            except Exception:
                pass

        entries.sort(key=lambda e: (e.backend, e.label.lower()))
        return entries

    def _detect_backend_for_path(self, path: Path) -> str:
        # Mirrors worker heuristics without importing diffusers.
        try:
            if path.is_dir():
                idx = path / "model_index.json"
                if idx.exists():
                    data = json.loads(idx.read_text(encoding="utf-8"))
                    cls = str(data.get("_class_name", "")).lower()
                    if "flux" in cls:
                        return "flux"
                    if "xl" in cls:
                        return "sdxl"
                name = path.name.lower()
                if "flux" in name:
                    return "flux"
                if "xl" in name:
                    return "sdxl"
                return "sd15"
            name = path.name.lower()
            if "flux" in name:
                return "flux"
            if "xl" in name:
                return "sdxl"
        except Exception:
            pass
        return "sd15"

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
        status = self._engine_status
        busy = status in (SystemStatus.LOADING, SystemStatus.RUNNING, SystemStatus.UNLOADING)
        has_results = bool(self._results)
        self.btn_generate.setEnabled(not busy)
        self.btn_load.setEnabled(not busy)
        self.btn_unload.setEnabled(not busy)
        self.btn_stop.setEnabled(bool(self.bridge) and bool(busy))
        self.btn_save_all.setEnabled(has_results)
        self.btn_save_selected.setEnabled(has_results and self._selected_result_idx is not None)

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
