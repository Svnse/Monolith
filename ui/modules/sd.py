import json

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QScrollArea, QFileDialog, QSpinBox, QDoubleSpinBox,
    QAbstractSpinBox
)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QPixmap, QImage

from core.state import SystemStatus
from core.paths import CONFIG_DIR, MONOLITH_ROOT
from monokernel.bridge import MonoBridge
from monokernel.guard import MonoGuard
from ui.components.atoms import MonoGroupBox, MonoButton, MonoTriangleButton, CollapsibleSection

class SDModule(QWidget):
    def __init__(self, bridge: MonoBridge, guard: MonoGuard):
        super().__init__()
        import core.style as s
        BG_INPUT, BORDER_DARK, FG_DIM, FG_TEXT, FG_ACCENT, FG_ERROR = (
            s.BG_INPUT, s.BORDER_DARK, s.FG_DIM, s.FG_TEXT, s.FG_ACCENT, s.FG_ERROR
        )
        self.bridge = bridge
        self.guard = guard

        self.config_path = CONFIG_DIR / "vision_config.json"
        self.legacy_config_path = CONFIG_DIR / "sd_config.json"
        self.artifacts_dir = MONOLITH_ROOT / "artifacts" / "vision"
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.config = self._load_config()
        self.model_path = self.config.get("model_path", "")
        self.is_model_loaded = False
        self.current_image = None
        self._engine_status = SystemStatus.READY
        self._config_timer = QTimer(self)
        self._config_timer.setInterval(1000)
        self._config_timer.setSingleShot(True)
        self._config_timer.timeout.connect(self._save_config)
        self._status_reset_timer = QTimer(self)
        self._status_reset_timer.setInterval(1000)
        self._status_reset_timer.setSingleShot(True)
        self._status_reset_timer.timeout.connect(self._reset_status)
        if self.model_path:
            self.bridge.submit(
                self.bridge.wrap(
                    "vision",
                    "set_path",
                    "vision",
                    payload={"path": self.model_path},
                )
            )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)

        grp = MonoGroupBox("VISION")
        inner = QVBoxLayout()
        inner.setSpacing(12)

        # Config Section
        config_section = CollapsibleSection("⚙ CONFIGURATION")
        config_layout = QVBoxLayout()
        config_layout.setSpacing(8)
        
        # Model Loader
        grp_loader = MonoGroupBox("MODEL LOADER")
        loader_layout = QVBoxLayout()
        loader_row = QHBoxLayout()
        lbl_model = QLabel("Model Path")
        lbl_model.setStyleSheet(f"color: {FG_DIM}; font-size: 10px;")
        self.inp_model = QLineEdit(self.model_path)
        self.inp_model.setReadOnly(True)
        self.inp_model.setPlaceholderText("Select a model file (.ckpt or .safetensors)...")
        self.inp_model.setToolTip(self.model_path)
        self.inp_model.setStyleSheet(f"""
            QLineEdit {{
                background: {BG_INPUT}; color: {FG_TEXT};
                border: 1px solid {BORDER_DARK}; padding: 4px;
            }}
        """)
        btn_browse = MonoButton("BROWSE...")
        btn_browse.setFixedWidth(90)
        btn_browse.clicked.connect(self._browse_model)
        loader_row.addWidget(lbl_model, 0)
        loader_row.addWidget(self.inp_model, 1)
        loader_row.addWidget(btn_browse, 0)
        self.btn_load = MonoButton("LOAD MODEL")
        self.btn_load.setCheckable(True)
        self.btn_load.setChecked(False)
        self.btn_load.clicked.connect(self._load_model)
        loader_layout.addLayout(loader_row)
        loader_layout.addWidget(self.btn_load)
        grp_loader.add_layout(loader_layout)
        config_layout.addWidget(grp_loader)
        
        # Steps
        steps_row = QHBoxLayout()
        lbl_steps = QLabel("Steps")
        lbl_steps.setStyleSheet(f"color: {FG_DIM}; font-size: 10px;")
        lbl_steps.setFixedWidth(80)
        self.inp_steps = QSpinBox()
        self.inp_steps.setRange(1, 150)
        self.inp_steps.setValue(self.config.get("steps", 25))
        self.inp_steps.setButtonSymbols(QAbstractSpinBox.NoButtons)
        self.inp_steps.setStyleSheet(f"""
            QSpinBox {{
                background: {BG_INPUT}; color: {FG_TEXT};
                border: 1px solid {BORDER_DARK}; padding: 4px;
            }}
        """)
        steps_row.addWidget(lbl_steps)
        btn_steps_down = MonoTriangleButton("◀")
        btn_steps_down.clicked.connect(self.inp_steps.stepDown)
        btn_steps_up = MonoTriangleButton("▶")
        btn_steps_up.clicked.connect(self.inp_steps.stepUp)
        steps_row.addWidget(btn_steps_down)
        steps_row.addWidget(self.inp_steps)
        steps_row.addWidget(btn_steps_up)
        steps_row.addStretch()
        config_layout.addLayout(steps_row)
        
        # Strength
        strength_row = QHBoxLayout()
        lbl_strength = QLabel("Strength")
        lbl_strength.setStyleSheet(f"color: {FG_DIM}; font-size: 10px;")
        lbl_strength.setFixedWidth(80)
        self.inp_strength = QDoubleSpinBox()
        self.inp_strength.setRange(1.0, 20.0)
        self.inp_strength.setValue(self.config.get("guidance_scale", 7.5))
        self.inp_strength.setSingleStep(0.5)
        self.inp_strength.setButtonSymbols(QAbstractSpinBox.NoButtons)
        self.inp_strength.setStyleSheet(f"""
            QDoubleSpinBox {{
                background: {BG_INPUT}; color: {FG_TEXT};
                border: 1px solid {BORDER_DARK}; padding: 4px;
            }}
        """)
        btn_strength_down = MonoTriangleButton("◀")
        btn_strength_down.clicked.connect(self.inp_strength.stepDown)
        btn_strength_up = MonoTriangleButton("▶")
        btn_strength_up.clicked.connect(self.inp_strength.stepUp)
        strength_row.addWidget(lbl_strength)
        strength_row.addWidget(btn_strength_down)
        strength_row.addWidget(self.inp_strength)
        strength_row.addWidget(btn_strength_up)
        strength_row.addStretch()
        config_layout.addLayout(strength_row)
        
        # Seed
        seed_row = QHBoxLayout()
        lbl_seed = QLabel("Seed")
        lbl_seed.setStyleSheet(f"color: {FG_DIM}; font-size: 10px;")
        lbl_seed.setFixedWidth(80)
        self.inp_seed = QSpinBox()
        self.inp_seed.setRange(-1, 2147483647)
        self.inp_seed.setSpecialValueText("RANDOM")
        self.inp_seed.setValue(self.config.get("seed", -1))
        self.inp_seed.setButtonSymbols(QAbstractSpinBox.NoButtons)
        self.inp_seed.setStyleSheet(f"""
            QSpinBox {{
                background: {BG_INPUT}; color: {FG_TEXT};
                border: 1px solid {BORDER_DARK}; padding: 4px;
            }}
        """)
        btn_seed_down = MonoTriangleButton("◀")
        btn_seed_down.clicked.connect(self.inp_seed.stepDown)
        btn_seed_up = MonoTriangleButton("▶")
        btn_seed_up.clicked.connect(self.inp_seed.stepUp)
        seed_row.addWidget(lbl_seed)
        seed_row.addWidget(btn_seed_down)
        seed_row.addWidget(self.inp_seed)
        seed_row.addWidget(btn_seed_up)
        seed_row.addStretch()
        config_layout.addLayout(seed_row)
        
        config_section.set_content_layout(config_layout)
        inner.addWidget(config_section)

        # Prompt
        lbl_prompt = QLabel("Prompt")
        lbl_prompt.setStyleSheet(f"color: {FG_DIM}; font-size: 10px;")

        self.inp_prompt = QLineEdit()
        self.inp_prompt.setPlaceholderText("Describe an image to generate...")
        self.inp_prompt.setStyleSheet(f"""
            QLineEdit {{
                background: {BG_INPUT}; color: {FG_TEXT};
                border: 1px solid {BORDER_DARK}; padding: 6px;
            }}
        """)

        # Buttons
        btn_row = QHBoxLayout()
        self.btn_generate = MonoButton("GENERATE", accent=True)
        self.btn_generate.clicked.connect(self._start_generate)
        self.btn_generate.setEnabled(self.is_model_loaded)
        self.btn_stop = MonoButton("STOP")
        self.btn_stop.clicked.connect(lambda: self.bridge.stop("vision"))
        self.btn_stop.setEnabled(False)
        self.btn_save = MonoButton("SAVE IMAGE")
        self.btn_save.clicked.connect(self._save_image)
        self.btn_save.setEnabled(False)
        btn_row.addWidget(self.btn_generate)
        btn_row.addWidget(self.btn_stop)
        btn_row.addWidget(self.btn_save)
        btn_row.addStretch()

        # Preview
        preview_scroll = QScrollArea()
        preview_scroll.setWidgetResizable(True)
        preview_scroll.setFixedHeight(400)
        preview_scroll.setStyleSheet(f"background: {BG_INPUT}; border: 1px solid {BORDER_DARK};")
        
        self.lbl_preview = QLabel("NO IMAGE")
        self.lbl_preview.setAlignment(Qt.AlignCenter)
        self.lbl_preview.setStyleSheet(f"color: {FG_DIM}; font-size: 12px;")
        preview_scroll.setWidget(self.lbl_preview)

        # Status
        status_row = QHBoxLayout()
        lbl_status_title = QLabel("Status")
        lbl_status_title.setStyleSheet(f"color: {FG_DIM}; font-size: 10px;")
        self.lbl_status = QLabel("IDLE")
        self.lbl_status.setStyleSheet(f"color: {FG_TEXT}; font-size: 10px; font-weight: bold;")
        status_row.addWidget(lbl_status_title)
        status_row.addStretch()
        status_row.addWidget(self.lbl_status)

        inner.addWidget(lbl_prompt)
        inner.addWidget(self.inp_prompt)
        inner.addLayout(btn_row)
        inner.addWidget(preview_scroll)
        inner.addLayout(status_row)
        inner.addStretch()

        grp.add_layout(inner)
        layout.addWidget(grp)

        self.inp_steps.valueChanged.connect(self._queue_save_config)
        self.inp_strength.valueChanged.connect(self._queue_save_config)
        self.inp_seed.valueChanged.connect(self._queue_save_config)
        self.guard.sig_image.connect(self._on_image)
        self.guard.sig_status.connect(self._on_status)
        self.guard.sig_trace.connect(lambda _engine_key, message: self._on_trace(message))

    def _load_config(self):
        if self.config_path.exists():
            try:
                with open(self.config_path, 'r') as f:
                    config = json.load(f)
                return self._normalize_config(config)
            except Exception:
                pass
        if self.legacy_config_path.exists():
            try:
                with open(self.legacy_config_path, 'r') as f:
                    config = json.load(f)
                config = self._normalize_config(config)
                with open(self.config_path, 'w') as f:
                    json.dump(config, f, indent=2)
                return config
            except Exception:
                pass
        return {
            "model_path": "",
            "steps": 25,
            "guidance_scale": 7.5,
            "seed": -1
        }

    def _normalize_config(self, config):
        use_seed = config.get("use_seed")
        if use_seed is False:
            config["seed"] = -1
        config.pop("use_seed", None)
        if "seed" not in config:
            config["seed"] = -1
        return config

    def _save_config(self):
        config = {
            "model_path": self.model_path,
            "steps": self.inp_steps.value(),
            "guidance_scale": self.inp_strength.value(),
            "seed": self.inp_seed.value()
        }
        with open(self.config_path, 'w') as f:
            json.dump(config, f, indent=2)
        self.config = config
        if self._engine_status not in (
            SystemStatus.LOADING,
            SystemStatus.RUNNING,
            SystemStatus.UNLOADING,
        ):
            import core.style as s
            self._set_status("CONFIG SAVED", s.FG_ACCENT)
            self._status_reset_timer.start()

    def _browse_model(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Vision Model",
            "",
            "Model Files (*.ckpt *.safetensors);;All Files (*)"
        )
        if path:
            self.inp_model.setText(path)
            self.inp_model.setToolTip(path)
            self.btn_load.setChecked(False)
            self.btn_load.setText("LOAD MODEL")
            self.is_model_loaded = False

    def _load_model(self):
        import core.style as s
        if self.btn_load.isChecked():
            path = self.inp_model.text().strip()
            if not path:
                self._set_status("ERROR: No model selected", s.FG_ERROR)
                self.btn_load.setChecked(False)
                return
            self.model_path = path
            self._queue_save_config()
            self.bridge.submit(
                self.bridge.wrap(
                    "vision",
                    "set_path",
                    "vision",
                    payload={"path": path},
                )
            )
            self.bridge.submit(self.bridge.wrap("vision", "load", "vision"))
        else:
            self.bridge.submit(self.bridge.wrap("vision", "unload", "vision"))

    def _queue_save_config(self):
        self._status_reset_timer.stop()
        self._config_timer.start()

    def _set_status(self, status, color=None):
        import core.style as s
        self.lbl_status.setText(status)
        self.lbl_status.setStyleSheet(f"color: {color or s.FG_TEXT}; font-size: 10px; font-weight: bold;")

    def _reset_status(self):
        self._set_status("IDLE")

    def _start_generate(self):
        import core.style as s
        prompt = self.inp_prompt.text().strip()
        if not prompt:
            self._set_status("ERROR: No prompt", s.FG_ERROR)
            return

        self.btn_generate.setEnabled(False)
        self.btn_save.setEnabled(False)
        self._set_status("REQUESTED", s.FG_ACCENT)

        seed_value = self.inp_seed.value()
        seed = None if seed_value < 0 else seed_value
        config = {
            "steps": self.inp_steps.value(),
            "guidance_scale": self.inp_strength.value(),
            "seed": seed,
        }
        self.bridge.submit(
            self.bridge.wrap(
                "vision",
                "generate",
                "vision",
                payload={"prompt": prompt, "config": config},
            )
        )

    def _save_image(self):
        if not self.current_image:
            return

        import time
        import core.style as s
        filename = f"vision_{int(time.time())}.png"
        filepath = self.artifacts_dir / filename

        try:
            if isinstance(self.current_image, QImage):
                self.current_image.save(str(filepath))
            else:
                self.current_image.save(filepath)
            self._set_status(f"SAVED: {filename}", s.FG_ACCENT)
        except Exception as e:
            self._set_status(f"SAVE ERROR: {str(e)}", s.FG_ERROR)

    def _on_image(self, image):
        self.current_image = image
        if isinstance(image, QImage):
            qimage = image
        else:
            pil_image = image.convert("RGB")
            data = pil_image.tobytes("raw", "RGB")
            qimage = QImage(
                data,
                pil_image.width,
                pil_image.height,
                QImage.Format_RGB888,
            ).copy()

        pixmap = QPixmap.fromImage(qimage)
        self.lbl_preview.setPixmap(pixmap.scaled(
            self.lbl_preview.width() - 20,
            self.lbl_preview.height() - 20,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        ))
        self._set_status("DONE")
        self.btn_save.setEnabled(True)

    def _on_status(self, engine_key: str, status: SystemStatus) -> None:
        if engine_key != "vision":
            return
        import core.style as s
        self._engine_status = status
        is_busy = status in (
            SystemStatus.LOADING,
            SystemStatus.RUNNING,
            SystemStatus.UNLOADING,
        )
        self.btn_generate.setEnabled(not is_busy and self.is_model_loaded)
        self.btn_load.setEnabled(not is_busy)
        self.btn_stop.setEnabled(is_busy)
        if status == SystemStatus.LOADING:
            self._set_status("LOADING", s.FG_ACCENT)
            self.btn_load.setText("LOADING...")
        elif status == SystemStatus.RUNNING:
            self._set_status("RUNNING", s.FG_ACCENT)
        elif status == SystemStatus.UNLOADING:
            self._set_status("UNLOADING", s.FG_ACCENT)
        elif status == SystemStatus.READY:
            self._set_status("READY")
            if self.btn_load.isChecked() and self.model_path:
                self.is_model_loaded = True
                self.btn_load.setText("UNLOAD MODEL")
            else:
                self.is_model_loaded = False
                self.btn_load.setChecked(False)
                self.btn_load.setText("LOAD MODEL")
            self.btn_generate.setEnabled(self.is_model_loaded and not is_busy)
        elif status == SystemStatus.ERROR:
            self._set_status("ERROR", s.FG_ERROR)
            self.is_model_loaded = False
            self.btn_load.setChecked(False)
            self.btn_load.setText("LOAD MODEL")
            self.btn_generate.setEnabled(False)

    def _on_trace(self, message: str) -> None:
        if "VISION: ERROR:" in message:
            import core.style as s
            self._set_status(message.replace("VISION: ", ""), s.FG_ERROR)
