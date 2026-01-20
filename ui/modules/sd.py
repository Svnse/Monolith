import os
import json
from pathlib import Path
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QFrame, QScrollArea, QFileDialog, QSpinBox, QDoubleSpinBox,
    QAbstractSpinBox
)
from PySide6.QtCore import Qt, QThread, Signal, QTimer
from PySide6.QtGui import QPixmap, QImage

from core.style import BG_INPUT, BORDER_DARK, FG_DIM, FG_TEXT, FG_ACCENT, FG_ERROR
from ui.components.atoms import SkeetGroupBox, SkeetButton, CollapsibleSection

DIFFUSERS_AVAILABLE = False
try:
    import diffusers
    DIFFUSERS_AVAILABLE = True
except ImportError:
    pass


class SDWorker(QThread):
    progress = Signal(str)
    finished = Signal(object)
    error = Signal(str)

    def __init__(self, prompt, model_path, steps, guidance, seed):
        super().__init__()
        self.prompt = prompt
        self.model_path = model_path
        self.steps = steps
        self.guidance = guidance
        self.seed = seed

    def run(self):
        try:
            import torch
            from diffusers import StableDiffusionPipeline
            
            self.progress.emit("Loading pipeline...")
            
            device = "cuda" if torch.cuda.is_available() else "cpu"
            dtype = torch.float16 if device == "cuda" else torch.float32
            
            pipe = StableDiffusionPipeline.from_pretrained(
                self.model_path,
                torch_dtype=dtype,
                safety_checker=None,
                requires_safety_checker=False
            )
            pipe = pipe.to(device)
            
            self.progress.emit("Generating image...")
            
            generator = None
            if self.seed is not None:
                generator = torch.Generator(device=device).manual_seed(self.seed)
            
            result = pipe(
                self.prompt,
                num_inference_steps=self.steps,
                guidance_scale=self.guidance,
                generator=generator
            )
            
            self.finished.emit(result.images[0])
            
        except Exception as e:
            self.error.emit(str(e))


class SDModule(QWidget):
    def __init__(self):
        super().__init__()
        
        self.config_path = Path("config/vision_config.json")
        self.legacy_config_path = Path("config/sd_config.json")
        self.artifacts_dir = Path("artifacts/vision")
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.config = self._load_config()
        self.model_path = self.config.get("model_path", "")
        self.current_image = None
        self.worker = None
        self._config_timer = QTimer(self)
        self._config_timer.setInterval(1000)
        self._config_timer.setSingleShot(True)
        self._config_timer.timeout.connect(self._save_config)
        self._status_reset_timer = QTimer(self)
        self._status_reset_timer.setInterval(1000)
        self._status_reset_timer.setSingleShot(True)
        self._status_reset_timer.timeout.connect(self._reset_status)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)

        grp = SkeetGroupBox("VISION")
        inner = QVBoxLayout()
        inner.setSpacing(12)

        # Config Section
        config_section = CollapsibleSection("⚙ CONFIGURATION")
        config_layout = QVBoxLayout()
        config_layout.setSpacing(8)
        
        # Model Loader
        grp_loader = SkeetGroupBox("MODEL LOADER")
        loader_layout = QVBoxLayout()
        loader_row = QHBoxLayout()
        lbl_model = QLabel("Model Path")
        lbl_model.setStyleSheet(f"color: {FG_DIM}; font-size: 10px;")
        self.inp_model = QLineEdit(self.model_path)
        self.inp_model.setReadOnly(True)
        self.inp_model.setPlaceholderText("Select a model file (.gguf or .ckpt)...")
        self.inp_model.setToolTip(self.model_path)
        self.inp_model.setStyleSheet(f"""
            QLineEdit {{
                background: {BG_INPUT}; color: {FG_TEXT};
                border: 1px solid {BORDER_DARK}; padding: 4px;
            }}
        """)
        btn_browse = SkeetButton("BROWSE...")
        btn_browse.setFixedWidth(90)
        btn_browse.clicked.connect(self._browse_model)
        loader_row.addWidget(lbl_model, 0)
        loader_row.addWidget(self.inp_model, 1)
        loader_row.addWidget(btn_browse, 0)
        self.btn_load = SkeetButton("LOAD MODEL")
        self.btn_load.setCheckable(True)
        self.btn_load.setChecked(bool(self.model_path))
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
        btn_steps_up = SkeetButton("+")
        btn_steps_up.setFixedWidth(26)
        btn_steps_up.clicked.connect(self.inp_steps.stepUp)
        btn_steps_down = SkeetButton("-")
        btn_steps_down.setFixedWidth(26)
        btn_steps_down.clicked.connect(self.inp_steps.stepDown)
        steps_row.addWidget(btn_steps_up)
        steps_row.addWidget(self.inp_steps)
        steps_row.addWidget(btn_steps_down)
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
        btn_strength_up = SkeetButton("+")
        btn_strength_up.setFixedWidth(26)
        btn_strength_up.clicked.connect(self.inp_strength.stepUp)
        btn_strength_down = SkeetButton("-")
        btn_strength_down.setFixedWidth(26)
        btn_strength_down.clicked.connect(self.inp_strength.stepDown)
        strength_row.addWidget(lbl_strength)
        strength_row.addWidget(btn_strength_up)
        strength_row.addWidget(self.inp_strength)
        strength_row.addWidget(btn_strength_down)
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
        btn_seed_down = SkeetButton("◀")
        btn_seed_down.setFixedSize(20, 20)
        btn_seed_down.setStyleSheet(f"""
            QPushButton {{
                background: #181818; border: 1px solid #333; color: {FG_DIM};
                padding: 0; font-size: 10px; font-weight: bold; border-radius: 2px;
            }}
            QPushButton:hover {{ background: #222; color: {FG_ACCENT}; border: 1px solid {FG_ACCENT}; }}
            QPushButton:disabled {{ background: #111; color: #333; border: 1px solid #222; }}
        """)
        btn_seed_down.clicked.connect(self.inp_seed.stepDown)
        btn_seed_up = SkeetButton("▶")
        btn_seed_up.setFixedSize(20, 20)
        btn_seed_up.setStyleSheet(f"""
            QPushButton {{
                background: #181818; border: 1px solid #333; color: {FG_DIM};
                padding: 0; font-size: 10px; font-weight: bold; border-radius: 2px;
            }}
            QPushButton:hover {{ background: #222; color: {FG_ACCENT}; border: 1px solid {FG_ACCENT}; }}
            QPushButton:disabled {{ background: #111; color: #333; border: 1px solid #222; }}
        """)
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
        self.btn_generate = SkeetButton("GENERATE", accent=True)
        self.btn_generate.clicked.connect(self._start_generate)
        self.btn_save = SkeetButton("SAVE IMAGE")
        self.btn_save.clicked.connect(self._save_image)
        self.btn_save.setEnabled(False)
        btn_row.addWidget(self.btn_generate)
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
        self._set_status("CONFIG SAVED", FG_ACCENT)
        self._status_reset_timer.start()

    def _browse_model(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Vision Model",
            "",
            "Model Files (*.gguf *.ckpt);;All Files (*)"
        )
        if path:
            self.inp_model.setText(path)
            self.inp_model.setToolTip(path)
            self.btn_load.setChecked(False)

    def _load_model(self):
        path = self.inp_model.text().strip()
        if not path:
            self._set_status("ERROR: No model selected", FG_ERROR)
            self.btn_load.setChecked(False)
            return
        self.model_path = path
        self.btn_load.setChecked(True)
        self._queue_save_config()

    def _queue_save_config(self):
        self._status_reset_timer.stop()
        self._config_timer.start()

    def _set_status(self, status, color):
        self.lbl_status.setText(status)
        self.lbl_status.setStyleSheet(f"color: {color}; font-size: 10px; font-weight: bold;")

    def _reset_status(self):
        self._set_status("IDLE", FG_TEXT)

    def _start_generate(self):
        if not DIFFUSERS_AVAILABLE:
            self._set_status("ERROR: diffusers not installed", FG_ERROR)
            return
            
        prompt = self.inp_prompt.text().strip()
        if not prompt:
            self._set_status("ERROR: No prompt", FG_ERROR)
            return
            
        model_path = self.model_path
        if not model_path or not os.path.exists(model_path):
            self._set_status("ERROR: Invalid model path", FG_ERROR)
            return

        self.btn_generate.setEnabled(False)
        self.btn_save.setEnabled(False)
        self._set_status("INITIALIZING", FG_ACCENT)

        seed_value = self.inp_seed.value()
        seed = None if seed_value < 0 else seed_value
        self.worker = SDWorker(
            prompt,
            model_path,
            self.inp_steps.value(),
            self.inp_strength.value(),
            seed
        )
        self.worker.progress.connect(self._on_progress)
        self.worker.finished.connect(self._on_finished)
        self.worker.error.connect(self._on_error)
        self.worker.start()

    def _on_progress(self, msg):
        self._set_status(msg, FG_ACCENT)

    def _on_finished(self, pil_image):
        self.current_image = pil_image
        
        # Convert PIL to QPixmap
        pil_image = pil_image.convert("RGB")
        data = pil_image.tobytes("raw", "RGB")
        qimage = QImage(data, pil_image.width, pil_image.height, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimage)
        
        self.lbl_preview.setPixmap(pixmap.scaled(
            self.lbl_preview.width() - 20,
            self.lbl_preview.height() - 20,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        ))
        
        self._set_status("DONE", FG_TEXT)
        self.btn_generate.setEnabled(True)
        self.btn_save.setEnabled(True)

    def _on_error(self, err_msg):
        self._set_status(f"ERROR: {err_msg}", FG_ERROR)
        self.btn_generate.setEnabled(True)

    def _save_image(self):
        if not self.current_image:
            return
            
        import time
        filename = f"vision_{int(time.time())}.png"
        filepath = self.artifacts_dir / filename
        
        try:
            self.current_image.save(filepath)
            self._set_status(f"SAVED: {filename}", FG_ACCENT)
        except Exception as e:
            self._set_status(f"SAVE ERROR: {str(e)}", FG_ERROR)
