import os
from pathlib import Path
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QFrame, QComboBox, QDoubleSpinBox, QFileDialog,
    QAbstractSpinBox, QScrollArea,
)
from PySide6.QtCore import Qt, QThread, Signal, QUrl, QTimer
from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput
from PySide6.QtGui import QPainter, QPen, QColor

import core.style as _s  # dynamic theme bridge
from ui.components.atoms import MonoGroupBox, MonoButton, MonoTriangleButton, CollapsibleSection
from core.config import get_config, update_config_section

AUDIOCRAFT_AVAILABLE = False
try:
    import importlib
    importlib.import_module("audiocraft")
    AUDIOCRAFT_AVAILABLE = True
except ImportError:
    AUDIOCRAFT_AVAILABLE = False


class WaveformWidget(QFrame):
    def __init__(self):
        super().__init__()
        self.setFixedHeight(120)
        import core.style as _s
        self.setStyleSheet(f"background: {_s.BG_INPUT}; border: 1px solid {_s.BORDER_DARK};")
        self.waveform_data = None
        
    def set_waveform(self, audio_array):
        if audio_array is not None and len(audio_array) > 0:
            # Downsample for display
            target_points = 500
            if len(audio_array) > target_points:
                step = len(audio_array) // target_points
                self.waveform_data = audio_array[::step]
            else:
                self.waveform_data = audio_array
        else:
            self.waveform_data = None
        self.update()
    
    def paintEvent(self, event):
        super().paintEvent(event)
        import core.style as _s
        if self.waveform_data is None:
            painter = QPainter(self)
            painter.setPen(QPen(QColor(_s.FG_DIM)))
            painter.drawText(self.rect(), Qt.AlignCenter, "NO WAVEFORM")
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        w = self.width()
        h = self.height()
        mid_y = h / 2

        pen = QPen(QColor(_s.FG_ACCENT), 1)
        painter.setPen(pen)
        
        data = self.waveform_data
        num_points = len(data)
        
        for i in range(num_points - 1):
            x1 = int((i / num_points) * w)
            x2 = int(((i + 1) / num_points) * w)
            
            y1 = int(mid_y - (data[i] * mid_y * 0.9))
            y2 = int(mid_y - (data[i + 1] * mid_y * 0.9))
            
            painter.drawLine(x1, y1, x2, y2)


class AudioGenWorker(QThread):
    progress = Signal(str)
    finished = Signal(object, int)
    error = Signal(str)

    def __init__(self, prompt, model_path, duration, sample_rate):
        super().__init__()
        self.prompt = prompt
        self.model_path = model_path
        self.duration = duration
        self.sample_rate = sample_rate

    def run(self):
        try:
            try:
                from audiocraft.models import MusicGen
            except ImportError:
                self.error.emit("ERROR: audiocraft not installed. pip install audiocraft")
                return
            import torchaudio
            import torch
            
            self.progress.emit("Loading model...")
            
            model = MusicGen.get_pretrained(self.model_path)
            model.set_generation_params(duration=self.duration)
            
            self.progress.emit("Generating audio...")
            
            wav = model.generate([self.prompt])
            
            audio_array = wav[0].cpu().numpy()
            
            self.finished.emit(audio_array, self.sample_rate)
            
        except Exception as e:
            self.error.emit(str(e))


class AudioGenModule(QWidget):
    def __init__(self, state=None, ui_bridge=None):
        super().__init__()
        import core.style as s
        self.state = state
        self._ui_bridge = ui_bridge

        self.artifacts_dir = Path("artifacts/audio")
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        
        self.config = self._load_config()
        self.model_path = self.config.get("model_path", "")
        self.current_audio = None
        self.current_sample_rate = None
        self.current_filepath = None
        self.worker = None
        self._soundtrap_ready_callback = None
        self._soundtrap_ready_metadata = None
        self._config_timer = QTimer(self)
        self._config_timer.setInterval(1000)
        self._config_timer.setSingleShot(True)
        self._config_timer.timeout.connect(self._save_config)
        self._status_reset_timer = QTimer(self)
        self._status_reset_timer.setInterval(1000)
        self._status_reset_timer.setSingleShot(True)
        self._status_reset_timer.timeout.connect(self._reset_status)
        
        self.player = QMediaPlayer()
        self.audio_output = QAudioOutput()
        self.player.setAudioOutput(self.audio_output)

        # Outer layout wraps a scroll area around the body so the controls
        # stay reachable when the companion pane is at its default 360px.
        # Drops the previous MonoGroupBox("AUDIO") wrapper (the companion
        # already titles the tab) and the boxes-inside-boxes nesting of
        # CollapsibleSection containing two MonoGroupBoxes.
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        outer.addWidget(scroll, 1)

        body = QWidget()
        inner = QVBoxLayout(body)
        inner.setContentsMargins(12, 12, 12, 12)
        inner.setSpacing(10)
        scroll.setWidget(body)

        _input_ss = (
            f"QLineEdit, QDoubleSpinBox, QComboBox {{"
            f" background: {s.BG_INPUT}; color: {s.FG_TEXT};"
            f" border: 1px solid {s.BORDER_DARK}; padding: 4px; }}"
        )

        # ── PRIMARY: prompt + generate/play/save ──────────────────────────
        self.inp_prompt = QLineEdit()
        self.inp_prompt.setPlaceholderText("Prompt - describe a sound to generate")
        self.inp_prompt.setStyleSheet(_input_ss)
        inner.addWidget(self.inp_prompt)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(6)
        self.btn_generate = MonoButton("GENERATE", accent=True)
        self.btn_generate.clicked.connect(self._start_generate)
        self.btn_play = MonoButton("PLAY")
        self.btn_play.clicked.connect(self._play_audio)
        self.btn_play.setEnabled(False)
        self.btn_save = MonoButton("SAVE")
        self.btn_save.clicked.connect(self._save_audio)
        self.btn_save.setEnabled(False)
        btn_row.addWidget(self.btn_generate)
        btn_row.addWidget(self.btn_play)
        btn_row.addWidget(self.btn_save)
        btn_row.addStretch()
        inner.addLayout(btn_row)

        # Waveform Display + status
        self.waveform_widget = WaveformWidget()
        inner.addWidget(self.waveform_widget)

        status_row = QHBoxLayout()
        lbl_status_title = QLabel("Status")
        lbl_status_title.setStyleSheet(f"color: {s.FG_DIM}; font-size: 10px;")
        self.lbl_status = QLabel("IDLE")
        self.lbl_status.setStyleSheet(f"color: {s.FG_TEXT}; font-size: 10px; font-weight: bold;")
        status_row.addWidget(lbl_status_title)
        status_row.addStretch()
        status_row.addWidget(self.lbl_status)
        inner.addLayout(status_row)

        # ── MODEL (collapsed) ─────────────────────────────────────────────
        # Single section consolidating MODEL PATH + MODEL ID (used to live
        # in MonoGroupBox("MODEL LOADER") nested inside a CollapsibleSection
        # nested inside MonoGroupBox("AUDIO")).
        model_section = CollapsibleSection("MODEL")
        model_inner = QVBoxLayout()
        model_inner.setContentsMargins(0, 4, 0, 4)
        model_inner.setSpacing(6)

        model_path_row = QHBoxLayout()
        self.inp_model_path = QLineEdit(self.model_path)
        self.inp_model_path.setReadOnly(True)
        self.inp_model_path.setPlaceholderText("Select an AudioGen model file...")
        self.inp_model_path.setToolTip(self.model_path)
        self.inp_model_path.setStyleSheet(_input_ss)
        btn_browse = MonoButton("BROWSE")
        btn_browse.setFixedWidth(80)
        btn_browse.clicked.connect(self._browse_model)
        model_path_row.addWidget(self.inp_model_path, 1)
        model_path_row.addWidget(btn_browse)
        model_inner.addLayout(model_path_row)

        self.inp_model_id = QLineEdit()
        self.inp_model_id.setPlaceholderText("facebook/musicgen-small")
        self.inp_model_id.setText(self.config.get("model_id", "facebook/musicgen-small"))
        self.inp_model_id.setReadOnly(True)
        self.inp_model_id.setStyleSheet(_input_ss)
        model_inner.addWidget(self.inp_model_id)

        model_section.set_content_layout(model_inner)
        inner.addWidget(model_section)

        # ── PARAMS (collapsed) ────────────────────────────────────────────
        # Duration + sample rate. Replaces MonoGroupBox("AUDIO CONFIG").
        params_section = CollapsibleSection("PARAMS")
        params_inner = QVBoxLayout()
        params_inner.setContentsMargins(0, 4, 0, 4)
        params_inner.setSpacing(6)

        duration_row = QHBoxLayout()
        lbl_duration = QLabel("Duration (s)")
        lbl_duration.setStyleSheet(f"color: {s.FG_DIM}; font-size: 10px;")
        lbl_duration.setFixedWidth(90)
        self.inp_duration = QDoubleSpinBox()
        self.inp_duration.setRange(1.0, 30.0)
        self.inp_duration.setValue(self.config.get("duration", 5.0))
        self.inp_duration.setSingleStep(0.5)
        self.inp_duration.setButtonSymbols(QAbstractSpinBox.NoButtons)
        self.inp_duration.setStyleSheet(_input_ss)
        btn_duration_down = MonoTriangleButton("◀")
        btn_duration_down.clicked.connect(self.inp_duration.stepDown)
        btn_duration_up = MonoTriangleButton("▶")
        btn_duration_up.clicked.connect(self.inp_duration.stepUp)
        duration_row.addWidget(lbl_duration)
        duration_row.addWidget(btn_duration_down)
        duration_row.addWidget(self.inp_duration, 1)
        duration_row.addWidget(btn_duration_up)
        params_inner.addLayout(duration_row)

        sr_row = QHBoxLayout()
        lbl_sr = QLabel("Sample Rate")
        lbl_sr.setStyleSheet(f"color: {s.FG_DIM}; font-size: 10px;")
        lbl_sr.setFixedWidth(90)
        self.cmb_sr = QComboBox()
        self.cmb_sr.addItems(["32000", "44100", "48000"])
        self.cmb_sr.setCurrentText(str(self.config.get("sample_rate", 32000)))
        self.cmb_sr.setStyleSheet(_input_ss)
        sr_row.addWidget(lbl_sr)
        sr_row.addWidget(self.cmb_sr, 1)
        params_inner.addLayout(sr_row)

        params_section.set_content_layout(params_inner)
        inner.addWidget(params_section)

        inner.addStretch()

        self.inp_duration.valueChanged.connect(self._queue_save_config)
        self.cmb_sr.currentTextChanged.connect(self._queue_save_config)
        self._set_audio_world_state("READY", task=None)

    def _load_config(self):
        return get_config().audio.model_dump()

    def _queue_save_config(self):
        self._status_reset_timer.stop()
        self._config_timer.start()

    def _save_config(self):
        import core.style as s
        config = {
            "model_path": self.model_path,
            "model_id": self.inp_model_id.text().strip() or self.inp_model_id.placeholderText(),
            "duration": self.inp_duration.value(),
            "sample_rate": int(self.cmb_sr.currentText())
        }
        update_config_section("audio", config, persist=True)
        self.config = config
        self._set_status("CONFIG SAVED", s.FG_ACCENT)
        self._status_reset_timer.start()

    def _browse_model(self):
        import core.style as s
        path, _ = QFileDialog.getOpenFileName(self, "Select Audio Model", "", "All Files (*)")
        if not path:
            return
        path = os.path.abspath(path)
        try:
            if not AUDIOCRAFT_AVAILABLE:
                self._set_status("ERROR: audiocraft not installed. pip install audiocraft", s.FG_ERROR)
                self.inp_model_path.setText(self.model_path)
                self.inp_model_path.setToolTip(self.model_path)
                return
            try:
                from audiocraft.models import MusicGen
            except ImportError:
                self._set_status("ERROR: audiocraft not installed. pip install audiocraft", s.FG_ERROR)
                self.inp_model_path.setText(self.model_path)
                self.inp_model_path.setToolTip(self.model_path)
                return
            MusicGen.get_pretrained(path)
        except Exception as exc:
            self._set_status(f"ERROR: {str(exc)}", s.FG_ERROR)
            self.inp_model_path.setText(self.model_path)
            self.inp_model_path.setToolTip(self.model_path)
            return
        self.model_path = path
        self.inp_model_path.setText(path)
        self.inp_model_path.setToolTip(path)
        self._queue_save_config()

    def _set_status(self, status, color=None):
        import core.style as s
        self.lbl_status.setText(status)
        self.lbl_status.setStyleSheet(f"color: {color or s.FG_TEXT}; font-size: 10px; font-weight: bold;")

    def _set_audio_world_state(self, status: str, task: dict | None) -> None:
        world_state = getattr(self.state, "world_state", None)
        if world_state is None:
            return
        world_state.set_engine_status("audio", str(status or "").upper())
        world_state.set_active_task("audio", task)

    def _reset_status(self):
        self._set_status("IDLE")

    def _start_generate(self):
        import core.style as s
        if not AUDIOCRAFT_AVAILABLE:
            self._set_status("ERROR: audiocraft not installed. pip install audiocraft", s.FG_ERROR)
            self._log("ERROR", "[audio] audiocraft not installed")
            return

        prompt = self.inp_prompt.text().strip()
        if not prompt:
            self._set_status("ERROR: No prompt", s.FG_ERROR)
            self._log("WARNING", "[audio] missing prompt")
            return

        self.btn_generate.setEnabled(False)
        self.btn_play.setEnabled(False)
        self.btn_save.setEnabled(False)
        self._set_status("INITIALIZING", s.FG_ACCENT)

        model_path = self.model_path
        if not model_path:
            self._set_status("ERROR: No model selected", s.FG_ERROR)
            self._log("WARNING", "[audio] missing model path")
            return

        self.worker = AudioGenWorker(
            prompt,
            model_path,
            self.inp_duration.value(),
            int(self.cmb_sr.currentText())
        )
        self.worker.progress.connect(self._on_progress)
        self.worker.finished.connect(self._on_finished)
        self.worker.error.connect(self._on_error)
        self.worker.start()
        self._set_audio_world_state(
            "RUNNING",
            {
                "command": "generate",
                "status": "RUNNING",
                "prompt": prompt[:160],
            },
        )
        self._log("INFO", f"[audio] generation started (duration={self.inp_duration.value()}s)")

    def _on_progress(self, msg):
        import core.style as s
        self._set_status(msg, s.FG_ACCENT)

    def _on_finished(self, audio_array, sample_rate):
        self.current_audio = audio_array
        self.current_sample_rate = sample_rate

        # Clean up the previous temp file so artifacts/audio/ doesn't grow
        # unbounded across the session. The SAVE button persists the audio
        # to a user-chosen path; temp files only exist for in-app playback.
        prior = self.current_filepath
        if prior is not None:
            try:
                if prior.exists() and prior.name.startswith("temp_audio_"):
                    prior.unlink()
            except Exception:
                pass

        # Save temporarily for playback
        import time
        temp_filename = f"temp_audio_{int(time.time())}.wav"
        self.current_filepath = self.artifacts_dir / temp_filename

        try:
            import torch
            import torchaudio
            audio_tensor = torch.from_numpy(audio_array).unsqueeze(0)
            torchaudio.save(str(self.current_filepath), audio_tensor, sample_rate)
        except Exception as e:
            import core.style as s
            self._set_status(f"SAVE ERROR: {str(e)}", s.FG_ERROR)
            return

        # Display waveform (use mono channel)
        if len(audio_array.shape) > 1:
            display_data = audio_array[0]
        else:
            display_data = audio_array
        self.waveform_widget.set_waveform(display_data)

        self._set_status("DONE")
        self.btn_generate.setEnabled(True)
        self.btn_play.setEnabled(True)
        self.btn_save.setEnabled(True)
        self._set_audio_world_state("READY", task=None)
        callback = self._soundtrap_ready_callback
        metadata = dict(self._soundtrap_ready_metadata or {})
        self._soundtrap_ready_callback = None
        self._soundtrap_ready_metadata = None
        if callable(callback):
            metadata.setdefault("sample_rate", sample_rate)
            metadata.setdefault("audio_path", str(self.current_filepath))
            try:
                callback(str(self.current_filepath), metadata)
            except Exception as exc:
                self._log("ERROR", f"[audio] soundtrap import callback failed: {exc}")
        self._log("INFO", f"[audio] generation finished ({sample_rate} Hz)")

    def _on_error(self, err_msg):
        import core.style as s
        self._set_status(f"ERROR: {err_msg}", s.FG_ERROR)
        self.btn_generate.setEnabled(True)
        self._set_audio_world_state("ERROR", task=None)
        self._soundtrap_ready_callback = None
        self._soundtrap_ready_metadata = None
        self._log("ERROR", f"[audio] generation error: {err_msg}")

    def _play_audio(self):
        if not self.current_filepath or not self.current_filepath.exists():
            return
            
        self.player.setSource(QUrl.fromLocalFile(str(self.current_filepath)))
        self.player.play()
        import core.style as s
        self._set_status("PLAYING", s.FG_ACCENT)
        self._log("INFO", "[audio] playback started")

    def _save_audio(self):
        if self.current_audio is None:
            return
        
        if not self.current_filepath or not self.current_filepath.exists():
            return
            
        import time
        filename = f"audio_{int(time.time())}.wav"
        filepath = self.artifacts_dir / filename
        
        try:
            import shutil
            shutil.copy(self.current_filepath, filepath)
            import core.style as s
            self._set_status(f"SAVED: {filename}", s.FG_ACCENT)
            self._log("INFO", f"[audio] saved {filename}")
        except Exception as e:
            import core.style as s
            self._set_status(f"SAVE ERROR: {str(e)}", s.FG_ERROR)
            self._log("ERROR", f"[audio] save error: {e}")

    def trigger_generation(self, params: dict) -> str:
        """Programmatic entry point for agent-driven audio generation."""
        callback = params.get("_soundtrap_ready_callback")
        if not callable(callback):
            callback = None
        if not AUDIOCRAFT_AVAILABLE:
            return "audiocraft not installed"

        prompt = str(params.get("prompt", "")).strip()
        if not prompt:
            return "no prompt provided"

        model_path = self.model_path
        if not model_path:
            model_id = self.config.get("model_id", "facebook/musicgen-small")
            if not model_id:
                return "no audio model configured"
            model_path = model_id

        duration = float(params.get("duration", self.config.get("duration", 5.0)))
        duration = max(1.0, min(30.0, duration))
        sample_rate = int(params.get("sample_rate", self.config.get("sample_rate", 32000)))
        if sample_rate not in (32000, 44100, 48000):
            sample_rate = 32000

        self.btn_generate.setEnabled(False)
        self.btn_play.setEnabled(False)
        self.btn_save.setEnabled(False)
        import core.style as s
        self._set_status("INITIALIZING", s.FG_ACCENT)

        self.worker = AudioGenWorker(prompt, model_path, duration, sample_rate)
        self.worker.progress.connect(self._on_progress)
        self.worker.finished.connect(self._on_finished)
        self.worker.error.connect(self._on_error)
        self._soundtrap_ready_callback = callback
        self._soundtrap_ready_metadata = {
            "soundtrap_pending_clip_id": str(params.get("soundtrap_pending_clip_id", "") or ""),
            "soundtrap_clip_name": str(params.get("soundtrap_clip_name", "") or ""),
            "prompt": prompt,
            "duration": duration,
            "sample_rate": sample_rate,
        } if callback is not None else None
        self.worker.start()
        self._set_audio_world_state(
            "RUNNING",
            {"command": "generate", "status": "RUNNING", "prompt": prompt[:160]},
        )
        self._log("INFO", f"[audio] agent-triggered generation (duration={duration}s)")
        return f"generation started - prompt={prompt[:80]}, duration={duration}s"

    def _log(self, severity: str, message: str) -> None:
        if self._ui_bridge is None:
            return
        self._ui_bridge.sig_monitor_log.emit(severity, message)
