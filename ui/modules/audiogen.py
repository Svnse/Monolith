import os
import json
from pathlib import Path
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QFrame, QComboBox, QDoubleSpinBox, QFileDialog,
    QAbstractSpinBox
)
from PySide6.QtCore import Qt, QThread, Signal, QUrl, QTimer
from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput
from PySide6.QtGui import QPainter, QPen, QColor

import core.style as _s  # dynamic theme bridge
from ui.components.atoms import MonoGroupBox, MonoButton, MonoTriangleButton, CollapsibleSection

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
    def __init__(self):
        super().__init__()
        import core.style as s

        self.config_path = Path("config/audiogen_config.json")
        self.artifacts_dir = Path("artifacts/audio")
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.config = self._load_config()
        self.model_path = self.config.get("model_path", "")
        self.current_audio = None
        self.current_sample_rate = None
        self.current_filepath = None
        self.worker = None
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

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)

        grp = MonoGroupBox("AUDIO")
        inner = QVBoxLayout()
        inner.setSpacing(12)

        # Config Section
        config_section = CollapsibleSection("⚙ CONFIGURATION")
        config_layout = QVBoxLayout()
        config_layout.setSpacing(8)

        grp_model = MonoGroupBox("MODEL LOADER")
        model_layout = QVBoxLayout()
        model_path_row = QHBoxLayout()
        lbl_model_path = QLabel("MODEL PATH")
        lbl_model_path.setStyleSheet(f"color: {s.FG_DIM}; font-size: 10px;")
        lbl_model_path.setFixedWidth(80)
        self.inp_model_path = QLineEdit(self.model_path)
        self.inp_model_path.setReadOnly(True)
        self.inp_model_path.setPlaceholderText("Select an AudioGen model file...")
        self.inp_model_path.setToolTip(self.model_path)
        self.inp_model_path.setStyleSheet(f"""
            QLineEdit {{
                background: {s.BG_INPUT}; color: {s.FG_TEXT};
                border: 1px solid {s.BORDER_DARK}; padding: 4px;
            }}
        """)
        btn_browse = MonoButton("BROWSE...")
        btn_browse.setFixedWidth(90)
        btn_browse.clicked.connect(self._browse_model)
        model_path_row.addWidget(lbl_model_path)
        model_path_row.addWidget(self.inp_model_path)
        model_path_row.addWidget(btn_browse)
        model_layout.addLayout(model_path_row)

        model_row = QHBoxLayout()
        lbl_model = QLabel("MODEL ID")
        lbl_model.setStyleSheet(f"color: {s.FG_DIM}; font-size: 10px;")
        lbl_model.setFixedWidth(80)
        self.inp_model_id = QLineEdit()
        self.inp_model_id.setPlaceholderText("facebook/musicgen-small")
        self.inp_model_id.setText(self.config.get("model_id", "facebook/musicgen-small"))
        self.inp_model_id.setReadOnly(True)
        self.inp_model_id.setStyleSheet(f"""
            QLineEdit {{
                background: {s.BG_INPUT}; color: {s.FG_TEXT};
                border: 1px solid {s.BORDER_DARK}; padding: 4px;
            }}
        """)
        model_row.addWidget(lbl_model)
        model_row.addWidget(self.inp_model_id)
        model_layout.addLayout(model_row)
        grp_model.add_layout(model_layout)
        config_layout.addWidget(grp_model)

        grp_audio = MonoGroupBox("AUDIO CONFIG")
        audio_layout = QVBoxLayout()

        duration_row = QHBoxLayout()
        lbl_duration = QLabel("Duration (s)")
        lbl_duration.setStyleSheet(f"color: {s.FG_DIM}; font-size: 10px;")
        lbl_duration.setFixedWidth(80)
        self.inp_duration = QDoubleSpinBox()
        self.inp_duration.setRange(1.0, 30.0)
        self.inp_duration.setValue(self.config.get("duration", 5.0))
        self.inp_duration.setSingleStep(0.5)
        self.inp_duration.setButtonSymbols(QAbstractSpinBox.NoButtons)
        self.inp_duration.setStyleSheet(f"""
            QDoubleSpinBox {{
                background: {s.BG_INPUT}; color: {s.FG_TEXT};
                border: 1px solid {s.BORDER_DARK}; padding: 4px;
            }}
        """)
        duration_row.addWidget(lbl_duration)
        btn_duration_down = MonoTriangleButton("◀")
        btn_duration_down.clicked.connect(self.inp_duration.stepDown)
        btn_duration_up = MonoTriangleButton("▶")
        btn_duration_up.clicked.connect(self.inp_duration.stepUp)
        duration_row.addWidget(btn_duration_down)
        duration_row.addWidget(self.inp_duration)
        duration_row.addWidget(btn_duration_up)
        duration_row.addStretch()
        audio_layout.addLayout(duration_row)

        sr_row = QHBoxLayout()
        lbl_sr = QLabel("Sample Rate")
        lbl_sr.setStyleSheet(f"color: {s.FG_DIM}; font-size: 10px;")
        lbl_sr.setFixedWidth(80)
        self.cmb_sr = QComboBox()
        self.cmb_sr.addItems(["32000", "44100", "48000"])
        self.cmb_sr.setCurrentText(str(self.config.get("sample_rate", 32000)))
        self.cmb_sr.setStyleSheet(f"""
            QComboBox {{
                background: {s.BG_INPUT}; color: {s.FG_TEXT};
                border: 1px solid {s.BORDER_DARK}; padding: 4px;
            }}
        """)
        sr_row.addWidget(lbl_sr)
        sr_row.addWidget(self.cmb_sr)
        sr_row.addStretch()
        audio_layout.addLayout(sr_row)

        grp_audio.add_layout(audio_layout)
        config_layout.addWidget(grp_audio)
        
        config_section.set_content_layout(config_layout)
        inner.addWidget(config_section)

        # Prompt
        lbl_prompt = QLabel("Prompt")
        lbl_prompt.setStyleSheet(f"color: {s.FG_DIM}; font-size: 10px;")

        self.inp_prompt = QLineEdit()
        self.inp_prompt.setPlaceholderText("Describe a sound to generate...")
        self.inp_prompt.setStyleSheet(f"""
            QLineEdit {{
                background: {s.BG_INPUT}; color: {s.FG_TEXT};
                border: 1px solid {s.BORDER_DARK}; padding: 6px;
            }}
        """)

        # Buttons
        btn_row = QHBoxLayout()
        self.btn_generate = MonoButton("GENERATE", accent=True)
        self.btn_generate.clicked.connect(self._start_generate)
        self.btn_play = MonoButton("PLAY")
        self.btn_play.clicked.connect(self._play_audio)
        self.btn_play.setEnabled(False)
        self.btn_save = MonoButton("SAVE AUDIO")
        self.btn_save.clicked.connect(self._save_audio)
        self.btn_save.setEnabled(False)
        btn_row.addWidget(self.btn_generate)
        btn_row.addWidget(self.btn_play)
        btn_row.addWidget(self.btn_save)
        btn_row.addStretch()

        # Waveform Display
        self.waveform_widget = WaveformWidget()

        # Status
        status_row = QHBoxLayout()
        lbl_status_title = QLabel("Status")
        lbl_status_title.setStyleSheet(f"color: {s.FG_DIM}; font-size: 10px;")
        self.lbl_status = QLabel("IDLE")
        self.lbl_status.setStyleSheet(f"color: {s.FG_TEXT}; font-size: 10px; font-weight: bold;")
        status_row.addWidget(lbl_status_title)
        status_row.addStretch()
        status_row.addWidget(self.lbl_status)

        inner.addWidget(lbl_prompt)
        inner.addWidget(self.inp_prompt)
        inner.addLayout(btn_row)
        inner.addWidget(self.waveform_widget)
        inner.addLayout(status_row)
        inner.addStretch()

        grp.add_layout(inner)
        layout.addWidget(grp)

        self.inp_duration.valueChanged.connect(self._queue_save_config)
        self.cmb_sr.currentTextChanged.connect(self._queue_save_config)

    def _load_config(self):
        if self.config_path.exists():
            try:
                with open(self.config_path, 'r') as f:
                    config = json.load(f)
                return config
            except:
                pass
        return {
            "model_path": "",
            "model_id": "facebook/musicgen-small",
            "duration": 5.0,
            "sample_rate": 32000
        }

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
        with open(self.config_path, 'w') as f:
            json.dump(config, f, indent=2)
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

    def _reset_status(self):
        self._set_status("IDLE")

    def _start_generate(self):
        import core.style as s
        if not AUDIOCRAFT_AVAILABLE:
            self._set_status("ERROR: audiocraft not installed. pip install audiocraft", s.FG_ERROR)
            return

        prompt = self.inp_prompt.text().strip()
        if not prompt:
            self._set_status("ERROR: No prompt", s.FG_ERROR)
            return

        self.btn_generate.setEnabled(False)
        self.btn_play.setEnabled(False)
        self.btn_save.setEnabled(False)
        self._set_status("INITIALIZING", s.FG_ACCENT)

        model_path = self.model_path
        if not model_path:
            self._set_status("ERROR: No model selected", s.FG_ERROR)
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

    def _on_progress(self, msg):
        import core.style as s
        self._set_status(msg, s.FG_ACCENT)

    def _on_finished(self, audio_array, sample_rate):
        self.current_audio = audio_array
        self.current_sample_rate = sample_rate
        
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

    def _on_error(self, err_msg):
        import core.style as s
        self._set_status(f"ERROR: {err_msg}", s.FG_ERROR)
        self.btn_generate.setEnabled(True)

    def _play_audio(self):
        if not self.current_filepath or not self.current_filepath.exists():
            return
            
        self.player.setSource(QUrl.fromLocalFile(str(self.current_filepath)))
        self.player.play()
        import core.style as s
        self._set_status("PLAYING", s.FG_ACCENT)

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
        except Exception as e:
            import core.style as s
            self._set_status(f"SAVE ERROR: {str(e)}", s.FG_ERROR)
