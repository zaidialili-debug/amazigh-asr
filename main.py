import csv
import sys
import os
import time
import numpy as np
import torch
import sounddevice as sd
import soundfile as sf
import tempfile
import shutil
import noisereduce as nr
import torchaudio
import platform
import subprocess


# Import for the stop criterion
from transformers import (
    AutoProcessor,
    AutoModelForSpeechSeq2Seq,
    WhisperFeatureExtractor,
    WhisperProcessor,
    WhisperTokenizer,
    StoppingCriteria,
    StoppingCriteriaList,
)

from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton, QVBoxLayout,
                            QHBoxLayout, QLabel, QTextEdit, QWidget, QFileDialog,
                            QComboBox, QProgressBar, QMessageBox, QInputDialog,
                            QMenu, QAction, QDialog, QSizePolicy, QDialogButtonBox)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer, QSize
from PyQt5.QtGui import QIcon, QPalette, QBrush, QPixmap, QFont
def load_audio_resampled(path, target_sr):
    """Charge un fichier audio en mono et le reechantillonne."""
    audio, sr = sf.read(path, dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    if sr != target_sr:
        tensor = torch.from_numpy(audio).unsqueeze(0)
        tensor = torchaudio.functional.resample(tensor, sr, target_sr)
        audio = tensor.squeeze(0).numpy()
    return audio, target_sr
# --- Punctuation stop criterion ---
class StopOnPunctuationCriteria(StoppingCriteria):
    def __init__(self, stop_token_ids):
        super().__init__()
        self.stop_token_ids = stop_token_ids

    def __call__(self, input_ids, scores, **kwargs):
        if input_ids[0][-1] in self.stop_token_ids:
            return True
        return False

# --- Tifinagh converter ---
class TifinaghConverter:
    def __init__(self):
        self.replacements = {
            "Gʷ": "ⴳⵯ", "Kʷ": "ⴽⵯ", "Ch": "ⵛ", "Gh": "ⵖ", "Kh": "ⵅ", "Sh": "ⵛ",
            "Ts": "ⵜⵙ", "Dz": "ⴷⵣ", "Ḍ": "ⴹ", "Ṭ": "ⵟ", "Ẓ": "ⵥ", "Ṣ": "ⵚ", "Ṛ": "ⵕ",
            "A": "ⴰ", "E": "ⴻ", "I": "ⵉ", "U": "ⵓ",
            "B": "ⴱ", "D": "ⴷ", "F": "ⴼ", "G": "ⴳ", "H": "ⵀ", "ḥ": "ⵃ",
            "J": "ⵊ", "K": "ⴽ", "L": "ⵍ", "M": "ⵎ", "N": "ⵏ",
            "Ɛ": "ⵄ", "V": "ⵠ", "Q": "ⵇ", "R": "ⵔ", "S": "ⵙ",
            "T": "ⵜ", "W": "ⵡ", "X": "ⵅ", "Y": "ⵢ", "Z": "ⵣ",
            "P": "ⵠ", "O": "ⵓ",
            "gʷ": "ⴳⵯ", "kʷ": "ⴽⵯ", "ch": "ⵛ", "gh": "ⴴ", "kh": "ⵅ", "X":"ⵅ","x":"ⵅ", "sh": "ⵛ",
            "ts": "ⵜⵙ", "dz": "ⴷⵣ", "ḍ": "ⴹ", "ṭ": "ⵟ", "ẓ": "ⵥ", "ṣ": "ⵚ", "ṛ": "ⵕ",
            "a": "ⴰ", "e": "ⴻ", "i": "ⵉ", "u": "ⵓ",
            "b": "ⴱ", "d": "ⴷ", "f": "ⴼ", "g": "ⴳ", "h": "ⵀ",
            "j": "ⵊ", "k": "ⴽ", "l": "ⵍ", "m": "ⵎ", "n": "ⵏ",
            "ɛ": "ⵄ", "v": "ⵠ", "q": "ⵇ", "r": "ⵔ", "s": "ⵙ",
            "t": "ⵜ", "w": "ⵡ", "x": "ⵅ", "y": "ⵢ", "z": "ⵣ",
            "p": "ⵠ", "o": "ⵓ",
            " ": " ", "\n": "\n", "\t": "\t", ".": ".", ",": ",", "?": "?", "!": "!",
            ":": ":", ";": ";", "(": ")", "-": "-", "'": "'", "`": "`", "/": "/",
            "0": "0", "1": "1", "2": "2", "3": "3", "4": "4", "5": "5", "6": "6", "7": "7", "8": "8", "9": "9",
        }
        self.keys_sorted_by_length = sorted(self.replacements.keys(), key=len, reverse=True)

    def convert(self, text):
        result = []
        i = 0
        while i < len(text):
            found_match = False
            for key in self.keys_sorted_by_length:
                if text[i:].startswith(key):
                    result.append(self.replacements[key])
                    i += len(key)
                    found_match = True
                    break
            if not found_match:
                result.append(text[i])
                i += 1
        return "".join(result)

# --- Recorder (Recording) ---
class AudioRecorder(QThread):
    update_time = pyqtSignal(int)
    finished = pyqtSignal(str)

    def __init__(self, sample_rate=44100):
        super().__init__()
        self.sample_rate = sample_rate
        self.recording = False
        self.audio_data = []
        self.temp_file = None

    def run(self):
        self.recording = True
        self.audio_data = []
        self.temp_file = os.path.join(tempfile.gettempdir(), f"amazigh_rec_{int(time.time())}.wav")
        start_time = time.time()

        def callback(indata, frames, time_info, status):
            if self.recording:
                if isinstance(indata, np.ndarray) and indata.size > 0:
                    self.audio_data.append(indata.copy())
                    elapsed = int(time.time() - start_time)
                    self.update_time.emit(elapsed)

        try:
            with sd.InputStream(samplerate=self.sample_rate, channels=1, callback=callback, dtype='float32'):
                while self.recording:
                    sd.sleep(100)
        except Exception as e:
            self.finished.emit(f"ERROR: {e}")
            return

        try:
            if len(self.audio_data) > 0:
                audio = np.concatenate(self.audio_data, axis=0)
                sf.write(self.temp_file, audio, self.sample_rate)
                time.sleep(0.5) 
                self.finished.emit(self.temp_file)
            else:
                self.finished.emit("ERROR: No audio recorded.")
        except Exception as e:
            self.finished.emit(f"ERROR_SAVE: {e}")

    def stop(self):
        self.recording = False

# --- AudioProcessor ---
class AudioProcessor(QThread):
    progress = pyqtSignal(int)
    processed = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, audio_file):
        super().__init__()
        self.audio_file = audio_file

    def run(self):
        try:
            self.progress.emit(10)
            audio_input, sr = sf.read(self.audio_file)
            
            if len(audio_input.shape) > 1:
                audio_input = np.mean(audio_input, axis=1)

            self.progress.emit(30)
            clean_audio = nr.reduce_noise(y=audio_input, sr=sr, stationary=False)
            self.progress.emit(60)
            
            max_val = np.max(np.abs(clean_audio))
            if max_val > 0:
                clean_audio = clean_audio / max_val
            
            self.progress.emit(80)
            
            temp_file = os.path.join(tempfile.gettempdir(), f"amazigh_proc_{int(time.time())}.wav")
            sf.write(temp_file, clean_audio, sr)
            
            self.progress.emit(100)
            self.processed.emit(temp_file)

        except Exception as e:
            self.error.emit(f"Processing error: {str(e)}")

# --- Transcription thread (FORCE CPU MODIFICATION) ---
class TranscriptionThread(QThread):
    progress_update = pyqtSignal(int)
    transcription_complete = pyqtSignal(str)

    def __init__(self, audio_file, model_name="openai/whisper-small"):
        super().__init__()
        self.audio_file = audio_file
        self.model_name = model_name
        self.processor = None
        self.model = None
        # --- MODIFICATION CRITIQUE ---
        # We explicitly force CPU to avoid the CUDA no kernel image bug
        self.device = "cpu"
        print(f"Force CPU Mode. Device: {self.device}")
        # -----------------------------

    def run(self):
        try:
            if not os.path.exists(self.audio_file):
                self.transcription_complete.emit(f"Error: The file {self.audio_file} does not exist.")
                return

            self.progress_update.emit(10)
            
            model_path = self.model_name
            print(f"Loading model from: {model_path}")

            # Load model
            try:
                self.processor = AutoProcessor.from_pretrained(model_path)
                self.model = AutoModelForSpeechSeq2Seq.from_pretrained(model_path).to(self.device)
            except Exception as e:
                error_text = str(e)
                if "feature extractor" in error_text or "preprocessor_config.json" in error_text:
                    try:
                        tokenizer = WhisperTokenizer.from_pretrained(model_path)
                        feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-base")
                        self.processor = WhisperProcessor(feature_extractor=feature_extractor, tokenizer=tokenizer)
                        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(model_path).to(self.device)
                    except Exception as fallback_error:
                        self.transcription_complete.emit(f"Model loading error: {fallback_error}")
                        return
                else:
                    self.transcription_complete.emit(f"Model loading error: {error_text}")
                    return

            self.progress_update.emit(40)
            
            # Load audio
            try:
                target_sr = 16000
                audio_input, sr = load_audio_resampled(self.audio_file, target_sr)
            except Exception as e:
                self.transcription_complete.emit(f"File reading error: {e}")
                return

            self.progress_update.emit(50)

            # Quick preprocessing
            try:
                clean_audio = nr.reduce_noise(y=audio_input, sr=sr, stationary=False)
                max_val = np.max(np.abs(clean_audio))
                if max_val > 0:
                    clean_audio = clean_audio / max_val
                audio_input = clean_audio
            except Exception as e:
                print(f"Warning noise reduction: {e}")

            self.progress_update.emit(70)

            processor_output = self.processor(audio_input, sampling_rate=sr, return_tensors="pt")
            input_features = processor_output.input_features.to(self.device)
            attention_mask = processor_output.attention_mask if hasattr(processor_output, "attention_mask") else None

            self.progress_update.emit(80)

            period_tokens = self.processor.tokenizer.encode(".", add_special_tokens=False)
            exclamation_tokens = self.processor.tokenizer.encode("!", add_special_tokens=False)
            question_tokens = self.processor.tokenizer.encode("?", add_special_tokens=False)

            stop_token_ids = []
            if period_tokens:
                stop_token_ids.append(period_tokens[0])
            if exclamation_tokens:
                stop_token_ids.append(exclamation_tokens[0])
            if question_tokens:
                stop_token_ids.append(question_tokens[0])

            stopping_criteria = None
            if stop_token_ids:
                stopping_criteria = StoppingCriteriaList([StopOnPunctuationCriteria(stop_token_ids)])

            prompt_ids = self.processor.get_prompt_ids("transcribe", return_tensors="pt")
            predicted_ids = self.model.generate(
                input_features,
                attention_mask=attention_mask,
                prompt_ids=prompt_ids,
                prompt_condition_type="first-segment",
                max_new_tokens=200,
                stopping_criteria=stopping_criteria,
            )

            self.progress_update.emit(90)

            transcription = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

            self.progress_update.emit(100)
            self.transcription_complete.emit(transcription)

        except Exception as e:
            error_message = f"Critical transcription error ({self.device}): {str(e)}"
            self.transcription_complete.emit(error_message)

# --- About Dialog ---
class AboutDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("About")
        self.setModal(True)
        self.resize(700, 650)
        self.setStyleSheet("""
            QDialog {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #0f172a, stop:1 #111827);
                color: #F8FAFC;
            }
            QLabel {
                color: #E2E8F0;
                font-size: 10pt;
            }
            QTextEdit {
                background-color: rgba(30, 41, 59, 0.95);
                border: 1px solid #475569;
                border-radius: 6px;
                color: #FFFFFF;
                padding: 10px;
            }
            QPushButton {
                background-color: #3B82F6;
                color: white;
                border: none;
                border-radius: 8px;
                padding: 8px 14px;
                font-weight: bold;
            }
            QPushButton:hover { background-color: #2563EB; }
        """)

        layout = QVBoxLayout(self)
        title = QLabel("<h2>Amaziɣ-ASR 1.0</h2>")
        title.setAlignment(Qt.AlignCenter)

        description_text = QTextEdit()
        description_text.setReadOnly(True)
        description_text.setHtml("""
            <p><b>Amaziɣ-ASR</b> is an innovative automatic speech recognition application
            dedicated to the Amazigh language. Developed from a rigorous research project
            <b>(code: 03/10/TTTAL/CRLCA/2023)</b>, this tool was created within the
            <b>Terminology, Translation and Automatic Processing of Natural Language (TTTAL)</b>
            division of the <b>Center for Research in Amazigh Language and Culture (CRLCA)</b> in Béjaïa.</p>
            
            <p>This strategic project aims to bridge the digital divide by equipping the Amazigh language
            with state-of-the-art technological tools, enabling precise conversion of speech
            into text and opening the way to many applications such as voice dictation and
            digital accessibility in Tamazight.</p>
            
            <h4>Technical Information</h4>
            <ul>
                <li><b>Version:</b> 1.0 (2026)</li>
                <li><b>Model:</b> whisper-small</li>
                <li><b>Goal:</b> facilitate the collection, processing and validation of Amazigh audio corpora</li>
            </ul>
            
            <h4>Authors</h4>
            <ul>
                <li>Dr. Ali ZAIDI</li>
                <li>Dr. Mourad AZI</li>
                <li>Dr. Saida MATOUB</li>
                <li>Dr. Lamine IDIR</li>
            </ul>
        """)

        close_button = QPushButton("Close")
        close_button.clicked.connect(self.close)

        layout.addWidget(title)
        layout.addWidget(description_text)
        layout.addWidget(close_button, 0, Qt.AlignRight)

# --- Guidelines Dialog ---
class GuidelinesDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Guidelines")
        self.setModal(True)
        self.resize(600, 500)
        self.setStyleSheet("""
            QDialog {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #0f172a, stop:1 #111827);
                color: #F8FAFC;
            }
            QLabel {
                color: #E2E8F0;
                font-size: 11pt;
            }
            QTextEdit {
                background-color: rgba(30, 41, 59, 0.95);
                border: 1px solid #475569;
                border-radius: 6px;
                color: #FFFFFF;
                padding: 10px;
            }
            QPushButton {
                background-color: #3B82F6;
                color: white;
                border: none;
                border-radius: 8px;
                padding: 8px 14px;
                font-weight: bold;
            }
            QPushButton:hover { background-color: #2563EB; }
        """)

        layout = QVBoxLayout(self)
        title = QLabel("<h2>Usage Guidelines</h2>")
        title.setAlignment(Qt.AlignCenter)

        guidelines_text = QTextEdit()
        guidelines_text.setReadOnly(True)
        guidelines_text.setHtml("""
            <h3>Amaziɣ-ASR Usage Guide</h3>
            <p><b>1. Audio recording:</b></p>
            <ul>
                <li>Click "Start Recording" to begin recording</li>
                <li>Speak clearly in Tamazight</li>
                <li>Click "Stop Recording" to finish</li>
            </ul>
            <p><b>2. Audio processing:</b></p>
            <ul>
                <li>Click "Process Audio" to reduce background noise</li>
                <li>This improves transcription quality</li>
            </ul>
            <p><b>3. Transcription:</b></p>
            <ul>
                <li>Select the desired model</li>
                <li>Click "Transcribe Audio"</li>
                <li>Results appear in Latin and Tifinagh</li>
            </ul>
            <p><b>4. Editing and validation:</b></p>
            <ul>
                <li>Use "Edit" to modify the transcription</li>
                <li>Use "Validate" to save to recordings.csv</li>
            </ul>
            <p><b>5. Keyboard shortcuts:</b></p>
            <ul>
                <li>Ctrl+T: Start transcription</li>
                <li>Ctrl+P: Process audio</li>
                <li>Ctrl+E: Edit transcription</li>
                <li>Ctrl+V: Validate</li>
                <li>Ctrl+L: Clear</li>
            </ul>
        """)

        close_button = QPushButton("Close")
        close_button.clicked.connect(self.close)

        layout.addWidget(title)
        layout.addWidget(guidelines_text)
        layout.addWidget(close_button, 0, Qt.AlignRight)

# --- Main Window ---
class AudioTranscriptionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Amaziɣ-ASR  1.0 (2026)")

        screen = QApplication.primaryScreen()
        if screen is not None:
            geom = screen.availableGeometry()
            width = min(int(geom.width() * 0.8), 1200)
            height = min(int(geom.height() * 0.8), 850)
            x = geom.x() + (geom.width() - width) // 2
            y = geom.y() + (geom.height() - height) // 2
            self.setGeometry(x, y, width, height + 40)
            self.setMinimumSize(900, 690)
        else:
            self.setGeometry(100, 100, 1024, 808)
            self.setMinimumSize(900, 690)

        self.recorder = None
        self.transcription_thread = None
        self.audio_processor = None
        self.recording_time = 0
        self.timer = QTimer()
        self.audio_file_path = None
        self.processed_audio_path = None
        self.icon_size = QSize(45, 45)
        self.tifinagh_converter = TifinaghConverter()

        self.set_background_image()
        self.init_ui()

    def set_background_image(self):
        background_path = "background.jpg"
        if os.path.exists(background_path):
            palette = QPalette()
            pixmap = QPixmap(background_path)
            scaled_pixmap = pixmap.scaled(self.size(), Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation)
            palette.setBrush(QPalette.Background, QBrush(scaled_pixmap))
            self.setPalette(palette)
        else:
            self.setStyleSheet("""
                QMainWindow {
                    background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                        stop:0 #020024, stop:0.5 #090979, stop:1 #00d4ff);
                }
            """)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        background_path = "background.jpg"
        if os.path.exists(background_path):
            palette = QPalette()
            pixmap = QPixmap(background_path)
            scaled_pixmap = pixmap.scaled(self.size(), Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation)
            palette.setBrush(QPalette.Background, QBrush(scaled_pixmap))
            self.setPalette(palette)

    def init_ui(self):
        self.menuBar().setNativeMenuBar(False)
        self.menuBar().setStyleSheet("""
            QMenuBar {
                background-color: #0F172A;
                color: #F8FAFC;
                border: none;
                padding: 4px;
            }
            QMenuBar::item {
                background: transparent;
                padding: 6px 10px;
                border-radius: 4px;
            }
            QMenuBar::item:selected {
                background-color: #2563EB;
            }
            QMenu {
                background-color: #111827;
                color: #F8FAFC;
                border: 1px solid #334155;
            }
            QMenu::item:selected {
                background-color: #2563EB;
            }
        """)

        file_menu = QMenu("File", self)
        exit_action = QAction("Exit", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        self.menuBar().addMenu(file_menu)

        edit_menu = QMenu("Edit", self)
        clear_action = QAction("Clear", self)
        clear_action.setShortcut("Ctrl+L")
        clear_action.triggered.connect(self.clear_interface)
        edit_menu.addAction(clear_action)

        validate_action = QAction("Validate", self)
        validate_action.setShortcut("Ctrl+V")
        validate_action.triggered.connect(self.validate_transcription)
        edit_menu.addAction(validate_action)

        edit_text_action = QAction("Edit transcription", self)
        edit_text_action.setShortcut("Ctrl+E")
        edit_text_action.triggered.connect(self.edit_transcription)
        edit_menu.addAction(edit_text_action)
        self.menuBar().addMenu(edit_menu)

        transcribe_menu = QMenu("Transcription", self)
        start_transcription_action = QAction("Start transcription", self)
        start_transcription_action.setShortcut("Ctrl+T")
        start_transcription_action.triggered.connect(self.start_transcription)
        transcribe_menu.addAction(start_transcription_action)

        process_audio_action = QAction("Process audio", self)
        process_audio_action.setShortcut("Ctrl+P")
        process_audio_action.triggered.connect(self.process_audio)
        transcribe_menu.addAction(process_audio_action)

        play_original_action = QAction("Play original audio", self)
        play_original_action.setShortcut("Ctrl+R")
        play_original_action.triggered.connect(lambda: self.play_file_external(self.audio_file_path))
        transcribe_menu.addAction(play_original_action)
        self.menuBar().addMenu(transcribe_menu)

        tools_menu = QMenu("Tools", self)
        open_audio_action = QAction("Open audio file", self)
        open_audio_action.setShortcut("Ctrl+O")
        open_audio_action.triggered.connect(self.open_audio_file)
        tools_menu.addAction(open_audio_action)
        self.menuBar().addMenu(tools_menu)

        help_menu = QMenu("Help", self)
        about_action = QAction("About", self)
        about_action.setShortcut("F1")
        about_action.triggered.connect(self.show_about_dialog)
        help_menu.addAction(about_action)
        
        guidelines_action = QAction("Guidelines", self)
        guidelines_action.setShortcut("F2")
        guidelines_action.triggered.connect(self.show_guidelines_dialog)
        help_menu.addAction(guidelines_action)
        
        self.menuBar().addMenu(help_menu)

        central_widget = QWidget()
        central_widget.setStyleSheet("""
            QWidget { 
                background-color: rgba(15, 23, 42, 0.90);
                border-radius: 15px; 
                margin: 10px; 
                color: #F1F5F9;
            }
            QLabel { 
                background-color: transparent; 
                color: #E2E8F0;
                font-weight: bold; 
            }
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #3B82F6, stop:1 #2563EB);
                color: white;
                border: 1px solid rgba(255, 255, 255, 0.18);
                border-radius: 12px;
                padding: 10px 16px;
                min-height: 48px;
                font-size: 11pt;
                font-weight: 600;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #60A5FA, stop:1 #3B82F6);
            }
            QPushButton:pressed {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #1D4ED8, stop:1 #1E40AF);
            }
            QPushButton:disabled {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 rgba(71, 85, 105, 0.65), stop:1 rgba(51, 65, 85, 0.75));
                color: rgba(241, 245, 249, 0.6);
                border: 1px solid rgba(148, 163, 184, 0.25);
            }
            #playButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #10B981, stop:1 #059669);
            }
            #playButton:hover { background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #34D399, stop:1 #10B981); }
            #playButton:pressed { background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #047857, stop:1 #065F46); }
            #playButton:disabled { background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 rgba(71, 85, 105, 0.65), stop:1 rgba(51, 65, 85, 0.75)); }
            #playProcessedButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #8B5CF6, stop:1 #7C3AED);
            }
            #playProcessedButton:hover { background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #A78BFA, stop:1 #8B5CF6); }
            #playProcessedButton:pressed { background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #6D28D9, stop:1 #5B21B6); }
            #playProcessedButton:disabled { background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 rgba(71, 85, 105, 0.65), stop:1 rgba(51, 65, 85, 0.75)); }
            #clearButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #EF4444, stop:1 #DC2626);
            }
            #clearButton:hover { background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #F87171, stop:1 #EF4444); }
            #clearButton:pressed { background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #B91C1C, stop:1 #991B1B); }
            #validateButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #22C55E, stop:1 #16A34A);
            }
            #validateButton:hover { background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #4ADE80, stop:1 #22C55E); }
            #validateButton:pressed { background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #15803D, stop:1 #166534); }
            #editButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #0EA5E9, stop:1 #0284C7);
            }
            #editButton:hover { background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #38BDF8, stop:1 #0EA5E9); }
            #editButton:pressed { background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #0369A1, stop:1 #075985); }
            QTextEdit {
                background-color: rgba(30, 41, 59, 0.95);
                border: 2px solid #475569;
                border-radius: 8px; 
                padding: 10px; 
                font-size: 11pt;
                color: #FFFFFF;
                selection-background-color: #3B82F6;
                min-height: 110px;
                max-height: 220px;
                overflow: hidden;
            }
            #tifinaghTextEdit { font-size: 20pt; }
            QComboBox {
                background-color: rgba(30, 41, 59, 0.9);
                color: white;
                border: 2px solid #475569;
                border-radius: 6px; 
                padding: 5px; 
                font-size: 10pt;
            }
            QComboBox QAbstractItemView {
                background-color: #0F172A;
                color: white;
                selection-background-color: #3B82F6;
            }
            QProgressBar {
                background-color: rgba(15, 23, 42, 0.9); 
                border: 1px solid #475569;
                border-radius: 6px; 
                text-align: center;
                color: white;
            }
            QProgressBar::chunk { background-color: #3B82F6; border-radius: 6px; }
        """)
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # Header
        header_widget = QWidget()
        header_widget.setStyleSheet("background-color: rgba(30, 41, 59, 0.95); border-radius: 15px; margin: 5px; padding: 10px; border: 1px solid #475569;")
        header_layout_inner = QHBoxLayout(header_widget)

        self.left_logo_label = QLabel()
        left_logo_path = "logo1.png"
        if os.path.exists(left_logo_path):
            logo_pixmap = QPixmap(left_logo_path)
            scaled_logo = logo_pixmap.scaled(120, 120, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.left_logo_label.setPixmap(scaled_logo)
        else:
            self.left_logo_label.setText("Logo 1")
            self.left_logo_label.setStyleSheet("font-size: 16px; color: #3B82F8;")
        self.left_logo_label.setAlignment(Qt.AlignCenter)
        header_layout_inner.addWidget(self.left_logo_label, 0, Qt.AlignLeft)

        title_label = QLabel("Asemmas n unadi deg tutlayt d yidles n tmaziɣt")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("font-size: 24pt; font-weight: bold; color: #FFFFFF; background-color: transparent; padding: 10px;")
        header_layout_inner.addWidget(title_label, 1)

        self.right_logo_label = QLabel()
        right_logo_path = "crlca.png"
        if os.path.exists(right_logo_path):
            logo_pixmap = QPixmap(right_logo_path)
            scaled_logo = logo_pixmap.scaled(120, 120, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.right_logo_label.setPixmap(scaled_logo)
        else:
            self.right_logo_label.setText("CRLCA")
            self.right_logo_label.setStyleSheet("font-size: 16px; color: #3B82F8;")
        self.right_logo_label.setAlignment(Qt.AlignCenter)
        header_layout_inner.addWidget(self.right_logo_label, 0, Qt.AlignRight)

        main_layout.addWidget(header_widget)

        # Model
        model_layout = QHBoxLayout()
        model_layout.addStretch()
        model_label = QLabel("Model:")
        self.model_combo = QComboBox()

        model_options = [
            "Alili113/whisper-tamazight-final1",
        ]
        self.model_combo.addItems(model_options)
        self.model_combo.setCurrentText("Alili113/whisper-tamazight-final1")

        model_layout.addWidget(model_label)
        model_layout.addWidget(self.model_combo)
        model_layout.addStretch()
        main_layout.addLayout(model_layout)

        # Controls
        control_layout = QHBoxLayout()
        control_layout.setSpacing(10)
        control_layout.setContentsMargins(8, 8, 8, 8)
        control_layout.addStretch(1)

        self.record_button = QPushButton(" Start Recording")
        self.record_button.clicked.connect(self.toggle_recording)
        self.record_button.setIconSize(self.icon_size)
        self.record_button.setIcon(QIcon(os.path.join("icons", "record.png")))
        self.record_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.record_button.setMinimumHeight(56)
        self.record_button.setMinimumWidth(240)
        self.record_button.setMaximumWidth(320)
        control_layout.addWidget(self.record_button)

        self.time_label = QLabel("00:00")
        self.time_label.setStyleSheet("font-size: 16pt; background-color: rgba(15, 23, 42, 0.8); color: #3B82F6; border-radius: 8px; padding: 8px 15px; border: 1px solid #475569;")
        self.time_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.time_label.setMinimumWidth(120)
        control_layout.addWidget(self.time_label)

        control_layout.addSpacing(40)

        self.open_button = QPushButton(" Open Audio File")
        self.open_button.clicked.connect(self.open_audio_file)
        self.open_button.setIconSize(self.icon_size)
        self.open_button.setIcon(QIcon(os.path.join("icons", "folder_open.png")))
        self.open_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.open_button.setMinimumHeight(56)
        self.open_button.setMinimumWidth(240)
        self.open_button.setMaximumWidth(320)
        control_layout.addWidget(self.open_button)
        self.file_button = self.open_button

        self.process_audio_button = QPushButton(" Process Audio")
        self.process_audio_button.clicked.connect(self.process_audio)
        self.process_audio_button.setEnabled(False)
        self.process_audio_button.setIconSize(self.icon_size)
        self.process_audio_button.setIcon(QIcon(os.path.join("icons", "save.png")))
        self.process_audio_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.process_audio_button.setMinimumHeight(56)
        self.process_audio_button.setMinimumWidth(240)
        self.process_audio_button.setMaximumWidth(320)
        control_layout.addWidget(self.process_audio_button)

        self.transcribe_button = QPushButton(" Transcribe Audio")
        self.transcribe_button.clicked.connect(self.start_transcription)
        self.transcribe_button.setEnabled(False)
        self.transcribe_button.setIconSize(self.icon_size)
        self.transcribe_button.setIcon(QIcon(os.path.join("icons", "transcribe.png")))
        self.transcribe_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.transcribe_button.setMinimumHeight(56)
        self.transcribe_button.setMinimumWidth(240)
        self.transcribe_button.setMaximumWidth(320)
        control_layout.addWidget(self.transcribe_button)

        control_layout.addStretch(1)
        main_layout.addLayout(control_layout)

        # Playback row with action buttons
        playback_layout = QHBoxLayout()
        playback_layout.setSpacing(10)
        playback_layout.setContentsMargins(8, 6, 8, 8)
        playback_layout.setAlignment(Qt.AlignTop)

        playback_layout.addStretch(1)

        self.play_button = QPushButton(" Play Original")
        self.play_button.setObjectName("playButton")
        self.play_button.clicked.connect(lambda: self.play_file_external(self.audio_file_path))
        self.play_button.setEnabled(False)
        self.play_button.setIconSize(self.icon_size)
        self.play_button.setIcon(QIcon(os.path.join("icons", "audio_file.png")))
        self.play_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.play_button.setMinimumHeight(56)
        self.play_button.setMinimumWidth(220)
        self.play_button.setMaximumWidth(320)
        playback_layout.addWidget(self.play_button, 1)
        
        self.play_processed_button = QPushButton(" Play Processed")
        self.play_processed_button.setObjectName("playProcessedButton")
        self.play_processed_button.clicked.connect(lambda: self.play_file_external(self.processed_audio_path))
        self.play_processed_button.setEnabled(False)
        self.play_processed_button.setIconSize(self.icon_size)
        self.play_processed_button.setIcon(QIcon(os.path.join("icons", "audio_file.png")))
        self.play_processed_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.play_processed_button.setMinimumHeight(56)
        self.play_processed_button.setMinimumWidth(220)
        self.play_processed_button.setMaximumWidth(320)
        playback_layout.addWidget(self.play_processed_button, 1)

        self.clear_button = QPushButton("Clear")
        self.clear_button.setObjectName("clearButton")
        self.clear_button.clicked.connect(self.clear_interface)
        self.clear_button.setIconSize(self.icon_size)
        self.clear_button.setIcon(QIcon(os.path.join("icons", "clear.png")))
        self.clear_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.clear_button.setMinimumHeight(56)
        self.clear_button.setMinimumWidth(220)
        self.clear_button.setMaximumWidth(320)
        playback_layout.addWidget(self.clear_button, 1)

        self.validate_button = QPushButton("Validate")
        self.validate_button.setObjectName("validateButton")
        self.validate_button.clicked.connect(self.validate_transcription)
        self.validate_button.setIconSize(self.icon_size)
        self.validate_button.setIcon(QIcon(os.path.join("icons", "save.png")))
        self.validate_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.validate_button.setMinimumHeight(56)
        self.validate_button.setMinimumWidth(220)
        self.validate_button.setMaximumWidth(320)
        playback_layout.addWidget(self.validate_button, 1)

        self.edit_button = QPushButton("Edit")
        self.edit_button.setObjectName("editButton")
        self.edit_button.clicked.connect(self.edit_transcription)
        self.edit_button.setIconSize(self.icon_size)
        self.edit_button.setIcon(QIcon(os.path.join("icons", "edit.png")))
        self.edit_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.edit_button.setMinimumHeight(56)
        self.edit_button.setMinimumWidth(220)
        self.edit_button.setMaximumWidth(320)
        playback_layout.addWidget(self.edit_button, 1)

        playback_layout.addStretch(1)
        main_layout.addLayout(playback_layout)

        self.file_label = QLabel("No file selected")
        self.file_label.setStyleSheet("background-color: rgba(15, 23, 42, 0.6); border-radius: 6px; padding: 8px 15px; border: 1px solid #475569; color: #94A3B8;")
        self.file_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.file_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(self.file_label)

        # Progress Bars
        progress_layout = QVBoxLayout()
        self.process_progress = QProgressBar()
        self.process_progress.setVisible(False)
        self.process_progress.setFormat("Audio processing: %p%")
        progress_layout.addWidget(self.process_progress)
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setFormat("Transcription: %p%")
        progress_layout.addWidget(self.progress_bar)
        main_layout.addLayout(progress_layout)

        # Outputs
        self.tifinagh_text_edit = QTextEdit()
        self.tifinagh_text_edit.setObjectName("tifinaghTextEdit")
        self.tifinagh_text_edit.setPlaceholderText("Tifinaɣ transcription will appear here...")
        self.tifinagh_text_edit.setReadOnly(True)
        tifinagh_font = QFont()
        tifinagh_font.setPointSize(22)
        self.tifinagh_text_edit.setFont(tifinagh_font)
        self.tifinagh_text_edit.setMinimumHeight(120)
        self.tifinagh_text_edit.setMaximumHeight(180)
        main_layout.addWidget(self.tifinagh_text_edit)

        self.text_edit = QTextEdit()
        self.text_edit.setPlaceholderText("Latin transcription will appear here...")
        self.text_edit.setReadOnly(True)
        latin_font = QFont()
        latin_font.setPointSize(14)
        self.text_edit.setFont(latin_font)
        self.text_edit.setMinimumHeight(120)
        self.text_edit.setMaximumHeight(180)
        main_layout.addWidget(self.text_edit)

        self.timer.timeout.connect(self.update_timer)

    def show_about_dialog(self):
        dialog = AboutDialog(self)
        dialog.exec_()

    def show_guidelines_dialog(self):
        dialog = GuidelinesDialog(self)
        dialog.exec_()

    def play_file_external(self, file_path):
        if not file_path or not os.path.exists(file_path):
            QMessageBox.warning(self, "Error", "Audio file not found or deleted.")
            return

        playback_error = None
        try:
            audio_data, sr = sf.read(file_path, dtype='float32')
            if audio_data.ndim > 1:
                audio_data = np.mean(audio_data, axis=1)
            sd.play(audio_data, sr)
            sd.wait()
            return
        except Exception as e:
            playback_error = f"sounddevice: {e}"

        try:
            if platform.system() == 'Windows':
                os.startfile(file_path)
            elif platform.system() == 'Darwin':
                subprocess.run(('open', file_path), check=True)
            else:
                # Use a simple audio player fallback on Linux
                if shutil.which('paplay'):
                    subprocess.run(('paplay', file_path), check=True)
                elif shutil.which('aplay'):
                    subprocess.run(('aplay', file_path), check=True)
                else:
                    subprocess.run(('xdg-open', file_path), check=True)
            return
        except Exception as e:
            error_message = f"Unable to play the audio file.\n{playback_error or ''}\n{e}"
            QMessageBox.critical(self, "Playback error", error_message)

    def update_timer(self):
        pass 

    def process_audio(self):
        if not self.audio_file_path:
            QMessageBox.warning(self, "Warning", "Select audio file first.")
            return
        
        self.process_progress.setVisible(True)
        self.process_progress.setValue(0)
        self.process_audio_button.setEnabled(False)
        self.play_button.setEnabled(False)
        self.play_processed_button.setEnabled(False)
        
        self.audio_processor = AudioProcessor(self.audio_file_path)
        self.audio_processor.progress.connect(self.update_process_progress)
        self.audio_processor.processed.connect(self.on_audio_processed)
        self.audio_processor.error.connect(self.on_process_error)
        self.audio_processor.start()

    def update_process_progress(self, value):
        self.process_progress.setValue(value)

    def on_audio_processed(self, processed_file_path):
        self.processed_audio_path = processed_file_path
        self.process_progress.setVisible(False)
        self.process_audio_button.setEnabled(True)
        self.play_button.setEnabled(True)
        self.play_processed_button.setEnabled(True)
        QMessageBox.information(self, "Success", "Audio processing completed.")

    def on_process_error(self, error_message):
        self.process_progress.setVisible(False)
        self.process_audio_button.setEnabled(True)
        self.play_button.setEnabled(True)
        QMessageBox.critical(self, "Processing error", error_message)

    def toggle_recording(self):
        if not self.recorder or not self.recorder.recording:
            self.start_recording()
        else:
            self.stop_recording()

    def start_recording(self):
        self.text_edit.clear()
        self.tifinagh_text_edit.clear()
        self.file_label.setText("Recording...")
        self.clean_old_files()

        self.audio_file_path = None
        self.processed_audio_path = None
        self.toggle_buttons(False)
        self.progress_bar.setVisible(False)
        self.process_progress.setVisible(False)

        self.recorder = AudioRecorder()
        self.recorder.update_time.connect(self.update_recording_time)
        self.recorder.finished.connect(self.on_recording_finished)
        self.recorder.start()

        self.record_button.setText("⏹️ Stop Recording")
        self.recording_time = 0
        self.timer.start(1000)

    def stop_recording(self):
        if self.recorder and self.recorder.recording:
            self.recorder.stop()
            self.recorder.wait()
            self.timer.stop()
            self.record_button.setText("🎤 Start Recording")

    def toggle_buttons(self, enabled):
        self.transcribe_button.setEnabled(enabled)
        self.play_button.setEnabled(enabled)
        self.play_processed_button.setEnabled(enabled)
        self.process_audio_button.setEnabled(enabled)
        self.file_button.setEnabled(not self.recorder or not self.recorder.recording)

    def update_recording_time(self, seconds):
        self.recording_time = seconds
        mins = seconds // 60
        secs = seconds % 60
        self.time_label.setText(f"{mins:02d}:{secs:02d}")

    def on_recording_finished(self, file_path):
        self.record_button.setText("🎤 Start Recording")
        
        if file_path.startswith("ERROR"):
            QMessageBox.critical(self, "Recording error", file_path)
            self.file_label.setText("Recording failed")
        else:
            self.audio_file_path = file_path
            self.file_label.setText(f"Recorded: {os.path.basename(file_path)}")
            self.toggle_buttons(True)
            self.play_processed_button.setEnabled(False)
        self.time_label.setText("00:00")

    def open_audio_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Audio", "", "Audio Files (*.wav *.flac *.mp3 *.ogg *.m4a)")
        if file_path:
            self.audio_file_path = file_path
            self.file_label.setText(f"File: {os.path.basename(file_path)}")
            self.text_edit.clear()
            self.tifinagh_text_edit.clear()
            self.toggle_buttons(True)
            self.processed_audio_path = None
            self.play_processed_button.setEnabled(False)

    def start_transcription(self):
        if not self.audio_file_path:
            QMessageBox.warning(self, "Warning", "No audio file.")
            return

        self.toggle_buttons(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.text_edit.setPlaceholderText("Transcription in progress (CPU - this may take a few moments)...")
        self.tifinagh_text_edit.setPlaceholderText("Transcription in progress...")

        model_name = self.model_combo.currentText()
        self.transcription_thread = TranscriptionThread(self.audio_file_path, model_name)
        self.transcription_thread.progress_update.connect(self.update_progress)
        self.transcription_thread.transcription_complete.connect(self.on_transcription_complete)
        self.transcription_thread.start()

    def update_progress(self, value):
        self.progress_bar.setValue(value)

    def on_transcription_complete(self, latin_transcription):
        self.text_edit.setPlainText(latin_transcription)
        
        valid = not latin_transcription.startswith("Error")

        if valid:
            tifinagh_transcription = self.tifinagh_converter.convert(latin_transcription)
            self.tifinagh_text_edit.setPlainText(tifinagh_transcription)
        else:
            self.tifinagh_text_edit.setPlainText("Conversion error.")

        self.progress_bar.setVisible(False)
        self.toggle_buttons(True)
        if not self.processed_audio_path:
            self.play_processed_button.setEnabled(False)
        
        if not valid:
            QMessageBox.critical(self, "Transcription error", latin_transcription)
            print(f"Received error: {latin_transcription}")

    def validate_transcription(self):
        if not self.audio_file_path:
            QMessageBox.warning(self, "Validation", "No audio file to save.")
            return

        latin_text = self.text_edit.toPlainText().strip()
        tifinagh_text = self.tifinagh_text_edit.toPlainText().strip()
        if not latin_text and not tifinagh_text:
            QMessageBox.warning(self, "Validation", "No transcription to validate.")
            return

        project_dir = os.path.dirname(os.path.abspath(__file__))
        audios_dir = os.path.join(project_dir, "audios")
        os.makedirs(audios_dir, exist_ok=True)

        audio_name = f"recording_{int(time.time())}_{os.path.basename(self.audio_file_path)}"
        dest_audio_path = os.path.join(audios_dir, audio_name)

        try:
            shutil.copy2(self.audio_file_path, dest_audio_path)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Unable to save the audio: {e}")
            return

        csv_path = os.path.join(project_dir, "recordings.csv")
        csv_audio_path = os.path.join("audios", audio_name)
        file_exists = os.path.exists(csv_path)

        try:
            with open(csv_path, "a", encoding="utf-8", newline="") as csv_file:
                writer = csv.writer(csv_file, delimiter=";")
                if not file_exists:
                    writer.writerow(["audios_path", "transcription_latin", "transcription_tifinagh"])
                writer.writerow([csv_audio_path, latin_text, tifinagh_text])
        except Exception as e:
            QMessageBox.critical(self, "CSV error", f"Unable to write the CSV file: {e}")
            return

        QMessageBox.information(
            self,
            "Validation",
            f"Audio saved to: {dest_audio_path}\nCSV updated: {csv_path}"
        )

    def edit_transcription(self):
        current_text = self.text_edit.toPlainText().strip()
        if not current_text:
            QMessageBox.warning(self, "Edit", "No transcription to edit.")
            return

        dialog = QDialog(self)
        dialog.setWindowTitle("Edit transcription")
        dialog.resize(900, 420)
        dialog.setMinimumSize(700, 320)

        layout = QVBoxLayout(dialog)
        label = QLabel("Edit the text:")
        label.setStyleSheet("font-weight: bold; color: #E2E8F0;")

        editor = QTextEdit()
        editor.setPlainText(current_text)
        editor.setMinimumHeight(240)
        editor.setStyleSheet("""
            QTextEdit {
                background-color: rgba(30, 41, 59, 0.95);
                border: 1px solid #475569;
                border-radius: 8px;
                color: white;
                padding: 10px;
                font-size: 11pt;
            }
        """)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)

        layout.addWidget(label)
        layout.addWidget(editor, 1)
        layout.addWidget(buttons)

        if dialog.exec_():
            new_text = editor.toPlainText()
            if new_text != current_text:
                self.text_edit.setPlainText(new_text)
                self.tifinagh_text_edit.setPlainText(self.tifinagh_converter.convert(new_text))

    def save_recording(self, user_id: int, duration: float):
        latin_text = self.text_edit.toPlainText().strip()
        tifinagh_text = self.tifinagh_text_edit.toPlainText().strip()

        if not self.audio_file_path or not latin_text or not tifinagh_text:
            QMessageBox.warning(self, "Warning", "Audio and transcriptions must be present to save.")
            return

        QMessageBox.information(self, "Success", "Recording is ready, but database saving is not configured.")

    def clean_old_files(self):
        pass

    def clear_interface(self):
        self.text_edit.clear()
        self.tifinagh_text_edit.clear()
        self.file_label.setText("No file")
        self.time_label.setText("00:00")
        self.audio_file_path = None
        self.processed_audio_path = None
        self.toggle_buttons(False)
        self.record_button.setEnabled(True)
        self.file_button.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.process_progress.setVisible(False)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = AudioTranscriptionApp()
    window.show()
    sys.exit(app.exec_())