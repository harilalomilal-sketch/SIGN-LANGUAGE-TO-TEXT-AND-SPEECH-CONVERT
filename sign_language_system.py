import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import cv2
import mediapipe as mp
import numpy as np
import threading
import time
import pyttsx3
import pickle
import os
import json
from PIL import Image, ImageTk
from collections import deque, Counter
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# 30 SENTENCES DICTIONARY
# Each key = sentence label, value = display text + description
# ─────────────────────────────────────────────────────────────────────────────
SENTENCES = {
    0:  {"text": "Hello",                    "gesture": "Open hand wave"},
    1:  {"text": "Thank you",                "gesture": "Flat hand from chin forward"},
    2:  {"text": "Please",                   "gesture": "Circular motion on chest"},
    3:  {"text": "Sorry",                    "gesture": "Fist circular on chest"},
    4:  {"text": "Yes",                      "gesture": "Fist nodding up-down"},
    5:  {"text": "No",                       "gesture": "Index + middle finger together"},
    6:  {"text": "Help me",                  "gesture": "Thumbs up lifted by other hand"},
    7:  {"text": "I love you",               "gesture": "Thumb + index + pinky extended"},
    8:  {"text": "Good morning",             "gesture": "Flat hand rising motion"},
    9:  {"text": "Good night",               "gesture": "Hand descending motion"},
    10: {"text": "How are you?",             "gesture": "Both hands bent, twist outward"},
    11: {"text": "I am fine",               "gesture": "Middle finger tap chest twice"},
    12: {"text": "My name is...",           "gesture": "H-hand taps twice"},
    13: {"text": "Nice to meet you",        "gesture": "Handshake motion"},
    14: {"text": "Where is the bathroom?",  "gesture": "R-hand shaken sideways"},
    15: {"text": "I need water",            "gesture": "W-hand taps chin"},
    16: {"text": "I am hungry",             "gesture": "C-hand down chest"},
    17: {"text": "I am tired",              "gesture": "Bent hands drop to chest"},
    18: {"text": "Call the doctor",         "gesture": "D-hand at ear then point"},
    19: {"text": "I need help",             "gesture": "A-hand lifted on palm"},
    20: {"text": "Stop",                    "gesture": "Edge of palm hits flat hand"},
    21: {"text": "Come here",               "gesture": "Index finger beckoning"},
    22: {"text": "Go away",                 "gesture": "Flat hand pushing outward"},
    23: {"text": "Wait",                    "gesture": "Spread fingers wiggle"},
    24: {"text": "Repeat please",           "gesture": "Index circles around other hand"},
    25: {"text": "I don't understand",      "gesture": "Index finger twist at temple"},
    26: {"text": "Slow down",               "gesture": "One hand slides over other slowly"},
    27: {"text": "That is correct",         "gesture": "Index fingers touch"},
    28: {"text": "I am deaf",               "gesture": "Index touches ear then mouth"},
    29: {"text": "Emergency!",              "gesture": "E-hand shaken rapidly"},
}

# ─────────────────────────────────────────────────────────────────────────────
# FEATURE EXTRACTOR
# ─────────────────────────────────────────────────────────────────────────────
class FeatureExtractor:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_styles = mp.solutions.drawing_styles

    def extract(self, frame):
        """Returns (landmarks_array, annotated_frame, hand_detected)"""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)
        annotated = frame.copy()
        features = np.zeros(21 * 3 * 2)  # 2 hands × 21 landmarks × (x,y,z)
        detected = False

        if results.multi_hand_landmarks:
            detected = True
            for i, hand_lm in enumerate(results.multi_hand_landmarks[:2]):
                self.mp_draw.draw_landmarks(
                    annotated, hand_lm,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_styles.get_default_hand_landmarks_style(),
                    self.mp_styles.get_default_hand_connections_style()
                )
                # Normalize landmarks relative to wrist
                wrist = hand_lm.landmark[0]
                for j, lm in enumerate(hand_lm.landmark):
                    base = i * 63 + j * 3
                    features[base]   = lm.x - wrist.x
                    features[base+1] = lm.y - wrist.y
                    features[base+2] = lm.z - wrist.z

        return features, annotated, detected

    def close(self):
        self.hands.close()


# ─────────────────────────────────────────────────────────────────────────────
# DEMO CLASSIFIER (rule-based simulation for demonstration)
# Replace with a real trained model for production use
# ─────────────────────────────────────────────────────────────────────────────
class DemoClassifier:
    """
    Demo-mode classifier: cycles through sentences based on hand pose patterns.
    For real use, train a model using the DataCollector and swap this class.
    """
    def __init__(self):
        self.last_label = -1
        self.hold_count = 0
        self.change_threshold = 30  # frames to hold before accepting

    def predict(self, features):
        """Returns (label, confidence) or (-1, 0) if no hand."""
        if np.sum(np.abs(features)) < 0.01:
            return -1, 0.0

        # Use feature vector statistics to deterministically map to a gesture
        # This is purely for demonstration — replace with real model predict()
        feature_sum = float(np.sum(np.abs(features[:21])))
        label = int(abs(feature_sum * 100)) % 30
        confidence = min(0.5 + abs(features[3]) * 2, 0.99)
        return label, round(confidence, 2)


# ─────────────────────────────────────────────────────────────────────────────
# TRAINED MODEL CLASSIFIER (used after data collection + training)
# ─────────────────────────────────────────────────────────────────────────────
class TrainedClassifier:
    def __init__(self, model_path="sign_model.pkl"):
        with open(model_path, "rb") as f:
            self.model = pickle.load(f)

    def predict(self, features):
        proba = self.model.predict_proba([features])[0]
        label = int(np.argmax(proba))
        confidence = float(proba[label])
        return label, round(confidence, 2)


# ─────────────────────────────────────────────────────────────────────────────
# TTS ENGINE  -- uses Windows SAPI5 directly via win32com
# Each speak() call creates a fresh SAPI voice = no state issues
# ─────────────────────────────────────────────────────────────────────────────
import queue as _queue
import subprocess

class TTSEngine:
    def __init__(self):
        self._rate   = 150
        self._volume = 100
        self._q      = _queue.Queue()
        self._worker = threading.Thread(target=self._loop, daemon=True)
        self._worker.start()

    def _speak_once(self, text):
        """Speak using PowerShell SAPI - most reliable on Windows, no state issues."""
        try:
            cmd = (
                f'Add-Type -AssemblyName System.Speech; '
                f'$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; '
                f'$s.Rate = {int((self._rate - 150) / 30)}; '
                f'$s.Volume = {self._volume}; '
                f'$s.Speak([string]"{text}"); '
                f'$s.Dispose()'
            )
            subprocess.Popen(
                ["powershell", "-WindowStyle", "Hidden", "-Command", cmd],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=0x08000000  # CREATE_NO_WINDOW
            )
        except Exception as e:
            print(f"[TTS ERROR] {e}")
            # fallback to pyttsx3
            try:
                engine = pyttsx3.init()
                engine.say(text)
                engine.runAndWait()
            except Exception:
                pass

    def _loop(self):
        while True:
            item = self._q.get()
            if item is None:
                break
            self._speak_once(item)

    def speak(self, text):
        # Drain queue — only speak the latest word
        while not self._q.empty():
            try:
                self._q.get_nowait()
            except Exception:
                break
        self._q.put(text)
        print(f"[TTS] Queued: {text}")

    def set_rate(self, rate):
        self._rate = int(float(rate))

    def set_volume(self, vol):
        self._volume = int(float(vol) * 100)

    def stop(self):
        self._q.put(None)


# ─────────────────────────────────────────────────────────────────────────────
# DATA COLLECTOR (for training a real model)
# ─────────────────────────────────────────────────────────────────────────────
class DataCollector:
    def __init__(self, save_dir="training_data"):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.X = []
        self.y = []
        self._load_existing()

    def _load_existing(self):
        path = os.path.join(self.save_dir, "data.pkl")
        if os.path.exists(path):
            with open(path, "rb") as f:
                data = pickle.load(f)
                self.X = data.get("X", [])
                self.y = data.get("y", [])

    def add_sample(self, features, label):
        self.X.append(features.copy())
        self.y.append(label)

    def save(self):
        path = os.path.join(self.save_dir, "data.pkl")
        with open(path, "wb") as f:
            pickle.dump({"X": self.X, "y": self.y}, f)
        return len(self.X)

    def train_model(self, save_path="sign_model.pkl"):
        if len(self.X) < 10:
            return False, "Not enough data (need at least 10 samples)"
        try:
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.preprocessing import StandardScaler
            from sklearn.pipeline import Pipeline

            X = np.array(self.X)
            y = np.array(self.y)
            model = Pipeline([
                ("scaler", StandardScaler()),
                ("clf", RandomForestClassifier(n_estimators=200, random_state=42))
            ])
            model.fit(X, y)
            with open(save_path, "wb") as f:
                pickle.dump(model, f)
            return True, f"Model trained on {len(X)} samples and saved to {save_path}"
        except Exception as e:
            return False, str(e)

    def get_counts(self):
        return Counter(self.y)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN APPLICATION
# ─────────────────────────────────────────────────────────────────────────────
class SignLanguageApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("🤟 Sign Language to Speech Converter")
        self.root.configure(bg="#1a1a2e")
        self.root.geometry("1280x800")
        self.root.minsize(1100, 700)

        # Core components
        self.extractor = FeatureExtractor()
        self.tts = TTSEngine()
        self.collector = DataCollector()

        # State
        self.cap = None
        self.running = False
        self.training_mode = False
        self.current_label = 0
        self.collect_active = False
        self.collect_count = 0
        self.collect_target = 50
        self.prediction_buffer = deque(maxlen=15)
        self.confidence_threshold = 0.55
        self.last_spoken = ""
        self.last_spoken_time = 0
        self.speak_cooldown = 2.5
        self.auto_speak = tk.BooleanVar(value=True)
        self._prev_stable = -1   # last spoken gesture label
        self.history = []

        # Load classifier
        if os.path.exists("sign_model.pkl"):
            try:
                self.classifier = TrainedClassifier()
            except Exception:
                self.classifier = DemoClassifier()
        else:
            self.classifier = DemoClassifier()

        self._build_ui()
        self._bind_keys()
        self._start_camera()

    # ── UI CONSTRUCTION ────────────────────────────────────────────────────
    def _build_ui(self):
        # Apply styles first
        self._style_progressbar()

        # Root uses pack for reliable layout
        self.root.configure(bg="#1a1a2e")

        # ── TOP TITLE BAR ──
        title_bar = tk.Frame(self.root, bg="#0d0d1a", height=50)
        title_bar.pack(side="top", fill="x")
        title_bar.pack_propagate(False)

        tk.Label(title_bar, text="🤟  Sign Language to Speech Converter",
                 font=("Courier", 15, "bold"), fg="#00d4aa", bg="#0d0d1a"
                 ).pack(side="left", padx=16, pady=10)

        self.mode_label = tk.Label(title_bar, text="● RECOGNITION MODE",
                                   font=("Courier", 10, "bold"),
                                   fg="#00d4aa", bg="#0d0d1a")
        self.mode_label.pack(side="right", padx=16)

        self.fps_var = tk.StringVar(value="FPS: --")
        tk.Label(title_bar, textvariable=self.fps_var,
                 font=("Courier", 10), fg="#555", bg="#0d0d1a"
                 ).pack(side="right", padx=8)

        # ── MAIN BODY ──
        body = tk.Frame(self.root, bg="#1a1a2e")
        body.pack(side="top", fill="both", expand=True, padx=10, pady=6)

        # LEFT: camera + controls (fixed 660px wide)
        left = tk.Frame(body, bg="#1a1a2e", width=660)
        left.pack(side="left", fill="both", expand=False, padx=(0, 6))
        left.pack_propagate(False)

        # Camera feed box (fixed 640x480)
        cam_outer = tk.Frame(left, bg="#0f3460", bd=2, relief="solid",
                             width=640, height=480)
        cam_outer.pack(side="top", padx=0, pady=0)
        cam_outer.pack_propagate(False)

        self.video_label = tk.Label(cam_outer, bg="#0d0d1a",
                                    text="📷  Starting camera...",
                                    font=("Courier", 13), fg="#555",
                                    width=640, height=480)
        self.video_label.pack(fill="both", expand=True)

        # Recognized sentence panel
        rec_panel = tk.Frame(left, bg="#16213e", height=56)
        rec_panel.pack(side="top", fill="x", pady=(6, 0))
        rec_panel.pack_propagate(False)

        tk.Label(rec_panel, text="RECOGNIZED",
                 font=("Courier", 8, "bold"), fg="#555", bg="#16213e"
                 ).place(x=10, y=4)

        self.recognized_var = tk.StringVar(value="— show your hand to the camera —")
        tk.Label(rec_panel, textvariable=self.recognized_var,
                 font=("Courier", 16, "bold"), fg="#00d4aa", bg="#16213e",
                 anchor="w"
                 ).place(x=10, y=20)

        # Confidence bar panel
        conf_panel = tk.Frame(left, bg="#1a1a2e", height=36)
        conf_panel.pack(side="top", fill="x", pady=(4, 0))
        conf_panel.pack_propagate(False)

        tk.Label(conf_panel, text="Confidence:",
                 font=("Courier", 9), fg="#555", bg="#1a1a2e"
                 ).place(x=0, y=10)

        self.conf_bar = ttk.Progressbar(conf_panel, length=530,
                                         mode="determinate",
                                         style="green.Horizontal.TProgressbar")
        self.conf_bar.place(x=90, y=10)

        self.conf_label = tk.Label(conf_panel, text="0%",
                                   font=("Courier", 9, "bold"),
                                   fg="#00d4aa", bg="#1a1a2e", width=5)
        self.conf_label.place(x=628, y=8)

        # Buttons panel
        btn_panel = tk.Frame(left, bg="#1a1a2e", height=50)
        btn_panel.pack(side="top", fill="x", pady=(6, 0))
        btn_panel.pack_propagate(False)

        btn_cfg = [
            ("🔊  SPEAK  [Space]", "#e94560", self.speak_current),
            ("🗑  CLEAR  [C]",     "#333a52", self.clear_history),
            ("📋  SENTENCES",      "#0f3460", self.show_sentences),
        ]
        for txt, col, cmd in btn_cfg:
            tk.Button(btn_panel, text=txt, bg=col, fg="white",
                      font=("Courier", 9, "bold"), relief="flat",
                      padx=14, pady=8, cursor="hand2", bd=0,
                      activebackground="#555", activeforeground="white",
                      command=cmd).pack(side="left", padx=(0, 6))

        tk.Checkbutton(btn_panel, text="Auto-speak",
                       variable=self.auto_speak,
                       bg="#1a1a2e", fg="#aaa",
                       selectcolor="#1a1a2e",
                       activebackground="#1a1a2e",
                       font=("Courier", 9)
                       ).pack(side="left", padx=10)

        # RIGHT: tabbed panel (fills remaining space)
        right = tk.Frame(body, bg="#16213e")
        right.pack(side="left", fill="both", expand=True)

        nb = ttk.Notebook(right, style="Custom.TNotebook")
        nb.pack(fill="both", expand=True, padx=4, pady=4)

        # Tab 1: History
        hist_tab = tk.Frame(nb, bg="#16213e")
        nb.add(hist_tab, text="  📜 History  ")
        hist_tab.rowconfigure(0, weight=1)
        hist_tab.columnconfigure(0, weight=1)

        self.history_box = scrolledtext.ScrolledText(
            hist_tab, bg="#0d0d1a", fg="#00d4aa",
            font=("Courier", 11),
            relief="flat", insertbackground="white",
            state="disabled", wrap=tk.WORD,
            selectbackground="#0f3460"
        )
        self.history_box.pack(fill="both", expand=True, padx=6, pady=6)

        # Tab 2: Training
        train_tab = tk.Frame(nb, bg="#16213e")
        nb.add(train_tab, text="  🏋 Training  ")
        self._build_training_tab(train_tab)

        # Tab 3: Settings
        settings_tab = tk.Frame(nb, bg="#16213e")
        nb.add(settings_tab, text="  ⚙ Settings  ")
        self._build_settings_tab(settings_tab)

        # ── STATUS BAR ──
        status_bar = tk.Frame(self.root, bg="#0d0d1a", height=26)
        status_bar.pack(side="bottom", fill="x")
        status_bar.pack_propagate(False)

        self.status_var = tk.StringVar(value="Ready — Demo mode active")
        tk.Label(status_bar, textvariable=self.status_var,
                 font=("Courier", 8), fg="#555", bg="#0d0d1a"
                 ).pack(side="left", padx=10)

        tk.Label(status_bar,
                 text="[SPACE] Speak   [C] Clear   [T] Train   [Q] Quit",
                 font=("Courier", 8), fg="#333a52", bg="#0d0d1a"
                 ).pack(side="right", padx=10)

    def _build_training_tab(self, parent):
        # Use a scrollable inner frame for reliability
        canvas = tk.Canvas(parent, bg="#16213e", highlightthickness=0)
        canvas.pack(fill="both", expand=True)
        inner = tk.Frame(canvas, bg="#16213e")
        canvas.create_window((0, 0), window=inner, anchor="nw")

        def _resize(e):
            canvas.configure(scrollregion=canvas.bbox("all"))
        inner.bind("<Configure>", _resize)

        PAD = {"padx": 14, "pady": 5, "sticky": "ew"}
        inner.columnconfigure(0, weight=1)

        tk.Label(inner, text="TRAIN YOUR OWN GESTURES",
                 font=("Courier", 11, "bold"), fg="#e94560", bg="#16213e"
                 ).grid(row=0, column=0, padx=14, pady=(14, 6), sticky="w")

        tk.Frame(inner, bg="#333a52", height=1
                 ).grid(row=1, column=0, padx=14, sticky="ew", pady=(0, 10))

        tk.Label(inner, text="Select sentence:",
                 font=("Courier", 9), fg="#888", bg="#16213e"
                 ).grid(row=2, column=0, padx=14, sticky="w")

        self.train_sentence_var = tk.StringVar()
        options = [f"{k}: {v['text']}" for k, v in SENTENCES.items()]
        self.train_combo = ttk.Combobox(inner,
                                         textvariable=self.train_sentence_var,
                                         values=options, state="readonly",
                                         font=("Courier", 10))
        self.train_combo.current(0)
        self.train_combo.grid(row=3, column=0, **PAD)

        # Samples row
        samp_row = tk.Frame(inner, bg="#16213e")
        samp_row.grid(row=4, column=0, padx=14, pady=5, sticky="w")
        tk.Label(samp_row, text="Samples to collect:",
                 font=("Courier", 9), fg="#888", bg="#16213e"
                 ).pack(side="left")
        self.samples_var = tk.IntVar(value=50)
        tk.Spinbox(samp_row, from_=10, to=200,
                   textvariable=self.samples_var,
                   width=6, bg="#0d0d1a", fg="#00d4aa",
                   insertbackground="white",
                   font=("Courier", 10), relief="flat"
                   ).pack(side="left", padx=10)

        self.record_btn = tk.Button(inner, text="⏺   START RECORDING",
                                    bg="#e94560", fg="white",
                                    font=("Courier", 11, "bold"),
                                    relief="flat", pady=10, cursor="hand2",
                                    bd=0, activebackground="#c73652",
                                    activeforeground="white",
                                    command=self.toggle_recording)
        self.record_btn.grid(row=5, column=0, padx=14, pady=8, sticky="ew")

        self.collect_progress = ttk.Progressbar(inner, mode="determinate",
                                                  style="green.Horizontal.TProgressbar")
        self.collect_progress.grid(row=6, column=0, padx=14, sticky="ew")

        self.collect_info = tk.Label(inner, text="0 / 0 samples",
                                     font=("Courier", 9), fg="#555", bg="#16213e")
        self.collect_info.grid(row=7, column=0, padx=14, sticky="w", pady=(2, 8))

        tk.Frame(inner, bg="#333a52", height=1
                 ).grid(row=8, column=0, padx=14, sticky="ew", pady=4)

        tk.Button(inner, text="🧠   TRAIN MODEL",
                  bg="#0f3460", fg="white",
                  font=("Courier", 11, "bold"),
                  relief="flat", pady=10, cursor="hand2",
                  bd=0, activebackground="#1a4a80",
                  activeforeground="white",
                  command=self.train_model
                  ).grid(row=9, column=0, padx=14, pady=8, sticky="ew")

        self.train_stats = tk.Label(inner, text="No training data yet.",
                                    font=("Courier", 9), fg="#555",
                                    bg="#16213e", justify="left",
                                    wraplength=280)
        self.train_stats.grid(row=10, column=0, padx=14, pady=4, sticky="w")
        self._update_train_stats()

    def _build_settings_tab(self, parent):
        inner = tk.Frame(parent, bg="#16213e")
        inner.pack(fill="both", expand=True, padx=14, pady=10)
        inner.columnconfigure(0, weight=1)

        def add_slider(row, label, var, from_, to, cmd=None):
            tk.Label(inner, text=label,
                     font=("Courier", 9), fg="#888", bg="#16213e"
                     ).grid(row=row, column=0, sticky="w", pady=(12, 0))
            s = tk.Scale(inner, from_=from_, to=to, orient="horizontal",
                         variable=var, bg="#16213e", fg="#00d4aa",
                         troughcolor="#0d0d1a", highlightthickness=0,
                         activebackground="#0f3460", sliderrelief="flat",
                         font=("Courier", 8), command=cmd)
            s.grid(row=row+1, column=0, sticky="ew")

        self.conf_thresh_var = tk.DoubleVar(value=self.confidence_threshold)
        add_slider(0, "Confidence Threshold (0.1 – 0.99)",
                   self.conf_thresh_var, 0.1, 0.99)

        self.tts_rate_var = tk.IntVar(value=150)
        add_slider(2, "Speech Rate (words per minute)",
                   self.tts_rate_var, 80, 280,
                   lambda v: self.tts.set_rate(int(float(v))))

        self.tts_vol_var = tk.DoubleVar(value=1.0)
        add_slider(4, "Speech Volume",
                   self.tts_vol_var, 0.1, 1.0,
                   lambda v: self.tts.set_volume(float(v)))

        self.cooldown_var = tk.DoubleVar(value=self.speak_cooldown)
        add_slider(6, "Auto-speak Cooldown (seconds)",
                   self.cooldown_var, 0.5, 10.0)

        tk.Frame(inner, bg="#333a52", height=1
                 ).grid(row=8, column=0, sticky="ew", pady=12)

        tk.Button(inner, text="✔   APPLY SETTINGS",
                  bg="#0f3460", fg="white",
                  font=("Courier", 10, "bold"),
                  relief="flat", pady=8, cursor="hand2",
                  bd=0, activebackground="#1a4a80",
                  activeforeground="white",
                  command=self._apply_settings
                  ).grid(row=9, column=0, sticky="ew")

    def _style_progressbar(self):
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("green.Horizontal.TProgressbar",
                        troughcolor="#0d0d1a",
                        background="#00d4aa",
                        thickness=12)
        style.configure("Custom.TNotebook",
                        background="#16213e",
                        borderwidth=0,
                        tabmargins=[0, 0, 0, 0])
        style.configure("Custom.TNotebook.Tab",
                        background="#0d0d1a",
                        foreground="#555",
                        font=("Courier", 9, "bold"),
                        padding=[12, 6])
        style.map("Custom.TNotebook.Tab",
                  background=[("selected", "#16213e")],
                  foreground=[("selected", "#00d4aa")])

    # ── CAMERA & INFERENCE LOOP ────────────────────────────────────────────
    def _start_camera(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("Camera Error",
                                  "Could not open webcam.\nPlease check your camera connection.")
            return
        self.running = True
        self._fps_counter = deque(maxlen=30)
        threading.Thread(target=self._camera_loop, daemon=True).start()

    def _camera_loop(self):
        while self.running:
            t0 = time.time()
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.05)
                continue

            frame = cv2.flip(frame, 1)
            features, annotated, detected = self.extractor.extract(frame)

            label, confidence = (-1, 0.0)
            if detected:
                label, confidence = self.classifier.predict(features)

            # Training data collection
            if self.collect_active and detected and label != -1:
                self.collector.add_sample(features, self.current_label)
                self.collect_count += 1
                pct = min(100, int(self.collect_count / self.collect_target * 100))
                self.root.after(0, self._update_collect_progress, pct)
                if self.collect_count >= self.collect_target:
                    self.collect_active = False
                    total = self.collector.save()
                    self.root.after(0, self._collection_done, total)

            # Update prediction buffer
            if detected and confidence >= self.confidence_threshold:
                self.prediction_buffer.append(label)
            else:
                self.prediction_buffer.append(-1)

            # Get stable prediction (majority vote)
            stable = self._stable_prediction()
            now = time.time()

            if stable != -1:
                sentence = SENTENCES[stable]["text"]
                self.root.after(0, self._update_recognition, sentence, confidence)
                if self.auto_speak.get():
                    diff_gesture = (stable != self._prev_stable)
                    time_ok = (now - self.last_spoken_time) > 2.0
                    print(f"[DEBUG] stable={stable} prev={self._prev_stable} diff={diff_gesture} time_ok={time_ok} elapsed={now-self.last_spoken_time:.1f}s")
                    if diff_gesture or time_ok:
                        print(f"[DEBUG] SPEAKING: {sentence}")
                        self.last_spoken = sentence
                        self.last_spoken_time = now
                        self._prev_stable = stable
                        self.tts.speak(sentence)
                        self.root.after(0, self._add_to_history, sentence)
            else:
                if not detected:
                    self.prediction_buffer.clear()
                    self._prev_stable = -1
                    self.last_spoken_time = 0
                    #print(f"[DEBUG] Hand gone - RESET")
                    self.root.after(0, self._update_recognition,
                                    "— no hand detected —", 0.0)

            if stable != -1:
                self._prev_stable = stable

            # Overlay info on frame
            self._draw_overlay(annotated, stable, confidence, detected)

            # Convert for tkinter
            img = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            img = img.resize((640, 480), Image.LANCZOS)
            photo = ImageTk.PhotoImage(img)
            self.root.after(0, self._update_frame, photo)

            # FPS
            self._fps_counter.append(time.time() - t0)
            fps = 1 / (sum(self._fps_counter) / len(self._fps_counter) + 1e-6)
            self.root.after(0, self.fps_var.set, f"FPS: {fps:.1f}")

    def _draw_overlay(self, frame, label, confidence, detected):
        h, w = frame.shape[:2]
        # Background bar
        cv2.rectangle(frame, (0, h-60), (w, h), (0, 0, 0), -1)

        if detected and label != -1:
            text = SENTENCES[label]["text"]
            cv2.putText(frame, f"{text}  ({confidence*100:.0f}%)",
                        (10, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 212, 170), 2)
        elif detected:
            cv2.putText(frame, "Detecting...",
                        (10, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 100), 2)
        else:
            cv2.putText(frame, "Show your hand to the camera",
                        (10, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 150, 150), 2)

        mode_txt = "TRAINING" if self.collect_active else "RECOGNITION"
        color = (0, 100, 255) if self.collect_active else (0, 200, 100)
        cv2.putText(frame, mode_txt, (w-160, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    def _stable_prediction(self):
        if len(self.prediction_buffer) < 10:
            return -1
        counts = Counter(self.prediction_buffer)
        most_common, count = counts.most_common(1)[0]
        if most_common == -1:
            return -1
        if count >= len(self.prediction_buffer) * 0.6:
            return most_common
        return -1

    # ── UI UPDATE CALLBACKS ────────────────────────────────────────────────
    def _update_frame(self, photo):
        self.video_label.configure(image=photo, text="")
        self.video_label.image = photo

    def _update_recognition(self, text, confidence):
        self.recognized_var.set(text)
        pct = int(confidence * 100)
        self.conf_bar["value"] = pct
        self.conf_label.configure(text=f"{pct}%")
        color = "#00d4aa" if pct >= 70 else "#ffaa00" if pct >= 40 else "#e94560"
        self.conf_label.configure(fg=color)

    def _add_to_history(self, sentence):
        self.history.append(sentence)
        ts = time.strftime("%H:%M:%S")
        self.history_box.configure(state="normal")
        self.history_box.insert("end", f"[{ts}] {sentence}\n")
        self.history_box.see("end")
        self.history_box.configure(state="disabled")

    def _update_collect_progress(self, pct):
        self.collect_progress["value"] = pct
        self.collect_info.configure(
            text=f"{self.collect_count} / {self.collect_target} samples ({pct}%)")

    def _collection_done(self, total):
        self.record_btn.configure(text="⏺  Start Recording", bg="#e94560")
        self.status_var.set(f"Collection done! Total samples saved: {total}")
        self._update_train_stats()

    def _update_train_stats(self):
        counts = self.collector.get_counts()
        total = sum(counts.values())
        if total == 0:
            self.train_stats.configure(text="No training data yet.")
            return
        lines = [f"Total samples: {total}"]
        for label, cnt in sorted(counts.items())[:8]:
            lines.append(f"  {SENTENCES[label]['text'][:20]}: {cnt}")
        if len(counts) > 8:
            lines.append(f"  ... and {len(counts)-8} more")
        self.train_stats.configure(text="\n".join(lines))

    # ── ACTIONS ────────────────────────────────────────────────────────────
    def speak_current(self):
        text = self.recognized_var.get()
        if text and "waiting" not in text and "no hand" not in text:
            self.tts.speak(text)
            self._add_to_history(f"[Manual] {text}")
            self.status_var.set(f"Speaking: {text}")

    def clear_history(self):
        self.history.clear()
        self.history_box.configure(state="normal")
        self.history_box.delete("1.0", "end")
        self.history_box.configure(state="disabled")
        self.status_var.set("History cleared.")

    def toggle_recording(self):
        if self.collect_active:
            self.collect_active = False
            self.record_btn.configure(text="⏺  Start Recording", bg="#e94560")
            self.status_var.set("Recording stopped.")
            return

        sel = self.train_combo.current()
        self.current_label = sel
        self.collect_target = self.samples_var.get()
        self.collect_count = 0
        self.collect_progress["value"] = 0
        self.collect_active = True
        self.record_btn.configure(text="⏹  Stop Recording", bg="#ff6600")
        self.status_var.set(
            f"Recording gesture for: {SENTENCES[sel]['text']} — {self.collect_target} samples")
        self.mode_label.configure(text="● TRAINING MODE", fg="#ff6600")

    def train_model(self):
        self.status_var.set("Training model… please wait.")
        self.root.update()

        def _train():
            ok, msg = self.collector.train_model()
            def _done():
                if ok:
                    try:
                        self.classifier = TrainedClassifier()
                        self.status_var.set(f"✅ {msg}")
                        messagebox.showinfo("Training Complete", msg)
                    except Exception as e:
                        self.status_var.set(f"Error loading model: {e}")
                else:
                    self.status_var.set(f"❌ Training failed: {msg}")
                    messagebox.showerror("Training Failed", msg)
            self.root.after(0, _done)

        threading.Thread(target=_train, daemon=True).start()

    def show_sentences(self):
        win = tk.Toplevel(self.root)
        win.title("30 Supported Sentences")
        win.configure(bg="#1a1a2e")
        win.geometry("600x600")

        tk.Label(win, text="Supported Sentences & Gestures",
                 font=("Helvetica", 14, "bold"), fg="#e94560", bg="#1a1a2e"
                 ).pack(pady=10)

        frame = tk.Frame(win, bg="#1a1a2e")
        frame.pack(fill="both", expand=True, padx=10, pady=4)

        canvas = tk.Canvas(frame, bg="#0d0d1a", highlightthickness=0)
        scrollbar = ttk.Scrollbar(frame, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)

        inner = tk.Frame(canvas, bg="#0d0d1a")
        canvas.create_window((0, 0), window=inner, anchor="nw")

        headers = ["#", "Sentence", "Gesture Description"]
        widths = [4, 22, 30]
        for c, (h, w) in enumerate(zip(headers, widths)):
            tk.Label(inner, text=h, font=("Helvetica", 10, "bold"),
                     fg="#00d4aa", bg="#0d0d1a", width=w, anchor="w"
                     ).grid(row=0, column=c, padx=4, pady=4, sticky="w")

        for i, (k, v) in enumerate(SENTENCES.items()):
            bg = "#0d0d1a" if i % 2 == 0 else "#101025"
            tk.Label(inner, text=str(k), font=("Courier", 10),
                     fg="#888", bg=bg, width=4, anchor="w"
                     ).grid(row=i+1, column=0, padx=4, sticky="w")
            tk.Label(inner, text=v["text"], font=("Helvetica", 10, "bold"),
                     fg="#ccc", bg=bg, width=22, anchor="w"
                     ).grid(row=i+1, column=1, padx=4, sticky="w")
            tk.Label(inner, text=v["gesture"], font=("Helvetica", 9),
                     fg="#888", bg=bg, width=30, anchor="w", wraplength=220
                     ).grid(row=i+1, column=2, padx=4, sticky="w")

        inner.update_idletasks()
        canvas.configure(scrollregion=canvas.bbox("all"))

    def _apply_settings(self):
        self.confidence_threshold = self.conf_thresh_var.get()
        self.speak_cooldown = self.cooldown_var.get()
        self.tts.set_rate(self.tts_rate_var.get())
        self.tts.set_volume(self.tts_vol_var.get())
        self.status_var.set("Settings applied.")

    # ── KEY BINDINGS ───────────────────────────────────────────────────────
    def _bind_keys(self):
        self.root.bind("<space>",  lambda e: self.speak_current())
        self.root.bind("<c>",      lambda e: self.clear_history())
        self.root.bind("<C>",      lambda e: self.clear_history())
        self.root.bind("<q>",      lambda e: self.quit())
        self.root.bind("<Q>",      lambda e: self.quit())
        self.root.bind("<Escape>", lambda e: self.quit())
        self.root.bind("<t>",      lambda e: self.toggle_recording())
        self.root.bind("<T>",      lambda e: self.toggle_recording())

    # ── LIFECYCLE ──────────────────────────────────────────────────────────
    def quit(self):
        self.running = False
        time.sleep(0.2)
        if self.cap:
            self.cap.release()
        self.extractor.close()
        self.tts.stop()
        self.root.destroy()

    def run(self):
        self.root.protocol("WM_DELETE_WINDOW", self.quit)
        self.root.mainloop()


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app = SignLanguageApp()
    app.run()
