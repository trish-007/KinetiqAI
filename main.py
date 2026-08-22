import os
import cv2
import time
import math
import csv
import queue
import threading
import numpy as np
from PIL import Image, ImageTk
import customtkinter as ctk
from dotenv import load_dotenv

# Optional local offline Voice Engine
try:
    import pyttsx3
    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False

load_dotenv()

# MediaPipe Pose
import mediapipe as mp
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


# -------------------------------------------------------------
# ASYNCHRONOUS OFFLINE VOICE SPOTTER ENGINE
# -------------------------------------------------------------
class VoiceSpotter:
    """Non-blocking, queued Text-To-Speech engine for audio form cues."""
    def __init__(self):
        self.queue = queue.Queue()
        self.last_spoken_time = 0
        self.cooldown = 2.2  # minimum seconds between audio alerts
        if TTS_AVAILABLE:
            self.thread = threading.Thread(target=self._worker, daemon=True)
            self.thread.start()

    def _worker(self):
        try:
            engine = pyttsx3.init()
            engine.setProperty('rate', 170)
            engine.setProperty('volume', 0.9)
            while True:
                text = self.queue.get()
                if text is None:
                    break
                engine.say(text)
                engine.runAndWait()
                self.queue.task_done()
        except Exception:
            pass

    def speak(self, text, force=False):
        now = time.time()
        if TTS_AVAILABLE and (force or (now - self.last_spoken_time > self.cooldown)):
            self.last_spoken_time = now
            self.queue.put(text)


# -------------------------------------------------------------
# BIOMECHANICS & KINEMATIC RESOLVER
# -------------------------------------------------------------
def calculate_angle(a, b, c):
    """Calculates angle at joint vertex b in 2D space."""
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360.0 - angle
    return angle


# -------------------------------------------------------------
# MAIN KINETIQ AI ENGINE
# -------------------------------------------------------------
class KinetiqAI(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Kinetiq AI - Edge Biomechanics & Injury Prevention Platform")
        self.geometry("1440x860")
        self.configure(fg_color="#070b12")

        # Initialize Voice Coach
        self.voice = VoiceSpotter()

        # Telemetry & State Tracking
        self.current_exercise = "squat"
        self.reps = 0
        self.stage = "UP"
        self.rep_start_time = None
        self.inflection_time = None
        self.rep_telemetry = []
        self.current_rep_min_angle = 180
        self.baseline_concentric_time = None
        self.live_angle = 0
        self.feedback_text = "Step into camera frame to calibrate..."

        # UI Exercise Map
        self.exercise_buttons = {}
        self.exercise_config = {
            "squat": {"label": "Knee Angle", "target": 90, "up": 160, "name": "SQUATS", "cue": "Push knees out"},
            "pushup": {"label": "Elbow Angle", "target": 90, "up": 155, "name": "PUSH-UPS", "cue": "Lock arms out"},
            "pullup": {"label": "Elbow Angle", "target": 70, "up": 155, "name": "PULL-UPS", "cue": "Chin over bar"},
            "curl": {"label": "Elbow Bend", "target": 45, "up": 150, "name": "BICEP CURLS", "cue": "Keep elbows pinned"}
        }

        # MediaPipe Pose Model
        self.pose = mp_pose.Pose(
            model_complexity=0,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        self.cap = cv2.VideoCapture(0)

        # Chat History
        self.chat_history = [
            {"role": "system", "content": "You are Kinetiq AI, an elite sports biomechanist and Indian nutrition scientist. Deliver sharp kinetic feedback, velocity loss interpretation, and precise regional Indian meal macros."}
        ]

        self.is_bot_thinking = False
        self.spinner_chars = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
        self.spinner_idx = 0

        self.build_ui()
        self.update_mode_ui_highlight("squat")

        self.is_running = True
        self.update_video_stream()

    # ---------------------------------------------------------
    # UI ARCHITECTURE
    # ---------------------------------------------------------
    def build_ui(self):
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=3)
        self.grid_columnconfigure(2, weight=2)
        self.grid_rowconfigure(0, weight=1)

        # ================= LEFT PANEL =================
        self.left_panel = ctk.CTkFrame(self, fg_color="#0e1526", corner_radius=14, border_width=1, border_color="#1e293b")
        self.left_panel.grid(row=0, column=0, padx=12, pady=12, sticky="nsew")

        ctk.CTkLabel(self.left_panel, text="KINEMATIC PROTOCOLS", font=ctk.CTkFont(size=13, weight="bold"), text_color="#06b6d4").pack(pady=(12, 4))

        self.active_mode_badge = ctk.CTkLabel(
            self.left_panel, text="⚡ ACTIVE: SQUATS", font=ctk.CTkFont(size=12, weight="bold"),
            fg_color="#1e293b", text_color="#06b6d4", corner_radius=6, height=28
        )
        self.active_mode_badge.pack(fill="x", padx=12, pady=(0, 8))

        # Exercise Selectors
        for ex in ["squat", "pushup", "pullup", "curl"]:
            btn = ctk.CTkButton(
                self.left_panel, text=self.exercise_config[ex]["name"],
                command=lambda e=ex: self.switch_exercise(e),
                fg_color="#1e293b", hover_color="#334155", font=ctk.CTkFont(size=12, weight="bold"),
                height=34, border_width=1, border_color="#334155"
            )
            btn.pack(fill="x", padx=12, pady=3)
            self.exercise_buttons[ex] = btn

        ctk.CTkLabel(self.left_panel, text="VELOCITY LOSS & FATIGUE", font=ctk.CTkFont(size=13, weight="bold"), text_color="#f59e0b").pack(pady=(14, 4))

        self.card_score = self.create_metric_card(self.left_panel, "Form Integrity Score", "100%", "#10b981")
        self.card_vloss = self.create_metric_card(self.left_panel, "Concentric Velocity Loss", "0.0%", "#38bdf8")
        self.card_breakdown = self.create_metric_card(self.left_panel, "Kinetic Collapse Point", "Stable Execution", "#10b981")
        self.card_tempo = self.create_metric_card(self.left_panel, "Concentric Drive Time", "-- s", "#f8fafc")

        self.btn_diagnose = ctk.CTkButton(
            self.left_panel, text="🤖 AI Biomechanical Diagnosis",
            command=self.ask_ai_to_diagnose, fg_color="#10b981", hover_color="#059669",
            text_color="#000", font=ctk.CTkFont(size=12, weight="bold"), height=34
        )
        self.btn_diagnose.pack(fill="x", padx=12, pady=(10, 4))

        self.btn_export = ctk.CTkButton(
            self.left_panel, text="📥 Export Session CSV",
            command=self.export_telemetry_csv, fg_color="#334155", hover_color="#475569",
            text_color="#fff", font=ctk.CTkFont(size=11, weight="bold"), height=28
        )
        self.btn_export.pack(fill="x", padx=12, pady=(0, 10))

        # ================= CENTER PANEL (VIDEO HUD) =================
        self.center_panel = ctk.CTkFrame(self, fg_color="#0e1526", corner_radius=14, border_width=1, border_color="#1e293b")
        self.center_panel.grid(row=0, column=1, padx=6, pady=12, sticky="nsew")

        self.video_header = ctk.CTkFrame(self.center_panel, fg_color="transparent")
        self.video_header.pack(fill="x", padx=12, pady=6)
        
        ctk.CTkLabel(self.video_header, text="KINETIQ EDGE OPTICAL ENGINE (30+ FPS)", font=ctk.CTkFont(size=13, weight="bold"), text_color="#06b6d4").pack(side="left")
        self.hud_mode_tag = ctk.CTkLabel(self.video_header, text="● MODE: SQUATS", font=ctk.CTkFont(size=12, weight="bold"), text_color="#10b981")
        self.hud_mode_tag.pack(side="right")

        self.video_label = ctk.CTkLabel(self.center_panel, text="")
        self.video_label.pack(padx=12, pady=4, expand=True)

        self.status_banner = ctk.CTkLabel(
            self.center_panel, text=self.feedback_text, font=ctk.CTkFont(size=14, weight="bold"),
            fg_color="#1e293b", corner_radius=8, height=38
        )
        self.status_banner.pack(fill="x", padx=12, pady=6)

        self.telemetry_strip = ctk.CTkFrame(self.center_panel, fg_color="transparent")
        self.telemetry_strip.pack(fill="x", padx=12, pady=(0, 8))
        self.telemetry_strip.grid_columnconfigure((0, 1, 2), weight=1)

        self.val_angle = self.create_strip_item(self.telemetry_strip, 0, "Knee Angle", "--°", "#06b6d4")
        self.val_phase = self.create_strip_item(self.telemetry_strip, 1, "Phase", "UP", "#f59e0b")
        self.val_reps = self.create_strip_item(self.telemetry_strip, 2, "Completed Reps", "0", "#10b981")

        # ================= RIGHT PANEL (CHATBOT) =================
        self.right_panel = ctk.CTkFrame(self, fg_color="#0e1526", corner_radius=14, border_width=1, border_color="#1e293b")
        self.right_panel.grid(row=0, column=2, padx=12, pady=12, sticky="nsew")

        self.chat_header_frame = ctk.CTkFrame(self.right_panel, fg_color="transparent")
        self.chat_header_frame.pack(fill="x", padx=10, pady=(10, 4))

        provider_name = os.getenv("LLM_PROVIDER", "offline").upper()
        ctk.CTkLabel(self.chat_header_frame, text=f"AI SPOTTER [{provider_name}]", font=ctk.CTkFont(size=13, weight="bold"), text_color="#8b5cf6").pack(side="left")
        self.thinking_indicator = ctk.CTkLabel(self.chat_header_frame, text="", font=ctk.CTkFont(size=11, weight="bold"), text_color="#f59e0b")
        self.thinking_indicator.pack(side="right")

        # Regional Diet Quick Prompts
        self.chips_frame = ctk.CTkFrame(self.right_panel, fg_color="transparent")
        self.chips_frame.pack(fill="x", padx=10, pady=2)
        
        btn_sattu = ctk.CTkButton(self.chips_frame, text="🌾 Sattu Shake", width=80, height=24, fg_color="#1e293b", text_color="#f59e0b", font=ctk.CTkFont(size=10), command=lambda: self.quick_prompt("What are the exact macro splits and leucine content of 50g Sattu?"))
        btn_sattu.pack(side="left", padx=2)
        btn_ragi = ctk.CTkButton(self.chips_frame, text="🥣 Ragi Mudde", width=80, height=24, fg_color="#1e293b", text_color="#f59e0b", font=ctk.CTkFont(size=10), command=lambda: self.quick_prompt("How does Ragi Mudde support bone density during heavy compound lifting?"))
        btn_ragi.pack(side="left", padx=2)
        btn_vloss = ctk.CTkButton(self.chips_frame, text="📉 Velocity Loss", width=80, height=24, fg_color="#1e293b", text_color="#38bdf8", font=ctk.CTkFont(size=10), command=lambda: self.quick_prompt("Explain why 20% velocity loss is the optimal threshold to stop a set."))
        btn_vloss.pack(side="left", padx=2)

        # Read-only protected chat display
        self.chat_display = ctk.CTkTextbox(self.right_panel, fg_color="#161f36", text_color="#f8fafc", font=ctk.CTkFont(size=12), wrap="word", state="disabled")
        self.chat_display.pack(fill="both", expand=True, padx=10, pady=6)
        
        self.append_to_chat("Kinetiq AI: Ready for session tracking. Real-time audio coaching is active. Start your first rep!\n\n")

        self.input_frame = ctk.CTkFrame(self.right_panel, fg_color="transparent")
        self.input_frame.pack(fill="x", padx=10, pady=(0, 10))

        self.chat_input = ctk.CTkEntry(self.input_frame, placeholder_text="Ask about velocity loss, cues, or Indian macros...", fg_color="#161f36", border_color="#334155")
        self.chat_input.pack(side="left", fill="x", expand=True, padx=(0, 6))
        self.chat_input.bind("<Return>", lambda event: self.send_chat())

        self.send_btn = ctk.CTkButton(self.input_frame, text="➤", width=36, fg_color="#8b5cf6", hover_color="#7c3aed", command=self.send_chat)
        self.send_btn.pack(side="right")

    def create_metric_card(self, parent, title, initial_val, color):
        frame = ctk.CTkFrame(parent, fg_color="#161f36", corner_radius=8, border_width=1, border_color="#24324f")
        frame.pack(fill="x", padx=12, pady=3)
        ctk.CTkLabel(frame, text=title, font=ctk.CTkFont(size=10), text_color="#94a3b8").pack(anchor="w", padx=8, pady=(3, 0))
        val_label = ctk.CTkLabel(frame, text=initial_val, font=ctk.CTkFont(size=14, weight="bold"), text_color=color)
        val_label.pack(anchor="w", padx=8, pady=(0, 3))
        return val_label

    def create_strip_item(self, parent, col, title, initial_val, color):
        frame = ctk.CTkFrame(parent, fg_color="#161f36", corner_radius=8)
        frame.grid(row=0, column=col, padx=4, sticky="nsew")
        ctk.CTkLabel(frame, text=title, font=ctk.CTkFont(size=10), text_color="#94a3b8").pack(pady=(3, 0))
        val_label = ctk.CTkLabel(frame, text=initial_val, font=ctk.CTkFont(size=15, weight="bold"), text_color=color)
        val_label.pack(pady=(0, 3))
        return val_label

    def update_mode_ui_highlight(self, selected_ex):
        for ex, btn in self.exercise_buttons.items():
            if ex == selected_ex:
                btn.configure(fg_color="#06b6d4", text_color="#000000", border_color="#06b6d4")
            else:
                btn.configure(fg_color="#1e293b", text_color="#94a3b8", border_color="#334155")
        mode_name = self.exercise_config[selected_ex]["name"]
        self.active_mode_badge.configure(text=f"⚡ ACTIVE: {mode_name}")
        self.hud_mode_tag.configure(text=f"● MODE: {mode_name}")

    def switch_exercise(self, ex):
        self.current_exercise = ex
        self.reps = 0
        self.stage = "UP"
        self.rep_telemetry.clear()
        self.baseline_concentric_time = None
        
        self.update_mode_ui_highlight(ex)
        self.val_reps.configure(text="0")
        self.card_score.configure(text="100%", text_color="#10b981")
        self.card_vloss.configure(text="0.0%", text_color="#38bdf8")
        self.card_breakdown.configure(text="Stable Execution", text_color="#10b981")
        self.card_tempo.configure(text="-- s")
        self.val_angle.master.winfo_children()[0].configure(text=self.exercise_config[ex]["label"])
        self.voice.speak(f"Switched protocol to {self.exercise_config[ex]['name']}")

    # ---------------------------------------------------------
    # CHAT HELPER METHODS (READ-ONLY PROTECTION)
    # ---------------------------------------------------------
    def append_to_chat(self, text):
        self.chat_display.configure(state="normal")
        self.chat_display.insert("end", text)
        self.chat_display.see("end")
        self.chat_display.configure(state="disabled")

    def delete_chat_placeholder(self):
        self.chat_display.configure(state="normal")
        self.chat_display.delete("end-2l", "end-1c")
        self.chat_display.configure(state="disabled")

    # ---------------------------------------------------------
    # COMPUTER VISION & VELOCITY LOSS TRACKING
    # ---------------------------------------------------------
    def update_video_stream(self):
        if not self.is_running:
            return

        ret, frame = self.cap.read()
        if ret:
            frame = cv2.resize(frame, (580, 430))
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image_rgb.flags.writeable = False
            results = self.pose.process(image_rgb)
            image_rgb.flags.writeable = True

            if results.pose_landmarks:
                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                lm = results.pose_landmarks.landmark

                if self.current_exercise == "squat":
                    hip = [lm[mp_pose.PoseLandmark.RIGHT_HIP.value].x, lm[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                    knee = [lm[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, lm[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
                    ankle = [lm[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x, lm[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
                    self.live_angle = int(calculate_angle(hip, knee, ankle))
                elif self.current_exercise in ["pushup", "pullup", "curl"]:
                    shoulder = [lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                    elbow = [lm[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, lm[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                    wrist = [lm[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, lm[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
                    self.live_angle = int(calculate_angle(shoulder, elbow, wrist))

                self.val_angle.configure(text=f"{self.live_angle}°")
                cfg = self.exercise_config[self.current_exercise]

                if self.live_angle < self.current_rep_min_angle:
                    self.current_rep_min_angle = self.live_angle

                # Bottom Inflection Point (Start of Concentric Phase)
                if self.live_angle <= cfg["target"] and self.stage == "UP":
                    self.stage = "DOWN"
                    self.inflection_time = time.time()
                    self.val_phase.configure(text="DOWN", text_color="#10b981")
                    self.status_banner.configure(text="Target Depth Achieved! Drive Up", text_color="#10b981")
                    self.voice.speak("Up!")

                # Top Lockout Point (Rep Completion & Concentric Speed Measurement)
                elif self.live_angle >= cfg["up"] and self.stage == "DOWN":
                    self.stage = "UP"
                    self.val_phase.configure(text="UP", text_color="#06b6d4")
                    self.reps += 1
                    self.val_reps.configure(text=str(self.reps))

                    concentric_duration = time.time() - self.inflection_time if self.inflection_time else 1.2
                    if self.baseline_concentric_time is None:
                        self.baseline_concentric_time = concentric_duration

                    # Calculate Concentric Velocity Loss %
                    v_loss = max(0.0, ((concentric_duration - self.baseline_concentric_time) / self.baseline_concentric_time) * 100)

                    # Form Score penalizes depth cuts
                    rep_score = 100
                    if self.current_rep_min_angle > cfg["target"]:
                        rep_score = max(40, 100 - (self.current_rep_min_angle - cfg["target"]) * 3)

                    self.rep_telemetry.append({
                        "rep": self.reps,
                        "exercise": self.current_exercise,
                        "min_angle": self.current_rep_min_angle,
                        "concentric_sec": round(concentric_duration, 2),
                        "v_loss_percent": round(v_loss, 1),
                        "integrity_score": int(rep_score)
                    })

                    self.current_rep_min_angle = 180
                    self.evaluate_fatigue_and_velocity()
                    self.status_banner.configure(text=f"Rep #{self.reps} Logged! (V-Loss: {round(v_loss, 1)}%)", text_color="#06b6d4")

                    # Voice Warning if Velocity Drops over 25% (RPE 9.5+)
                    if v_loss >= 25.0:
                        self.voice.speak("High fatigue detected. Rack the weight.")
                    else:
                        self.voice.speak(f"{self.reps}")
            else:
                self.status_banner.configure(text="Align in frame...", text_color="#94a3b8")

            rgb_display = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb_display)
            ctk_img = ctk.CTkImage(light_image=img, dark_image=img, size=(580, 430))
            self.video_label.configure(image=ctk_img)
            self.video_label.image = ctk_img

        self.after(20, self.update_video_stream)

    # ---------------------------------------------------------
    # FATIGUE & VELOCITY LOSS ANALYTICS
    # ---------------------------------------------------------
    def evaluate_fatigue_and_velocity(self):
        if not self.rep_telemetry:
            return

        scores = [r["integrity_score"] for r in self.rep_telemetry]
        avg_score = int(sum(scores) / len(scores))
        self.card_score.configure(text=f"{avg_score}%")
        self.card_score.configure(text_color="#10b981" if avg_score >= 85 else "#f59e0b" if avg_score >= 70 else "#ef4444")

        latest = self.rep_telemetry[-1]
        self.card_vloss.configure(text=f"{latest['v_loss_percent']}%")
        self.card_tempo.configure(text=f"{latest['concentric_sec']} s")

        breakdown = next((r for r in self.rep_telemetry if r["integrity_score"] < 75 or r["v_loss_percent"] > 25.0), None)
        if breakdown:
            self.card_breakdown.configure(
                text=f"Rep #{breakdown['rep']} (V-Loss: {breakdown['v_loss_percent']}%)",
                text_color="#ef4444"
            )
        else:
            self.card_breakdown.configure(text="Stable Execution", text_color="#10b981")

    # ---------------------------------------------------------
    # CSV TELEMETRY EXPORT
    # ---------------------------------------------------------
    def export_telemetry_csv(self):
        if not self.rep_telemetry:
            self.status_banner.configure(text="No reps recorded to export.", text_color="#f59e0b")
            return

        filename = f"kinetiq_session_{self.current_exercise}_{int(time.time())}.csv"
        try:
            with open(filename, mode='w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=["rep", "exercise", "min_angle", "concentric_sec", "v_loss_percent", "integrity_score"])
                writer.writeheader()
                writer.writerows(self.rep_telemetry)
            self.status_banner.configure(text=f"Saved CSV: {filename}", text_color="#10b981")
            self.voice.speak("Session telemetry exported.")
        except Exception as e:
            self.status_banner.configure(text=f"Export failed: {str(e)}", text_color="#ef4444")

    # ---------------------------------------------------------
    # CHATBOT & LLM LOGIC
    # ---------------------------------------------------------
    def animate_thinking(self):
        if self.is_bot_thinking:
            char = self.spinner_chars[self.spinner_idx % len(self.spinner_chars)]
            self.thinking_indicator.configure(text=f"Analyzing {char}")
            self.spinner_idx += 1
            self.after(90, self.animate_thinking)
        else:
            self.thinking_indicator.configure(text="")

    def quick_prompt(self, text):
        self.chat_input.delete(0, "end")
        self.chat_input.insert(0, text)
        self.send_chat()

    def ask_ai_to_diagnose(self):
        if not self.rep_telemetry:
            self.quick_prompt(f"What is the ideal eccentric-to-concentric tempo for {self.current_exercise.upper()}?")
        else:
            scores = [r["integrity_score"] for r in self.rep_telemetry]
            avg_score = int(sum(scores) / len(scores))
            max_vloss = max(r["v_loss_percent"] for r in self.rep_telemetry)
            prompt = (
                f"Analyze my set of {self.current_exercise}: {self.reps} reps, "
                f"Form Integrity Score: {avg_score}%, Peak Velocity Loss: {max_vloss}%. "
                f"Provide athletic neuromuscular diagnosis and optimal recovery advice."
            )
            self.quick_prompt(prompt)

    def send_chat(self):
        user_text = self.chat_input.get().strip()
        if not user_text:
            return

        self.chat_input.delete(0, "end")
        self.append_to_chat(f"You: {user_text}\n\n")

        context = f"[Context: Mode={self.current_exercise.upper()}, Reps={self.reps}, Integrity={self.card_score.cget('text')}, VLoss={self.card_vloss.cget('text')}]"
        self.chat_history.append({"role": "user", "content": f"{context} {user_text}"})

        self.is_bot_thinking = True
        self.animate_thinking()
        threading.Thread(target=self.stream_llm_response, args=(user_text,), daemon=True).start()

    def stream_llm_response(self, user_text):
        provider = os.getenv("LLM_PROVIDER", "offline").lower()

        placeholder = "Kinetiq AI: [Analyzing biomechanical vectors...]\n"
        self.append_to_chat(placeholder)

        try:
            if provider == "gemini" and os.getenv("GEMINI_API_KEY"):
                from google import genai
                client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

                # Read from .env if defined, otherwise default to gemini-3.6-flash
                gemini_model = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")

                response = client.models.generate_content_stream(
                    model=gemini_model,
                    contents=[m["content"] for m in self.chat_history if m["role"] != "system"]
                )

                self.delete_chat_placeholder()
                self.is_bot_thinking = False
                self.append_to_chat("Kinetiq AI: ")

                accumulated = ""
                for chunk in response:
                    text = chunk.text
                    accumulated += text
                    self.append_to_chat(text)
                self.chat_history.append({"role": "assistant", "content": accumulated})

            elif provider == "openai" and os.getenv("OPENAI_API_KEY"):
                from openai import OpenAI
                client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
                stream = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=self.chat_history,
                    stream=True
                )
                self.delete_chat_placeholder()
                self.is_bot_thinking = False
                self.append_to_chat("Kinetiq AI: ")

                accumulated = ""
                for chunk in stream:
                    if chunk.choices[0].delta.content:
                        text = chunk.choices[0].delta.content
                        accumulated += text
                        self.append_to_chat(text)
                self.chat_history.append({"role": "assistant", "content": accumulated})

            elif provider == "ollama":
                import requests
                import json
                base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
                model = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
                res = requests.post(
                    f"{base_url}/api/chat",
                    json={"model": model, "messages": self.chat_history, "stream": True},
                    stream=True
                )
                self.delete_chat_placeholder()
                self.is_bot_thinking = False
                self.append_to_chat("Kinetiq AI: ")

                accumulated = ""
                for line in res.iter_lines():
                    if line:
                        chunk = json.loads(line.decode("utf-8"))
                        text = chunk.get("message", {}).get("content", "")
                        accumulated += text
                        self.append_to_chat(text)
                self.chat_history.append({"role": "assistant", "content": accumulated})

            else:
                time.sleep(0.3)
                self.delete_chat_placeholder()
                self.is_bot_thinking = False
                self.stream_offline_fallback(user_text)

        except Exception as e:
            self.is_bot_thinking = False
            self.delete_chat_placeholder()
            self.append_to_chat(f"\n[LLM Warning: {str(e)}. Using Edge Rule Engine]\n")
            self.stream_offline_fallback(user_text)

        self.append_to_chat("\n\n")

    def stream_offline_fallback(self, q):
        q = q.lower()
        self.append_to_chat("Kinetiq AI: ")
        
        if "sattu" in q or "protein" in q:
            reply = "🌾 Sattu Profile: 50g yields 13g bioavailable protein + 33g complex carbs. Mix with chilled chaas (buttermilk) and roasted jeera to complement lysine and methionine ratios."
        elif "ragi" in q or "mudde" in q:
            reply = "🥣 Ragi Mudde: Provides 344mg Calcium per 100g, fortifying bone mineral density against spinal compressive loads during heavy squats."
        elif "velocity" in q or "vloss" in q or "rpe" in q:
            reply = "📉 Velocity Loss Science: Terminating sets at 20-25% concentric velocity loss maximizes mechanical tension while minimizing excessive CNS fatigue and connective tissue breakdown."
        elif "diagnose" in q or "fatigue" in q:
            reply = f"📊 Biomechanical Set Diagnosis: Completed {self.reps} reps. Average Form Integrity is {self.card_score.cget('text')} with a peak Concentric Velocity Loss of {self.card_vloss.cget('text')}."
        else:
            reply = "💡 Kinematic Cue: Maintain rigid intra-abdominal bracing (Valsalva), ensure even ground reaction force distribution through the mid-foot, and terminate repetitions before compensatory valgus occurs."

        for char in reply:
            self.append_to_chat(char)
            time.sleep(0.01)

        self.chat_history.append({"role": "assistant", "content": reply})

    def destroy(self):
        self.is_running = False
        if self.cap.isOpened():
            self.cap.release()
        super().destroy()


# -------------------------------------------------------------
# ENTRY POINT
# -------------------------------------------------------------
if __name__ == "__main__":
    app = KinetiqAI()
    app.mainloop()