# Kinetiq AI
### Real-Time Edge Vision Biomechanics, Velocity Loss Index and Indigenous Nutrition Engine

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10+-teal.svg)](https://developers.google.com/mediapipe)
[![CustomTkinter](https://img.shields.io/badge/GUI-CustomTkinter-blueviolet.svg)](https://github.com/TomSchimansky/CustomTkinter)
[![Google Gemini API](https://img.shields.io/badge/AI-Gemini%203.6%20Flash-orange.svg)](https://aistudio.google.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Kinetiq AI is an on-device, vision-based athletic assistant and biomechanical telemetry engine. It converts a standard laptop webcam or mobile camera into an athletic spotter with zero cloud video latency, joint vector tracking, concentric velocity loss quantification, and voice-guided real-time cues.

---

## Key Differentiators and Problem Solved

| Traditional Fitness Trackers | Kinetiq AI Engine |
| :--- | :--- |
| Basic rep counting (increments counter even on improper form) | **Kinetic Collapse Index:** Detects the exact repetition where form breakdown begins |
| High latency (streams raw video frames to remote cloud servers) | **100% On-Device Edge Pipeline:** 30+ FPS locally, fully private, zero video uploaded |
| Silent visual-only UI (requires user to look away from equipment) | **Asynchronous Audio Voice Spotter:** Speaks live cues ("Knees out", "Drive up") |
| Westernized diet plans (Avocado, Whey Isolate, Berries) | **Indigenous Macro Complementation Engine:** Tailored to accessible staples (Sattu, Ragi, Sprouts) |

---

## Core Features and Architecture

### 1. Edge Kinematic Joint Resolvers (MediaPipe + NumPy)
- Real-time trigonometric joint angle computation across hips, knees, ankles, shoulders, and elbows.
- Live angle visualizer with target threshold detection for squats, push-ups, pull-ups, and bicep curls.

### 2. Concentric Velocity Loss ($V_{\text{loss}}\%$) and Fatigue Engine
- Calculates velocity drop across successive repetitions:
  $$\Delta V_{\text{loss}} = \left( \frac{T_{\text{rep}} - T_{\text{baseline}}}{T_{\text{baseline}}} \right) \times 100$$
- Triggers fatigue warnings when $V_{\text{loss}} \ge 25\%$ (RPE 9.5+), preventing central nervous system (CNS) overtraining and musculoskeletal injury.

### 3. Integrated AI Spotter and Indigenous Nutrition Engine
- Context-aware LLM pipeline connected to Google Gemini 3.6 Flash (with fallback support for OpenAI and local Ollama).
- Automatically passes current exercise telemetry (completed repetitions, integrity score, velocity loss) directly into the AI prompt for precise recovery analysis.

### 4. Asynchronous Threaded Voice Coach (pyttsx3)
- Non-blocking text-to-speech engine with cooldown queues to deliver verbal cues without reducing video stream frame rates.

### 5. Telemetry CSV Data Logger
- Export of repetition timestamps, inflection angles, concentric duration, and integrity scores for athlete record keeping.

---

## System Architecture
