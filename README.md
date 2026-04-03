# Phoneme Trainer

An Android app for real-time pronunciation analysis targeting the **ELPAC** (English Language Proficiency Assessment for California). It uses on-device AI to detect phonemes, align them against expected pronunciations, and produce scored feedback — no cloud required.

## Features

- Record speech and receive instant phoneme-level pronunciation feedback
- AI-powered phoneme detection using a fine-tuned Wav2Vec2 model (Facebook, trained on 60k hours of speech)
- Phoneme alignment against 140K+ words via the CMU Pronouncing Dictionary
- ELPAC scoring rubric (Levels 1–4) with accuracy, fluency, and completeness scores
- Live waveform visualization and word-level feedback
- Fully on-device — no data leaves the phone

## ELPAC Scoring Rubric

| Score | Level |
|-------|-------|
| ≥ 85  | Level 4 |
| ≥ 70  | Level 3 |
| ≥ 50  | Level 2 |
| < 50  | Level 1 |

## Requirements

- Android API 26+ (Android 8.0 Oreo or higher)
- Java 17 toolchain
- Android SDK 34

## Setup

### 1. Clone the repo

```bash
git clone https://github.com/MA0610/Elpac-training-app.git
cd Elpac-training-app/Phoneme_Trainer
```

### 2. Get the Wav2Vec2 model

The ONNX model (~95 MB) is too large for git and is downloaded automatically by the app at first launch from GitHub Releases.

To generate it yourself instead:

```bash
pip install transformers optimum[onnxruntime] torch
python export_model.py   # downloads ~1.5 GB from HuggingFace, outputs ~95 MB ONNX
```

Then place `wav2vec2_phoneme.onnx` in `Phoneme_Trainer/app/src/main/assets/`.

### 3. Open in Android Studio

Open the `Phoneme_Trainer/` folder in Android Studio, sync Gradle, and run on a device or emulator.

## Build

```bash
./gradlew assembleDebug      # Debug APK
./gradlew assembleRelease    # Release APK
./gradlew test               # Unit tests
```

## Architecture

```
AudioRecorder (16kHz PCM)
    └─> MainViewModel
            ├─> Wav2Vec2PhonemeDetector  — ONNX inference, CTC decode → IPA phonemes + timing
            ├─> PhonemeDetector          — Needleman-Wunsch alignment vs CMU dict
            └─> MainScreen (Compose)     — Waveform, phoneme timeline, score rings
```

**Key files:**

| File | Role |
|------|------|
| `MainViewModel.kt` | Recording workflow and analysis pipeline orchestrator |
| `Wav2Vec2PhonemeDetector.kt` | ONNX inference engine with CTC decoding |
| `PhonemeDetector.kt` | Alignment, scoring, ELPAC level mapping |
| `PhonemeModels.kt` | Data classes and ARPABET↔IPA mappings |
| `AudioRecorder.kt` | Real-time PCM streaming via Kotlin Flow |
| `MainScreen.kt` / `PhonemeViews.kt` | Compose UI |

## Tech Stack

- **Kotlin** + Jetpack Compose
- **ONNX Runtime** for on-device Wav2Vec2 inference
- **Vosk ASR** for word-boundary timing
- **CMU Pronouncing Dictionary** for expected phoneme lookup
- StateFlow / Coroutines for async audio processing
