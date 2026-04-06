# Phoneme Trainer

An Android app for real-time pronunciation analysis targeting the **ELPAC** (English Language Proficiency Assessment for California). It uses on-device AI to detect phonemes, align them against expected pronunciations, and produce scored feedback — no cloud required.

## Features

- Record speech and receive instant phoneme-level pronunciation feedback
- AI-powered phoneme detection using a fine-tuned Wav2Vec2 model (Facebook, trained on 60k hours of speech)
- Phoneme alignment against 140K+ words via the CMU Pronouncing Dictionary
- ELPAC scoring rubric (Levels 1–4) with accuracy, fluency, and completeness sub-scores
- 22 preset ELPAC-aligned target phrases, plus custom phrase input
- Live waveform visualization and RMS level meter during recording
- Word-level and phoneme-level feedback with expected vs. actual comparison
- Fully on-device — no data leaves the phone

## How It Works

### Recording

Speech is captured via Android `AudioRecord` at **16kHz mono, 16-bit PCM**. Samples stream in 100ms chunks via a Kotlin Flow while a live waveform and level meter update in real time. A persistent `AudioRecord` instance is reused across sessions to avoid audio capture issues on emulators.

### Phoneme Detection (Wav2Vec2)

The core model is `facebook/wav2vec2-lv-60-espeak-cv-ft` exported to ONNX (~95 MB). It runs fully on-device via ONNX Runtime Android:

1. PCM samples are normalized to float `[-1, 1]` and fed to the ONNX model
2. The model outputs logits `[1, num_frames, 43]` — 43 IPA phoneme tokens per time frame
3. **CTC greedy decoding**: softmax → argmax per frame → collapse consecutive duplicate tokens → skip PAD tokens → produce a list of IPA phonemes with real acoustic timing and per-phoneme confidence (softmax posterior probability)

### Word Timing (Vosk)

Vosk ASR runs in parallel to extract word-boundary timestamps (start/end ms per word). Vosk confidence values are discarded — only the timing is used.

### Expected Phoneme Lookup (CMU Dictionary)

The target phrase is tokenized into words, looked up in the bundled CMU Pronouncing Dictionary (~140K entries), and converted from ARPABET to IPA using eSpeak-compatible mappings.

### Alignment (Needleman-Wunsch)

Detected IPA phonemes are globally aligned against expected phonemes using the Needleman-Wunsch algorithm:

| Alignment type | Score |
|---|---|
| Exact match | +2 |
| Near-miss (voiced/unvoiced pair, long/short vowel) | +1 |
| Substitution | -1 |
| Gap (insertion or deletion) | -1 |

The traceback annotates each detected phoneme with whether it was correct, a near-miss, wrong, or an insertion.

### Scoring

| Component | Weight | How computed |
|---|---|---|
| Accuracy | 55% | Weighted match rate: exact=1.0, near-miss=0.7, insertion=0.85, wrong=0.0 |
| Fluency | 30% | Gap analysis — penalties for hesitations >300ms or overlaps |
| Completeness | 15% | Ratio of expected phonemes that were produced |
| **Overall** | — | Weighted blend → mapped to ELPAC level |

### ELPAC Level Mapping

| Overall Score | Level |
|---|---|
| ≥ 85 | Level 4 |
| ≥ 70 | Level 3 |
| ≥ 50 | Level 2 |
| < 50 | Level 1 |

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

The ONNX model (~95 MB) is downloaded automatically by the app at first launch from GitHub Releases.

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
AudioRecorder (16kHz PCM, 100ms chunks via Flow)
    └─> MainViewModel
            ├─> Wav2Vec2PhonemeDetector  — ONNX inference, CTC decode → IPA phonemes + timing
            ├─> PhonemeDetector          — Vosk word timing, CMU dict lookup, Needleman-Wunsch
            │                              alignment, accuracy/fluency/completeness scoring
            └─> MainScreen (Compose)     — Waveform, phoneme timeline, score rings,
                                           word-level and phoneme-level feedback
```

**Key files:**

| File | Role |
|---|---|
| `MainViewModel.kt` | Recording workflow and analysis pipeline orchestrator |
| `Wav2Vec2PhonemeDetector.kt` | ONNX inference engine with CTC greedy decoding |
| `PhonemeDetector.kt` | Vosk timing, CMU dict lookup, NW alignment, scoring, ELPAC mapping |
| `PhonemeModels.kt` | Data classes, ARPABET↔IPA mappings, ELPAC preset phrases |
| `AudioRecorder.kt` | Real-time PCM streaming via Kotlin Flow |
| `MainScreen.kt` | Primary Compose UI (phrase selector, record button, results) |
| `PhonemeViews.kt` | Canvas-based waveform, phoneme timeline, score rings |
| `TranscriptFeedbackSection.kt` | Word-level and phoneme-level feedback UI |

## Tech Stack

- **Kotlin** + Jetpack Compose (Material Design 3)
- **ONNX Runtime Android 1.20.0** for on-device Wav2Vec2 inference
- **Vosk ASR 0.3.47** for word-boundary timing
- **CMU Pronouncing Dictionary** for expected phoneme lookup
- **StateFlow / Coroutines** for async audio processing
- **Accompanist Permissions** for runtime RECORD_AUDIO handling
