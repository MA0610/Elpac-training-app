# Phoneme Trainer

An Android app for on-device pronunciation analysis targeting the **ELPAC** (English Language Proficiency Assessment for California). It records speech, detects phonemes with a fine-tuned **WavLM-for-CTC** model, aligns them against a CMU-dict reference sequence, and scores accuracy, fluency and completeness against the ELPAC rubric — all on-device, no cloud.

## Features

- Record live speech or upload a WAV file for analysis
- AI-powered phoneme detection using a fine-tuned **WavLM** (Microsoft, ~310 M params) CTC head producing ~52 IPA-like tokens
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
- ~400 MB free storage for the downloaded model

## Setup

### 1. Clone the repo

```bash
git clone https://github.com/MA0610/Elpac-training-app.git
cd Elpac-training-app/Phoneme_Trainer
```

### 2. Get the WavLM model

The ONNX model (~95 MB) is downloaded automatically by the app at first launch from GitHub Releases.

To generate the ONNX yourself instead, place the HuggingFace checkpoint at the repo root as `age aware base +/` and run:

```bash
pip install transformers torch onnx
python export_model.py
```

This produces the ONNX weights and a token-id → symbol vocab JSON. Place the vocab at `Phoneme_Trainer/app/src/main/assets/wavlm_vocab.json`, and either:

- upload the ONNX to GitHub Releases and update `WAVLM_MODEL_URL` / `WAVLM_MODEL_SHA256` in `app/build.gradle`, **or**
- drop it into the app's `filesDir` manually via `adb push` for local testing

### 3. Open in Android Studio

Open the `Phoneme_Trainer/` folder in Android Studio, sync Gradle, and run on a device or emulator.

## Build

```bash
./gradlew assembleDebug      # Debug APK
./gradlew assembleRelease    # Release APK
./gradlew test               # Unit tests
./gradlew clean              # Clean build outputs
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

**Key files**

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
