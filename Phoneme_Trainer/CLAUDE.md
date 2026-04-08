# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build Commands

```bash
./gradlew assembleDebug          # Build debug APK
./gradlew assembleRelease        # Build release APK
./gradlew build                  # Full build (all variants)
./gradlew test                   # Run unit tests
./gradlew connectedAndroidTest   # Run instrumented tests (requires device/emulator)
./gradlew clean                  # Clean build outputs
```

**Requirements:** Java 17 toolchain, Android SDK 34 (minSdk 26)

## Architecture Overview

ELPAC Training App is an Android app for real-time pronunciation analysis targeting ELPAC (English Language Proficiency Assessment for California). It uses on-device AI to detect phonemes, align them against expected pronunciations, and produce scored feedback.

### Key Components

**`MainViewModel.kt`** — Central orchestrator using StateFlow/coroutines:
- Recording workflow: `startRecording()` → `stopRecording()` → `analyzeRecording()`
- Analysis pipeline: CMU dict lookup → WavLM phoneme detection → Needleman-Wunsch alignment → score computation → ELPAC level mapping
- Model download state machine: `CHECKING → DOWNLOADING → READY | FAILED`

**`PhonemeDetector.kt`** — ML engine orchestrator:
- Primary: `WavLMPhonemeDetector` using `age aware base +` checkpoint (WavLM-base + CTC head, ONNX, ~360 MB)
- Secondary: Vosk ASR (vosk-model-small-en-us-0.15) for word-boundary timing only — confidence values discarded
- CMU Pronouncing Dictionary (~140K words) for expected phoneme lookup
- ARPABET↔IPA mappings, weighted accuracy scoring
- ELPAC rubric: score ≥85→Level 4, ≥70→Level 3, ≥50→Level 2, <50→Level 1

**`WavLMPhonemeDetector.kt`** — ONNX inference engine:
- Input: `input_values` `[1, num_samples]`, output: `logits` `[1, num_frames, 52]`
- CTC greedy decoding → IPA phoneme sequence with real acoustic timing
- Length-mark merge: `ː` token (ID 50) merged into preceding vowel to recover long-vowel forms
- Per-phoneme confidence = mean softmax probability; segments < 0.08 posterior discarded
- SHA-256 model verification via `BuildConfig.WAVLM_MODEL_SHA256` (empty = skip, debug only)

**`PhonemeModels.kt`** — Data classes + `PhonemeInventory`:
- `ARPABET_TO_IPA` maps CMU dict → IPA long vowels (ɑː, ɔː, ɜː) for alignment
- `ESPEAK_TO_ARPABET` reverse map for normalisation

**`AudioRecorder.kt`** — Real-time PCM streaming at 16kHz mono, 16-bit, 100ms chunks via `Flow<ShortArray>`

**`MainScreen.kt`** + **`PhonemeViews.kt`** + **`TranscriptFeedbackSection.kt`** — Compose UI stack with waveform visualization, interactive phoneme timeline, score rings, and word-level feedback

### Data Flow

1. `AudioRecorder` emits PCM chunks → `MainViewModel` accumulates samples + updates live waveform/level meter
2. On stop: `WavLMPhonemeDetector.detectPhonemes()` runs ONNX inference → CTC decode + length-mark merge → IPA phonemes with timing
3. Vosk extracts word-boundary timings (used for word-level UI only)
4. `PhonemeDetector.alignPhonemes()` runs Needleman-Wunsch against CMU dict expected phonemes
5. Results flow back as `AnalysisSession` → `MainUiState` StateFlow → Compose UI recomposes

### State Model

`RecordingState` enum: `IDLE → RECORDING → PROCESSING → DONE | ERROR`

`MainUiState` in `MainViewModel` holds all UI state including waveform points, phoneme results, scores, and ELPAC level.

### Key Data Classes (`PhonemeModels.kt`)

- `PhonemeResult` — single detected phoneme (timing, confidence, score, IPA symbol)
- `PronunciationScore` — aggregate scores (overall, accuracy, fluency, completeness)
- `AnalysisSession` — complete session (audio buffer, phonemes, scores, waveform)
- `PhonemeComparison` — expected vs actual side-by-side

### Assets Required

- `wavlm_phoneme.onnx` — WavLM CTC phoneme model (**downloaded to `filesDir` on first launch — not checked into git**; SHA-256 and URL pinned in `app/build.gradle`)
- `wavlm_vocab.json` — 52-token IPA vocab (**in `app/src/main/assets/`**)
- `vosk-model-small-en-us-0.15/` — Vosk ASR model (for word timing only, copied from assets on first run)
- `cmudict-0.7b` — CMU Pronouncing Dictionary

### Generating Model Files (one-time setup)

```bash
pip install transformers torch
python export_model.py   # from repo root — requires ./age\ aware\ base\ +/ checkpoint, outputs ~360 MB ONNX
```

This produces `wavlm_phoneme.onnx` and `wavlm_vocab.json`. After export:
1. Copy `wavlm_vocab.json` to `app/src/main/assets/`
2. Upload `wavlm_phoneme.onnx` to GitHub Releases (tag `v2.0`)
3. `WAVLM_MODEL_URL` and `WAVLM_MODEL_SHA256` are already set in `app/build.gradle` for v2.0 — update only if uploading a new release

### Known Issues

**AudioRecord emulator behavior:** Never call `AudioRecord.stop()` between recording sessions on the emulator — it switches to synthetic audio. Keep instance alive and drain buffer with `READ_NON_BLOCKING` between sessions. The recording restart bug (constant feedback, no audio on second press) was fixed by this approach; do not re-introduce `stop()` calls.
