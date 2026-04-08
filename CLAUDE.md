# Phoneme Trainer — Claude Context

## Project Overview

Android app (Kotlin + Jetpack Compose) for on-device pronunciation analysis targeting ELPAC. Uses a fine-tuned **WavLM-base + CTC head** ONNX model for phoneme detection, Vosk ASR for word timing, CMU Pronouncing Dictionary for expected phonemes, and Needleman-Wunsch alignment to score accuracy, fluency, and completeness.

Single-activity architecture. All state managed in `MainViewModel` via `StateFlow<MainUiState>`.

## Key Source Files

| File | What it does |
|---|---|
| `MainViewModel.kt` | Orchestrates recording → analysis → UI state. Central coroutine scope. |
| `WavLMPhonemeDetector.kt` | ONNX inference + CTC greedy decode + length-mark merge → `List<PhonemeResult>` with IPA, timing, confidence |
| `PhonemeDetector.kt` | Vosk word timing extraction, CMU dict lookup, Needleman-Wunsch alignment, scoring |
| `PhonemeModels.kt` | All data classes + enums + `ElpacPhrases` (22 presets) + `PhonemeInventory` (ARPABET↔IPA) |
| `AudioRecorder.kt` | Streams 16kHz PCM via `Flow<ShortArray>` in 100ms chunks; persists `AudioRecord` instance |
| `MainScreen.kt` | Root composable: phrase selector, record button, score display, results |
| `PhonemeViews.kt` | Canvas-based: `WaveformCanvas`, `PhonemeTimeline`, `ScoreRings` |
| `TranscriptFeedbackSection.kt` | Word-level and phoneme-level feedback with expected vs. actual comparison |

## Audio Recording

- `AudioRecorder` maintains a **persistent `AudioRecord` instance** across sessions (do not call `stop()` between sessions — see known issue below).
- Streams 1600-sample (100ms) chunks to `MainViewModel`, which accumulates them in `allSamples: MutableList<Short>`.
- Minimum valid recording: 8000 samples (0.5s).
- On stop, `MainViewModel` cancels the recording coroutine, then calls `analyzeRecording()`.

## Detection Pipeline

1. **WavLM ONNX** (`WavLMPhonemeDetector`):
   - Model: `age aware base +` checkpoint — WavLM-base (~94M params) fine-tuned with CTC head
   - Input: float32 PCM normalized to `[-1, 1]`, shape `[1, numSamples]`
   - Output: logits `[1, numFrames, 52]`
   - CTC greedy decode: softmax → argmax per frame → collapse consecutive duplicates → skip PAD (ID 0)
   - **Length-mark merge**: `ː` token (ID 50) merged into preceding vowel to recover long-vowel forms (`iː`, `uː`, `ɑː`, `ɔː`, `ɜː`); dangling marks dropped
   - Min confidence threshold: 0.08 (mean posterior per segment)
   - Frame→time mapping derived from `samples / frames`

2. **Vosk ASR** (`PhonemeDetector.extractWordTimings()`):
   - Used only for word-boundary timing (start/end ms per word)
   - Vosk confidence values are discarded

3. **CMU Dict lookup** (`PhonemeDetector.getPhraseExpectedPhonemes()`):
   - Tokenizes phrase → looks up each word in `cmudict-0.7b` (140K entries, bundled as asset)
   - Converts ARPABET → IPA via `PhonemeInventory.ARPABET_TO_IPA` (eSpeak-compatible long vowel forms)
   - OOV words contribute zero phonemes (no fallback)

4. **Needleman-Wunsch alignment** (`PhonemeDetector.alignPhonemes()`):
   - Global alignment of actual vs. expected IPA sequences
   - Scoring: exact match +2, near-miss +1, mismatch -1, gap -1
   - Near-miss pairs: voiced/unvoiced obstruent pairs only; phonemically contrastive pairs (e.g. `ɪ/iː`, `ɛ/æ`) are **not** near-misses

5. **Overall score** (`PhonemeDetector.computeOverallScore()`):
   - Accuracy 55% + Fluency 30% + Completeness 15%
   - Fluency: gaps measured between **words** (Vosk `WordTiming`) when ≥2 words available, else inter-phoneme; hesitation >300ms (−8 pts each, cap 40), overlap <−10ms (−5 pts each), floor 30; returns fixed 50 for <2 phonemes
   - Completeness: counts only `isCorrect` phonemes (exact + near-miss); substitutions do not count
   - NW tie-breaking: diagonal preferred over gap on equal DP score (documented in `docs/SCORING.md`)
   - ELPAC mapping: ≥85→L4, ≥70→L3, ≥50→L2, <50→L1

## Models & Assets

| Asset | Location | Size | Notes |
|---|---|---|---|
| `wavlm_phoneme.onnx` | Downloaded to `filesDir` at first launch | ~360 MB | IR version 10, requires ONNX Runtime 1.20.0+ |
| `wavlm_vocab.json` | `app/src/main/assets/` | ~6 KB | Token ID → IPA symbol mapping (52 tokens) |
| `cmudict-0.7b` | `app/src/main/assets/` | ~3.7 MB | CMU Pronouncing Dictionary |
| `vosk-model-small-en-us-0.15/` | `app/src/main/assets/` | ~50 MB | Copied to `filesDir` on first run |

## Scoring Details

```
accuracy     = weighted match rate (weightedAccuracy() is single source of truth)
               exact match   → weight 1.0
               near-miss     → weight 0.7
               insertion     → weight 0.0
               wrong/missing → weight 0.0

fluency      = 100 - gap_penalty, floor 30
               gaps measured between words (Vosk WordTiming) when ≥2 words exist,
               else between consecutive phonemes
               gap > 300ms   → hesitation penalty (8 pts each, cap 40)
               gap < -10ms   → overlap penalty (5 pts each, cap 40)
               <2 phonemes   → fixed 50 (independent of accuracy)

completeness = (isCorrect phonemes) / (total expected phonemes)
               substitutions do NOT count; only exact matches and near-misses

overall      = 0.55 * accuracy + 0.30 * fluency + 0.15 * completeness
```

## Model Verification

- SHA-256 hash checked against `BuildConfig.WAVLM_MODEL_SHA256`
- Mismatch → file deleted + `SecurityException`
- Empty hash disables verification (debug only)
- One-time migration renames legacy `wav2vec2_phoneme.onnx` → `wavlm_phoneme.onnx` on first run

## UI State Machine

`RecordingState` enum controls UI:
- `IDLE` → user can select phrase and press record
- `RECORDING` → live waveform + level meter, stop button shown
- `PROCESSING` → spinner shown while analysis runs
- `DONE` → score rings, phoneme timeline, transcript feedback shown
- `ERROR` → error message shown

## Known Issues

### AudioRecord emulator behavior
Never call `AudioRecord.stop()` between recording sessions on the emulator. The emulator switches to a synthetic tone fallback if `AudioRecord` is stopped and recreated. Instead, keep the `AudioRecord` instance alive and drain the buffer using `READ_NON_BLOCKING` between sessions.

### Recording restart/reset
There was a bug where pressing record again (after a session) caused constant audio feedback and no audio captured on subsequent presses. Fixed by maintaining persistent `AudioRecord` state. Do not re-introduce `stop()` calls between sessions.

## Build

- `compileSdk 34`, `minSdk 26`, `targetSdk 34`
- Kotlin jvmToolchain 17
- Key dependencies: `onnxruntime-android:1.20.0`, `vosk-android:0.3.47`, `accompanist-permissions:0.32.0`
- Compose BOM: `2023.10.01`
- `buildConfig true` required — `WAVLM_MODEL_URL` and `WAVLM_MODEL_SHA256` must be set in `app/build.gradle` before release

## Permissions

- `RECORD_AUDIO` — requested at runtime via Accompanist
- `INTERNET` — for one-time model download at first launch
