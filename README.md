# Phoneme Trainer

An Android app for on-device pronunciation analysis targeting the **ELPAC** (English Language Proficiency Assessment for California). It records speech or accepts a WAV file upload, detects phonemes with a fine-tuned **WavLM-for-CTC** model, aligns them against a CMU-dict reference sequence, and scores accuracy, fluency and completeness against the ELPAC rubric — all on-device, no cloud.

## Features

- Record live speech or upload a WAV file for analysis
- AI-powered phoneme detection using a fine-tuned **WavLM** (Microsoft, ~310 M params) CTC head producing ~52 IPA-like tokens
- Phoneme alignment against 140K+ words via the CMU Pronouncing Dictionary
- ELPAC rubric (Levels 1–4) with accuracy, fluency and completeness sub-scores
- 22 preset ELPAC-aligned target phrases, plus custom phrase input
- Live waveform visualisation and RMS level meter during recording
- Word-level and phoneme-level feedback with expected vs. actual comparison
- SHA-256 pinning of the downloaded model — a tampered or corrupted file is rejected rather than executed
- Fully on-device — no data leaves the phone

## How It Works

### Audio Input

The app accepts audio from two sources, both feeding the same analysis pipeline:

**Live recording** — Speech is captured via Android `AudioRecord` at **16 kHz mono, 16-bit PCM**. Samples stream in 100 ms chunks via a Kotlin `Flow` while a live waveform and level meter update in real time. A persistent `AudioRecord` instance is reused across sessions to avoid the emulator falling back to its synthetic tone source when `stop()` is called between recordings.

**WAV file upload** — Tap "Upload audio file" (visible in IDLE and DONE states) to pick any WAV file from the device. `MainViewModel.readWavSamples()` parses the RIFF/WAVE header, scans for `fmt ` and `data` chunks, and validates the format before reading. Accepted format: **uncompressed PCM, mono, 16 kHz, 16-bit**. Files that don't match produce a clear error message. The resulting `ShortArray` is handed directly to `analyzeRecording()` — identical to what the live recorder produces.

### Phoneme Detection (WavLM-for-CTC)

The core model is a fine-tuned **WavLM-base + CTC head** (`age aware base +`) exported to ONNX (~360 MB float32). It runs fully on-device via ONNX Runtime Android 1.20.0:

1. PCM samples are normalised to float `[-1, 1]` and fed as `input_values` of shape `[1, num_samples]`
2. The model outputs `logits` of shape `[1, num_frames, 52]` — one distribution over 52 IPA-like tokens per time frame
3. **CTC greedy decoding**: softmax → argmax per frame → collapse consecutive duplicate tokens (PAD/blank resets the run) → skip special/non-phoneme tokens
4. **Length-mark merge**: the vocabulary includes `ː` (id 50) as a standalone token. It is merged into the preceding vowel segment so the decoder output (`iː`, `uː`, `ɑː`, `ɔː`, `ɜː`) matches the long-vowel forms produced by CMU-dict lookup. A dangling `ː` at the start of an utterance is dropped
5. **Posterior threshold**: segments whose mean softmax posterior falls below `0.08` are discarded as noise/boundary artefacts
6. Frame→time mapping is derived from the actual `samples / frames` ratio, so it stays correct even if the feature-extractor stride changes

Each emitted phoneme carries a real acoustic timestamp and a confidence equal to the mean softmax posterior of the assigned token across its frames — a direct Bayesian posterior, not a proxy.

### Model Verification

Before the ONNX session is created, the downloaded file is hashed with SHA-256 and compared against a pinned value in `BuildConfig.WAVLM_MODEL_SHA256`. A mismatch deletes the file and raises a `SecurityException`, which the ViewModel surfaces as a clear "model verification failed" error — the app will **not** auto-retry or execute an unverified graph. Leaving the pinned hash blank disables verification for debug-only convenience.

A one-time migration renames any pre-existing `wav2vec2_phoneme.onnx` in `filesDir` to `wavlm_phoneme.onnx` so users on old builds don't have to re-download.

### Word Timing (Vosk)

Vosk ASR (`vosk-model-small-en-us-0.15`) runs in parallel to extract word-boundary timestamps (start/end ms per word). **Vosk confidence values are intentionally discarded** — Vosk contributes timing only, never phoneme or word identity.

### Expected Phoneme Lookup (CMU Dictionary)

The target phrase is tokenised into words, looked up in the bundled CMU Pronouncing Dictionary (~140K entries), and converted from ARPABET to IPA using eSpeak-compatible mappings (`ɑː`, `ɔː`, `ɜː`, etc.). Out-of-vocabulary words contribute **zero** expected phonemes — there is no naive grapheme-to-phoneme fallback, because the previous one silently corrupted the reference sequence for any novel vocabulary.

### Alignment (Needleman-Wunsch)

Detected IPA is globally aligned against expected IPA using Needleman-Wunsch:

| Alignment type | DP score |
|---|---|
| Exact match | +2 |
| Near-miss (voiced/unvoiced pair, long/short vowel safety-net) | +1 |
| Substitution | −1 |
| Gap (insertion or deletion) | −1 |

The traceback annotates each detected phoneme with `isCorrect` and `expectedPhoneme` for the feedback UI.

**Near-miss set is deliberately narrow.** It contains only voiced/unvoiced obstruent pairs (`p/b`, `t/d`, `k/ɡ`, `f/v`, `s/z`, `ʃ/ʒ`, `θ/ð`, `tʃ/dʒ`) and long/short forms of the *same* vowel as a failsafe for length-mark merging. Phonemically contrastive pairs like `ɪ/iː` (ship/sheep), `ɛ/æ` (bed/bad), and `ʌ/ɑ` (cut/cot) are **not** near-misses — they alter word identity and must be scored as wrong.

### Scoring

| Component | Weight | How computed |
|---|---|---|
| Accuracy | 55% | Weighted match rate over expected phonemes. `exact=1.00`, `near-miss=0.70`, `insertion=0.00`, `wrong/missing=0.00` |
| Fluency | 30% | `100 − gap_penalty`, floored at 30. Penalty: 8 per hesitation (gap > 300 ms), 5 per overlap (gap < −10 ms), capped at 40 |
| Completeness | 15% | Fraction of expected phonemes that were produced (had a non-null `expectedPhoneme`) |
| **Overall** | — | `0.55·accuracy + 0.30·fluency + 0.15·completeness`, clamped to `[0, 100]` |

In free-form mode (no target phrase) accuracy degenerates to the mean acoustic posterior.

`PhonemeDetector.weightedAccuracy()` is the single source of truth, and is called both by `computeOverallScore()` and by `MainViewModel.buildComparison()` so the top-line ELPAC score and the per-phrase accuracy card can never drift out of sync.

### ELPAC Level Mapping

| Overall score | Level |
|---|---|
| ≥ 85 | Level 4 – Minimal errors |
| ≥ 70 | Level 3 – Generally intelligible |
| ≥ 50 | Level 2 – Some communication impact |
| < 50 | Level 1 – Significant communication impact |

> These thresholds and the component weights are **placeholder defaults** derived from the rubric descriptors. They should be re-calibrated against expert ratings on a held-out dev set before any published claim about overall scores.

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

At first launch the app downloads `wavlm_phoneme.onnx` (~360 MB) from the pinned GitHub Releases URL, verifies its SHA-256, and caches it in the app's `filesDir`.

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
AudioRecorder (16 kHz PCM, 100 ms chunks via Flow)  ──┐
WAV file upload (Uri → readWavSamples())             ──┤
                                                       ▼
                                    MainViewModel  (StateFlow<MainUiState>)
                                            ├─> PhonemeDetector
                                            │       ├─> WavLMPhonemeDetector  — ONNX inference, CTC decode,
                                            │       │                            length-mark merge, posterior threshold
                                            │       ├─> Vosk (vosk-model-small-en-us-0.15)  — word-boundary timing only
                                            │       ├─> CMU Pronouncing Dictionary (cmudict-0.7b)
                                            │       ├─> Needleman-Wunsch alignment
                                            │       └─> Weighted accuracy / fluency / completeness → ELPAC level
                                            └─> MainScreen (Compose)
                                                    ├─> Waveform + level meter
                                                    ├─> PhonemeTimeline + ScoreRings
                                                    └─> TranscriptFeedbackSection  (word + phoneme feedback)
```

**Key files**

| File | Role |
|---|---|
| `MainViewModel.kt` | Recording and file-upload workflows; WAV parsing (`readWavSamples`); analysis-pipeline orchestrator; model status state machine |
| `WavLMPhonemeDetector.kt` | ONNX download + SHA-256 verify + session init; PCM normalisation; CTC greedy decode with length-mark merge and posterior threshold |
| `PhonemeDetector.kt` | Vosk word timing, CMU-dict lookup, Needleman-Wunsch alignment, weighted accuracy / fluency / completeness, ELPAC mapping |
| `PhonemeModels.kt` | Data classes, ARPABET↔IPA mappings, 22 ELPAC preset phrases |
| `AudioRecorder.kt` | Real-time PCM streaming via `Flow<ShortArray>`; persistent `AudioRecord` |
| `MainScreen.kt` | Primary Compose UI (phrase selector, record button, file upload button, results) |
| `PhonemeViews.kt` | Canvas-based waveform, phoneme timeline, score rings |
| `TranscriptFeedbackSection.kt` | Word-level and phoneme-level feedback UI |

## Tech Stack

- **Kotlin** + Jetpack Compose (Material Design 3)
- **ONNX Runtime Android 1.20.0** for on-device WavLM inference (ONNX IR v10)
- **Vosk ASR 0.3.47** for word-boundary timing only
- **CMU Pronouncing Dictionary** for expected phoneme lookup
- **StateFlow / Coroutines** for async audio processing, with a recording mutex to serialise start/stop tap sequences
- **Accompanist Permissions 0.32.0** for runtime `RECORD_AUDIO` handling

## Known Issues

### AudioRecord emulator behaviour

Never call `AudioRecord.stop()` between recording sessions on the emulator — the emulator switches to a synthetic tone fallback if `AudioRecord` is stopped and recreated. The fix is to keep the `AudioRecord` instance alive and drain the buffer between sessions. Previous symptoms were a constant audio feedback loop and no audio being captured on subsequent presses after the first recording.

## What Still Needs to Be Done

- [ ] **Calibrate the placeholders.** `MIN_POSTERIOR = 0.08`, the three component weights (0.55 / 0.30 / 0.15), and the ELPAC threshold table are all placeholder defaults. They need to be re-tuned on a held-out dev set against expert ratings before any published score is meaningful (see `docs/SCORING.md`)
- [ ] **Consider swapping the hand-tuned near-miss set** for a distinctive-feature distance (e.g. PanPhon) so partial-credit decisions are principled rather than enumerated
