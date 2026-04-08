# ACM Submission — Issues To Fix

## Critical

- [x] **Verify vocab is IPA, not character-level**
  Verified: tokens 3–24 are single-character IPA symbols (b, d, f, h, i, j, k, l, m, n, p, s, t, u, v, w, z are valid IPA glyphs). The mixed vocab (ASCII-compatible IPA symbols + IPA-specific glyphs like ŋ, ɑ, ʃ) is consistent with the eSpeak IPA encoding used by the checkpoint. The model is a phoneme classifier, not character-level ASR.

- [x] **Fix affricate Unicode mismatch**
  Fixed: `wavlm_vocab.json` tokens 48 and 49 changed from ligatures `ʤ` (U+02A4) and `ʧ` (U+02A7) to two-character sequences `dʒ` and `tʃ`, matching `PhonemeModels.kt` `ARPABET_TO_IPA` and `SIMILAR_PAIRS`.

- [x] **Fix completeness counting substitutions as "produced"**
  Fixed: `PhonemeDetector.kt` `computeCompleteness` now uses `nonSil.count { it.isCorrect }` (exact matches and near-misses only) instead of `nonSil.count { it.expectedPhoneme != null }`.

- [x] **Fix or replace `export_model.py`**
  Fixed: script now exports from the local `./age aware base +/` checkpoint (not `facebook/wav2vec2-lv-60-espeak-cv-ft`), uses correct output filenames (`wavlm_phoneme.onnx`, `wavlm_vocab.json`), and drops the `attention_mask` input (not required by WavLM-base for inference).

- [x] **Fix model parameter count — inconsistent across documents**
  Fixed: `CLAUDE.md` updated to ~94M (WavLM-base actual count). Previous figures of ~310M / ~300M were wav2vec2-large figures incorrectly carried over.

## Important

- [x] **Fix fluency metric — penalizes coarticulation, not pauses**
  Fixed: `computeFluency` now uses inter-word gaps from Vosk `WordTiming` when ≥2 words are available, falling back to inter-phoneme gaps only when no word timings exist.

- [x] **Remove or justify fluency fallback for short utterances**
  Fixed: `computeFluency` now returns `50f` (fixed mid-range constant) for utterances with fewer than 2 detected phonemes, instead of coupling to accuracy via `accuracy * 0.9f`.

- [x] **Fix zero-speech fluency violating the documented floor**
  Fixed: `computeOverallScore` early return for zero non-silence phonemes now uses `MIN_FLUENCY_FLOOR` (30f) for the fluency component instead of 0f.

- [x] **Document NW traceback tie-breaking**
  Documented in `docs/SCORING.md` (Alignment section): diagonal move preferred over gaps on ties; this is the standard greedy convention; results in the paper use this convention.

- [x] **Fix OOV phantom phoneme in per-word UI**
  Fixed: `MainViewModel.kt` `perWordPhonemeCount` no longer applies `coerceAtLeast(1)` to OOV words. OOV words now contribute 0 expected phoneme slots.


Still needs human attention (not code):                                                                               
                                                                                                                      
  1. Paper write-up — The NW tie-breaking convention, word-level fluency (with Kormos 2006 / Cucchiarini et al. 2002    
  citations), and updated completeness definition should be reflected in your ACM submission text.                    
  2. Model re-export (optional for demo) — If you want to produce a clean reproducible artifact, run the updated        
  export_model.py with the age aware base + checkpoint and upload a new GitHub Release. The currently deployed v2.0     
  model still works fine at runtime since the inference code only passes input_values.
  3. SHA-256 in build.gradle — If you re-export the model, update WAVLM_MODEL_SHA256 to match the new file.    