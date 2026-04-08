# Scoring Reference

Documents the scoring formulas, constants, and rationale used in `PhonemeDetector.kt`. All constants referenced here correspond directly to named constants in that file.

---

## Overall Score

```
overall = W_ACCURACY · accuracy + W_FLUENCY · fluency + W_COMPLETENESS · completeness
        = 0.55 · accuracy + 0.30 · fluency + 0.15 · completeness
```

Clamped to `[0, 100]`.

The accuracy-heavy weighting mirrors the ELPAC Speaking rubric, which prioritises pronunciation intelligibility above fluency and task completion. These weights are **placeholder defaults** — they should be re-tuned against expert ratings on a held-out dev set before any published claim about overall scores.

---

## Accuracy (`W_ACCURACY = 0.55`)

```
accuracy = (Σ weight(phoneme_i)) / totalExpected × 100
```

Per-phoneme weights (`WEIGHT_*` constants):

| Alignment class | Weight | Rationale |
|---|---|---|
| Exact match | 1.00 | Correct production |
| Near-miss | 0.70 | Acoustically similar; likely intelligible |
| Insertion (extra phoneme, no expected counterpart) | 0.00 | Does not correspond to any expected phoneme |
| Substitution / missing | 0.00 | Wrong or absent phoneme |

`totalExpected` is the number of expected phonemes from the CMU dict lookup, floored at 1 to avoid division by zero.

In free-form mode (no target phrase), accuracy degenerates to the mean softmax posterior of the detected phonemes, scaled to `[0, 100]`.

### Near-miss pairs (`SIMILAR_PAIRS`)

Near-misses receive partial credit (0.70) rather than zero. The set is deliberately narrow:

**Voiced/unvoiced obstruent pairs** — minor allophonic errors that rarely change word meaning in context:

| Voiceless | Voiced |
|---|---|
| p | b |
| t | d |
| k | ɡ |
| f | v |
| s | z |
| ʃ | ʒ |
| θ | ð |
| tʃ | dʒ |

**Long/short vowel pairs** — safety net for length-mark merge failures in the CTC decoder:

| Short | Long |
|---|---|
| ɑ | ɑː |
| ɔ | ɔː |
| i | iː |
| u | uː |
| ɝ | ɜː |

**Intentionally excluded** — phonemically contrastive pairs that alter word identity and must be scored as wrong:

- `ɪ / iː` (ship vs. sheep)
- `ɛ / æ` (bed vs. bad)
- `ʌ / ɑ` (cut vs. cot)
- `ʊ / uː` (book vs. boot)

For a principled alternative, swap `SIMILAR_PAIRS` out for a distinctive-feature distance (e.g. [PanPhon](https://github.com/dmort27/panphon)) so partial-credit decisions are data-driven rather than enumerated.

---

## Fluency (`W_FLUENCY = 0.30`)

```
gap_penalty = min(hesitations × 8 + rushes × 5, 40)
fluency     = max(100 − gap_penalty, 30)
```

Constants:

| Constant | Value | Meaning |
|---|---|---|
| `HESITATION_GAP_MS` | 300 ms | Inter-word gap longer than this is counted as a hesitation |
| `OVERLAP_GAP_MS` | −10 ms | Negative gap larger in magnitude than this is counted as a rush/overlap |
| `PENALTY_PER_HESITATION` | 8 pts | Deducted per hesitation |
| `PENALTY_PER_OVERLAP` | 5 pts | Deducted per overlap |
| `MAX_GAP_PENALTY` | 40 pts | Cap on total gap penalty |
| `MIN_FLUENCY_FLOOR` | 30 pts | Minimum fluency score (prevents zero) |

Gaps are measured between **words** (using Vosk `WordTiming` start/end boundaries) when ≥2 words are detected. Inter-word measurement avoids penalising normal consonant clusters and coarticulation as rushes. The phoneme-level gap measurement is used only as a fallback when no word timings are available. The 300 ms threshold is consistent with typical pause-duration distributions in fluent L2 English speech (Kormos 2006; Cucchiarini et al. 2002). Values should be re-calibrated on a representative corpus.

When fewer than two non-silence phonemes are detected, fluency returns a fixed mid-range constant of `50` (no gaps to measure; the two dimensions are intentionally kept independent).

---

## Completeness (`W_COMPLETENESS = 0.15`)

```
completeness = produced / totalExpected × 100
```

`produced` = number of detected phonemes where `isCorrect = true` (exact matches and near-misses only). Substitutions are **not** counted — a learner who substitutes every phoneme receives 0% completeness, not 100%. This requires that phonemes be passed through `alignPhonemes()` before scoring.

In free-form mode (no target phrase), completeness falls back to `min(phoneme_count / 5, 1) × 100` — a rough proxy for utterance length relative to a 5-phoneme baseline.

---

## ELPAC Level Mapping (`ELPAC_THRESHOLDS`)

| Overall score | Level |
|---|---|
| ≥ 85 | Level 4 – Minimal errors |
| ≥ 70 | Level 3 – Generally intelligible |
| ≥ 50 | Level 2 – Some communication impact |
| < 50 | Level 1 – Significant communication impact |

These thresholds are **placeholder defaults** derived from the ELPAC Speaking rubric descriptors. They must be calibrated against expert ratings on a held-out dev set before any published score is meaningful.

---

## Alignment (Needleman-Wunsch)

Global alignment of detected IPA against expected IPA. DP scores:

| Event | Score |
|---|---|
| Exact match | +2 |
| Near-miss | +1 |
| Mismatch | −1 |
| Gap (insertion or deletion) | −1 |

The traceback produces an `expectedPhoneme` annotation on each detected phoneme. Deletions (expected phoneme with no detected counterpart) do not appear in the output list — they reduce `totalExpected` indirectly via the `PhonemeComparison.totalExpected` field.

**Tie-breaking convention:** When two or more traceback moves yield an equal DP score (e.g., diagonal match vs. a gap), the diagonal (match/mismatch) is always preferred over a gap move. This is the standard greedy convention but it is not the only valid choice — preferring gaps in ties would produce different per-phoneme error counts on ambiguous alignments. Results in the paper were produced with the diagonal-first convention.

---

## Posterior Threshold (`MIN_POSTERIOR = 0.08`)

WavLM-detected phoneme segments with a mean softmax posterior below `0.08` are discarded before scoring. This threshold was set empirically on L2 English speech to suppress noise and boundary artefacts. It should be re-tuned per checkpoint.
