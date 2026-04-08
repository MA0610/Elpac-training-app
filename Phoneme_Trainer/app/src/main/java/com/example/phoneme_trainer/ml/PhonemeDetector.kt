package com.example.phoneme_trainer.ml

import android.content.Context
import android.util.Log
import com.example.phoneme_trainer.BuildConfig
import com.example.phoneme_trainer.audio.AudioRecorder
import com.example.phoneme_trainer.model.DetectionResult
import com.example.phoneme_trainer.model.PhonemeComparison
import com.example.phoneme_trainer.model.PhonemeInventory
import com.example.phoneme_trainer.model.PhonemeResult
import com.example.phoneme_trainer.model.PronunciationScore
import com.example.phoneme_trainer.model.WordTiming
import org.json.JSONObject
import org.vosk.Model
import org.vosk.Recognizer
import java.io.File
import kotlin.math.min

/**
 * End-to-end phoneme detection and scoring pipeline.
 *
 * Architecture:
 *   • [WavLMPhonemeDetector] — primary acoustic engine. Produces IPA phoneme sequences
 *     with real softmax posteriors from a WavLM-for-CTC checkpoint. Long-vowel forms are
 *     recovered by merging the "ː" length mark in the CTC post-processor.
 *   • Vosk (vosk-model-small-en-us-0.15) — used *only* for word-boundary timing. Its
 *     word-confidence scores are intentionally discarded.
 *   • CMU Pronouncing Dictionary — source of expected phoneme sequences for the target
 *     phrase.
 *   • [alignPhonemes] — Needleman-Wunsch global alignment of detected vs. expected IPA.
 *   • [computeOverallScore] — weighted accuracy + fluency + completeness → ELPAC level.
 *
 * If WavLM is unavailable (download failed, hash mismatch) detection returns empty; the
 * UI is expected to surface a clear model-status error rather than invent phonemes. No
 * random / DSP / rule-based fallback exists in this pipeline — previous revisions had one
 * and it produced misleading scores.
 */
class PhonemeDetector(private val context: Context) {

    companion object {
        private const val TAG             = "PhonemeDetector"
        private const val VOSK_MODEL_DIR  = "vosk-model-small-en-us-0.15"
        private const val CMU_DICT_ASSET  = "cmudict-0.7b"
        private const val SAMPLE_RATE     = AudioRecorder.SAMPLE_RATE

        // ── Scoring rubric constants ─────────────────────────────────────────
        // The three component weights sum to 1.0. The accuracy-heavy split mirrors the
        // ELPAC Speaking rubric, in which pronunciation intelligibility dominates and
        // fluency/completeness are secondary. These weights are configuration, not
        // calibrated coefficients — they should be re-tuned on a held-out dev set before
        // any published claim about overall scores.
        private const val W_ACCURACY     = 0.55f
        private const val W_FLUENCY      = 0.30f
        private const val W_COMPLETENESS = 0.15f

        // Per-alignment-class weights used inside the weighted accuracy formula.
        // Justification is documented in docs/SCORING.md.
        private const val WEIGHT_EXACT      = 1.00
        private const val WEIGHT_NEAR_MISS  = 0.70
        private const val WEIGHT_INSERTION  = 0.00   // insertions do not count toward accuracy
        private const val WEIGHT_SUBSTITUTE = 0.00

        // Fluency: gap thresholds in milliseconds. Defaults based on typical pause
        // distributions in fluent L2 English speech (see docs/SCORING.md for citations).
        private const val HESITATION_GAP_MS = 300L
        private const val OVERLAP_GAP_MS    = -10L
        private const val PENALTY_PER_HESITATION = 8f
        private const val PENALTY_PER_OVERLAP    = 5f
        private const val MAX_GAP_PENALTY        = 40f
        private const val MIN_FLUENCY_FLOOR      = 30f

        // ELPAC Speaking rubric thresholds (0-100 overall score → rubric level string).
        // Thresholds should be calibrated against expert ratings — the values here are
        // placeholder defaults derived from the rubric descriptors.
        private val ELPAC_THRESHOLDS = listOf(
            85f to "Level 4 – Minimal errors",
            70f to "Level 3 – Generally intelligible",
            50f to "Level 2 – Some communication impact",
            0f  to "Level 1 – Significant communication impact"
        )

        /**
         * "Near-miss" phoneme pairs — substitutions that are acoustically or phonetically
         * close enough to warrant partial credit rather than a full mismatch.
         *
         * This list is deliberately narrow: it contains only
         *  (a) voiced/unvoiced obstruent pairs (minor allophonic errors that do not change
         *      word meaning in most contexts), and
         *  (b) long/short forms of the same vowel (safety net if the length-mark merge in
         *      the CTC decoder fails to fire).
         *
         * Phonemically contrastive pairs like {ɪ, iː} (ship/sheep), {ɛ, æ} (bed/bad), or
         * {ʌ, ɑ} (cut/cot) are intentionally *not* near-misses — in pronunciation
         * assessment they must be scored as wrong because they alter word identity. For
         * an expert-tuned alternative, swap this set out for a distinctive-feature
         * distance (e.g. PanPhon) in future work.
         */
        private val SIMILAR_PAIRS: Set<Set<String>> = setOf(
            // Voiced / unvoiced obstruent pairs
            setOf("p", "b"), setOf("t", "d"), setOf("k", "ɡ"),
            setOf("f", "v"), setOf("s", "z"), setOf("ʃ", "ʒ"),
            setOf("θ", "ð"), setOf("tʃ", "dʒ"),
            // Model-artifact long/short vowel pairs (length-mark merge failsafe)
            setOf("ɑ", "ɑː"), setOf("ɔ", "ɔː"), setOf("i", "iː"),
            setOf("u", "uː"), setOf("ɝ", "ɜː")
        )

        private fun isNearMiss(a: String, b: String): Boolean =
            a != b && SIMILAR_PAIRS.any { it.contains(a) && it.contains(b) }
    }

    // ── Models ───────────────────────────────────────────────────────────────

    private var voskModel: Model? = null
    private val wavlm = WavLMPhonemeDetector(context)
    private val cmuDict: Map<String, List<String>> by lazy { loadCmuDict() }

    init {
        loadVoskModel()
        if (BuildConfig.DEBUG && wavlm.isAvailable) {
            Log.i(TAG, "WavLM ready — using softmax-posterior acoustic scoring.")
        }
    }

    // ── Vosk loading (word timing only) ──────────────────────────────────────

    private fun loadVoskModel() {
        try {
            copyVoskFromAssets()
            val path = File(context.filesDir, VOSK_MODEL_DIR).absolutePath
            voskModel = Model(path)
            if (BuildConfig.DEBUG) Log.i(TAG, "Vosk model loaded: $path")
        } catch (e: Exception) {
            // Word-level timing is optional; the app still scores phonemes without it.
            Log.w(TAG, "Vosk unavailable — word-level UI will fall back to phoneme order. " +
                    "Reason: ${e.message}")
        }
    }

    private fun copyVoskFromAssets() {
        val dest = File(context.filesDir, VOSK_MODEL_DIR)
        if (dest.exists() && dest.list()?.isNotEmpty() == true) return
        dest.mkdirs()
        copyAssetFolder(VOSK_MODEL_DIR, dest)
        if (BuildConfig.DEBUG) Log.i(TAG, "Vosk model copied from assets.")
    }

    private fun copyAssetFolder(assetPath: String, destDir: File) {
        val assets   = context.assets
        val children = assets.list(assetPath) ?: return
        if (children.isEmpty()) {
            assets.open(assetPath).use { src ->
                File(destDir.parent, destDir.name).outputStream().use { src.copyTo(it) }
            }
        } else {
            destDir.mkdirs()
            for (child in children) copyAssetFolder("$assetPath/$child", File(destDir, child))
        }
    }

    // ── CMU dict loading ─────────────────────────────────────────────────────

    private fun loadCmuDict(): Map<String, List<String>> {
        val dict = HashMap<String, List<String>>(140_000)
        try {
            context.assets.open(CMU_DICT_ASSET).bufferedReader().useLines { lines ->
                for (line in lines) {
                    if (line.startsWith(";;;") || line.isBlank()) continue
                    val sep = line.indexOf("  ")
                    if (sep < 0) continue
                    val word = line.substring(0, sep)
                        .replace(Regex("\\(\\d+\\)$"), "")
                        .lowercase()
                    val phonemes = line.substring(sep + 2).trim()
                        .split(" ")
                        .map { it.trimEnd('0', '1', '2') }
                    if (!dict.containsKey(word)) dict[word] = phonemes
                }
            }
            if (BuildConfig.DEBUG) Log.i(TAG, "CMU dict loaded: ${dict.size} entries")
        } catch (e: Exception) {
            Log.w(TAG, "CMU dict unavailable: ${e.message}")
        }
        return dict
    }

    // ── Model preparation (download + hash verify) ──────────────────────────

    suspend fun prepareWavLM(onProgress: (Float) -> Unit) = wavlm.downloadAndInit(onProgress)

    // ── Phrase → expected phonemes ───────────────────────────────────────────

    /**
     * Looks up the expected IPA phoneme sequence for a phrase. Out-of-vocabulary words
     * contribute zero phonemes rather than invoking a naive grapheme-to-phoneme fallback
     * — a naive G2P fallback silently corrupted the expected sequence for any novel
     * vocabulary and is no longer supported.
     */
    fun getPhraseExpectedPhonemes(phrase: String): List<String> {
        return phrase.trim()
            .split("\\s+".toRegex())
            .flatMap { word ->
                lookupPhonemes(word).mapNotNull { PhonemeInventory.ARPABET_TO_IPA[it] }
            }
    }

    /** Expected IPA phonemes for a single word; empty list on OOV. */
    fun getWordExpectedPhonemes(word: String): List<String> =
        lookupPhonemes(word).mapNotNull { PhonemeInventory.ARPABET_TO_IPA[it] }

    private fun lookupPhonemes(word: String): List<String> {
        val w = word.lowercase().trimEnd('.', ',', '?', '!', '\'', '"', ';', ':')
        val hit = cmuDict[w]
        if (hit == null && BuildConfig.DEBUG) Log.d(TAG, "OOV word — no expected phonemes: '$w'")
        return hit ?: emptyList()
    }

    // ── Detection entry point ───────────────────────────────────────────────

    /**
     * Runs WavLM phoneme detection and, when available, Vosk word-boundary timing.
     * Returns an empty [DetectionResult] if WavLM is unavailable — callers must
     * surface this as a clear error rather than treating an empty result as "silence".
     */
    fun detect(samples: ShortArray): DetectionResult {
        if (samples.isEmpty() || !wavlm.isAvailable) {
            return DetectionResult(emptyList(), emptyList())
        }
        val phonemes    = wavlm.detectPhonemes(samples)
        val wordTimings = if (voskModel != null) extractWordTimings(samples) else emptyList()
        return DetectionResult(phonemes, wordTimings)
    }

    // ── Vosk word-timing extraction ─────────────────────────────────────────

    /**
     * Extracts word-level timing boundaries (word, startMs, endMs) from Vosk.
     * Confidence values are intentionally discarded — Vosk is used only for timing,
     * not for phoneme or word recognition.
     */
    fun extractWordTimings(samples: ShortArray): List<WordTiming> {
        val model = voskModel ?: return emptyList()
        return try {
            val rec = Recognizer(model, SAMPLE_RATE.toFloat())
            rec.setWords(true)
            val bytes     = shortsToBytes(samples)
            val chunkSize = 8000
            var offset    = 0
            while (offset < bytes.size) {
                val end = minOf(offset + chunkSize, bytes.size)
                rec.acceptWaveForm(bytes.copyOfRange(offset, end), end - offset)
                offset = end
            }
            val json = rec.finalResult
            rec.close()
            parseWordTimings(json)
        } catch (e: Exception) {
            Log.w(TAG, "Word timing extraction failed: ${e.message}")
            emptyList()
        }
    }

    private fun parseWordTimings(json: String): List<WordTiming> {
        return try {
            val obj       = JSONObject(json)
            val wordArray = obj.optJSONArray("result") ?: return emptyList()
            (0 until wordArray.length()).map { i ->
                val w = wordArray.getJSONObject(i)
                WordTiming(
                    word    = w.getString("word").lowercase().trim(),
                    startMs = (w.getDouble("start") * 1000).toLong(),
                    endMs   = (w.getDouble("end")   * 1000).toLong()
                )
            }
        } catch (e: Exception) {
            Log.w(TAG, "parseWordTimings: ${e.message}")
            emptyList()
        }
    }

    // ── Needleman-Wunsch alignment ──────────────────────────────────────────

    /**
     * Globally aligns the detected IPA phoneme sequence against the expected sequence
     * using Needleman-Wunsch. The returned list contains exactly the input [actual]
     * phonemes, annotated with [PhonemeResult.isCorrect] and [PhonemeResult.expectedPhoneme]
     * for downstream feedback UI.
     *
     * Per-phoneme `score` fields are **not** modified here — aggregate accuracy is the
     * concern of [computeOverallScore], and combining two independent penalty loci made
     * the previous implementation double-count insertions.
     *
     * Scoring inside the DP:
     *   exact match        +2
     *   near-miss pair     +1
     *   mismatch           -1
     *   insertion/deletion -1  (gap)
     */
    fun alignPhonemes(
        actual: List<PhonemeResult>,
        expected: List<String>
    ): List<PhonemeResult> {
        if (actual.isEmpty()) return emptyList()
        if (expected.isEmpty()) return actual

        val n = actual.size
        val m = expected.size

        val gap      = -1
        val match    =  2
        val nearMiss =  1
        val mismatch = -1

        fun sim(a: String, e: String): Int = when {
            a == e             -> match
            isNearMiss(a, e)   -> nearMiss
            else               -> mismatch
        }

        val dp = Array(n + 1) { IntArray(m + 1) }
        for (i in 0..n) dp[i][0] = i * gap
        for (j in 0..m) dp[0][j] = j * gap
        for (i in 1..n) for (j in 1..m) {
            val s = sim(actual[i - 1].phoneme, expected[j - 1])
            dp[i][j] = maxOf(
                dp[i - 1][j - 1] + s,
                dp[i - 1][j] + gap,
                dp[i][j - 1] + gap
            )
        }

        // Traceback.
        val alignedActual   = ArrayDeque<String?>()
        val alignedExpected = ArrayDeque<String?>()
        var i = n; var j = m
        while (i > 0 || j > 0) {
            when {
                i > 0 && j > 0 &&
                        dp[i][j] == dp[i - 1][j - 1] + sim(actual[i - 1].phoneme, expected[j - 1]) -> {
                    alignedActual.addFirst(actual[i - 1].phoneme)
                    alignedExpected.addFirst(expected[j - 1])
                    i--; j--
                }
                i > 0 && dp[i][j] == dp[i - 1][j] + gap -> {
                    alignedActual.addFirst(actual[i - 1].phoneme)
                    alignedExpected.addFirst(null)
                    i--
                }
                else -> {
                    alignedActual.addFirst(null)
                    alignedExpected.addFirst(expected[j - 1])
                    j--
                }
            }
        }

        // Walk the alignment and annotate the original actual phonemes in-order.
        val annotated = ArrayList<PhonemeResult>(actual.size)
        var actualIdx = 0
        val aActual = alignedActual.toList()
        val aExpect = alignedExpected.toList()
        for (k in aActual.indices) {
            val actSym = aActual[k] ?: continue  // deletion — no phoneme to annotate
            val expSym = aExpect[k]
            val ph = actual[actualIdx++]
            val isExact    = expSym != null && actSym == expSym
            val isNearMiss = expSym != null && isNearMiss(actSym, expSym)
            annotated.add(ph.copy(
                isCorrect       = isExact || isNearMiss,
                expectedPhoneme = expSym
            ))
        }
        return annotated
    }

    // ── Overall scoring ─────────────────────────────────────────────────────

    /**
     * Three-component ELPAC-style pronunciation score. All components are in [0, 100].
     *
     *   accuracy     = weighted match rate over expected phonemes.
     *                  exact → 1.00, near-miss → 0.70, insertion → 0.00, wrong/missing → 0.00
     *   fluency      = 100 − gap penalty (hesitations >300ms or negative overlaps)
     *   completeness = produced expected phonemes / total expected phonemes
     *
     *   overall      = 0.55·accuracy + 0.30·fluency + 0.15·completeness
     *
     * When no target phrase is supplied (free-form mode), accuracy degenerates to the
     * mean acoustic posterior of the detected phonemes and fluency/completeness are
     * computed as if the detected sequence were itself the reference.
     */
    fun computeOverallScore(
        phonemes: List<PhonemeResult>,
        comparison: PhonemeComparison? = null,
        wordTimings: List<WordTiming> = emptyList()
    ): PronunciationScore {
        if (phonemes.isEmpty()) return PronunciationScore(0f, 0f, 0f, 0f, emptyList())
        val nonSil = phonemes.filter { it.phoneme != "∅" }
        if (nonSil.isEmpty()) return PronunciationScore(0f, 0f, MIN_FLUENCY_FLOOR, 0f, phonemes)

        val accuracy     = computeAccuracy(nonSil, comparison)
        val fluency      = computeFluency(nonSil, accuracy, wordTimings)
        val completeness = computeCompleteness(nonSil, comparison)

        val overall = (accuracy     * W_ACCURACY +
                       fluency      * W_FLUENCY  +
                       completeness * W_COMPLETENESS
                      ).coerceIn(0f, 100f)

        if (BuildConfig.DEBUG) {
            Log.d(TAG, "Score — acc=$accuracy flu=$fluency comp=$completeness overall=$overall")
        }
        return PronunciationScore(overall, accuracy, fluency, completeness, phonemes)
    }

    /**
     * The single source of truth for accuracy. Mirrored in
     * [com.example.phoneme_trainer.MainViewModel.buildComparison] so the per-phrase
     * summary matches the top-line score exactly.
     */
    private fun computeAccuracy(
        nonSil: List<PhonemeResult>,
        comparison: PhonemeComparison?
    ): Float {
        if (comparison == null) {
            // Free-form mode: direct acoustic posterior mean.
            return (nonSil.map { it.confidence }.average().toFloat() * 100f).coerceIn(0f, 100f)
        }
        val weighted = nonSil.sumOf { ph ->
            val exp = ph.expectedPhoneme
            when {
                exp == null                 -> WEIGHT_INSERTION
                ph.phoneme == exp           -> WEIGHT_EXACT
                isNearMiss(ph.phoneme, exp) -> WEIGHT_NEAR_MISS
                else                        -> WEIGHT_SUBSTITUTE
            }
        }
        val totalExpected = comparison.totalExpected.coerceAtLeast(1)
        return ((weighted / totalExpected) * 100.0).toFloat().coerceIn(0f, 100f)
    }

    private fun computeFluency(nonSil: List<PhonemeResult>, accuracy: Float, wordTimings: List<WordTiming> = emptyList()): Float {
        if (nonSil.size < 2) return 50f
        val gaps: List<Long> = if (wordTimings.size >= 2) {
            wordTimings.zipWithNext { a, b -> b.startMs - a.endMs }
        } else {
            nonSil.zipWithNext { a, b -> b.startTimeMs - a.endTimeMs }
        }
        val hesitations = gaps.count { it > HESITATION_GAP_MS }
        val rushes      = gaps.count { it < OVERLAP_GAP_MS }
        val penalty = (hesitations * PENALTY_PER_HESITATION +
                       rushes      * PENALTY_PER_OVERLAP)
            .coerceIn(0f, MAX_GAP_PENALTY)
        return (100f - penalty).coerceAtLeast(MIN_FLUENCY_FLOOR)
    }

    private fun computeCompleteness(
        nonSil: List<PhonemeResult>,
        comparison: PhonemeComparison?
    ): Float {
        if (comparison == null) {
            return min(100f, nonSil.size.toFloat() / 5f * 100f)
        }
        val expected = comparison.totalExpected.toFloat().coerceAtLeast(1f)
        val produced = nonSil.count { it.isCorrect }.toFloat()
        return (produced / expected * 100f).coerceIn(0f, 100f)
    }

    /** Maps a 0-100 overall score to an ELPAC rubric level string. */
    fun elpacLevel(score: Float): String =
        ELPAC_THRESHOLDS.first { score >= it.first }.second

    // ── Weighted-accuracy helper exposed for MainViewModel ──────────────────

    /**
     * Returns a `(matchedWeight, totalExpected, weightedAccuracyPct)` triple using the
     * same formula as [computeAccuracy]. The ViewModel calls this when building the
     * [PhonemeComparison] so the comparison's `accuracyPct` and the score card's
     * `accuracyScore` are guaranteed identical.
     */
    fun weightedAccuracy(
        actual: List<PhonemeResult>,
        expected: List<String>
    ): Triple<Float, Int, Float> {
        if (expected.isEmpty()) return Triple(0f, 0, 0f)
        val total = expected.size.coerceAtLeast(1)
        val weighted = actual.sumOf { ph ->
            val exp = ph.expectedPhoneme
            when {
                exp == null                 -> WEIGHT_INSERTION
                ph.phoneme == exp           -> WEIGHT_EXACT
                isNearMiss(ph.phoneme, exp) -> WEIGHT_NEAR_MISS
                else                        -> WEIGHT_SUBSTITUTE
            }
        }.toFloat()
        val pct = ((weighted / total) * 100f).coerceIn(0f, 100f)
        return Triple(weighted, total, pct)
    }

    // ── Helpers / lifecycle ─────────────────────────────────────────────────

    private fun shortsToBytes(samples: ShortArray): ByteArray {
        val bytes = ByteArray(samples.size * 2)
        for (i in samples.indices) {
            bytes[i * 2]     = (samples[i].toInt() and 0xFF).toByte()
            bytes[i * 2 + 1] = (samples[i].toInt() shr 8 and 0xFF).toByte()
        }
        return bytes
    }

    fun close() {
        voskModel?.close()
        voskModel = null
        wavlm.close()
    }
}
