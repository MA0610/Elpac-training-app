package com.example.vosk_elpac.ml

import android.content.Context
import android.util.Log
import com.example.vosk_elpac.audio.AudioRecorder
import com.example.vosk_elpac.model.DetectionResult
import com.example.vosk_elpac.model.PhonemeComparison
import com.example.vosk_elpac.model.PhonemeInventory
import com.example.vosk_elpac.model.PhonemeResult
import com.example.vosk_elpac.model.PronunciationScore
import com.example.vosk_elpac.model.WordTiming
import org.json.JSONObject
import org.vosk.Model
import org.vosk.Recognizer
import java.io.File
import kotlin.math.abs
import kotlin.math.ln
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

/**
 * On-device phoneme detection using Vosk + CMU Pronouncing Dictionary.
 *
 * Scoring model:
 *   Each phoneme receives a GOP-style score derived from:
 *     1. Vosk word confidence  (proxy for acoustic posterior)
 *     2. Category-aware duration score  (stops are naturally short)
 *     3. Alignment penalty  (mismatched phoneme vs expected)
 *     4. Context bonus  (phoneme appears in correct word position)
 *
 *  Overall score maps to ELPAC Speaking rubric levels 1-4.
 */
class PhonemeDetector(private val context: Context) {

    companion object {
        private const val TAG            = "PhonemeDetector"
        private const val MODEL_DIR      = "vosk-model-small-en-us-0.15"
        private const val CMU_DICT_ASSET = "cmudict-0.7b"
        private const val SAMPLE_RATE    = AudioRecorder.SAMPLE_RATE
        private const val MIN_PHONEME_MS = 20L

        // ── Typical duration ranges (ms) per phoneme category ──────────────
        // Based on English speech research. Used for duration scoring.
        private val DURATION_RANGE = mapOf(
            "VOWEL"     to (60L  to 250L),
            "STOP"      to (20L  to 100L),   // stops are naturally short
            "FRICATIVE" to (50L  to 200L),
            "NASAL"     to (50L  to 150L),
            "LIQUID"    to (40L  to 130L),
            "AFFRICATE" to (50L  to 180L),
            "SILENCE"   to (0L   to 500L)
        )

        // ── Voicing pairs: close enough to be "near miss" not "wrong" ──────
        private val SIMILAR_PAIRS = setOf(
            setOf("ɪ", "iː"), setOf("ʌ", "ɑ"),  setOf("ɛ", "æ"),
            setOf("ʊ", "uː"), setOf("ɔ", "oʊ"),  setOf("ð", "θ"),
            setOf("s", "z"),  setOf("f", "v"),    setOf("p", "b"),
            setOf("t", "d"),  setOf("k", "ɡ"),    setOf("ʃ", "ʒ"),
            setOf("m", "n"),  setOf("tʃ", "dʒ"),  setOf("ɝ", "ʌ"),
            // eSpeak long/short vowel equivalents (model may produce either form)
            setOf("ɑ", "ɑː"), setOf("ɔ", "ɔː"),  setOf("ɜː", "ɝ"),
            setOf("iː", "i"), setOf("uː", "u")
        )

        // ── ELPAC level thresholds (0-100 mapped to rubric 1-4) ────────────
        // Level 4: minimal pronunciation errors
        // Level 3: some errors but generally intelligible
        // Level 2: frequent errors, impedes communication at times
        // Level 1: errors impede communication throughout
        private val ELPAC_THRESHOLDS = listOf(
            85f to "Level 4 – Minimal errors",
            70f to "Level 3 – Generally intelligible",
            50f to "Level 2 – Some communication impact",
            0f  to "Level 1 – Significant communication impact"
        )

        private val VOWELS     = setOf("AA","AE","AH","AO","AW","AY","EH","ER","EY","IH","IY","OW","OY","UH","UW")
        private val NASALS     = setOf("M","N","NG")
        private val FRICATIVES = setOf("F","V","S","Z","SH","ZH","TH","DH","HH")
        private val STOPS      = setOf("P","B","T","D","K","G")
        private val AFFRICATES = setOf("CH","JH")

        private fun arpabetCategory(a: String): String = when (a) {
            in VOWELS     -> "VOWEL"
            in STOPS      -> "STOP"
            in FRICATIVES -> "FRICATIVE"
            in NASALS     -> "NASAL"
            in AFFRICATES -> "AFFRICATE"
            "L","R","W","Y" -> "LIQUID"
            else          -> "VOWEL"
        }

        private val BUILTIN = mapOf(
            "the"   to listOf("DH","AH"), "a"    to listOf("AH"),
            "an"    to listOf("AE","N"),  "i"    to listOf("AY"),
            "is"    to listOf("IH","Z"),  "it"   to listOf("IH","T"),
            "in"    to listOf("IH","N"),  "and"  to listOf("AE","N","D"),
            "of"    to listOf("AH","V"),  "to"   to listOf("T","UW"),
            "that"  to listOf("DH","AE","T"), "this" to listOf("DH","IH","S"),
            "was"   to listOf("W","AH","Z"), "for" to listOf("F","AO","R"),
            "on"    to listOf("AO","N"),  "are"  to listOf("AA","R"),
            "he"    to listOf("HH","IY"), "she"  to listOf("SH","IY"),
            "they"  to listOf("DH","EY"), "we"   to listOf("W","IY"),
            "you"   to listOf("Y","UW"),  "be"   to listOf("B","IY"),
            "yes"   to listOf("Y","EH","S"), "no" to listOf("N","OW"),
            "hi"    to listOf("HH","AY"), "hello" to listOf("HH","AH","L","OW"),
            "ok"    to listOf("OW","K","EY"), "okay" to listOf("OW","K","EY"),
            "my"    to listOf("M","AY"),  "name" to listOf("N","EY","M"),
            "how"   to listOf("HH","AW"), "what" to listOf("W","AH","T"),
            "good"  to listOf("G","UH","D"), "morning" to listOf("M","AO","R","N","IH","NG"),
            "thank" to listOf("TH","AE","NG","K"), "very" to listOf("V","EH","R","IY"),
            "much"  to listOf("M","AH","CH"), "go"  to listOf("G","OW"),
            "see"   to listOf("S","IY"),  "do"   to listOf("D","UW"),
            "at"    to listOf("AE","T"),  "up"   to listOf("AH","P"),
            "day"   to listOf("D","EY"),  "her"  to listOf("HH","ER"),
            "his"   to listOf("HH","IH","Z"), "have" to listOf("HH","AE","V"),
        )
    }

    private var model: Model? = null
    val useFallback: Boolean

    private val cmuDict: Map<String, List<String>> by lazy { loadCmuDict() }

    /** Wav2Vec2 ONNX detector — primary acoustic engine. */
    private val wav2vec2 = Wav2Vec2PhonemeDetector(context)

    init {
        useFallback = !loadModel()
        if (useFallback) Log.w(TAG, "Vosk model not found — will use DSP fallback.")
        if (wav2vec2.isAvailable) Log.i(TAG, "Wav2Vec2 model ready — using real acoustic scoring.")
    }

    // ── Vosk model loading ─────────────────────────────────────────────────

    private fun loadModel(): Boolean {
        return try {
            copyModelFromAssets()
            val path = File(context.filesDir, MODEL_DIR).absolutePath
            model = Model(path)
            Log.i(TAG, "Vosk model loaded: $path")
            true
        } catch (e: Exception) {
            Log.w(TAG, "Vosk model load failed: ${e.message}")
            false
        }
    }

    private fun copyModelFromAssets() {
        val dest = File(context.filesDir, MODEL_DIR)
        if (dest.exists() && dest.list()?.isNotEmpty() == true) return
        dest.mkdirs()
        copyAssetFolder(MODEL_DIR, dest)
        Log.i(TAG, "Vosk model copied from assets.")
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

    // ── CMU dict loading ───────────────────────────────────────────────────

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
            Log.i(TAG, "CMU dict loaded: ${dict.size} entries")
        } catch (e: Exception) {
            Log.w(TAG, "CMU dict unavailable: ${e.message}")
        }
        return dict
    }

    // ── Model download ─────────────────────────────────────────────────────

    suspend fun prepareWav2Vec2(onProgress: (Float) -> Unit) {
        wav2vec2.downloadAndInit(onProgress)
    }

    // ── Public: expected IPA phonemes for a phrase ─────────────────────────

    fun getPhraseExpectedPhonemes(phrase: String): List<String> {
        return phrase.trim()
            .split("\\s+".toRegex())
            .flatMap { word ->
                val arpabets = lookupPhonemes(word)
                arpabets.mapNotNull { PhonemeInventory.ARPABET_TO_IPA[it] }
            }
    }

    // ── Phoneme lookup ─────────────────────────────────────────────────────

    private fun lookupPhonemes(word: String): List<String> {
        val w = word.lowercase().trimEnd('.', ',', '?', '!', '\'', '"', ';', ':')
        return cmuDict[w] ?: BUILTIN[w] ?: graphemeToPhoneme(w)
    }

    // ── Main entry point ───────────────────────────────────────────────────

    /**
     * Detect phonemes and word timing from audio samples.
     *
     * Priority:
     *   1. Wav2Vec2 (real per-phoneme acoustic scores) + Vosk (word timing)
     *   2. Vosk only (word-confidence proxy scores)
     *   3. DSP fallback (energy-based, last resort)
     */
    fun detect(samples: ShortArray): DetectionResult {
        if (samples.isEmpty()) return DetectionResult(emptyList(), emptyList())

        return when {
            wav2vec2.isAvailable -> runHybridDetection(samples)
            !useFallback && model != null ->
                DetectionResult(runVoskRecognition(samples), emptyList())
            else ->
                DetectionResult(runDspFallback(samples), emptyList())
        }
    }

    private fun runHybridDetection(samples: ShortArray): DetectionResult {
        // Primary: Wav2Vec2 for real per-phoneme acoustic scores
        val acousticPhonemes = wav2vec2.detectPhonemes(samples)

        // Secondary: Vosk for word-boundary timing only (confidence ignored)
        val wordTimings = if (!useFallback && model != null) {
            extractWordTimings(samples)
        } else emptyList()

        val phonemes = if (acousticPhonemes.isNotEmpty()) {
            acousticPhonemes
        } else {
            Log.w(TAG, "Wav2Vec2 returned empty — falling back to Vosk")
            if (!useFallback && model != null) runVoskRecognition(samples)
            else runDspFallback(samples)
        }

        return DetectionResult(phonemes, wordTimings)
    }

    /**
     * Extracts word-level timing boundaries from Vosk (word, startMs, endMs).
     * Only used for word highlighting — confidence values are discarded.
     */
    fun extractWordTimings(samples: ShortArray): List<WordTiming> {
        val voskModel = model ?: return emptyList()
        return try {
            val rec = Recognizer(voskModel, SAMPLE_RATE.toFloat())
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
        } catch (e: Exception) { emptyList() }
    }

    /** Returns expected IPA phonemes for a single word (used to enrich WordTiming). */
    fun getWordExpectedPhonemes(word: String): List<String> =
        lookupPhonemes(word).mapNotNull { PhonemeInventory.ARPABET_TO_IPA[it] }

    // ── Vosk recognition ───────────────────────────────────────────────────

    private fun runVoskRecognition(samples: ShortArray): List<PhonemeResult> {
        val voskModel = model ?: return runDspFallback(samples)
        return try {
            val rec = Recognizer(voskModel, SAMPLE_RATE.toFloat())
            rec.setWords(true)

            val bytes = shortsToBytes(samples)
            val chunkSize = 8000
            var offset = 0
            while (offset < bytes.size) {
                val end = minOf(offset + chunkSize, bytes.size)
                val chunk = bytes.copyOfRange(offset, end)
                rec.acceptWaveForm(chunk, chunk.size)
                offset = end
            }

            val json = rec.finalResult
            rec.close()
            Log.d(TAG, "Vosk raw: $json")

            val results = parseVoskResult(json, samples)
            if (results.isEmpty()) runDspFallback(samples) else results

        } catch (e: Exception) {
            Log.e(TAG, "Vosk error: ${e.message}")
            runDspFallback(samples)
        }
    }

    private fun parseVoskResult(json: String, samples: ShortArray): List<PhonemeResult> {
        val totalMs = samples.size.toLong() * 1000L / SAMPLE_RATE
        return try {
            val obj  = JSONObject(json)
            val text = obj.optString("text", "").trim()
            Log.d(TAG, "Recognized: \"$text\"")
            if (text.isEmpty()) return emptyList()
            if (!obj.has("result")) return spreadAcrossDuration(text, 0L, totalMs)

            val wordArray = obj.getJSONArray("result")
            val results   = mutableListOf<PhonemeResult>()

            for (i in 0 until wordArray.length()) {
                val w       = wordArray.getJSONObject(i)
                val word    = w.getString("word").lowercase().trim()
                val startMs = (w.getDouble("start") * 1000).toLong()
                val endMs   = (w.getDouble("end")   * 1000).toLong()
                val conf    = w.optDouble("conf", 0.75).toFloat().coerceIn(0f, 1f)
                val phonemes = lookupPhonemes(word)
                if (phonemes.isNotEmpty()) {
                    results.addAll(distributePhonemes(phonemes, startMs, endMs, conf))
                }
            }
            results
        } catch (e: Exception) {
            Log.e(TAG, "Parse error: ${e.message}")
            emptyList()
        }
    }

    /**
     * Distributes word-level confidence across phonemes using category-aware
     * duration weighting. Each phoneme also gets a per-category duration score
     * component so that short stops aren't unfairly penalized.
     */
    private fun distributePhonemes(
        phonemes: List<String>,
        startMs: Long,
        endMs: Long,
        wordConf: Float
    ): List<PhonemeResult> {
        val totalDuration = (endMs - startMs).coerceAtLeast(phonemes.size * MIN_PHONEME_MS)

        // Duration weights: vowels and fricatives are longer; stops are shorter
        val weights = phonemes.map { a ->
            when (arpabetCategory(a)) {
                "VOWEL"     -> 1.5f
                "FRICATIVE" -> 1.2f
                "NASAL"     -> 1.1f
                "LIQUID"    -> 1.0f
                "AFFRICATE" -> 1.0f
                "STOP"      -> 0.6f
                else        -> 1.0f
            }
        }
        val totalWeight = weights.sum()
        val results     = mutableListOf<PhonemeResult>()
        var cursor      = startMs

        phonemes.forEachIndexed { idx, arpabet ->
            val dur = ((weights[idx] / totalWeight) * totalDuration)
                .toLong().coerceAtLeast(MIN_PHONEME_MS)
            val segEnd = if (idx == phonemes.lastIndex) endMs else cursor + dur
            val ipa    = PhonemeInventory.ARPABET_TO_IPA[arpabet] ?: arpabet.lowercase()
            val cat    = arpabetCategory(arpabet)

            // ── GOP-style per-phoneme score ─────────────────────────────────
            // Component 1: acoustic posterior proxy (log-transformed confidence)
            // Using -ln(1 - conf) normalised to 0-1 amplifies high-confidence
            // signals more than a linear scale would.
            val acousticScore = gopAcousticScore(wordConf)

            // Component 2: duration plausibility for this phoneme category
            val durScore = durationScore(segEnd - cursor, cat)

            // Component 3: word position bonus — word-final phonemes tend to be
            // reduced; reward word-initial stressed phonemes slightly more.
            val positionBonus = when (idx) {
                0                  -> 5f   // word-initial
                phonemes.lastIndex -> 0f   // word-final (often reduced)
                else               -> 2f
            }

            // Blend into 0-100
            val rawScore = (acousticScore * 0.65f + durScore * 0.30f + positionBonus * 0.05f)
                .coerceIn(0f, 100f)

            results.add(PhonemeResult(
                phoneme     = ipa,
                startTimeMs = cursor,
                endTimeMs   = segEnd,
                confidence  = wordConf,
                score       = rawScore
            ))
            cursor = segEnd
        }
        return results
    }

    // ── GOP acoustic score ─────────────────────────────────────────────────

    /**
     * Transforms Vosk word confidence into a GOP-like acoustic score.
     *
     * Real GOP = -log P(correct_phone | audio)
     * We approximate this using Vosk's word confidence as a proxy for the
     * acoustic model's log-likelihood. A log transform amplifies the
     * difference between high/low confidence more accurately than linear.
     *
     * conf=1.0 → score≈100, conf=0.75 → score≈72, conf=0.5 → score≈50
     */
    private fun gopAcousticScore(conf: Float): Float {
        if (conf <= 0f) return 0f
        if (conf >= 1f) return 100f
        // Sigmoid-like mapping that's harsher on low confidence
        val logScore = (1.0 + ln(conf.toDouble())) * 100.0
        return logScore.toFloat().coerceIn(0f, 100f)
    }

    // ── Duration plausibility score ────────────────────────────────────────

    /**
     * Scores how plausible the phoneme duration is for its category.
     * Uses a trapezoid function: full score within typical range,
     * graceful falloff outside. Stops (20-100ms) score as well as
     * vowels (60-250ms) when their durations are appropriate.
     */
    private fun durationScore(durationMs: Long, category: String): Float {
        val (minMs, maxMs) = DURATION_RANGE[category] ?: (40L to 200L)
        val midMs = (minMs + maxMs) / 2

        return when {
            durationMs < minMs -> {
                // Below minimum: linear decay
                val ratio = durationMs.toFloat() / minMs.toFloat()
                (ratio * 70f).coerceIn(0f, 70f)
            }
            durationMs in minMs..maxMs -> {
                // Within range: full score, slight bonus near midpoint
                val distFromMid = abs(durationMs - midMs).toFloat()
                val rangeHalf   = (maxMs - midMs).toFloat().coerceAtLeast(1f)
                100f - (distFromMid / rangeHalf) * 15f
            }
            else -> {
                // Above maximum: slower decay (elongation is less penalised
                // than truncation in English EL speakers)
                val overshoot = (durationMs - maxMs).toFloat()
                (100f - overshoot / 20f).coerceIn(30f, 85f)
            }
        }
    }

    private fun spreadAcrossDuration(text: String, startMs: Long, endMs: Long): List<PhonemeResult> {
        val phonemes = text.trim().split("\\s+".toRegex()).flatMap { lookupPhonemes(it) }
        if (phonemes.isEmpty()) return emptyList()
        val msEach   = (endMs - startMs) / phonemes.size

        return phonemes.mapIndexed { idx, arpabet ->
            val s   = startMs + idx * msEach
            val e   = if (idx == phonemes.lastIndex) endMs else s + msEach
            val ipa = PhonemeInventory.ARPABET_TO_IPA[arpabet] ?: arpabet.lowercase()
            val cat = arpabetCategory(arpabet)
            PhonemeResult(
                phoneme     = ipa,
                startTimeMs = s,
                endTimeMs   = e,
                confidence  = 0.7f,
                score       = (gopAcousticScore(0.7f) * 0.7f + durationScore(e - s, cat) * 0.3f)
                    .coerceIn(0f, 100f)
            )
        }
    }

    // ── Needleman-Wunsch alignment ─────────────────────────────────────────

    /**
     * Aligns actual phoneme sequence against expected using global sequence
     * alignment (Needleman-Wunsch algorithm). This is far more robust than
     * the previous greedy sequential approach — one insertion or deletion
     * no longer cascades into mismatches for the rest of the utterance.
     *
     * Scoring:
     *   match         = +2   (exact or acoustically similar)
     *   near-miss     = +1   (voiced/unvoiced pair)
     *   substitution  = -1
     *   gap           = -1   (insertion or deletion)
     */
    fun alignPhonemes(
        actual: List<PhonemeResult>,
        expected: List<String>
    ): List<PhonemeResult> {
        if (actual.isEmpty()) return emptyList()
        if (expected.isEmpty()) return actual

        val n = actual.size
        val m = expected.size

        // Gap and match penalties
        val GAP_PENALTY = -1
        val MATCH       =  2
        val NEAR_MISS   =  1
        val MISMATCH    = -1

        fun similarity(act: String, exp: String): Int = when {
            act == exp                                                          -> MATCH
            SIMILAR_PAIRS.any { it.contains(act) && it.contains(exp) }        -> NEAR_MISS
            else                                                               -> MISMATCH
        }

        // Build DP matrix
        val dp = Array(n + 1) { IntArray(m + 1) }
        for (i in 0..n) dp[i][0] = i * GAP_PENALTY
        for (j in 0..m) dp[0][j] = j * GAP_PENALTY

        for (i in 1..n) {
            for (j in 1..m) {
                val score = similarity(actual[i - 1].phoneme, expected[j - 1])
                dp[i][j] = maxOf(
                    dp[i - 1][j - 1] + score,
                    dp[i - 1][j] + GAP_PENALTY,
                    dp[i][j - 1] + GAP_PENALTY
                )
            }
        }

        // Traceback
        val alignedActual   = mutableListOf<String?>() // null = gap in actual
        val alignedExpected = mutableListOf<String?>() // null = gap in expected
        var i = n; var j = m
        while (i > 0 || j > 0) {
            when {
                i > 0 && j > 0 &&
                        dp[i][j] == dp[i - 1][j - 1] + similarity(actual[i - 1].phoneme, expected[j - 1]) -> {
                    alignedActual.add(0, actual[i - 1].phoneme)
                    alignedExpected.add(0, expected[j - 1])
                    i--; j--
                }
                i > 0 && dp[i][j] == dp[i - 1][j] + GAP_PENALTY -> {
                    alignedActual.add(0, actual[i - 1].phoneme)
                    alignedExpected.add(0, null)
                    i--
                }
                else -> {
                    alignedActual.add(0, null)
                    alignedExpected.add(0, expected[j - 1])
                    j--
                }
            }
        }

        // Reconstruct annotated PhonemeResult list from alignment
        val annotated = mutableListOf<PhonemeResult>()
        var actualIdx = 0

        for (k in alignedActual.indices) {
            val actPhoneme = alignedActual[k]
            val expPhoneme = alignedExpected[k]

            if (actPhoneme != null && actualIdx < actual.size) {
                val ph         = actual[actualIdx++]
                val isExact    = actPhoneme == expPhoneme
                val isNearMiss = expPhoneme != null &&
                        SIMILAR_PAIRS.any { it.contains(actPhoneme) && it.contains(expPhoneme) }
                val isCorrect  = isExact || isNearMiss

                // Apply alignment-aware score adjustment
                val alignedScore = when {
                    expPhoneme == null -> ph.score * 0.85f  // insertion: slight penalty
                    isExact            -> ph.score          // perfect match: keep score
                    isNearMiss         -> ph.score * 0.80f  // near miss: small penalty
                    else               -> ph.score * 0.40f  // substitution: heavy penalty
                }

                annotated.add(ph.copy(
                    isCorrect      = isCorrect,
                    expectedPhoneme = expPhoneme,
                    score          = alignedScore.coerceIn(0f, 100f)
                ))
            }
            // Gaps in actual (expected phoneme with no actual) are skipped —
            // they reduce completenessScore in computeOverallScore()
        }

        return annotated
    }

    // ── Overall score computation ──────────────────────────────────────────

    /**
     * Computes a 4-component ELPAC-aware pronunciation score:
     *
     *  Accuracy     = weighted phoneme match rate (exact matches score full,
     *                 near-misses score partial, substitutions score low)
     *  Fluency      = inter-phoneme gap analysis (hesitations, restarts)
     *  Completeness = ratio of expected phonemes actually produced
     *  Overall      = ELPAC rubric blend (accuracy-heavy, maps to levels 1-4)
     */
    fun computeOverallScore(
        phonemes: List<PhonemeResult>,
        comparison: PhonemeComparison? = null
    ): PronunciationScore {
        if (phonemes.isEmpty()) return PronunciationScore(0f, 0f, 0f, 0f, emptyList())
        val nonSil = phonemes.filter { it.phoneme != "∅" }
        if (nonSil.isEmpty()) return PronunciationScore(0f, 0f, 0f, 0f, phonemes)

        // ── Accuracy ──────────────────────────────────────────────────────
        val accuracyScore: Float = if (comparison != null) {
            // Weight: exact matches = 1.0, near-miss = 0.7, wrong = 0.0
            val weightedMatches = nonSil.sumOf { ph ->
                when {
                    ph.expectedPhoneme == null      -> 0.85  // insertion
                    ph.phoneme == ph.expectedPhoneme -> 1.0   // exact
                    SIMILAR_PAIRS.any {
                        it.contains(ph.phoneme) &&
                                it.contains(ph.expectedPhoneme!!)
                    }                               -> 0.70   // near-miss
                    else                            -> 0.0    // wrong
                }
            }
            val totalExpected = comparison.totalExpected.coerceAtLeast(1)
            ((weightedMatches / totalExpected) * 100.0).toFloat().coerceIn(0f, 100f)
        } else {
            // No target phrase: use acoustic confidence as proxy
            nonSil.map { gopAcousticScore(it.confidence) }.average().toFloat()
        }

        // ── Fluency ────────────────────────────────────────────────────────
        // Penalise large inter-phoneme gaps (hesitations) and unusually
        // short gaps (rushed speech). Typical gap: 10-40ms.
        val fluencyScore: Float = if (nonSil.size < 2) {
            accuracyScore * 0.9f
        } else {
            val gaps = nonSil.zipWithNext { a, b -> b.startTimeMs - a.endTimeMs }
            val hesitations = gaps.count { it > 300 }   // >300ms gap = hesitation
            val rushes      = gaps.count { it < -10 }   // overlap = restart
            val gapPenalty  = (hesitations * 8f + rushes * 5f).coerceIn(0f, 40f)
            (100f - gapPenalty).coerceIn(30f, 100f)
        }

        // ── Completeness ───────────────────────────────────────────────────
        // How many expected phonemes were actually produced?
        val completenessScore: Float = if (comparison != null) {
            val expected = comparison.totalExpected.toFloat().coerceAtLeast(1f)
            // Count actual non-silent, non-inserted phonemes
            val produced = nonSil.count { it.expectedPhoneme != null }.toFloat()
            (produced / expected * 100f).coerceIn(0f, 100f)
        } else {
            min(100f, nonSil.size.toFloat() / 5f * 100f)
        }

        // ── ELPAC-weighted overall score ───────────────────────────────────
        // ELPAC Speaking rubric weights pronunciation errors by their impact
        // on communication: accuracy matters most, then fluency, then completeness.
        val overall = (
                accuracyScore     * 0.55f +
                        fluencyScore      * 0.30f +
                        completenessScore * 0.15f
                ).coerceIn(0f, 100f)

        Log.d(TAG, "Score — accuracy: $accuracyScore, fluency: $fluencyScore, " +
                "completeness: $completenessScore, overall: $overall")
        Log.d(TAG, "ELPAC level: ${elpacLevel(overall)}")

        return PronunciationScore(overall, accuracyScore, fluencyScore, completenessScore, phonemes)
    }

    /** Maps a 0-100 score to an ELPAC rubric level string. */
    fun elpacLevel(score: Float): String =
        ELPAC_THRESHOLDS.first { score >= it.first }.second

    // ── G2P rules ──────────────────────────────────────────────────────────

    private fun graphemeToPhoneme(word: String): List<String> {
        val phonemes = mutableListOf<String>()
        var i = 0
        val w = word.lowercase()
        while (i < w.length) {
            val r = w.substring(i)
            when {
                r.startsWith("th") -> { phonemes.add("DH"); i += 2 }
                r.startsWith("sh") -> { phonemes.add("SH"); i += 2 }
                r.startsWith("ch") -> { phonemes.add("CH"); i += 2 }
                r.startsWith("ph") -> { phonemes.add("F");  i += 2 }
                r.startsWith("wh") -> { phonemes.add("W");  i += 2 }
                r.startsWith("ng") -> { phonemes.add("NG"); i += 2 }
                r.startsWith("ck") -> { phonemes.add("K");  i += 2 }
                r.startsWith("ee") -> { phonemes.add("IY"); i += 2 }
                r.startsWith("ea") -> { phonemes.add("IY"); i += 2 }
                r.startsWith("oo") -> { phonemes.add("UW"); i += 2 }
                r.startsWith("ou") -> { phonemes.add("AW"); i += 2 }
                r.startsWith("ow") -> { phonemes.add("OW"); i += 2 }
                r.startsWith("oi") -> { phonemes.add("OY"); i += 2 }
                r.startsWith("ay") -> { phonemes.add("EY"); i += 2 }
                r.startsWith("ai") -> { phonemes.add("EY"); i += 2 }
                r.startsWith("au") -> { phonemes.add("AO"); i += 2 }
                r.startsWith("aw") -> { phonemes.add("AO"); i += 2 }
                w[i] == 'a' -> { phonemes.add("AE"); i++ }
                w[i] == 'e' -> { phonemes.add("EH"); i++ }
                w[i] == 'i' -> { phonemes.add("IH"); i++ }
                w[i] == 'o' -> { phonemes.add("OW"); i++ }
                w[i] == 'u' -> { phonemes.add("AH"); i++ }
                w[i] == 'b' -> { phonemes.add("B");  i++ }
                w[i] == 'c' -> { phonemes.add("K");  i++ }
                w[i] == 'd' -> { phonemes.add("D");  i++ }
                w[i] == 'f' -> { phonemes.add("F");  i++ }
                w[i] == 'g' -> { phonemes.add("G");  i++ }
                w[i] == 'h' -> { phonemes.add("HH"); i++ }
                w[i] == 'j' -> { phonemes.add("JH"); i++ }
                w[i] == 'k' -> { phonemes.add("K");  i++ }
                w[i] == 'l' -> { phonemes.add("L");  i++ }
                w[i] == 'm' -> { phonemes.add("M");  i++ }
                w[i] == 'n' -> { phonemes.add("N");  i++ }
                w[i] == 'p' -> { phonemes.add("P");  i++ }
                w[i] == 'q' -> { phonemes.add("K");  i++ }
                w[i] == 'r' -> { phonemes.add("R");  i++ }
                w[i] == 's' -> { phonemes.add("S");  i++ }
                w[i] == 't' -> { phonemes.add("T");  i++ }
                w[i] == 'v' -> { phonemes.add("V");  i++ }
                w[i] == 'w' -> { phonemes.add("W");  i++ }
                w[i] == 'x' -> { phonemes.addAll(listOf("K","S")); i++ }
                w[i] == 'y' -> { phonemes.add("Y");  i++ }
                w[i] == 'z' -> { phonemes.add("Z");  i++ }
                else -> i++
            }
        }
        return phonemes
    }

    // ── DSP fallback ───────────────────────────────────────────────────────

    private fun runDspFallback(samples: ShortArray): List<PhonemeResult> {
        val results   = mutableListOf<PhonemeResult>()
        val frameSize = SAMPLE_RATE / 100

        data class Frame(val timeMs: Long, val energy: Float, val zcr: Float)

        val frames = mutableListOf<Frame>()
        var i = 0
        while (i + frameSize < samples.size) {
            val frame = samples.slice(i until i + frameSize)
            val rms   = sqrt(frame.fold(0.0) { acc, s ->
                acc + (s.toDouble() / Short.MAX_VALUE).let { it * it }
            } / frame.size).toFloat()
            var zc = 0
            for (j in 1 until frame.size) if ((frame[j] >= 0) != (frame[j - 1] >= 0)) zc++
            frames.add(Frame(i.toLong() * 1000L / SAMPLE_RATE, rms, zc.toFloat() / frame.size))
            i += frameSize
        }

        var segStart = 0L
        var prevCat  = ""
        for (feat in frames) {
            val cat = when {
                feat.energy < 0.01f                       -> "SIL"
                feat.zcr > 0.3f && feat.energy < 0.05f   -> "FRIC"
                feat.energy > 0.15f && feat.zcr < 0.15f  -> "VOW"
                feat.energy > 0.05f && feat.zcr > 0.15f  -> "CON"
                else                                       -> "MID"
            }
            if (cat != prevCat) {
                if (prevCat.isNotEmpty() && prevCat != "SIL") {
                    val dur = feat.timeMs - segStart
                    if (dur >= MIN_PHONEME_MS) {
                        val (ph, conf) = heuristicPhoneme(prevCat)
                        val phonemeCat = when (prevCat) {
                            "VOW"  -> "VOWEL"
                            "FRIC" -> "FRICATIVE"
                            "CON"  -> "STOP"
                            else   -> "VOWEL"
                        }
                        results.add(PhonemeResult(
                            phoneme     = ph,
                            startTimeMs = segStart,
                            endTimeMs   = feat.timeMs,
                            confidence  = conf,
                            score       = (gopAcousticScore(conf) * 0.7f +
                                    durationScore(dur, phonemeCat) * 0.3f)
                                .coerceIn(0f, 100f)
                        ))
                    }
                }
                segStart = feat.timeMs
                prevCat  = cat
            }
        }
        return results
    }

    private fun heuristicPhoneme(cat: String): Pair<String, Float> = when (cat) {
        "VOW"  -> Pair(listOf("æ","ɑ","ɛ","ɪ","ʌ","oʊ").random(), 0.65f)
        "FRIC" -> Pair(listOf("s","ʃ","f","h","z").random(),        0.55f)
        "CON"  -> Pair(listOf("t","d","k","p","b","m","n").random(), 0.60f)
        else   -> Pair("∅", 0.40f)
    }

    // ── Helpers ────────────────────────────────────────────────────────────

    private fun shortsToBytes(samples: ShortArray): ByteArray {
        val bytes = ByteArray(samples.size * 2)
        for (i in samples.indices) {
            bytes[i * 2]     = (samples[i].toInt() and 0xFF).toByte()
            bytes[i * 2 + 1] = (samples[i].toInt() shr 8 and 0xFF).toByte()
        }
        return bytes
    }

    fun close() {
        model?.close()
        model = null
        wav2vec2.close()
    }
}