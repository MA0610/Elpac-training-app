package com.example.vosk_elpac.ml

import android.content.Context
import android.util.Log
import com.example.vosk_elpac.audio.AudioRecorder
import com.example.vosk_elpac.model.PhonemeComparison
import com.example.vosk_elpac.model.PhonemeInventory
import com.example.vosk_elpac.model.PhonemeResult
import com.example.vosk_elpac.model.PronunciationScore
import org.json.JSONObject
import org.vosk.Model
import org.vosk.Recognizer
import java.io.File
import kotlin.math.abs
import kotlin.math.min
import kotlin.math.sqrt

/**
 * On-device phoneme detection using Vosk + CMU Pronouncing Dictionary.
 *
 * Setup:
 *   1. Vosk model at: app/src/main/assets/vosk-model-small-en-us-0.15/
 *   2. CMU dict at:   app/src/main/assets/cmudict-0.7b
 *
 * Lookup chain: CMU dict (~134k words) → built-in table → G2P rules → DSP fallback
 */
class PhonemeDetector(private val context: Context) {

    companion object {
        private const val TAG            = "PhonemeDetector"
        private const val MODEL_DIR      = "vosk-model-small-en-us-0.15"
        private const val CMU_DICT_ASSET = "cmudict-0.7b"
        private const val SAMPLE_RATE    = AudioRecorder.SAMPLE_RATE
        private const val MIN_PHONEME_MS = 30L

        private val VOWELS     = setOf("AA","AE","AH","AO","AW","AY","EH","ER","EY","IH","IY","OW","OY","UH","UW")
        private val NASALS     = setOf("M","N","NG")
        private val FRICATIVES = setOf("F","V","S","Z","SH","ZH","TH","DH","HH")
        private val STOPS      = setOf("P","B","T","D","K","G")

        private fun isVowel(a: String)     = a in VOWELS
        private fun isNasal(a: String)     = a in NASALS
        private fun isFricative(a: String) = a in FRICATIVES
        private fun isStop(a: String)      = a in STOPS

        private val BUILTIN = mapOf(
            "the" to listOf("DH","AH"), "a" to listOf("AH"), "an" to listOf("AE","N"),
            "i" to listOf("AY"), "is" to listOf("IH","Z"), "it" to listOf("IH","T"),
            "in" to listOf("IH","N"), "and" to listOf("AE","N","D"),
            "of" to listOf("AH","V"), "to" to listOf("T","UW"),
            "that" to listOf("DH","AE","T"), "this" to listOf("DH","IH","S"),
            "was" to listOf("W","AH","Z"), "for" to listOf("F","AO","R"),
            "on" to listOf("AO","N"), "are" to listOf("AA","R"),
            "he" to listOf("HH","IY"), "she" to listOf("SH","IY"),
            "they" to listOf("DH","EY"), "we" to listOf("W","IY"),
            "you" to listOf("Y","UW"), "be" to listOf("B","IY"),
            "yes" to listOf("Y","EH","S"), "no" to listOf("N","OW"),
            "hi" to listOf("HH","AY"), "hello" to listOf("HH","AH","L","OW"),
            "ok" to listOf("OW","K","EY"), "okay" to listOf("OW","K","EY"),
        )
    }

    private var model: Model? = null
    val useFallback: Boolean

    private val cmuDict: Map<String, List<String>> by lazy { loadCmuDict() }

    init {
        useFallback = !loadModel()
        if (useFallback) Log.w(TAG, "Vosk model not found — will use DSP fallback.")
    }

    // ─── Vosk model loading ───────────────────────────────────────────────────

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

    // ─── CMU dict loading ─────────────────────────────────────────────────────

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

    // ─── Public: get expected IPA phonemes for a phrase ───────────────────────

    /**
     * Given a phrase like "this is a test", returns the full expected IPA
     * phoneme sequence using the CMU dict.
     * e.g. ["ð","ɪ","s","ɪ","z","ʌ","t","ɛ","s","t"]
     */
    fun getPhraseExpectedPhonemes(phrase: String): List<String> {
        return phrase.trim()
            .split("\\s+".toRegex())
            .flatMap { word ->
                val arpabets = lookupPhonemes(word)
                arpabets.mapNotNull { PhonemeInventory.ARPABET_TO_IPA[it] }
            }
    }

    // ─── Phoneme lookup ───────────────────────────────────────────────────────

    private fun lookupPhonemes(word: String): List<String> {
        val w = word.lowercase().trimEnd('.', ',', '?', '!', '\'', '"')
        return cmuDict[w] ?: BUILTIN[w] ?: graphemeToPhoneme(w)
    }

    // ─── Main entry point ─────────────────────────────────────────────────────

    fun detect(samples: ShortArray): List<PhonemeResult> {
        if (samples.isEmpty()) return emptyList()
        return if (!useFallback && model != null) runVoskRecognition(samples)
        else runDspFallback(samples)
    }

    // ─── Vosk recognition ─────────────────────────────────────────────────────

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
                // FIX: pass the correct slice, not the full array every time
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

    private fun distributePhonemes(
        phonemes: List<String>,
        startMs: Long,
        endMs: Long,
        conf: Float
    ): List<PhonemeResult> {
        val duration = (endMs - startMs).coerceAtLeast(phonemes.size * MIN_PHONEME_MS)
        val weights  = phonemes.map { a ->
            when {
                isVowel(a)     -> 1.5f
                isFricative(a) -> 1.2f
                isNasal(a)     -> 1.1f
                isStop(a)      -> 0.6f
                else           -> 1.0f
            }
        }
        val total   = weights.sum()
        val results = mutableListOf<PhonemeResult>()
        var cursor  = startMs

        phonemes.forEachIndexed { idx, arpabet ->
            val dur = ((weights[idx] / total) * duration).toLong().coerceAtLeast(MIN_PHONEME_MS)
            val end = if (idx == phonemes.lastIndex) endMs else cursor + dur
            val ipa = PhonemeInventory.ARPABET_TO_IPA[arpabet] ?: arpabet.lowercase()

            results.add(PhonemeResult(
                phoneme     = ipa,
                startTimeMs = cursor,
                endTimeMs   = end,
                confidence  = conf,
                score       = computePhonemeScore(conf, end - cursor, arpabet)
            ))
            cursor = end
        }
        return results
    }

    private fun spreadAcrossDuration(text: String, startMs: Long, endMs: Long): List<PhonemeResult> {
        val phonemes = text.trim().split("\\s+".toRegex()).flatMap { lookupPhonemes(it) }
        if (phonemes.isEmpty()) return emptyList()
        val msEach = (endMs - startMs) / phonemes.size

        return phonemes.mapIndexed { idx, arpabet ->
            val s   = startMs + idx * msEach
            val e   = if (idx == phonemes.lastIndex) endMs else s + msEach
            val ipa = PhonemeInventory.ARPABET_TO_IPA[arpabet] ?: arpabet.lowercase()
            PhonemeResult(
                phoneme     = ipa,
                startTimeMs = s,
                endTimeMs   = e,
                confidence  = 0.7f,
                score       = computePhonemeScore(0.7f, e - s, arpabet)
            )
        }
    }

    // ─── G2P rules ────────────────────────────────────────────────────────────

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

    // ─── DSP fallback ─────────────────────────────────────────────────────────

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
                feat.energy < 0.01f                      -> "SIL"
                feat.zcr > 0.3f && feat.energy < 0.05f  -> "FRIC"
                feat.energy > 0.15f && feat.zcr < 0.15f -> "VOW"
                feat.energy > 0.05f && feat.zcr > 0.15f -> "CON"
                else                                      -> "MID"
            }
            if (cat != prevCat) {
                if (prevCat.isNotEmpty() && prevCat != "SIL") {
                    val dur = feat.timeMs - segStart
                    if (dur >= MIN_PHONEME_MS) {
                        val (ph, conf) = heuristicPhoneme(prevCat)
                        results.add(PhonemeResult(
                            phoneme     = ph,
                            startTimeMs = segStart,
                            endTimeMs   = feat.timeMs,
                            confidence  = conf,
                            score       = computePhonemeScore(conf, dur, "")
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

    // ─── Scoring ──────────────────────────────────────────────────────────────

    private fun computePhonemeScore(conf: Float, durationMs: Long, arpabet: String): Float {
        val confScore = conf * 70f
        val durScore  = when {
            durationMs < 20L         -> 0f
            durationMs in 20L..300L -> 30f * (1f - abs(durationMs - 100L).toFloat() / 300f)
            else                     -> 10f
        }.coerceIn(0f, 30f)
        return (confScore + durScore).coerceIn(0f, 100f)
    }

    /**
     * Computes overall score, boosting/penalising based on phrase comparison
     * accuracy when a target phrase is provided.
     */
    fun computeOverallScore(
        phonemes: List<PhonemeResult>,
        comparison: PhonemeComparison? = null
    ): PronunciationScore {
        if (phonemes.isEmpty()) return PronunciationScore(0f, 0f, 0f, 0f, emptyList())
        val nonSil = phonemes.filter { it.phoneme != "∅" }
        if (nonSil.isEmpty()) return PronunciationScore(0f, 0f, 0f, 0f, phonemes)

        val baseAccuracy = nonSil.map { it.confidence * 100f }.average().toFloat()

        // If we have a comparison, blend Vosk confidence with match accuracy
        val accuracyScore = if (comparison != null) {
            (baseAccuracy * 0.4f + comparison.accuracyPct * 0.6f).coerceIn(0f, 100f)
        } else {
            baseAccuracy
        }

        val fluencyScore = if (nonSil.size < 2) accuracyScore else {
            val avgGap = nonSil.zipWithNext { a, b -> b.startTimeMs - a.endTimeMs }.average().toFloat()
            (100f - (avgGap / 50f).coerceIn(0f, 50f)).coerceIn(0f, 100f)
        }

        val expectedCount = comparison?.totalExpected?.toFloat() ?: 5f
        val completenessScore = min(100f, nonSil.size.toFloat() / expectedCount * 100f)

        val overall = (accuracyScore * 0.5f + fluencyScore * 0.3f + completenessScore * 0.2f)
            .coerceIn(0f, 100f)

        return PronunciationScore(overall, accuracyScore, fluencyScore, completenessScore, phonemes)
    }

    // ─── Helpers ──────────────────────────────────────────────────────────────

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
    }
}