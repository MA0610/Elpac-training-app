package com.example.phoneme_trainer.ml

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import android.util.Log
import com.example.phoneme_trainer.BuildConfig
import com.example.phoneme_trainer.audio.AudioRecorder
import com.example.phoneme_trainer.model.PhonemeResult
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.json.JSONObject
import java.io.Closeable
import java.io.File
import java.net.HttpURLConnection
import java.net.URL
import java.nio.FloatBuffer
import java.security.MessageDigest
import kotlin.math.exp
import kotlin.math.ln

/**
 * On-device phoneme detection using a fine-tuned WavLM-for-CTC ONNX model.
 *
 * The underlying architecture is Microsoft's WavLM (base) with a CTC head trained to emit
 * an IPA-like phoneme inventory of ~52 tokens. CTC greedy decoding converts per-frame
 * logits to a phoneme sequence with real acoustic timing and per-phoneme softmax
 * posteriors.
 *
 * Two correctness details worth noting:
 *  1. The vocabulary includes the length mark "ː" (token id 50) as a standalone token.
 *     Naive CTC decoding emits it as its own phoneme, which then fails to align against
 *     long-vowel forms (iː, uː, ɑː, ɔː, ɜː) produced by the CMU dict lookup. We
 *     post-process segments to merge "ː" into the preceding vowel so the decoder output
 *     matches the expected-sequence inventory.
 *  2. The downloaded ONNX file is verified against a pinned SHA-256 hash before the
 *     session is initialised. A mismatch aborts loading rather than executing an
 *     attacker-controlled graph.
 *
 * Legacy note: remote and asset file names retain the "wav2vec2_" prefix for backward
 * compatibility with the v2.0 GitHub release. The class name and API use "WavLM" which
 * is the correct architecture.
 */
class WavLMPhonemeDetector(private val context: Context) : Closeable {

    companion object {
        private const val TAG              = "WavLMDetector"
        private const val MODEL_FILE_NAME  = "wavlm_phoneme.onnx"
        private const val LEGACY_FILE_NAME = "wav2vec2_phoneme.onnx"  // renamed on first run
        private const val VOCAB_ASSET      = "wavlm_vocab.json"
        private const val SAMPLE_RATE      = AudioRecorder.SAMPLE_RATE

        // Special token ids. PAD (CTC blank) is id 0 per the exported vocab.
        private const val PAD_TOKEN_ID    = 0
        private const val UNK_TOKEN_ID    = 1

        // IPA length mark. Must be merged into the preceding vowel in the CTC post-processor,
        // otherwise it is emitted as a standalone phoneme and breaks alignment with CMU dict
        // long-vowel forms (iː, uː, ɑː, ɔː, ɜː).
        private const val LENGTH_MARK     = "ː"

        // Minimum mean softmax posterior for a phoneme segment to be emitted. Segments below
        // this are almost always acoustic noise or boundary artefacts. Calibrated empirically
        // on L2 English speech; should be re-tuned per checkpoint.
        private const val MIN_POSTERIOR   = 0.08f

        // Special / non-phoneme tokens in the vocabulary that must not appear in output.
        private val SKIP_TOKENS = setOf(
            "<pad>", "<unk>", "|", "<s>", "</s>", "[PAD]", "[UNK]",
            " "  // word-boundary separator, not a phoneme
        )

        // ONNX tensor names — must match WavLMOnnxWrapper in export_model.py.
        private const val INPUT_NAME  = "input_values"
        private const val OUTPUT_NAME = "logits"

        /**
         * Pinned SHA-256 of the published ONNX model. Set via BuildConfig at build time
         * (see app/build.gradle buildConfigField) so releases can update the hash without
         * editing source. An empty string disables verification (debug-only convenience).
         */
        val MODEL_SHA256: String = BuildConfig.WAVLM_MODEL_SHA256

        /**
         * Public download URL for the ONNX weights. The remote file name still reflects the
         * old "wav2vec2_" naming because it corresponds to the v2.0 release asset.
         */
        val MODEL_DOWNLOAD_URL: String = BuildConfig.WAVLM_MODEL_URL
    }

    private var env: OrtEnvironment? = null
    private var session: OrtSession? = null

    /** Maps token id → IPA string (e.g. 25 → "æ", 50 → "ː"). */
    val vocab: Map<Int, String>

    /** True once the ONNX model is downloaded, verified, and the session is initialised. */
    @Volatile var isAvailable: Boolean = false

    init {
        vocab = loadVocab()
        // Session init deferred to downloadAndInit() which runs on a background dispatcher:
        // loading a ~380 MB ONNX graph on the main thread would freeze the UI.
    }

    // ── Vocab loading ────────────────────────────────────────────────────────

    private fun loadVocab(): Map<Int, String> {
        return try {
            val json = context.assets.open(VOCAB_ASSET).bufferedReader().readText()
            val obj  = JSONObject(json)
            val map  = mutableMapOf<Int, String>()
            obj.keys().forEach { key -> map[key.toInt()] = obj.getString(key) }
            if (BuildConfig.DEBUG) Log.i(TAG, "Vocab loaded: ${map.size} tokens")
            map
        } catch (e: Exception) {
            Log.e(TAG, "Vocab load failed: ${e.message}")
            emptyMap()
        }
    }

    // ── Model download + verification + session init ────────────────────────

    /**
     * Downloads the model if not already present, verifies its SHA-256 against the pinned
     * hash, and initialises the ONNX session on a background thread.
     */
    suspend fun downloadAndInit(onProgress: (Float) -> Unit) {
        if (isAvailable) { onProgress(1f); return }
        val modelFile = File(context.filesDir, MODEL_FILE_NAME)
        withContext(Dispatchers.IO) {
            // One-time migration: rename the legacy file if it exists, so users who
            // installed prior builds do not have to re-download.
            val legacy = File(context.filesDir, LEGACY_FILE_NAME)
            if (legacy.exists() && !modelFile.exists()) {
                if (legacy.renameTo(modelFile)) {
                    if (BuildConfig.DEBUG) Log.i(TAG, "Migrated $LEGACY_FILE_NAME → $MODEL_FILE_NAME")
                }
            }

            if (!modelFile.exists() || modelFile.length() <= 1_000_000L) {
                downloadModel(modelFile, onProgress)
            }

            if (!verifyHash(modelFile)) {
                modelFile.delete()
                throw SecurityException(
                    "Downloaded model hash does not match pinned SHA-256 — refusing to load."
                )
            }

            isAvailable = initSession(modelFile)
        }
        onProgress(1f)
    }

    private fun initSession(modelFile: File): Boolean {
        return try {
            env = OrtEnvironment.getEnvironment()
            val options = OrtSession.SessionOptions().apply {
                setIntraOpNumThreads(2)
                setOptimizationLevel(OrtSession.SessionOptions.OptLevel.BASIC_OPT)
            }
            session = env!!.createSession(modelFile.absolutePath, options)
            if (BuildConfig.DEBUG) {
                logTensorNames()
                Log.i(TAG, "WavLM model loaded (${modelFile.length() / 1_048_576} MB, " +
                        "vocab: ${vocab.size} tokens)")
            }
            true
        } catch (e: Exception) {
            Log.e(TAG, "WavLM model load failed: ${e.message}")
            false
        }
    }

    private fun downloadModel(dest: File, onProgress: (Float) -> Unit) {
        val tmp = File(dest.parent, "${dest.name}.tmp")
        try {
            val conn = URL(MODEL_DOWNLOAD_URL).openConnection() as HttpURLConnection
            conn.connectTimeout = 15_000
            conn.readTimeout    = 60_000
            conn.connect()
            val total = conn.contentLengthLong
            var downloaded = 0L
            conn.inputStream.use { input ->
                tmp.outputStream().use { output ->
                    val buf = ByteArray(32_768)
                    var n: Int
                    while (input.read(buf).also { n = it } != -1) {
                        output.write(buf, 0, n)
                        downloaded += n
                        if (total > 0) onProgress(downloaded.toFloat() / total)
                    }
                }
            }
            if (!tmp.renameTo(dest)) {
                throw java.io.IOException("Failed to finalise download to ${dest.absolutePath}")
            }
            if (BuildConfig.DEBUG) Log.i(TAG, "Model downloaded: ${dest.length() / 1_048_576} MB")
        } catch (e: Exception) {
            tmp.delete()
            Log.e(TAG, "Download failed: ${e.message}")
            throw e
        }
    }

    /**
     * Verifies the on-disk model file against the pinned SHA-256 hash. Returns true when
     * the hashes match OR the pinned hash is blank (useful for debug builds with unreleased
     * checkpoints). Returns false on mismatch; callers must delete the file in that case.
     */
    private fun verifyHash(modelFile: File): Boolean {
        val pinned = MODEL_SHA256.trim().lowercase()
        if (pinned.isEmpty()) {
            if (BuildConfig.DEBUG) Log.w(TAG, "WAVLM_MODEL_SHA256 is blank — skipping verification.")
            return true
        }
        val actual = sha256Hex(modelFile)
        val ok = actual == pinned
        if (!ok) Log.e(TAG, "SHA-256 mismatch: expected=$pinned actual=$actual")
        return ok
    }

    private fun sha256Hex(file: File): String {
        val md = MessageDigest.getInstance("SHA-256")
        file.inputStream().use { input ->
            val buf = ByteArray(65_536)
            var n: Int
            while (input.read(buf).also { n = it } != -1) md.update(buf, 0, n)
        }
        return md.digest().joinToString("") { "%02x".format(it) }
    }

    private fun logTensorNames() {
        val s = session ?: return
        Log.i(TAG, "=== WavLM ONNX tensor names ===")
        s.inputNames.forEachIndexed  { i, n -> Log.i(TAG, "  Input  [$i]: \"$n\"") }
        s.outputNames.forEachIndexed { i, n -> Log.i(TAG, "  Output [$i]: \"$n\"") }
    }

    // ── Public inference API ─────────────────────────────────────────────────

    /**
     * Detect phonemes in raw 16 kHz mono PCM audio.
     *
     * Returns a list of [PhonemeResult] where each entry carries:
     *  - the IPA symbol (long vowels are reconstructed by merging the ː length mark)
     *  - real acoustic start/end times derived from the model's frame rate
     *  - [PhonemeResult.confidence] = mean softmax posterior of the assigned token across
     *    the frames belonging to the phoneme (a direct Bayesian posterior, not a proxy).
     *  - [PhonemeResult.score] = confidence × 100, unadjusted. Alignment-aware adjustments
     *    live in [PhonemeDetector.computeOverallScore], not here.
     */
    fun detectPhonemes(samples: ShortArray): List<PhonemeResult> {
        val s = session ?: return emptyList()
        val e = env     ?: return emptyList()
        if (!isAvailable || samples.isEmpty()) return emptyList()

        return try {
            val floats      = normalizePcm(samples)
            val inputShape  = longArrayOf(1L, floats.size.toLong())
            val inputTensor = OnnxTensor.createTensor(e, FloatBuffer.wrap(floats), inputShape)

            val output = s.run(mapOf(INPUT_NAME to inputTensor))
            @Suppress("UNCHECKED_CAST")
            val logits = output[OUTPUT_NAME].get().value as Array<Array<FloatArray>>
            inputTensor.close()
            output.close()

            val frameLogits = logits[0]
            if (BuildConfig.DEBUG) {
                Log.d(TAG, "Inference: ${samples.size} samples → ${frameLogits.size} frames")
            }
            decodeCtc(frameLogits, samples.size)
        } catch (ex: Exception) {
            Log.e(TAG, "Inference error: ${ex.message}")
            emptyList()
        }
    }

    // ── PCM normalization ────────────────────────────────────────────────────

    private fun normalizePcm(samples: ShortArray): FloatArray {
        val scale = 1f / Short.MAX_VALUE.toFloat()
        return FloatArray(samples.size) { samples[it] * scale }
    }

    // ── CTC greedy decoding ──────────────────────────────────────────────────

    /**
     * Intermediate segment emitted by the collapse pass. The `symbol` field lets us
     * rewrite a segment's label independently of its token id, which is how length-mark
     * merging is implemented.
     */
    private data class PhonemeSegment(
        val symbol: String,
        val startFrame: Int,
        val endFrame: Int,            // exclusive
        val avgProb: Float
    )

    /**
     * CTC greedy decode:
     *  1. Softmax + argmax per frame.
     *  2. Collapse consecutive identical tokens into runs; PAD (blank) resets the run so
     *     genuinely repeated phonemes separated by blanks are still emitted twice.
     *  3. Drop non-phoneme tokens (padding, BOS/EOS, word boundary, unknown).
     *  4. Merge the length mark "ː" into the preceding vowel segment so long-vowel forms
     *     are recovered from the model's short-vowel + length-mark token stream.
     *  5. Drop any segment whose mean posterior falls below MIN_POSTERIOR.
     *
     * Frame→time mapping is derived from the actual samples/frames ratio rather than
     * hard-coding WavLM's nominal 20 ms stride; this stays correct if the feature extractor
     * changes.
     */
    private fun decodeCtc(frameLogits: Array<FloatArray>, totalSamples: Int): List<PhonemeResult> {
        val numFrames = frameLogits.size
        if (numFrames == 0) return emptyList()
        val msPerFrame = totalSamples.toDouble() / numFrames / SAMPLE_RATE * 1000.0

        // Pass 1: softmax + argmax per frame.
        data class FrameDecision(val tokenId: Int, val prob: Float)
        val decisions = frameLogits.map { logits ->
            val probs  = softmax(logits)
            val argmax = probs.indices.maxByOrNull { probs[it] } ?: PAD_TOKEN_ID
            FrameDecision(argmax, probs[argmax])
        }

        // Pass 2: collapse runs into raw segments (still keyed by token id).
        val raw = mutableListOf<PhonemeSegment>()
        var prevToken = -1
        var runStart  = 0
        val runProbs  = mutableListOf<Float>()

        fun flushRun(endFrame: Int) {
            if (prevToken != -1 && prevToken != PAD_TOKEN_ID && runProbs.isNotEmpty()) {
                val symbol = vocab[prevToken]
                if (symbol != null && symbol !in SKIP_TOKENS && prevToken != UNK_TOKEN_ID) {
                    raw.add(PhonemeSegment(
                        symbol     = symbol,
                        startFrame = runStart,
                        endFrame   = endFrame,
                        avgProb    = runProbs.average().toFloat()
                    ))
                }
                runProbs.clear()
            }
        }

        for (i in decisions.indices) {
            val d = decisions[i]
            when {
                d.tokenId == PAD_TOKEN_ID -> {
                    flushRun(i)
                    prevToken = PAD_TOKEN_ID
                }
                d.tokenId != prevToken -> {
                    flushRun(i)
                    prevToken = d.tokenId
                    runStart  = i
                    runProbs.add(d.prob)
                }
                else -> runProbs.add(d.prob)
            }
        }
        flushRun(numFrames)

        // Pass 3: merge length marks into the preceding vowel segment.
        // If the length mark appears without a preceding vowel (rare edge case at
        // utterance start) we simply drop it — it is meaningless on its own.
        val merged = mutableListOf<PhonemeSegment>()
        for (seg in raw) {
            if (seg.symbol == LENGTH_MARK) {
                if (merged.isNotEmpty()) {
                    val prev = merged.removeAt(merged.lastIndex)
                    val blendedProb = (prev.avgProb + seg.avgProb) / 2f
                    merged.add(prev.copy(
                        symbol   = prev.symbol + LENGTH_MARK,
                        endFrame = seg.endFrame,
                        avgProb  = blendedProb
                    ))
                }
                // else: dangling length mark — discarded.
                continue
            }
            merged.add(seg)
        }

        // Pass 4: apply posterior threshold and convert to PhonemeResult.
        return merged.mapNotNull { seg ->
            if (seg.avgProb < MIN_POSTERIOR) return@mapNotNull null
            val startMs = (seg.startFrame * msPerFrame).toLong()
            val endMs   = ((seg.endFrame * msPerFrame).toLong()).coerceAtLeast(startMs + 20L)
            PhonemeResult(
                phoneme     = seg.symbol,
                startTimeMs = startMs,
                endTimeMs   = endMs,
                confidence  = seg.avgProb,
                score       = (seg.avgProb * 100f).coerceIn(0f, 100f)
            )
        }
    }

    // ── Math helpers ─────────────────────────────────────────────────────────

    private fun softmax(logits: FloatArray): FloatArray {
        val maxVal = logits.max()
        val exps   = FloatArray(logits.size) { exp((logits[it] - maxVal).toDouble()).toFloat() }
        val sum    = exps.sum().coerceAtLeast(1e-9f)
        return FloatArray(exps.size) { exps[it] / sum }
    }

    /** Natural-log acoustic score for downstream GOP-style computation. */
    internal fun logPosterior(posterior: Float): Float =
        if (posterior <= 0f) Float.NEGATIVE_INFINITY else ln(posterior)

    // ── Lifecycle ────────────────────────────────────────────────────────────

    override fun close() {
        session?.close()
        env?.close()
        session = null
        env     = null
    }
}
