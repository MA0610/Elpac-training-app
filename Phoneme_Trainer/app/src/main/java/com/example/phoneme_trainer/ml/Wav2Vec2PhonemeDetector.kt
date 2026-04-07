package com.example.phoneme_trainer.ml

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import android.util.Log
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
import kotlin.math.exp

/**
 * On-device phoneme detection using a fine-tuned WavLM ONNX model.
 *
 * The model outputs per-frame logits over 52 token classes: 26 ASCII letters (a-z) for basic phonemes
 * and 26 IPA symbols for more complex phonetic distinctions. CTC greedy decoding converts this to
 * a phoneme sequence (mix of ASCII and IPA) with real acoustic timing and per-phoneme softmax
 * confidence (a true acoustic posterior).
 *
 * This replaces Vosk's word-level confidence proxy with real per-phoneme scores.
 */
class Wav2Vec2PhonemeDetector(private val context: Context) : Closeable {

    companion object {
        private const val TAG          = "Wav2Vec2Detector"
        private const val MODEL_ASSET  = "wav2vec2_phoneme.onnx"
        private const val VOCAB_ASSET  = "wav2vec2_vocab.json"
        private const val SAMPLE_RATE  = AudioRecorder.SAMPLE_RATE
        private const val PAD_TOKEN_ID = 0       // <pad> is the CTC blank token
        private const val MIN_CONF     = 0.08f   // WavLM model confidence threshold

        // WavLM special tokens to strip from decoded output
        // Note: ASCII letters (a-z) are kept as they represent valid phonemes
        private val SKIP_TOKENS = setOf(
            "<pad>", "<unk>", "|", "<s>", "</s>", "[PAD]", "[UNK]",
            " "  // word boundary token (not a phoneme)
        )

        // Tensor names matching WavLMOnnxWrapper in export_model.py
        private const val INPUT_NAME  = "input_values"
        private const val OUTPUT_NAME = "logits"

        // Upload wav2vec2_phoneme.onnx to GitHub Releases and paste the URL here.
        // Example: "https://github.com/MA0610/Elpac-training-app/releases/download/v2.0/wav2vec2_phoneme.onnx"
        const val MODEL_DOWNLOAD_URL = "https://github.com/MA0610/Elpac-training-app/releases/download/v2.0/wav2vec2_phoneme.onnx"
    }

    private var env: OrtEnvironment? = null
    private var session: OrtSession? = null

    /** Maps token index → IPA string (e.g. 2 → "æ"). */
    val vocab: Map<Int, String>

    /** True if the ONNX model is loaded and ready. Updated after downloadAndInit(). */
    @Volatile var isAvailable: Boolean = false

    init {
        vocab = loadVocab()
        // Session init is deferred to downloadAndInit() which runs on a background thread.
        // Loading a 1+ GB model on the main thread would freeze the UI.
    }

    // ── Model loading ─────────────────────────────────────────────────────────

    private fun initSession(modelFile: File = File(context.filesDir, MODEL_ASSET)): Boolean {
        return try {
            env = OrtEnvironment.getEnvironment()
            val sessionOptions = OrtSession.SessionOptions().apply {
                setIntraOpNumThreads(2)
                setOptimizationLevel(OrtSession.SessionOptions.OptLevel.BASIC_OPT)
            }
            session = env!!.createSession(modelFile.absolutePath, sessionOptions)
            logTensorNames()
            Log.i(TAG, "Wav2Vec2 model loaded (${modelFile.length() / 1_048_576} MB, vocab: ${vocab.size} tokens)")
            true
        } catch (e: Exception) {
            Log.e(TAG, "Wav2Vec2 model load failed: ${e.message}")
            false
        }
    }

    // ── Download ──────────────────────────────────────────────────────────────

    /** Downloads the model if not present, then initialises the ONNX session on a background thread. */
    suspend fun downloadAndInit(onProgress: (Float) -> Unit) {
        if (isAvailable) { onProgress(1f); return }
        val modelFile = File(context.filesDir, MODEL_ASSET)
        withContext(Dispatchers.IO) {
            if (!modelFile.exists() || modelFile.length() <= 1_000_000L) {
                downloadModel(modelFile, onProgress)
            }
            // initSession() on IO thread — 1+ GB model must not block main thread
            isAvailable = initSession(modelFile)
        }
        onProgress(1f)
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
            tmp.renameTo(dest)
            Log.i(TAG, "Model downloaded: ${dest.length() / 1_048_576} MB")
        } catch (e: Exception) {
            tmp.delete()
            Log.e(TAG, "Download failed: ${e.message}")
            throw e
        }
    }

    /** Logs input/output tensor names so you can verify they match INPUT_NAME/OUTPUT_NAME. */
    private fun logTensorNames() {
        val s = session ?: return
        Log.i(TAG, "=== Wav2Vec2 ONNX tensor names ===")
        s.inputNames.forEachIndexed  { i, n -> Log.i(TAG, "  Input  [$i]: \"$n\"") }
        s.outputNames.forEachIndexed { i, n -> Log.i(TAG, "  Output [$i]: \"$n\"") }
        Log.i(TAG, "If names differ from \"$INPUT_NAME\"/\"$OUTPUT_NAME\", " +
                "update the constants in Wav2Vec2PhonemeDetector.kt")
    }

    private fun loadVocab(): Map<Int, String> {
        return try {
            val json = context.assets.open(VOCAB_ASSET).bufferedReader().readText()
            val obj  = JSONObject(json)
            val map  = mutableMapOf<Int, String>()
            // Keys are token IDs (as strings), values are IPA symbols
            obj.keys().forEach { key -> map[key.toInt()] = obj.getString(key) }
            Log.i(TAG, "Vocab loaded: ${map.size} tokens")
            map
        } catch (e: Exception) {
            Log.e(TAG, "Vocab load failed: ${e.message}")
            emptyMap()
        }
    }

    // ── Public API ────────────────────────────────────────────────────────────

    /**
     * Detect phonemes in raw 16 kHz mono PCM audio.
     *
     * Returns a list of [PhonemeResult] with:
     * - Real acoustic timing derived from model output frame count
     * - Per-phoneme confidence = mean softmax probability of winning token
     * - Initial score = confidence × 100 (refined later by alignment)
     */
    fun detectPhonemes(samples: ShortArray): List<PhonemeResult> {
        val s = session ?: return emptyList()
        val e = env    ?: return emptyList()
        if (!isAvailable || samples.isEmpty()) return emptyList()

        return try {
            val floats      = normalizePcm(samples)
            val inputShape  = longArrayOf(1L, floats.size.toLong())
            val inputTensor = OnnxTensor.createTensor(e, FloatBuffer.wrap(floats), inputShape)

            val output  = s.run(mapOf(INPUT_NAME to inputTensor))
            // logits shape: [1, num_frames, vocab_size]
            val logits  = output[OUTPUT_NAME].get().value as Array<Array<FloatArray>>

            inputTensor.close()
            output.close()

            val frameLogits = logits[0]   // [num_frames, vocab_size]
            Log.d(TAG, "Wav2Vec2 inference: ${samples.size} samples → ${frameLogits.size} frames")

            decodeCtc(frameLogits, samples.size)
                .also { Log.d(TAG, "Wav2Vec2 detected ${it.size} phonemes") }

        } catch (ex: Exception) {
            Log.e(TAG, "Wav2Vec2 inference error: ${ex.message}")
            emptyList()
        }
    }

    // ── PCM normalization ─────────────────────────────────────────────────────

    private fun normalizePcm(samples: ShortArray): FloatArray {
        val scale = 1f / Short.MAX_VALUE.toFloat()
        return FloatArray(samples.size) { samples[it] * scale }
    }

    // ── CTC greedy decoding ───────────────────────────────────────────────────

    private data class PhonemeSegment(
        val tokenId: Int,
        val startFrame: Int,
        val endFrame: Int,        // exclusive
        val avgProb: Float
    )

    /**
     * CTC greedy decode:
     * 1. Softmax + argmax per frame
     * 2. Collapse consecutive identical tokens
     * 3. Remove PAD (blank) tokens
     * 4. Group frames into phoneme segments with timing
     */
    private fun decodeCtc(frameLogits: Array<FloatArray>, totalSamples: Int): List<PhonemeResult> {
        val numFrames = frameLogits.size
        if (numFrames == 0) return emptyList()

        // Frame → time mapping: derive stride from actual samples/frames ratio
        // (more robust than hard-coding 320-sample stride)
        val msPerFrame = totalSamples.toDouble() / numFrames / SAMPLE_RATE * 1000.0

        // Step 1: softmax + argmax per frame
        data class FrameDecision(val tokenId: Int, val prob: Float)
        val decisions = frameLogits.map { logits ->
            val probs  = softmax(logits)
            val argmax = probs.indices.maxByOrNull { probs[it] } ?: PAD_TOKEN_ID
            FrameDecision(argmax, probs[argmax])
        }

        // Step 2: collect runs (PAD resets the previous token, enabling repeated phonemes)
        val segments = mutableListOf<PhonemeSegment>()
        var prevToken = -1
        var runStart  = 0
        val runProbs  = mutableListOf<Float>()

        fun flushRun(endFrame: Int) {
            if (prevToken != -1 && prevToken != PAD_TOKEN_ID && runProbs.isNotEmpty()) {
                val avgProb = runProbs.average().toFloat()
                if (avgProb >= MIN_CONF) {
                    segments.add(PhonemeSegment(prevToken, runStart, endFrame, avgProb))
                }
                runProbs.clear()
            }
        }

        for (i in decisions.indices) {
            val d = decisions[i]
            when {
                d.tokenId == PAD_TOKEN_ID -> {
                    // PAD (blank) resets current run
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

        // Step 3: convert segments to PhonemeResult
        return segments.mapNotNull { seg ->
            val ipaSymbol = vocab[seg.tokenId] ?: return@mapNotNull null
            val isSkipped = ipaSymbol in SKIP_TOKENS
            Log.d(TAG, "Token ${seg.tokenId} → '$ipaSymbol' (conf: ${"%.3f".format(seg.avgProb)}, skipped: $isSkipped)")
            if (isSkipped) return@mapNotNull null

            val startMs = (seg.startFrame * msPerFrame).toLong()
            val endMs   = ((seg.endFrame * msPerFrame).toLong()).coerceAtLeast(startMs + 20L)

            PhonemeResult(
                phoneme     = ipaSymbol,
                startTimeMs = startMs,
                endTimeMs   = endMs,
                confidence  = seg.avgProb,
                score       = (seg.avgProb * 100f).coerceIn(0f, 100f)
            )
        }
    }

    // ── Math helpers ──────────────────────────────────────────────────────────

    private fun softmax(logits: FloatArray): FloatArray {
        val maxVal = logits.max()
        val exps   = FloatArray(logits.size) { exp((logits[it] - maxVal).toDouble()).toFloat() }
        val sum    = exps.sum().coerceAtLeast(1e-9f)
        return FloatArray(exps.size) { exps[it] / sum }
    }

    // ── Lifecycle ─────────────────────────────────────────────────────────────

    override fun close() {
        session?.close()
        env?.close()
        session = null
        env     = null
    }
}
