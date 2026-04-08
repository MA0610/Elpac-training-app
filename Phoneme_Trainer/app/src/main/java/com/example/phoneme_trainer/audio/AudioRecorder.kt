package com.example.phoneme_trainer.audio

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import android.util.Log
import androidx.annotation.RequiresPermission
import androidx.core.content.ContextCompat
import com.example.phoneme_trainer.BuildConfig
import com.example.phoneme_trainer.model.WaveformPoint
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.flow.flowOn
import kotlinx.coroutines.isActive
import kotlin.coroutines.coroutineContext

private const val TAG = "AudioREC"

/**
 * Real-time audio recorder that streams PCM samples at 16 kHz mono.
 *
 * The underlying [AudioRecord] is created once and kept in RECORDING state permanently
 * (never stopped between sessions). Stopping and restarting it on the emulator causes
 * the host-mic bridge to drop and fall back to a synthetic 440 Hz tone; on real devices
 * this is also more efficient than tearing down and rebuilding the input pipeline.
 *
 * Between sessions the stale hardware buffer is drained with non-blocking reads so
 * previous audio cannot leak into the next recording.
 *
 * Samples are accumulated into a growable primitive [ShortArray] rather than a
 * `MutableList<Short>` to avoid boxing — a 10-second recording is 160k samples, which
 * would otherwise produce 160k boxed `Short` objects and a full O(n) copy in
 * [getAllSamples].
 *
 * Call [release] from `ViewModel.onCleared()` when the recorder is no longer needed.
 */
class AudioRecorder {

    companion object {
        const val SAMPLE_RATE    = 16000
        const val CHANNEL_CONFIG = AudioFormat.CHANNEL_IN_MONO
        const val AUDIO_FORMAT   = AudioFormat.ENCODING_PCM_16BIT
        const val CHUNK_SIZE_MS  = 100
        val CHUNK_SAMPLES        = SAMPLE_RATE * CHUNK_SIZE_MS / 1000  // 1600 samples
        val MIN_BUFFER           = AudioRecord.getMinBufferSize(SAMPLE_RATE, CHANNEL_CONFIG, AUDIO_FORMAT)
        val BUFFER_SIZE          = maxOf(MIN_BUFFER, CHUNK_SAMPLES * 2 * 4)

        // Initial capacity for the sample accumulator: ~3 s of audio. Grown geometrically
        // as needed, so long recordings still only see a handful of reallocations.
        private const val INITIAL_SAMPLE_CAPACITY = SAMPLE_RATE * 3
    }

    private var audioRecord: AudioRecord? = null
    @Volatile private var isRecording = false

    // Primitive-backed growable sample buffer. Access is confined to the single IO
    // coroutine that owns the recording flow, so no synchronization is required.
    private var samples: ShortArray = ShortArray(INITIAL_SAMPLE_CAPACITY)
    private var sampleCount: Int = 0

    @RequiresPermission(Manifest.permission.RECORD_AUDIO)
    fun recordingFlow(context: Context): Flow<ShortArray> = flow {
        if (ContextCompat.checkSelfPermission(context, Manifest.permission.RECORD_AUDIO)
            != PackageManager.PERMISSION_GRANTED
        ) {
            throw SecurityException("RECORD_AUDIO permission not granted")
        }

        val record = audioRecord?.takeIf { it.state == AudioRecord.STATE_INITIALIZED } ?: run {
            val ar = try {
                AudioRecord(
                    MediaRecorder.AudioSource.MIC,
                    SAMPLE_RATE,
                    CHANNEL_CONFIG,
                    AUDIO_FORMAT,
                    BUFFER_SIZE
                )
            } catch (e: SecurityException) {
                throw SecurityException("Microphone permission not granted", e)
            }
            if (ar.state != AudioRecord.STATE_INITIALIZED) {
                ar.release()
                throw IllegalStateException("AudioRecord failed to initialize — microphone may be unavailable")
            }
            ar.startRecording()
            audioRecord = ar
            ar
        }

        resetSampleBuffer()

        // Drain stale audio from the hardware buffer with a bounded non-blocking loop.
        val drainBuf = ShortArray(CHUNK_SAMPLES)
        var drained = 0
        while (drained < 100) {
            val n = record.read(drainBuf, 0, drainBuf.size, AudioRecord.READ_NON_BLOCKING)
            if (n <= 0) break
            drained++
        }
        if (BuildConfig.DEBUG) Log.d(TAG, "Drained $drained stale chunks from buffer")

        isRecording = true
        val buffer = ShortArray(CHUNK_SAMPLES)

        try {
            while (coroutineContext.isActive && isRecording) {
                val read = record.read(buffer, 0, buffer.size)
                if (read > 0 && isRecording) {
                    val chunk = buffer.copyOf(read)
                    appendSamples(chunk)
                    emit(chunk)
                } else if (read < 0 && BuildConfig.DEBUG) {
                    Log.w(TAG, "AudioRecord.read returned $read")
                }
            }
        } finally {
            isRecording = false
            // Intentionally NOT calling record.stop() — see class comment.
        }
    }.flowOn(Dispatchers.IO)

    /** Signals the collection loop to stop. The AudioRecord hardware stays active. */
    fun stop() {
        isRecording = false
    }

    /** Fully releases the AudioRecord hardware. Call only from ViewModel.onCleared(). */
    fun release() {
        isRecording = false
        val ar = audioRecord
        audioRecord = null
        ar?.stop()
        ar?.release()
    }

    /** Returns all accumulated PCM samples from the current session. */
    fun getAllSamples(): ShortArray = samples.copyOf(sampleCount)

    /** Builds a downsampled waveform suitable for display. */
    fun buildWaveform(samples: ShortArray, targetPoints: Int = 300): List<WaveformPoint> {
        if (samples.isEmpty()) return emptyList()
        val step = maxOf(1, samples.size / targetPoints)
        return (0 until targetPoints).mapNotNull { i ->
            val sampleIndex = i * step
            if (sampleIndex >= samples.size) return@mapNotNull null
            val amplitude = samples[sampleIndex].toFloat() / Short.MAX_VALUE
            val timeMs = sampleIndex.toLong() * 1000L / SAMPLE_RATE
            WaveformPoint(timeMs, amplitude)
        }
    }

    /** RMS energy of a chunk in [0, 1] for the live level meter. */
    fun chunkRmsLevel(chunk: ShortArray): Float {
        if (chunk.isEmpty()) return 0f
        val sumSq = chunk.fold(0.0) { acc, s ->
            acc + (s.toDouble() / Short.MAX_VALUE).let { it * it }
        }
        return kotlin.math.sqrt(sumSq / chunk.size).toFloat().coerceIn(0f, 1f)
    }

    // ── Sample buffer management ────────────────────────────────────────────

    private fun resetSampleBuffer() {
        sampleCount = 0
        // Reuse the existing allocation when possible; zero-fill is unnecessary since
        // sampleCount tracks the valid region.
    }

    private fun appendSamples(chunk: ShortArray) {
        ensureCapacity(sampleCount + chunk.size)
        System.arraycopy(chunk, 0, samples, sampleCount, chunk.size)
        sampleCount += chunk.size
    }

    private fun ensureCapacity(required: Int) {
        if (required <= samples.size) return
        var newSize = samples.size
        while (newSize < required) newSize = (newSize * 2).coerceAtLeast(required)
        samples = samples.copyOf(newSize)
    }
}
