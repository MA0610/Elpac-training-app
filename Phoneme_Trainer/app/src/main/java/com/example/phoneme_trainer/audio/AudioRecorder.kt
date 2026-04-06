package com.example.phoneme_trainer.audio

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import android.util.Log
import androidx.core.content.ContextCompat
import com.example.phoneme_trainer.model.WaveformPoint
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.flow.flowOn
import kotlinx.coroutines.isActive
import kotlin.coroutines.coroutineContext
import androidx.annotation.RequiresPermission

private const val TAG = "AudioREC"

/**
 * Real-time audio recorder that streams PCM samples at 16kHz mono.
 *
 * The AudioRecord is created once and kept in RECORDING state permanently (never stopped
 * between sessions). This prevents the emulator's host-mic bridge from dropping when
 * stop() is called, which causes it to fall back to a synthetic 440Hz tone.
 *
 * Between recording sessions the accumulated buffer is drained with non-blocking reads
 * so stale audio doesn't contaminate the next session.
 *
 * Call release() when the recorder is no longer needed (e.g. ViewModel.onCleared()).
 */
class AudioRecorder {

    companion object {
        const val SAMPLE_RATE = 16000
        const val CHANNEL_CONFIG = AudioFormat.CHANNEL_IN_MONO
        const val AUDIO_FORMAT = AudioFormat.ENCODING_PCM_16BIT
        const val CHUNK_SIZE_MS = 100
        val CHUNK_SAMPLES = SAMPLE_RATE * CHUNK_SIZE_MS / 1000  // 1600 samples
        val MIN_BUFFER = AudioRecord.getMinBufferSize(SAMPLE_RATE, CHANNEL_CONFIG, AUDIO_FORMAT)
        val BUFFER_SIZE = maxOf(MIN_BUFFER, CHUNK_SAMPLES * 2 * 4)
    }

    private var audioRecord: AudioRecord? = null
    @Volatile private var isRecording = false
    private val allSamples = mutableListOf<Short>()
    private var cycleCount = 0

    /**
     * Returns a flow of audio chunks (Short arrays). Each chunk is CHUNK_SAMPLES long.
     *
     * Creates the AudioRecord on first call and immediately calls startRecording() — it
     * stays in RECORDING state for all subsequent sessions. Between sessions the stale
     * buffer is drained rather than stopping and restarting the hardware.
     */
    @RequiresPermission(Manifest.permission.RECORD_AUDIO)
    fun recordingFlow(context: Context): Flow<ShortArray> = flow {
        if (ContextCompat.checkSelfPermission(context, Manifest.permission.RECORD_AUDIO)
            != PackageManager.PERMISSION_GRANTED
        ) {
            throw SecurityException("RECORD_AUDIO permission not granted")
        }

        cycleCount++
        val cycle = cycleCount

        // Create the AudioRecord once and immediately start it. On subsequent sessions
        // the same instance is reused — we never call stop() on it between sessions.
        val record = audioRecord?.takeIf { it.state == AudioRecord.STATE_INITIALIZED } ?: run {
            Log.d(TAG, "[$cycle] Creating new AudioRecord (first session or after release)")
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
                Log.e(TAG, "[$cycle] AudioRecord failed to initialize! state=${ar.state}")
                ar.release()
                throw IllegalStateException("AudioRecord failed to initialize — microphone may be unavailable")
            }
            ar.startRecording()
            Log.d(TAG, "[$cycle] AudioRecord ${ar.hashCode()} created and startRecording() called, recordingState=${ar.recordingState}")
            audioRecord = ar
            ar
        }

        Log.d(TAG, "[$cycle] Flow START — AR=${record.hashCode()}, state=${record.state}, recordingState=${record.recordingState}, isRecording=$isRecording")

        allSamples.clear()

        // Drain any audio that accumulated in the hardware buffer while we were idle.
        // Use READ_NON_BLOCKING so this completes instantly rather than blocking.
        val drainBuf = ShortArray(CHUNK_SAMPLES)
        var drained = 0
        while (true) {
            val n = record.read(drainBuf, 0, drainBuf.size, AudioRecord.READ_NON_BLOCKING)
            if (n <= 0 || drained >= 100) break
            drained++
        }
        Log.d(TAG, "[$cycle] Drained $drained stale chunks from buffer")

        isRecording = true
        val buffer = ShortArray(CHUNK_SAMPLES)

        try {
            var chunkNum = 0
            while (coroutineContext.isActive && isRecording) {
                val read = record.read(buffer, 0, buffer.size)
                // Re-check after the blocking read — stop() may have been called while blocked.
                if (read > 0 && isRecording) {
                    val chunk = buffer.copyOf(read)
                    val rms = chunkRmsLevel(chunk)
                    chunkNum++
                    if (chunkNum <= 5 || chunkNum % 20 == 0) {
                        Log.d(TAG, "[$cycle] chunk #$chunkNum: read=$read, rms=${"%.3f".format(rms)}, max=${chunk.maxOrNull()}, min=${chunk.minOrNull()}")
                    }
                    allSamples.addAll(chunk.toList())
                    emit(chunk)
                } else if (read <= 0) {
                    Log.d(TAG, "[$cycle] read returned $read (error or stopped)")
                }
            }
            Log.d(TAG, "[$cycle] Loop exited: isActive=${coroutineContext.isActive}, isRecording=$isRecording, totalChunks=$chunkNum")
        } finally {
            isRecording = false
            // IMPORTANT: Do NOT call record.stop() here. Keeping AudioRecord in RECORDING
            // state preserves the emulator's host-mic bridge. Stopping it causes the bridge
            // to drop, and startRecording() on the same instance then produces a 440Hz
            // synthetic tone instead of real microphone audio.
            Log.d(TAG, "[$cycle] Flow FINALLY: isRecording=false, AR stays in RECORDING state")
        }
    }.flowOn(Dispatchers.IO)

    /** Signals the collection loop to stop. The AudioRecord hardware keeps running. */
    fun stop() {
        Log.d(TAG, "stop() — setting isRecording=false (was $isRecording)")
        isRecording = false
    }

    /** Fully releases the AudioRecord hardware. Call only from ViewModel.onCleared(). */
    fun release() {
        Log.d(TAG, "release() — stopping and releasing AR=${audioRecord?.hashCode()}")
        isRecording = false
        val ar = audioRecord
        audioRecord = null
        ar?.stop()
        ar?.release()
    }

    /** Returns all accumulated PCM samples from this session. */
    fun getAllSamples(): ShortArray = allSamples.toShortArray()

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

    /** Compute RMS energy of a chunk (0.0–1.0) for live level meter. */
    fun chunkRmsLevel(chunk: ShortArray): Float {
        if (chunk.isEmpty()) return 0f
        val sumSq = chunk.fold(0.0) { acc, s ->
            acc + (s.toDouble() / Short.MAX_VALUE).let { it * it }
        }
        return kotlin.math.sqrt(sumSq / chunk.size).toFloat().coerceIn(0f, 1f)
    }
}
