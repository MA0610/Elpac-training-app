package com.example.phoneme_trainer.audio

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import androidx.core.content.ContextCompat
import com.example.phoneme_trainer.model.WaveformPoint
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.flow.flowOn
import kotlinx.coroutines.isActive
import kotlin.coroutines.coroutineContext
import androidx.annotation.RequiresPermission


/**
 * Real-time audio recorder that streams PCM samples at 16kHz mono.
 *
 * Usage:
 *   val recorder = AudioRecorder()
 *   recorder.recordingFlow(context).collect { chunk -> process(chunk) }
 *   recorder.stop()
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
    private var isRecording = false
    private val allSamples = mutableListOf<Short>()

    /**
     * Returns a flow of audio chunks (Short arrays). Each chunk is CHUNK_SAMPLES long.
     * Collect this in a coroutine; cancel the coroutine to stop recording.
     */
    @RequiresPermission(Manifest.permission.RECORD_AUDIO)
    fun recordingFlow(context: Context): Flow<ShortArray> = flow {
        // Explicitly check permission before constructing AudioRecord
        if (ContextCompat.checkSelfPermission(context, Manifest.permission.RECORD_AUDIO)
            != PackageManager.PERMISSION_GRANTED
        ) {
            throw SecurityException("RECORD_AUDIO permission not granted")
        }

        val record = try {
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

        audioRecord = record
        allSamples.clear()

        try {
            record.startRecording()
            isRecording = true
            val buffer = ShortArray(CHUNK_SAMPLES)

            while (coroutineContext.isActive && isRecording) {
                val read = record.read(buffer, 0, buffer.size)
                if (read > 0) {
                    val chunk = buffer.copyOf(read)
                    allSamples.addAll(chunk.toList())
                    emit(chunk)
                }
            }
        } finally {
            record.stop()
            record.release()
            audioRecord = null
            isRecording = false
        }
    }.flowOn(Dispatchers.IO)

    fun stop() {
        isRecording = false
        audioRecord?.stop()
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
