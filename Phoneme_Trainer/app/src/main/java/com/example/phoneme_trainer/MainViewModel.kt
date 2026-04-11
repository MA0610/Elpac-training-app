package com.example.phoneme_trainer

import android.Manifest
import android.app.Application
import android.content.pm.PackageManager
import android.media.MediaCodec
import android.media.MediaExtractor
import android.media.MediaFormat
import android.net.Uri
import android.util.Log
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.example.phoneme_trainer.audio.AudioRecorder
import com.example.phoneme_trainer.ml.PhonemeDetector
import com.example.phoneme_trainer.model.AnalysisSession
import com.example.phoneme_trainer.model.PhonemeComparison
import com.example.phoneme_trainer.model.PhonemeResult
import com.example.phoneme_trainer.model.RecordingState
import com.example.phoneme_trainer.model.TargetPhrase
import com.example.phoneme_trainer.model.WaveformPoint
import com.example.phoneme_trainer.model.WordTiming
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext
import java.io.ByteArrayOutputStream
import java.io.IOException
import java.nio.ByteBuffer
import java.nio.ByteOrder

private const val TAG = "MainVM"

enum class ModelStatus { CHECKING, DOWNLOADING, READY, FAILED }

data class MainUiState(
    val recordingState: RecordingState = RecordingState.IDLE,
    val liveLevel: Float = 0f,
    val liveWaveform: List<WaveformPoint> = emptyList(),
    val session: AnalysisSession? = null,
    val selectedPhoneme: PhonemeResult? = null,
    val errorMessage: String? = null,
    val recordingDurationMs: Long = 0L,
    val elpacLevel: String? = null,
    val targetPhrase: TargetPhrase? = null,
    val customPhraseText: String = "",
    val showPhraseSelector: Boolean = false,
    val modelStatus: ModelStatus = ModelStatus.CHECKING,
    val modelDownloadProgress: Float = 0f
)

class MainViewModel(application: Application) : AndroidViewModel(application) {

    private val recorder = AudioRecorder()
    private val detector = PhonemeDetector(application)
    private val _uiState = MutableStateFlow(MainUiState())
    val uiState: StateFlow<MainUiState> = _uiState.asStateFlow()

    // Serialises startRecording / stopRecording so a rapid tap-stop-tap sequence cannot
    // run two recording flows concurrently. Replaces the previous "join the old job from
    // inside the new job" pattern, which was subtle and prone to regressions.
    private val recordingMutex = Mutex()
    private var recordingJob: Job? = null
    private var startTimeMs = 0L
    private val liveWaveformBuffer = mutableListOf<WaveformPoint>()

    // Holds a file URI that arrived before the model was ready; processed once READY.
    private var pendingFileUri: Uri? = null

    init {
        viewModelScope.launch {
            try {
                detector.prepareWavLM { progress ->
                    _uiState.update {
                        it.copy(
                            modelStatus = ModelStatus.DOWNLOADING,
                            modelDownloadProgress = progress
                        )
                    }
                }
                _uiState.update { it.copy(modelStatus = ModelStatus.READY) }
                // Process any file that was queued while the model was loading.
                pendingFileUri?.let { uri ->
                    pendingFileUri = null
                    analyzeFromFile(uri)
                }
            } catch (e: SecurityException) {
                // Hash mismatch — never auto-retry, always surface to the user.
                Log.e(TAG, "Model verification failed", e)
                _uiState.update {
                    it.copy(
                        modelStatus = ModelStatus.FAILED,
                        errorMessage = "Model verification failed. The downloaded file did " +
                                "not match the expected hash and will not be loaded."
                    )
                }
            } catch (e: Exception) {
                Log.e(TAG, "Model download failed", e)
                _uiState.update {
                    it.copy(
                        modelStatus = ModelStatus.FAILED,
                        errorMessage = "Model download failed. Check internet connection and try again."
                    )
                }
            }
        }
    }

    // ── Permission ─────────────────────────────────────────────────────────

    fun onPermissionDenied() {
        _uiState.update {
            it.copy(errorMessage = "Microphone permission is required to analyze pronunciation.")
        }
    }

    // ── Phrase selection ───────────────────────────────────────────────────

    fun showPhraseSelector() { _uiState.update { it.copy(showPhraseSelector = true) } }
    fun hidePhraseSelector() { _uiState.update { it.copy(showPhraseSelector = false) } }

    fun selectPresetPhrase(phrase: TargetPhrase) {
        _uiState.update {
            it.copy(
                targetPhrase = phrase,
                customPhraseText = phrase.text,
                showPhraseSelector = false,
                session = null,
                selectedPhoneme = null,
                elpacLevel = null
            )
        }
    }

    fun setCustomPhrase(text: String) { _uiState.update { it.copy(customPhraseText = text) } }

    fun confirmCustomPhrase() {
        val text = _uiState.value.customPhraseText.trim()
        if (text.isBlank()) return
        _uiState.update {
            it.copy(
                targetPhrase = TargetPhrase(text, "Custom"),
                showPhraseSelector = false,
                session = null,
                selectedPhoneme = null,
                elpacLevel = null
            )
        }
    }

    fun clearPhrase() {
        _uiState.update {
            it.copy(
                targetPhrase = null,
                customPhraseText = "",
                session = null,
                selectedPhoneme = null,
                elpacLevel = null
            )
        }
    }

    // ── Recording control ──────────────────────────────────────────────────

    fun startRecording() {
        val ctx = getApplication<Application>()
        if (ctx.checkSelfPermission(Manifest.permission.RECORD_AUDIO)
            != PackageManager.PERMISSION_GRANTED
        ) {
            _uiState.update { it.copy(errorMessage = "Microphone permission not granted.") }
            return
        }
        if (_uiState.value.recordingState == RecordingState.RECORDING) return
        if (_uiState.value.modelStatus != ModelStatus.READY) return

        liveWaveformBuffer.clear()
        startTimeMs = System.currentTimeMillis()

        _uiState.update {
            it.copy(
                recordingState = RecordingState.RECORDING,
                session = null,
                selectedPhoneme = null,
                errorMessage = null,
                liveWaveform = emptyList(),
                liveLevel = 0f,
                recordingDurationMs = 0L,
                elpacLevel = null
            )
        }

        recordingJob = viewModelScope.launch {
            // Mutex guarantees the previous recording's flow has fully exited (finally
            // block ran, isRecording=false) before this one starts, without having to
            // join a job from inside another job.
            recordingMutex.withLock {
                try {
                    recorder.recordingFlow(ctx).collect { chunk ->
                        val rms       = recorder.chunkRmsLevel(chunk)
                        val timeMs    = System.currentTimeMillis() - startTimeMs
                        val amplitude = chunk.maxOrNull()?.toFloat()?.div(Short.MAX_VALUE) ?: 0f
                        liveWaveformBuffer.add(WaveformPoint(timeMs, amplitude))
                        _uiState.update {
                            it.copy(
                                liveLevel = rms,
                                liveWaveform = liveWaveformBuffer.toList(),
                                recordingDurationMs = timeMs
                            )
                        }
                    }
                } catch (e: CancellationException) {
                    throw e
                } catch (e: Exception) {
                    Log.e(TAG, "Recording failed", e)
                    _uiState.update {
                        it.copy(
                            recordingState = RecordingState.ERROR,
                            errorMessage = "Recording failed: ${e.message}"
                        )
                    }
                }
            }
        }
    }

    fun stopRecording() {
        recorder.stop()
        recordingJob?.cancel()
        val samples = recorder.getAllSamples()
        _uiState.update { it.copy(recordingState = RecordingState.PROCESSING, liveLevel = 0f) }
        viewModelScope.launch { analyzeRecording(samples) }
    }

    // ── File upload ────────────────────────────────────────────────────────

    /** Sets the target phrase (if non-blank) then analyzes the given audio file URI. */
    fun analyzeFromFileWithPhrase(uri: Uri, phraseText: String) {
        val trimmed = phraseText.trim()
        if (trimmed.isNotBlank()) {
            _uiState.update {
                it.copy(
                    targetPhrase = TargetPhrase(trimmed, "Custom"),
                    customPhraseText = trimmed
                )
            }
        }
        analyzeFromFile(uri)
    }

    fun analyzeFromFile(uri: Uri) {
        if (_uiState.value.modelStatus != ModelStatus.READY) {
            // Model is still loading — queue the file and process it once READY.
            pendingFileUri = uri
            return
        }
        _uiState.update {
            it.copy(
                recordingState = RecordingState.PROCESSING,
                session = null,
                selectedPhoneme = null,
                errorMessage = null,
                elpacLevel = null
            )
        }
        viewModelScope.launch {
            try {
                val samples = withContext(Dispatchers.IO) { readAudioSamples(uri) }
                analyzeRecording(samples)
            } catch (e: Exception) {
                Log.e(TAG, "File load failed", e)
                _uiState.update {
                    it.copy(
                        recordingState = RecordingState.ERROR,
                        errorMessage = "Could not read audio file: ${e.message}"
                    )
                }
            }
        }
    }

    /**
     * Reads any audio file Android can decode and returns 16 kHz mono 16-bit PCM samples.
     * Handles WAV (PCM, any sample rate, mono or stereo) via direct parsing, and all other
     * formats (AAC/M4A, MP3, OGG, FLAC, …) via MediaExtractor + MediaCodec.
     */
    private fun readAudioSamples(uri: Uri): ShortArray {
        val ctx = getApplication<Application>()

        // Read the file once; peek at the header to decide which path to take.
        val bytes = ctx.contentResolver.openInputStream(uri)?.use { it.readBytes() }
            ?: throw IOException("Could not open file")

        if (bytes.size >= 4 && String(bytes, 0, 4) == "RIFF") {
            return readWavAndConvert(bytes)
        }

        return decodeWithMediaCodec(uri)
    }

    /**
     * Parses a PCM WAV byte array and converts to 16 kHz mono 16-bit samples.
     * Handles stereo (downmixes to mono) and arbitrary sample rates (resamples).
     */
    private fun readWavAndConvert(bytes: ByteArray): ShortArray {
        val buf = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)

        val riff = ByteArray(4).also { buf.get(it) }
        if (String(riff) != "RIFF") throw IOException("Not a WAV file")
        buf.int  // file size
        val wave = ByteArray(4).also { buf.get(it) }
        if (String(wave) != "WAVE") throw IOException("Not a WAV file (missing WAVE marker)")

        var audioFormat   = 0
        var numChannels   = 0
        var sampleRate    = 0
        var bitsPerSample = 0
        var dataSize      = 0
        var foundFmt      = false
        var foundData     = false

        while (buf.remaining() >= 8 && !foundData) {
            val chunkId    = ByteArray(4).also { buf.get(it) }
            val chunkSize  = buf.int
            val chunkStart = buf.position()
            when (String(chunkId)) {
                "fmt " -> {
                    audioFormat   = buf.short.toInt() and 0xFFFF
                    numChannels   = buf.short.toInt() and 0xFFFF
                    sampleRate    = buf.int
                    buf.int   // byte rate
                    buf.short // block align
                    bitsPerSample = buf.short.toInt() and 0xFFFF
                    foundFmt = true
                    buf.position(chunkStart + chunkSize + (chunkSize and 1))
                }
                "data" -> {
                    dataSize  = chunkSize
                    foundData = true
                }
                else -> buf.position(chunkStart + chunkSize + (chunkSize and 1))
            }
        }

        if (!foundFmt)  throw IOException("WAV file is missing fmt chunk")
        if (!foundData) throw IOException("WAV file is missing data chunk")
        if (audioFormat != 1)
            throw IOException("Compressed WAV is not supported (format $audioFormat). Use PCM WAV, M4A, or MP3.")
        if (bitsPerSample != 16)
            throw IOException("Only 16-bit WAV files are supported (file is $bitsPerSample-bit)")

        val totalSamples = dataSize / 2
        val rawSamples   = ShortArray(totalSamples) { buf.short }

        // Downmix stereo (or higher) to mono.
        val monoSamples = if (numChannels > 1) {
            ShortArray(totalSamples / numChannels) { i ->
                var sum = 0
                for (c in 0 until numChannels) sum += rawSamples[i * numChannels + c]
                (sum / numChannels).toShort()
            }
        } else rawSamples

        // Resample to 16 kHz.
        return if (sampleRate != AudioRecorder.SAMPLE_RATE) {
            resampleLinear(monoSamples, sampleRate, AudioRecorder.SAMPLE_RATE)
        } else monoSamples
    }

    /**
     * Uses MediaExtractor + MediaCodec to decode any Android-supported compressed format
     * (AAC/M4A, MP3, OGG, FLAC, …) to raw PCM, then converts to 16 kHz mono 16-bit.
     */
    private fun decodeWithMediaCodec(uri: Uri): ShortArray {
        val ctx       = getApplication<Application>()
        val extractor = MediaExtractor()
        extractor.setDataSource(ctx, uri, null)

        var audioTrackIndex = -1
        var format: MediaFormat? = null
        for (i in 0 until extractor.trackCount) {
            val trackFormat = extractor.getTrackFormat(i)
            val mime = trackFormat.getString(MediaFormat.KEY_MIME) ?: continue
            if (mime.startsWith("audio/")) {
                audioTrackIndex = i
                format = trackFormat
                break
            }
        }
        if (audioTrackIndex < 0 || format == null) {
            extractor.release()
            throw IOException("No audio track found in file")
        }

        extractor.selectTrack(audioTrackIndex)
        val mime          = format.getString(MediaFormat.KEY_MIME)!!
        val sourceSR      = format.getInteger(MediaFormat.KEY_SAMPLE_RATE)
        val channelCount  = format.getInteger(MediaFormat.KEY_CHANNEL_COUNT)

        val codec = MediaCodec.createDecoderByType(mime)
        codec.configure(format, null, null, 0)
        codec.start()

        val pcmOut   = ByteArrayOutputStream()
        val info     = MediaCodec.BufferInfo()
        var inputDone  = false
        var outputDone = false

        while (!outputDone) {
            if (!inputDone) {
                val inIndex = codec.dequeueInputBuffer(10_000L)
                if (inIndex >= 0) {
                    val inBuf     = codec.getInputBuffer(inIndex)!!
                    val sampleSz  = extractor.readSampleData(inBuf, 0)
                    if (sampleSz < 0) {
                        codec.queueInputBuffer(inIndex, 0, 0, 0L, MediaCodec.BUFFER_FLAG_END_OF_STREAM)
                        inputDone = true
                    } else {
                        codec.queueInputBuffer(inIndex, 0, sampleSz, extractor.sampleTime, 0)
                        extractor.advance()
                    }
                }
            }

            when (val outIndex = codec.dequeueOutputBuffer(info, 10_000L)) {
                MediaCodec.INFO_TRY_AGAIN_LATER -> { /* spin */ }
                MediaCodec.INFO_OUTPUT_FORMAT_CHANGED -> { /* ignore */ }
                else -> if (outIndex >= 0) {
                    val outBuf = codec.getOutputBuffer(outIndex)!!
                    val chunk  = ByteArray(info.size)
                    outBuf.get(chunk)
                    pcmOut.write(chunk)
                    codec.releaseOutputBuffer(outIndex, false)
                    if (info.flags and MediaCodec.BUFFER_FLAG_END_OF_STREAM != 0) {
                        outputDone = true
                    }
                }
            }
        }

        codec.stop()
        codec.release()
        extractor.release()

        // Convert decoded bytes → ShortArray (little-endian 16-bit PCM).
        val pcmBytes    = pcmOut.toByteArray()
        val shortBuf    = ByteBuffer.wrap(pcmBytes).order(ByteOrder.LITTLE_ENDIAN)
        val totalShorts = pcmBytes.size / 2
        val allSamples  = ShortArray(totalShorts) { shortBuf.short }

        // Downmix to mono.
        val monoSamples = if (channelCount > 1) {
            ShortArray(totalShorts / channelCount) { i ->
                var sum = 0
                for (c in 0 until channelCount) sum += allSamples[i * channelCount + c]
                (sum / channelCount).toShort()
            }
        } else allSamples

        // Resample to 16 kHz.
        return if (sourceSR != AudioRecorder.SAMPLE_RATE) {
            resampleLinear(monoSamples, sourceSR, AudioRecorder.SAMPLE_RATE)
        } else monoSamples
    }

    /** Linear interpolation resampler. */
    private fun resampleLinear(input: ShortArray, srcRate: Int, dstRate: Int): ShortArray {
        val ratio        = srcRate.toDouble() / dstRate
        val outputLength = (input.size / ratio).toInt()
        return ShortArray(outputLength) { i ->
            val srcPos   = i * ratio
            val srcIndex = srcPos.toInt()
            val frac     = srcPos - srcIndex
            val a        = input.getOrElse(srcIndex) { 0 }.toDouble()
            val b        = input.getOrElse(srcIndex + 1) { 0 }.toDouble()
            (a + frac * (b - a)).toInt().toShort()
        }
    }

    // ── Analysis ───────────────────────────────────────────────────────────

    private suspend fun analyzeRecording(samples: ShortArray) {
        if (samples.size < AudioRecorder.SAMPLE_RATE / 2) {
            _uiState.update {
                it.copy(
                    recordingState = RecordingState.ERROR,
                    errorMessage = "Recording too short. Please speak for at least 0.5 seconds."
                )
            }
            return
        }

        try {
            val targetPhrase = _uiState.value.targetPhrase

            val expectedPhonemes: List<String> = targetPhrase?.let {
                detector.getPhraseExpectedPhonemes(it.text)
            } ?: emptyList()

            val perWordPhonemeCount: List<Int> = targetPhrase?.let { phrase ->
                phrase.text.trim().split("\\s+".toRegex()).map { word ->
                    detector.getWordExpectedPhonemes(word).size
                }
            } ?: emptyList()

            val detectionResult = withContext(Dispatchers.Default) { detector.detect(samples) }
            val rawPhonemes    = detectionResult.phonemes
            val nonSilPhonemes = rawPhonemes.filter { it.phoneme != "∅" }

            val wordTimings = if (targetPhrase != null) {
                enrichWordTimings(detectionResult.wordTimings, targetPhrase.text)
            } else detectionResult.wordTimings

            val annotatedPhonemes = if (expectedPhonemes.isNotEmpty()) {
                detector.alignPhonemes(nonSilPhonemes, expectedPhonemes)
            } else {
                nonSilPhonemes
            }

            val comparison = if (expectedPhonemes.isNotEmpty()) {
                buildComparison(expectedPhonemes, annotatedPhonemes, perWordPhonemeCount)
            } else null

            val score      = detector.computeOverallScore(annotatedPhonemes, comparison, wordTimings)
            val elpacLevel = detector.elpacLevel(score.overallScore)
            val waveform   = recorder.buildWaveform(samples)
            val session    = AnalysisSession(
                sampleRate   = AudioRecorder.SAMPLE_RATE,
                phonemes     = annotatedPhonemes,
                score        = score,
                waveform     = waveform,
                targetPhrase = targetPhrase,
                comparison   = comparison,
                wordTimings  = wordTimings
            )

            _uiState.update {
                it.copy(
                    recordingState = RecordingState.DONE,
                    session        = session,
                    elpacLevel     = elpacLevel
                )
            }
        } catch (e: Exception) {
            Log.e(TAG, "Analysis failed", e)
            _uiState.update {
                it.copy(
                    recordingState = RecordingState.ERROR,
                    errorMessage   = "Analysis failed: ${e.message}"
                )
            }
        }
    }

    /**
     * Builds the [PhonemeComparison] shown to the user. The `accuracyPct` field MUST be
     * computed via [PhonemeDetector.weightedAccuracy] so that the card's accuracy number
     * is the same one that feeds into the top-line ELPAC score.
     */
    private fun buildComparison(
        expected: List<String>,
        actual: List<PhonemeResult>,
        perWordCounts: List<Int> = emptyList()
    ): PhonemeComparison {
        val (_, total, weightedPct) = detector.weightedAccuracy(actual, expected)
        val matched = actual.count { it.isCorrect }
        return PhonemeComparison(
            expectedPhonemes      = expected,
            actualPhonemes        = actual,
            matchedCount          = matched,
            totalExpected         = total,
            accuracyPct           = weightedPct,
            perWordExpectedCounts = perWordCounts
        )
    }

    private fun enrichWordTimings(
        timings: List<WordTiming>,
        phraseText: String
    ): List<WordTiming> {
        if (timings.isEmpty()) return timings
        val words = phraseText.trim().split("\\s+".toRegex())
        return timings.mapIndexed { i, wt ->
            val word = words.getOrElse(i) { wt.word }
            wt.copy(expectedPhonemes = detector.getWordExpectedPhonemes(word))
        }
    }

    // ── UI interaction ─────────────────────────────────────────────────────

    fun selectPhoneme(phoneme: PhonemeResult?) {
        _uiState.update { it.copy(selectedPhoneme = phoneme) }
    }

    fun reset() {
        recorder.stop()
        recordingJob?.cancel()
        liveWaveformBuffer.clear()
        val phrase = _uiState.value.targetPhrase
        val text   = _uiState.value.customPhraseText
        _uiState.update {
            MainUiState(
                targetPhrase = phrase,
                customPhraseText = text,
                modelStatus = ModelStatus.READY
            )
        }
    }

    fun dismissError() {
        _uiState.update { it.copy(errorMessage = null, recordingState = RecordingState.IDLE) }
    }

    override fun onCleared() {
        super.onCleared()
        recorder.release()
        detector.close()
    }
}
