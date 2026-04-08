package com.example.phoneme_trainer

import android.Manifest
import android.app.Application
import android.content.pm.PackageManager
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

    fun analyzeFromFile(uri: Uri) {
        if (_uiState.value.modelStatus != ModelStatus.READY) return
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
                val samples = withContext(Dispatchers.IO) { readWavSamples(uri) }
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
     * Reads a WAV file from a content URI and returns raw 16kHz mono 16-bit PCM samples.
     * Rejects files that are not uncompressed PCM, not mono, or not 16 kHz.
     */
    private fun readWavSamples(uri: Uri): ShortArray {
        val ctx = getApplication<Application>()
        val bytes = ctx.contentResolver.openInputStream(uri)?.use { it.readBytes() }
            ?: throw IOException("Could not open file")
        val buf = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)

        val riff = ByteArray(4).also { buf.get(it) }
        if (String(riff) != "RIFF") throw IOException("Not a WAV file (missing RIFF header)")
        buf.int  // file size
        val wave = ByteArray(4).also { buf.get(it) }
        if (String(wave) != "WAVE") throw IOException("Not a WAV file (missing WAVE marker)")

        var audioFormat  = 0
        var numChannels  = 0
        var sampleRate   = 0
        var bitsPerSample = 0
        var dataSize     = 0
        var foundFmt     = false
        var foundData    = false

        while (buf.remaining() >= 8 && !foundData) {
            val chunkId   = ByteArray(4).also { buf.get(it) }
            val chunkSize = buf.int
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
                    // buffer position is now at the start of PCM data
                }
                else -> buf.position(chunkStart + chunkSize + (chunkSize and 1))
            }
        }

        if (!foundFmt)  throw IOException("WAV file is missing fmt chunk")
        if (!foundData) throw IOException("WAV file is missing data chunk")
        if (audioFormat != 1)
            throw IOException("Only uncompressed PCM WAV files are supported (got format $audioFormat)")
        if (numChannels != 1)
            throw IOException("Only mono audio is supported (file has $numChannels channels)")
        if (sampleRate != AudioRecorder.SAMPLE_RATE)
            throw IOException("Only 16 kHz audio is supported (file is $sampleRate Hz)")
        if (bitsPerSample != 16)
            throw IOException("Only 16-bit audio is supported (file is $bitsPerSample-bit)")

        val numSamples = dataSize / 2
        return ShortArray(numSamples) { buf.short }
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
