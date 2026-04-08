package com.example.phoneme_trainer

import android.Manifest
import android.app.Application
import android.content.pm.PackageManager
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
        _uiState.update { it.copy(recordingState = RecordingState.PROCESSING, liveLevel = 0f) }
        viewModelScope.launch { analyzeRecording() }
    }

    // ── Analysis ───────────────────────────────────────────────────────────

    private suspend fun analyzeRecording() {
        val samples = recorder.getAllSamples()

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
                    detector.getWordExpectedPhonemes(word).size.coerceAtLeast(1)
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

            val score      = detector.computeOverallScore(annotatedPhonemes, comparison)
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
