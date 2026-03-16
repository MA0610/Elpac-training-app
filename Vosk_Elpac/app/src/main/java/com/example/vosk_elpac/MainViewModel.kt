package com.example.vosk_elpac

import android.Manifest
import android.app.Application
import android.content.pm.PackageManager
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.example.vosk_elpac.audio.AudioRecorder
import com.example.vosk_elpac.ml.PhonemeDetector
import com.example.vosk_elpac.model.*
import kotlinx.coroutines.Job
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch

data class MainUiState(
    val recordingState: RecordingState = RecordingState.IDLE,
    val liveLevel: Float = 0f,
    val liveWaveform: List<WaveformPoint> = emptyList(),
    val session: AnalysisSession? = null,
    val selectedPhoneme: PhonemeResult? = null,
    val errorMessage: String? = null,
    val recordingDurationMs: Long = 0L,
    val elpacLevel: String? = null,          // e.g. "Level 3 – Generally intelligible"
    val targetPhrase: TargetPhrase? = null,
    val customPhraseText: String = "",
    val showPhraseSelector: Boolean = false
)

class MainViewModel(application: Application) : AndroidViewModel(application) {

    private val recorder = AudioRecorder()
    private val detector = PhonemeDetector(application)
    private val _uiState = MutableStateFlow(MainUiState())
    val uiState: StateFlow<MainUiState> = _uiState.asStateFlow()

    private var recordingJob: Job? = null
    private var startTimeMs = 0L
    private val liveWaveformBuffer = mutableListOf<WaveformPoint>()
    private var sampleCount = 0L

    // ── Permission ─────────────────────────────────────────────────────────

    fun onPermissionDenied() {
        _uiState.update {
            it.copy(errorMessage = "Microphone permission is required to analyze pronunciation.")
        }
    }

    // ── Phrase selection ───────────────────────────────────────────────────

    fun showPhraseSelector()  { _uiState.update { it.copy(showPhraseSelector = true) } }
    fun hidePhraseSelector()  { _uiState.update { it.copy(showPhraseSelector = false) } }

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

        liveWaveformBuffer.clear()
        sampleCount = 0L
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
            try {
                recorder.recordingFlow(getApplication()).collect { chunk ->
                    val rms       = recorder.chunkRmsLevel(chunk)
                    val timeMs    = System.currentTimeMillis() - startTimeMs
                    val amplitude = chunk.maxOrNull()?.toFloat()?.div(Short.MAX_VALUE) ?: 0f
                    liveWaveformBuffer.add(WaveformPoint(timeMs, amplitude))
                    sampleCount += chunk.size
                    _uiState.update {
                        it.copy(
                            liveLevel = rms,
                            liveWaveform = liveWaveformBuffer.toList(),
                            recordingDurationMs = timeMs
                        )
                    }
                }
            } catch (e: kotlinx.coroutines.CancellationException) {
                throw e
            } catch (e: Exception) {
                _uiState.update {
                    it.copy(
                        recordingState = RecordingState.ERROR,
                        errorMessage = "Recording failed: ${e.message}"
                    )
                }
            }
        }
    }

    fun stopRecording() {
        recorder.stop()
        recordingJob?.cancel()
        recordingJob = null
        _uiState.update { it.copy(recordingState = RecordingState.PROCESSING, liveLevel = 0f) }
        viewModelScope.launch { analyzeRecording() }
    }

    // ── Analysis ───────────────────────────────────────────────────────────

    private suspend fun analyzeRecording() {
        val samples = recorder.getAllSamples()

        android.util.Log.d("PhonemeDEBUG",
            "Samples: ${samples.size}, max=${samples.maxOrNull()}, " +
                    "nonzero=${samples.count { it != 0.toShort() }}")

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

            // Expected phoneme sequence from CMU dict
            val expectedPhonemes: List<String> = targetPhrase?.let {
                detector.getPhraseExpectedPhonemes(it.text)
            } ?: emptyList()

            // Raw Vosk recognition
            val rawPhonemes    = detector.detect(samples)
            val nonSilPhonemes = rawPhonemes.filter { it.phoneme != "∅" }

            android.util.Log.d("PhonemeDEBUG", "=== TARGET: ${targetPhrase?.text} ===")
            android.util.Log.d("PhonemeDEBUG",
                "Expected (${expectedPhonemes.size}): $expectedPhonemes")
            android.util.Log.d("PhonemeDEBUG",
                "Actual   (${nonSilPhonemes.size}): ${nonSilPhonemes.map { it.phoneme }}")

            // ── Needleman-Wunsch alignment (replaces greedy annotateWithExpected) ──
            val annotatedPhonemes = if (expectedPhonemes.isNotEmpty()) {
                detector.alignPhonemes(nonSilPhonemes, expectedPhonemes)
            } else {
                nonSilPhonemes
            }

            // Build comparison
            val comparison = if (expectedPhonemes.isNotEmpty()) {
                buildComparison(expectedPhonemes, annotatedPhonemes)
            } else null

            android.util.Log.d("PhonemeDEBUG",
                "Matched: ${comparison?.matchedCount} / ${comparison?.totalExpected}, " +
                        "accuracy=${comparison?.accuracyPct}")

            val score = detector.computeOverallScore(annotatedPhonemes, comparison)

            // ELPAC level derived from overall score
            val elpacLevel = detector.elpacLevel(score.overallScore)

            val waveform = recorder.buildWaveform(samples)
            val session  = AnalysisSession(
                audioSamples = samples,
                sampleRate   = AudioRecorder.SAMPLE_RATE,
                phonemes     = annotatedPhonemes,
                score        = score,
                waveform     = waveform,
                targetPhrase = targetPhrase,
                comparison   = comparison
            )

            _uiState.update {
                it.copy(
                    recordingState = RecordingState.DONE,
                    session        = session,
                    elpacLevel     = elpacLevel
                )
            }
        } catch (e: Exception) {
            _uiState.update {
                it.copy(
                    recordingState = RecordingState.ERROR,
                    errorMessage   = "Analysis failed: ${e.message}"
                )
            }
        }
    }

    private fun buildComparison(
        expected: List<String>,
        actual: List<PhonemeResult>
    ): PhonemeComparison {
        val matched  = actual.count { it.isCorrect }
        val accuracy = if (expected.isEmpty()) 0f
        else (matched.toFloat() / expected.size * 100f).coerceIn(0f, 100f)
        return PhonemeComparison(
            expectedPhonemes = expected,
            actualPhonemes   = actual,
            matchedCount     = matched,
            totalExpected    = expected.size,
            accuracyPct      = accuracy
        )
    }

    // ── UI interaction ─────────────────────────────────────────────────────

    fun selectPhoneme(phoneme: PhonemeResult?) {
        _uiState.update { it.copy(selectedPhoneme = phoneme) }
    }

    fun reset() {
        recorder.stop()
        recordingJob?.cancel()
        recordingJob = null
        liveWaveformBuffer.clear()
        val phrase = _uiState.value.targetPhrase
        val text   = _uiState.value.customPhraseText
        _uiState.update { MainUiState(targetPhrase = phrase, customPhraseText = text) }
    }

    fun dismissError() {
        _uiState.update { it.copy(errorMessage = null, recordingState = RecordingState.IDLE) }
    }

    override fun onCleared() {
        super.onCleared()
        recorder.stop()
        detector.close()
    }
}