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
    // Target phrase state
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

    // ─── Permission ───────────────────────────────────────────────────────────

    fun onPermissionDenied() {
        _uiState.update {
            it.copy(errorMessage = "Microphone permission is required to analyze pronunciation.")
        }
    }

    // ─── Phrase selection ─────────────────────────────────────────────────────

    fun showPhraseSelector() {
        _uiState.update { it.copy(showPhraseSelector = true) }
    }

    fun hidePhraseSelector() {
        _uiState.update { it.copy(showPhraseSelector = false) }
    }

    fun selectPresetPhrase(phrase: TargetPhrase) {
        _uiState.update {
            it.copy(
                targetPhrase = phrase,
                customPhraseText = phrase.text,
                showPhraseSelector = false,
                session = null,
                selectedPhoneme = null
            )
        }
    }

    fun setCustomPhrase(text: String) {
        _uiState.update { it.copy(customPhraseText = text) }
    }

    fun confirmCustomPhrase() {
        val text = _uiState.value.customPhraseText.trim()
        if (text.isBlank()) return
        _uiState.update {
            it.copy(
                targetPhrase = TargetPhrase(text, "Custom"),
                showPhraseSelector = false,
                session = null,
                selectedPhoneme = null
            )
        }
    }

    fun clearPhrase() {
        _uiState.update {
            it.copy(
                targetPhrase = null,
                customPhraseText = "",
                session = null,
                selectedPhoneme = null
            )
        }
    }

    // ─── Recording control ────────────────────────────────────────────────────

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
                recordingDurationMs = 0L
            )
        }

        recordingJob = viewModelScope.launch {
            try {
                recorder.recordingFlow(getApplication()).collect { chunk ->
                    val rms = recorder.chunkRmsLevel(chunk)
                    val timeMs = System.currentTimeMillis() - startTimeMs
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

    // ─── Analysis ─────────────────────────────────────────────────────────────

    private suspend fun analyzeRecording() {
        val samples = recorder.getAllSamples()

        // DEBUG — remove after fixing
        android.util.Log.d("PhonemeDEBUG", "Samples: ${samples.size}, max=${samples.maxOrNull()}, nonzero=${samples.count { it != 0.toShort() }}")

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

            // Get expected phonemes from the target phrase (if set)
            val expectedPhonemes: List<String> = targetPhrase?.let {
                detector.getPhraseExpectedPhonemes(it.text)
            } ?: emptyList()

            // Run Vosk recognition
            val rawPhonemes = detector.detect(samples)

            // DEBUG — remove after fixing
            android.util.Log.d("PhonemeDEBUG", "=== TARGET PHRASE: ${targetPhrase?.text} ===")
            android.util.Log.d("PhonemeDEBUG", "Expected phonemes (${expectedPhonemes.size}): $expectedPhonemes")
            android.util.Log.d("PhonemeDEBUG", "Raw actual phonemes (${rawPhonemes.size}): ${rawPhonemes.map { it.phoneme }}")

            // FIX 1: Strip silence phonemes before alignment — silences burn
            // through expected slots and cause everything to mismatch
            val nonSilentPhonemes = rawPhonemes.filter { it.phoneme != "∅" }
            android.util.Log.d("PhonemeDEBUG", "After silence strip (${nonSilentPhonemes.size}): ${nonSilentPhonemes.map { it.phoneme }}")

            // If we have a target, annotate each actual phoneme with correctness
            val annotatedPhonemes = if (expectedPhonemes.isNotEmpty()) {
                annotateWithExpected(nonSilentPhonemes, expectedPhonemes)
            } else {
                nonSilentPhonemes
            }

            // Build comparison if target phrase is set
            val comparison = if (expectedPhonemes.isNotEmpty()) {
                buildComparison(expectedPhonemes, annotatedPhonemes)
            } else null

            android.util.Log.d("PhonemeDEBUG", "Matched: ${comparison?.matchedCount} / ${comparison?.totalExpected}, accuracy=${comparison?.accuracyPct}")

            val score = detector.computeOverallScore(annotatedPhonemes, comparison)
            val waveform = recorder.buildWaveform(samples)

            val session = AnalysisSession(
                audioSamples = samples,
                sampleRate = AudioRecorder.SAMPLE_RATE,
                phonemes = annotatedPhonemes,
                score = score,
                waveform = waveform,
                targetPhrase = targetPhrase,
                comparison = comparison
            )

            _uiState.update {
                it.copy(recordingState = RecordingState.DONE, session = session)
            }
        } catch (e: Exception) {
            _uiState.update {
                it.copy(
                    recordingState = RecordingState.ERROR,
                    errorMessage = "Analysis failed: ${e.message}"
                )
            }
        }
    }

    /**
     * Aligns actual phonemes against the expected sequence using a simple
     * greedy alignment and marks each as correct/incorrect.
     *
     * NOTE: silences must be stripped from [actual] before calling this.
     */
    private fun annotateWithExpected(
        actual: List<PhonemeResult>,
        expected: List<String>
    ): List<PhonemeResult> {
        if (actual.isEmpty() || expected.isEmpty()) return actual

        val annotated = mutableListOf<PhonemeResult>()
        var expIdx = 0

        for (act in actual) {
            if (expIdx < expected.size) {
                val exp = expected[expIdx]
                val isCorrect = phonemesMatch(act.phoneme, exp)
                annotated.add(
                    act.copy(
                        isCorrect = isCorrect,
                        expectedPhoneme = exp,
                        score = if (isCorrect) act.score else (act.score * 0.4f)
                    )
                )
                expIdx++
            } else {
                // Extra phonemes beyond expected — mark as inserted
                annotated.add(act.copy(isCorrect = false, expectedPhoneme = null))
            }
        }
        return annotated
    }

    /**
     * Two phonemes "match" if identical or acoustically close (voicing pairs,
     * reduced vowels, etc.).
     */
    private fun phonemesMatch(actual: String, expected: String): Boolean {
        if (actual == expected) return true
        val similar = setOf(
            setOf("ɪ", "iː"), setOf("ʌ", "ɑ"), setOf("ɛ", "æ"),
            setOf("ʊ", "uː"), setOf("ɔ", "oʊ"), setOf("ð", "θ"),
            setOf("s", "z"),  setOf("f", "v"),   setOf("p", "b"),
            setOf("t", "d"),  setOf("k", "ɡ"),   setOf("ʃ", "ʒ")
        )
        return similar.any { it.contains(actual) && it.contains(expected) }
    }

    private fun buildComparison(
        expected: List<String>,
        actual: List<PhonemeResult>
    ): PhonemeComparison {
        val matched = actual.count { it.isCorrect }
        val accuracy = if (expected.isEmpty()) 0f
        else (matched.toFloat() / expected.size * 100f).coerceIn(0f, 100f)
        return PhonemeComparison(
            expectedPhonemes = expected,
            actualPhonemes = actual,
            matchedCount = matched,
            totalExpected = expected.size,
            accuracyPct = accuracy
        )
    }

    // ─── UI interaction ───────────────────────────────────────────────────────

    fun selectPhoneme(phoneme: PhonemeResult?) {
        _uiState.update { it.copy(selectedPhoneme = phoneme) }
    }

    fun reset() {
        recorder.stop()
        recordingJob?.cancel()
        recordingJob = null
        liveWaveformBuffer.clear()
        // Keep the target phrase so student can try again immediately
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