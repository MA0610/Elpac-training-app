package com.example.phoneme_trainer.model

/**
 * Represents a single detected phoneme with timing and confidence.
 */
data class PhonemeResult(
    val phoneme: String,             // IPA symbol e.g. "æ", "t", "ʃ"
    val startTimeMs: Long,
    val endTimeMs: Long,
    val confidence: Float,           // 0.0 – 1.0
    val score: Float,                // 0–100
    val isCorrect: Boolean = true,   // did it match the expected phoneme?
    val expectedPhoneme: String? = null  // what was expected here
) {
    val durationMs: Long get() = endTimeMs - startTimeMs
}

/**
 * Scoring breakdown for an entire utterance.
 */
data class PronunciationScore(
    val overallScore: Float,
    val accuracyScore: Float,
    val fluencyScore: Float,
    val completenessScore: Float,
    val phonemeScores: List<PhonemeResult>
)

/**
 * A target phrase the student is asked to read aloud.
 */
data class TargetPhrase(
    val text: String,
    val category: String = "Custom"
)

/**
 * Side-by-side comparison of expected vs actual phonemes.
 */
data class PhonemeComparison(
    val expectedPhonemes: List<String>,    // IPA from dictionary
    val actualPhonemes: List<PhonemeResult>,
    val matchedCount: Int,
    val totalExpected: Int,
    val accuracyPct: Float,
    val perWordExpectedCounts: List<Int> = emptyList()  // phoneme count per word from CMU dict
)

/**
 * Word-level timing from Vosk ASR, enriched with CMU dict expected phonemes.
 * Used to map WavLM-detected phonemes back to individual words for word highlighting.
 */
data class WordTiming(
    val word: String,
    val startMs: Long,
    val endMs: Long,
    val expectedPhonemes: List<String> = emptyList()  // IPA phonemes expected for this word
)

/**
 * Combined result from the hybrid WavLM + Vosk detection pipeline.
 * WavLM produces the phoneme sequence; Vosk supplies word-boundary timings only.
 */
data class DetectionResult(
    val phonemes: List<PhonemeResult>,
    val wordTimings: List<WordTiming>
)

enum class RecordingState {
    IDLE, RECORDING, PROCESSING, DONE, ERROR
}

data class WaveformPoint(
    val timeMs: Long,
    val amplitude: Float
)

/**
 * The raw PCM buffer is intentionally NOT held in UI state — a 30 s recording is ~1 MB
 * and would be retained across every Compose recomposition. The waveform field holds a
 * 300-point downsampled representation that is sufficient for visualisation.
 */
data class AnalysisSession(
    val sampleRate: Int,
    val phonemes: List<PhonemeResult>,
    val score: PronunciationScore,
    val waveform: List<WaveformPoint>,
    val targetPhrase: TargetPhrase? = null,
    val comparison: PhonemeComparison? = null,
    val wordTimings: List<WordTiming> = emptyList()
)

data class PhonemeInfo(
    val symbol: String,
    val exampleWord: String,
    val category: PhonemeCategory
)

enum class PhonemeCategory {
    VOWEL, CONSONANT_STOP, CONSONANT_FRICATIVE,
    CONSONANT_NASAL, CONSONANT_LIQUID, CONSONANT_AFFRICATE, SILENCE
}

object PhonemeInventory {

    // ARPABET → IPA using long-vowel forms so that CMU dict expected phonemes align
    // with the WavLM model's output. The WavLM checkpoint emits short vowel tokens
    // plus a standalone "ː" length mark, which the CTC decoder (WavLMPhonemeDetector)
    // merges into long-vowel forms before alignment.
    val ARPABET_TO_IPA = mapOf(
        "AA" to "ɑː", "AE" to "æ",  "AH" to "ʌ",  "AO" to "ɔː",
        "AW" to "aʊ", "AY" to "aɪ", "B"  to "b",   "CH" to "tʃ",
        "D"  to "d",  "DH" to "ð",  "EH" to "ɛ",   "ER" to "ɜː",
        "EY" to "eɪ", "F"  to "f",  "G"  to "ɡ",   "HH" to "h",
        "IH" to "ɪ",  "IY" to "iː", "JH" to "dʒ",  "K"  to "k",
        "L"  to "l",  "M"  to "m",  "N"  to "n",   "NG" to "ŋ",
        "OW" to "oʊ", "OY" to "ɔɪ", "P"  to "p",   "R"  to "ɹ",
        "S"  to "s",  "SH" to "ʃ",  "T"  to "t",   "TH" to "θ",
        "UH" to "ʊ",  "UW" to "uː", "V"  to "v",   "W"  to "w",
        "Y"  to "j",  "Z"  to "z",  "ZH" to "ʒ",   "SIL" to "∅"
    )

    // Reverse map: eSpeak IPA → ARPABET (handles short/long variants eSpeak may produce)
    val ESPEAK_TO_ARPABET = mapOf(
        "æ"  to "AE", "ɑː" to "AA", "ɑ"  to "AA", "ʌ"  to "AH",
        "ɔː" to "AO", "ɔ"  to "AO", "aʊ" to "AW", "aɪ" to "AY",
        "ɛ"  to "EH", "ɜː" to "ER", "ɝ"  to "ER", "eɪ" to "EY",
        "ɪ"  to "IH", "iː" to "IY", "i"  to "IY", "oʊ" to "OW",
        "ɔɪ" to "OY", "ʊ"  to "UH", "uː" to "UW", "u"  to "UW",
        "b"  to "B",  "tʃ" to "CH", "d"  to "D",  "ð"  to "DH",
        "f"  to "F",  "ɡ"  to "G",  "h"  to "HH", "dʒ" to "JH",
        "k"  to "K",  "l"  to "L",  "m"  to "M",  "n"  to "N",
        "ŋ"  to "NG", "p"  to "P",  "ɹ"  to "R",  "r"  to "R",
        "s"  to "S",  "ʃ"  to "SH", "t"  to "T",  "θ"  to "TH",
        "v"  to "V",  "w"  to "W",  "j"  to "Y",  "z"  to "Z",  "ʒ" to "ZH"
    )

    val PHONEME_INFO = mapOf(
        // Short vowel forms (may appear in fast/reduced speech from eSpeak)
        "æ"  to PhonemeInfo("æ",  "cat",       PhonemeCategory.VOWEL),
        "ɑ"  to PhonemeInfo("ɑ",  "father",    PhonemeCategory.VOWEL),
        "ʌ"  to PhonemeInfo("ʌ",  "cup",       PhonemeCategory.VOWEL),
        "ɔ"  to PhonemeInfo("ɔ",  "thought",   PhonemeCategory.VOWEL),
        "aʊ" to PhonemeInfo("aʊ", "cow",       PhonemeCategory.VOWEL),
        "aɪ" to PhonemeInfo("aɪ", "buy",       PhonemeCategory.VOWEL),
        "ɛ"  to PhonemeInfo("ɛ",  "bed",       PhonemeCategory.VOWEL),
        "ɝ"  to PhonemeInfo("ɝ",  "bird",      PhonemeCategory.VOWEL),
        "eɪ" to PhonemeInfo("eɪ", "day",       PhonemeCategory.VOWEL),
        "ɪ"  to PhonemeInfo("ɪ",  "sit",       PhonemeCategory.VOWEL),
        "iː" to PhonemeInfo("iː", "see",       PhonemeCategory.VOWEL),
        "oʊ" to PhonemeInfo("oʊ", "go",        PhonemeCategory.VOWEL),
        "ɔɪ" to PhonemeInfo("ɔɪ", "boy",       PhonemeCategory.VOWEL),
        "ʊ"  to PhonemeInfo("ʊ",  "book",      PhonemeCategory.VOWEL),
        "uː" to PhonemeInfo("uː", "food",      PhonemeCategory.VOWEL),
        // Long-vowel forms (reconstructed by the CTC decoder via length-mark merging)
        "ɑː" to PhonemeInfo("ɑː", "father",    PhonemeCategory.VOWEL),
        "ɔː" to PhonemeInfo("ɔː", "thought",   PhonemeCategory.VOWEL),
        "ɜː" to PhonemeInfo("ɜː", "bird",      PhonemeCategory.VOWEL),
        "b"  to PhonemeInfo("b",  "bat",       PhonemeCategory.CONSONANT_STOP),
        "d"  to PhonemeInfo("d",  "dog",       PhonemeCategory.CONSONANT_STOP),
        "ɡ"  to PhonemeInfo("ɡ",  "go",        PhonemeCategory.CONSONANT_STOP),
        "k"  to PhonemeInfo("k",  "cat",       PhonemeCategory.CONSONANT_STOP),
        "p"  to PhonemeInfo("p",  "pat",       PhonemeCategory.CONSONANT_STOP),
        "t"  to PhonemeInfo("t",  "top",       PhonemeCategory.CONSONANT_STOP),
        "f"  to PhonemeInfo("f",  "fish",      PhonemeCategory.CONSONANT_FRICATIVE),
        "v"  to PhonemeInfo("v",  "van",       PhonemeCategory.CONSONANT_FRICATIVE),
        "s"  to PhonemeInfo("s",  "sun",       PhonemeCategory.CONSONANT_FRICATIVE),
        "z"  to PhonemeInfo("z",  "zoo",       PhonemeCategory.CONSONANT_FRICATIVE),
        "ʃ"  to PhonemeInfo("ʃ",  "shoe",      PhonemeCategory.CONSONANT_FRICATIVE),
        "ʒ"  to PhonemeInfo("ʒ",  "measure",   PhonemeCategory.CONSONANT_FRICATIVE),
        "θ"  to PhonemeInfo("θ",  "think",     PhonemeCategory.CONSONANT_FRICATIVE),
        "ð"  to PhonemeInfo("ð",  "the",       PhonemeCategory.CONSONANT_FRICATIVE),
        "h"  to PhonemeInfo("h",  "hat",       PhonemeCategory.CONSONANT_FRICATIVE),
        "tʃ" to PhonemeInfo("tʃ", "chat",      PhonemeCategory.CONSONANT_AFFRICATE),
        "dʒ" to PhonemeInfo("dʒ", "judge",     PhonemeCategory.CONSONANT_AFFRICATE),
        "m"  to PhonemeInfo("m",  "man",       PhonemeCategory.CONSONANT_NASAL),
        "n"  to PhonemeInfo("n",  "net",       PhonemeCategory.CONSONANT_NASAL),
        "ŋ"  to PhonemeInfo("ŋ",  "sing",      PhonemeCategory.CONSONANT_NASAL),
        "l"  to PhonemeInfo("l",  "lip",       PhonemeCategory.CONSONANT_LIQUID),
        "ɹ"  to PhonemeInfo("ɹ",  "red",       PhonemeCategory.CONSONANT_LIQUID),
        "w"  to PhonemeInfo("w",  "wet",       PhonemeCategory.CONSONANT_LIQUID),
        "j"  to PhonemeInfo("j",  "yes",       PhonemeCategory.CONSONANT_LIQUID),
        "∅"  to PhonemeInfo("∅",  "(silence)", PhonemeCategory.SILENCE)
    )

    fun categoryOf(ipa: String): PhonemeCategory? = PHONEME_INFO[ipa]?.category
    fun exampleWordOf(ipa: String): String = PHONEME_INFO[ipa]?.exampleWord ?: ipa
}

/**
 * Preset ELPAC phrases grouped by category.
 */
object ElpacPhrases {
    val ALL = listOf(
        TargetPhrase("this is a test",                 "Basics"),
        TargetPhrase("my name is",                     "Basics"),
        TargetPhrase("how are you",                    "Basics"),
        TargetPhrase("i am happy",                     "Basics"),
        TargetPhrase("good morning",                   "Basics"),
        TargetPhrase("thank you very much",            "Basics"),
        TargetPhrase("the three thin things",          "TH Sounds"),
        TargetPhrase("think about that",               "TH Sounds"),
        TargetPhrase("this and that",                  "TH Sounds"),
        TargetPhrase("i think therefore i am",         "TH Sounds"),
        TargetPhrase("she sells seashells",            "SH / CH"),
        TargetPhrase("the ship reached the shore",     "SH / CH"),
        TargetPhrase("check the chair",                "SH / CH"),
        TargetPhrase("the red rabbit ran",             "R Sounds"),
        TargetPhrase("around the world",               "R Sounds"),
        TargetPhrase("the cat sat on the mat",         "Vowels"),
        TargetPhrase("i see a big green tree",         "Vowels"),
        TargetPhrase("go home and eat food",           "Vowels"),
        TargetPhrase("i go to school every day",       "Sentences"),
        TargetPhrase("she plays with her friends",     "Sentences"),
        TargetPhrase("we eat lunch at noon",           "Sentences"),
        TargetPhrase("the dog runs in the park",       "Sentences"),
        TargetPhrase("my teacher helps me learn",      "Sentences"),
    )

    val CATEGORIES: List<String> = ALL.map { it.category }.distinct()
}