package com.example.vosk_elpac.model

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
    val accuracyPct: Float
)

enum class RecordingState {
    IDLE, RECORDING, PROCESSING, DONE, ERROR
}

data class WaveformPoint(
    val timeMs: Long,
    val amplitude: Float
)

data class AnalysisSession(
    val audioSamples: ShortArray,
    val sampleRate: Int,
    val phonemes: List<PhonemeResult>,
    val score: PronunciationScore,
    val waveform: List<WaveformPoint>,
    val targetPhrase: TargetPhrase? = null,
    val comparison: PhonemeComparison? = null
) {
    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (other !is AnalysisSession) return false
        return sampleRate == other.sampleRate &&
                audioSamples.contentEquals(other.audioSamples) &&
                phonemes == other.phonemes &&
                score == other.score &&
                waveform == other.waveform
    }
    override fun hashCode(): Int {
        var result = audioSamples.contentHashCode()
        result = 31 * result + sampleRate
        result = 31 * result + phonemes.hashCode()
        result = 31 * result + score.hashCode()
        result = 31 * result + waveform.hashCode()
        return result
    }
}

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

    val ARPABET_TO_IPA = mapOf(
        "AA" to "ɑ",  "AE" to "æ",  "AH" to "ʌ",  "AO" to "ɔ",
        "AW" to "aʊ", "AY" to "aɪ", "B"  to "b",   "CH" to "tʃ",
        "D"  to "d",  "DH" to "ð",  "EH" to "ɛ",   "ER" to "ɝ",
        "EY" to "eɪ", "F"  to "f",  "G"  to "ɡ",   "HH" to "h",
        "IH" to "ɪ",  "IY" to "iː", "JH" to "dʒ",  "K"  to "k",
        "L"  to "l",  "M"  to "m",  "N"  to "n",   "NG" to "ŋ",
        "OW" to "oʊ", "OY" to "ɔɪ", "P"  to "p",   "R"  to "ɹ",
        "S"  to "s",  "SH" to "ʃ",  "T"  to "t",   "TH" to "θ",
        "UH" to "ʊ",  "UW" to "uː", "V"  to "v",   "W"  to "w",
        "Y"  to "j",  "Z"  to "z",  "ZH" to "ʒ",   "SIL" to "∅"
    )

    val PHONEME_INFO = mapOf(
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