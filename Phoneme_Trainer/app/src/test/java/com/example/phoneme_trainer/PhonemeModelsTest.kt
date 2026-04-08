package com.example.phoneme_trainer

import com.example.phoneme_trainer.model.PhonemeInventory
import com.example.phoneme_trainer.model.PhonemeResult
import com.example.phoneme_trainer.model.ElpacPhrases
import org.junit.Assert.*
import org.junit.Test

class PhonemeModelsTest {

    // ── PhonemeResult ────────────────────────────────────────────────────────

    @Test
    fun `durationMs is endTimeMs minus startTimeMs`() {
        val ph = PhonemeResult("æ", startTimeMs = 100L, endTimeMs = 250L, confidence = 0.9f, score = 90f)
        assertEquals(150L, ph.durationMs)
    }

    @Test
    fun `durationMs is zero when start equals end`() {
        val ph = PhonemeResult("t", startTimeMs = 500L, endTimeMs = 500L, confidence = 0.5f, score = 50f)
        assertEquals(0L, ph.durationMs)
    }

    @Test
    fun `isCorrect defaults to true`() {
        val ph = PhonemeResult("p", startTimeMs = 0L, endTimeMs = 80L, confidence = 0.8f, score = 80f)
        assertTrue(ph.isCorrect)
    }

    @Test
    fun `expectedPhoneme defaults to null`() {
        val ph = PhonemeResult("p", startTimeMs = 0L, endTimeMs = 80L, confidence = 0.8f, score = 80f)
        assertNull(ph.expectedPhoneme)
    }

    @Test
    fun `copy preserves all fields and allows override`() {
        val ph = PhonemeResult("p", 0L, 80L, 0.8f, 80f)
        val annotated = ph.copy(isCorrect = false, expectedPhoneme = "b")
        assertEquals("p", annotated.phoneme)
        assertFalse(annotated.isCorrect)
        assertEquals("b", annotated.expectedPhoneme)
    }

    // ── PhonemeInventory ─────────────────────────────────────────────────────

    @Test
    fun `ARPABET_TO_IPA contains all standard ARPABET vowel codes`() {
        val vowels = listOf("AA", "AE", "AH", "AO", "AW", "AY", "EH", "ER", "EY", "IH", "IY", "OW", "OY", "UH", "UW")
        vowels.forEach { code ->
            assertNotNull("Missing ARPABET code: $code", PhonemeInventory.ARPABET_TO_IPA[code])
        }
    }

    @Test
    fun `ARPABET_TO_IPA contains all standard ARPABET consonant codes`() {
        val consonants = listOf("B", "CH", "D", "DH", "F", "G", "HH", "JH", "K", "L", "M", "N", "NG", "P", "R", "S", "SH", "T", "TH", "V", "W", "Y", "Z", "ZH")
        consonants.forEach { code ->
            assertNotNull("Missing ARPABET code: $code", PhonemeInventory.ARPABET_TO_IPA[code])
        }
    }

    @Test
    fun `ARPABET_TO_IPA maps key vowels to expected long-vowel IPA forms`() {
        assertEquals("ɑː", PhonemeInventory.ARPABET_TO_IPA["AA"])
        assertEquals("iː", PhonemeInventory.ARPABET_TO_IPA["IY"])
        assertEquals("uː", PhonemeInventory.ARPABET_TO_IPA["UW"])
        assertEquals("ɔː", PhonemeInventory.ARPABET_TO_IPA["AO"])
        assertEquals("ɜː", PhonemeInventory.ARPABET_TO_IPA["ER"])
    }

    @Test
    fun `ARPABET_TO_IPA maps key consonants correctly`() {
        assertEquals("t",  PhonemeInventory.ARPABET_TO_IPA["T"])
        assertEquals("ʃ",  PhonemeInventory.ARPABET_TO_IPA["SH"])
        assertEquals("tʃ", PhonemeInventory.ARPABET_TO_IPA["CH"])
        assertEquals("dʒ", PhonemeInventory.ARPABET_TO_IPA["JH"])
        assertEquals("ŋ",  PhonemeInventory.ARPABET_TO_IPA["NG"])
        assertEquals("θ",  PhonemeInventory.ARPABET_TO_IPA["TH"])
        assertEquals("ð",  PhonemeInventory.ARPABET_TO_IPA["DH"])
    }

    // ── ElpacPhrases ─────────────────────────────────────────────────────────

    @Test
    fun `ElpacPhrases ALL has at least 22 entries`() {
        assertTrue("Expected ≥22 ELPAC phrases, got ${ElpacPhrases.ALL.size}", ElpacPhrases.ALL.size >= 22)
    }

    @Test
    fun `ElpacPhrases CATEGORIES contains all unique categories from ALL`() {
        val expected = ElpacPhrases.ALL.map { it.category }.distinct()
        assertEquals(expected, ElpacPhrases.CATEGORIES)
    }

    @Test
    fun `every ELPAC phrase has non-blank text and category`() {
        ElpacPhrases.ALL.forEach { phrase ->
            assertTrue("Blank text in phrase: $phrase", phrase.text.isNotBlank())
            assertTrue("Blank category in phrase: $phrase", phrase.category.isNotBlank())
        }
    }
}
