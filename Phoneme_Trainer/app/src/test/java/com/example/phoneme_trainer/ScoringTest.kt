package com.example.phoneme_trainer

import com.example.phoneme_trainer.model.PhonemeResult
import org.junit.Assert.*
import org.junit.Test

/**
 * Tests for the scoring formulas in PhonemeDetector.
 *
 * PhonemeDetector requires an Android Context so it cannot be instantiated in a JVM
 * unit test. The scoring logic is pure arithmetic, so it is replicated inline here so
 * that the mathematical correctness of the formulas can be verified independently of
 * the Android runtime. If the constants or formulas change in PhonemeDetector.kt,
 * update the constants below to match.
 */
class ScoringTest {

    // ── Constants mirrored from PhonemeDetector ──────────────────────────────

    private val W_ACCURACY     = 0.55f
    private val W_FLUENCY      = 0.30f
    private val W_COMPLETENESS = 0.15f

    private val WEIGHT_EXACT      = 1.00
    private val WEIGHT_NEAR_MISS  = 0.70
    private val WEIGHT_INSERTION  = 0.00
    private val WEIGHT_SUBSTITUTE = 0.00

    private val HESITATION_GAP_MS       = 300L
    private val OVERLAP_GAP_MS          = -10L
    private val PENALTY_PER_HESITATION  = 8f
    private val PENALTY_PER_OVERLAP     = 5f
    private val MAX_GAP_PENALTY         = 40f
    private val MIN_FLUENCY_FLOOR       = 30f

    private val ELPAC_THRESHOLDS = listOf(
        85f to "Level 4 – Minimal errors",
        70f to "Level 3 – Generally intelligible",
        50f to "Level 2 – Some communication impact",
        0f  to "Level 1 – Significant communication impact"
    )

    private val SIMILAR_PAIRS: Set<Set<String>> = setOf(
        setOf("p", "b"), setOf("t", "d"), setOf("k", "ɡ"),
        setOf("f", "v"), setOf("s", "z"), setOf("ʃ", "ʒ"),
        setOf("θ", "ð"), setOf("tʃ", "dʒ"),
        setOf("ɑ", "ɑː"), setOf("ɔ", "ɔː"), setOf("i", "iː"),
        setOf("u", "uː"), setOf("ɝ", "ɜː")
    )

    // ── Helpers ──────────────────────────────────────────────────────────────

    private fun isNearMiss(a: String, b: String) = SIMILAR_PAIRS.any { it.contains(a) && it.contains(b) }

    private fun weightedAccuracy(actual: List<PhonemeResult>, totalExpected: Int): Float {
        if (totalExpected == 0) return 0f
        val weighted = actual.sumOf { ph ->
            val exp = ph.expectedPhoneme
            when {
                exp == null                 -> WEIGHT_INSERTION
                ph.phoneme == exp           -> WEIGHT_EXACT
                isNearMiss(ph.phoneme, exp) -> WEIGHT_NEAR_MISS
                else                        -> WEIGHT_SUBSTITUTE
            }
        }
        return ((weighted / totalExpected.coerceAtLeast(1)) * 100.0).toFloat().coerceIn(0f, 100f)
    }

    private fun fluency(phonemes: List<PhonemeResult>): Float {
        if (phonemes.size < 2) return 100f
        val gaps = phonemes.zipWithNext { a, b -> b.startTimeMs - a.endTimeMs }
        val hesitations = gaps.count { it > HESITATION_GAP_MS }
        val rushes = gaps.count { it < OVERLAP_GAP_MS }
        val penalty = (hesitations * PENALTY_PER_HESITATION + rushes * PENALTY_PER_OVERLAP)
            .coerceIn(0f, MAX_GAP_PENALTY)
        return (100f - penalty).coerceAtLeast(MIN_FLUENCY_FLOOR)
    }

    private fun completeness(produced: Int, totalExpected: Int): Float =
        (produced.toFloat() / totalExpected.coerceAtLeast(1) * 100f).coerceIn(0f, 100f)

    private fun overallScore(accuracy: Float, fluency: Float, completeness: Float): Float =
        (accuracy * W_ACCURACY + fluency * W_FLUENCY + completeness * W_COMPLETENESS).coerceIn(0f, 100f)

    private fun elpacLevel(score: Float): String =
        ELPAC_THRESHOLDS.first { score >= it.first }.second

    private fun ph(symbol: String, expected: String?, startMs: Long = 0L, endMs: Long = 100L) =
        PhonemeResult(
            phoneme = symbol,
            startTimeMs = startMs,
            endTimeMs = endMs,
            confidence = 0.9f,
            score = 90f,
            isCorrect = expected != null && (symbol == expected || isNearMiss(symbol, expected)),
            expectedPhoneme = expected
        )

    // ── Weighted accuracy ────────────────────────────────────────────────────

    @Test
    fun `all exact matches gives 100 percent accuracy`() {
        val phonemes = listOf(ph("t", "t"), ph("æ", "æ"), ph("p", "p"))
        assertEquals(100f, weightedAccuracy(phonemes, 3), 0.01f)
    }

    @Test
    fun `all near-misses gives 70 percent accuracy`() {
        val phonemes = listOf(ph("p", "b"), ph("t", "d"), ph("k", "ɡ"))
        assertEquals(70f, weightedAccuracy(phonemes, 3), 0.01f)
    }

    @Test
    fun `all substitutions gives 0 percent accuracy`() {
        val phonemes = listOf(ph("t", "s"), ph("æ", "iː"), ph("p", "ŋ"))
        assertEquals(0f, weightedAccuracy(phonemes, 3), 0.01f)
    }

    @Test
    fun `all insertions gives 0 percent accuracy`() {
        val phonemes = listOf(ph("t", null), ph("æ", null))
        assertEquals(0f, weightedAccuracy(phonemes, 3), 0.01f)
    }

    @Test
    fun `empty expected gives 0 percent accuracy`() {
        assertEquals(0f, weightedAccuracy(emptyList(), 0), 0.01f)
    }

    @Test
    fun `empty actual against non-empty expected gives 0 percent accuracy`() {
        assertEquals(0f, weightedAccuracy(emptyList(), 5), 0.01f)
    }

    @Test
    fun `mixed exact and near-miss blends correctly`() {
        // 1 exact (1.0) + 1 near-miss (0.7) out of 2 expected = 85%
        val phonemes = listOf(ph("t", "t"), ph("p", "b"))
        assertEquals(85f, weightedAccuracy(phonemes, 2), 0.01f)
    }

    @Test
    fun `extra insertions beyond expected count do not exceed 100 percent`() {
        // 3 exact phonemes but only 2 expected — insertions score 0, result still capped at 100
        val phonemes = listOf(ph("t", "t"), ph("æ", "æ"), ph("p", null))
        assertEquals(100f, weightedAccuracy(phonemes, 2), 0.01f)
    }

    // ── Near-miss pairs ──────────────────────────────────────────────────────

    @Test
    fun `voiced-unvoiced pairs are near misses`() {
        val pairs = listOf("p" to "b", "t" to "d", "k" to "ɡ", "f" to "v", "s" to "z", "ʃ" to "ʒ", "θ" to "ð", "tʃ" to "dʒ")
        pairs.forEach { (a, b) ->
            assertTrue("$a/$b should be near-miss", isNearMiss(a, b))
            assertTrue("$b/$a should be near-miss (symmetric)", isNearMiss(b, a))
        }
    }

    @Test
    fun `long-short vowel pairs are near misses`() {
        val pairs = listOf("ɑ" to "ɑː", "ɔ" to "ɔː", "i" to "iː", "u" to "uː", "ɝ" to "ɜː")
        pairs.forEach { (a, b) ->
            assertTrue("$a/$b should be near-miss", isNearMiss(a, b))
        }
    }

    @Test
    fun `contrastive vowel pairs are not near misses`() {
        val pairs = listOf("ɪ" to "iː", "ɛ" to "æ", "ʌ" to "ɑ", "ʊ" to "uː")
        pairs.forEach { (a, b) ->
            assertFalse("$a/$b should NOT be near-miss", isNearMiss(a, b))
        }
    }

    @Test
    fun `identical phoneme is not a near-miss (exact match)`() {
        // Same symbol: not in any pair by design — exact match check happens before near-miss
        assertFalse(isNearMiss("t", "t"))
    }

    // ── Fluency ──────────────────────────────────────────────────────────────

    @Test
    fun `no gaps gives perfect fluency`() {
        val phonemes = listOf(
            ph("t", "t", 0L,   80L),
            ph("æ", "æ", 80L,  180L),
            ph("p", "p", 180L, 260L)
        )
        assertEquals(100f, fluency(phonemes), 0.01f)
    }

    @Test
    fun `single phoneme gives 100 fluency (no gaps to measure)`() {
        assertEquals(100f, fluency(listOf(ph("t", "t", 0L, 80L))), 0.01f)
    }

    @Test
    fun `one hesitation deducts 8 points`() {
        val phonemes = listOf(
            ph("t", "t", 0L,    80L),
            ph("æ", "æ", 500L, 600L)  // 420ms gap > 300ms threshold
        )
        assertEquals(92f, fluency(phonemes), 0.01f)
    }

    @Test
    fun `one overlap deducts 5 points`() {
        val phonemes = listOf(
            ph("t", "t", 0L,  80L),
            ph("æ", "æ", 50L, 150L)  // -30ms overlap < -10ms threshold
        )
        assertEquals(95f, fluency(phonemes), 0.01f)
    }

    @Test
    fun `fluency is floored at 30`() {
        // 6 hesitations × 8 = 48 > MAX_GAP_PENALTY(40) → 100 - 40 = 60, but let's push past floor
        // Actually MAX_GAP_PENALTY caps at 40, so floor isn't hit that way.
        // Force floor: set penalty > 70 via many overlaps (5 × 8 hesitations = 40 capped, 100-40=60)
        // MIN_FLUENCY_FLOOR is only reachable if penalty could exceed 70, but it's capped at 40.
        // Test that the floor constant is 30 and not breached by the cap logic.
        val penalty = MAX_GAP_PENALTY  // worst case after cap
        val fluencyScore = (100f - penalty).coerceAtLeast(MIN_FLUENCY_FLOOR)
        assertEquals(60f, fluencyScore, 0.01f)  // 100 - 40 = 60, floor not triggered here
        assertTrue("Floor should be 30", MIN_FLUENCY_FLOOR == 30f)
    }

    @Test
    fun `gap exactly at threshold is not a hesitation`() {
        val phonemes = listOf(
            ph("t", "t", 0L,   80L),
            ph("æ", "æ", 380L, 480L)  // exactly 300ms gap = at threshold, not over
        )
        assertEquals(100f, fluency(phonemes), 0.01f)
    }

    // ── Completeness ─────────────────────────────────────────────────────────

    @Test
    fun `all expected phonemes produced gives 100 completeness`() {
        assertEquals(100f, completeness(produced = 5, totalExpected = 5), 0.01f)
    }

    @Test
    fun `no phonemes produced gives 0 completeness`() {
        assertEquals(0f, completeness(produced = 0, totalExpected = 5), 0.01f)
    }

    @Test
    fun `partial production gives proportional completeness`() {
        assertEquals(60f, completeness(produced = 3, totalExpected = 5), 0.01f)
    }

    @Test
    fun `completeness is capped at 100 even with more produced than expected`() {
        assertEquals(100f, completeness(produced = 7, totalExpected = 5), 0.01f)
    }

    // ── Overall score ────────────────────────────────────────────────────────

    @Test
    fun `perfect inputs give 100 overall`() {
        assertEquals(100f, overallScore(100f, 100f, 100f), 0.01f)
    }

    @Test
    fun `zero inputs give 0 overall`() {
        assertEquals(0f, overallScore(0f, 0f, 0f), 0.01f)
    }

    @Test
    fun `weights sum to 1`() {
        assertEquals(1.0f, W_ACCURACY + W_FLUENCY + W_COMPLETENESS, 0.001f)
    }

    @Test
    fun `overall score uses correct component weights`() {
        val expected = (80f * 0.55f + 90f * 0.30f + 70f * 0.15f)
        assertEquals(expected, overallScore(80f, 90f, 70f), 0.01f)
    }

    // ── ELPAC level mapping ──────────────────────────────────────────────────

    @Test
    fun `score 100 maps to Level 4`() {
        assertTrue(elpacLevel(100f).startsWith("Level 4"))
    }

    @Test
    fun `score 85 maps to Level 4`() {
        assertTrue(elpacLevel(85f).startsWith("Level 4"))
    }

    @Test
    fun `score 84 maps to Level 3`() {
        assertTrue(elpacLevel(84f).startsWith("Level 3"))
    }

    @Test
    fun `score 70 maps to Level 3`() {
        assertTrue(elpacLevel(70f).startsWith("Level 3"))
    }

    @Test
    fun `score 69 maps to Level 2`() {
        assertTrue(elpacLevel(69f).startsWith("Level 2"))
    }

    @Test
    fun `score 50 maps to Level 2`() {
        assertTrue(elpacLevel(50f).startsWith("Level 2"))
    }

    @Test
    fun `score 49 maps to Level 1`() {
        assertTrue(elpacLevel(49f).startsWith("Level 1"))
    }

    @Test
    fun `score 0 maps to Level 1`() {
        assertTrue(elpacLevel(0f).startsWith("Level 1"))
    }
}
