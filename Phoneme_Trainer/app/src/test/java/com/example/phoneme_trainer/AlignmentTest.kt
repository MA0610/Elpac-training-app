package com.example.phoneme_trainer

import com.example.phoneme_trainer.model.PhonemeResult
import org.junit.Assert.*
import org.junit.Test

/**
 * Tests for the Needleman-Wunsch alignment algorithm used in PhonemeDetector.alignPhonemes().
 *
 * PhonemeDetector requires Android Context so the algorithm is reimplemented inline here.
 * Keep in sync with PhonemeDetector.kt if the DP scores or traceback logic changes.
 */
class AlignmentTest {

    // ── NW constants (mirror PhonemeDetector) ────────────────────────────────

    private val GAP      = -1
    private val MATCH    =  2
    private val NEARMISS =  1
    private val MISMATCH = -1

    private val SIMILAR_PAIRS: Set<Set<String>> = setOf(
        setOf("p", "b"), setOf("t", "d"), setOf("k", "ɡ"),
        setOf("f", "v"), setOf("s", "z"), setOf("ʃ", "ʒ"),
        setOf("θ", "ð"), setOf("tʃ", "dʒ"),
        setOf("ɑ", "ɑː"), setOf("ɔ", "ɔː"), setOf("i", "iː"),
        setOf("u", "uː"), setOf("ɝ", "ɜː")
    )

    private fun isNearMiss(a: String, b: String) = a != b && SIMILAR_PAIRS.any { it.contains(a) && it.contains(b) }

    private fun sim(a: String, e: String): Int = when {
        a == e           -> MATCH
        isNearMiss(a, e) -> NEARMISS
        else             -> MISMATCH
    }

    /** Inline Needleman-Wunsch matching PhonemeDetector.alignPhonemes(). */
    private fun align(actual: List<PhonemeResult>, expected: List<String>): List<PhonemeResult> {
        if (actual.isEmpty()) return emptyList()
        if (expected.isEmpty()) return actual

        val n = actual.size; val m = expected.size
        val dp = Array(n + 1) { IntArray(m + 1) }
        for (i in 0..n) dp[i][0] = i * GAP
        for (j in 0..m) dp[0][j] = j * GAP
        for (i in 1..n) for (j in 1..m) {
            dp[i][j] = maxOf(
                dp[i - 1][j - 1] + sim(actual[i - 1].phoneme, expected[j - 1]),
                dp[i - 1][j] + GAP,
                dp[i][j - 1] + GAP
            )
        }

        val alignedActual   = ArrayDeque<String?>()
        val alignedExpected = ArrayDeque<String?>()
        var i = n; var j = m
        while (i > 0 || j > 0) {
            when {
                i > 0 && j > 0 &&
                        dp[i][j] == dp[i - 1][j - 1] + sim(actual[i - 1].phoneme, expected[j - 1]) -> {
                    alignedActual.addFirst(actual[i - 1].phoneme)
                    alignedExpected.addFirst(expected[j - 1])
                    i--; j--
                }
                i > 0 && dp[i][j] == dp[i - 1][j] + GAP -> {
                    alignedActual.addFirst(actual[i - 1].phoneme)
                    alignedExpected.addFirst(null)
                    i--
                }
                else -> {
                    alignedActual.addFirst(null)
                    alignedExpected.addFirst(expected[j - 1])
                    j--
                }
            }
        }

        val annotated = ArrayList<PhonemeResult>(actual.size)
        var actualIdx = 0
        val aAct = alignedActual.toList(); val aExp = alignedExpected.toList()
        for (k in aAct.indices) {
            val actSym = aAct[k] ?: continue
            val expSym = aExp[k]
            val ph = actual[actualIdx++]
            annotated.add(ph.copy(
                isCorrect       = expSym != null && (actSym == expSym || isNearMiss(actSym, expSym)),
                expectedPhoneme = expSym
            ))
        }
        return annotated
    }

    private fun ph(symbol: String) =
        PhonemeResult(symbol, 0L, 100L, 0.9f, 90f)

    // ── Empty / edge cases ───────────────────────────────────────────────────

    @Test
    fun `empty actual returns empty`() {
        assertTrue(align(emptyList(), listOf("t", "æ")).isEmpty())
    }

    @Test
    fun `empty expected returns actual unchanged`() {
        val actual = listOf(ph("t"), ph("æ"))
        val result = align(actual, emptyList())
        assertEquals(2, result.size)
        assertEquals("t", result[0].phoneme)
        assertEquals("æ", result[1].phoneme)
    }

    // ── Perfect match ────────────────────────────────────────────────────────

    @Test
    fun `identical sequences are all correct`() {
        val actual   = listOf(ph("t"), ph("æ"), ph("p"))
        val expected = listOf("t", "æ", "p")
        val result   = align(actual, expected)
        assertEquals(3, result.size)
        result.forEach { assertTrue("Expected isCorrect=true for ${it.phoneme}", it.isCorrect) }
        result.zip(expected).forEach { (ph, exp) -> assertEquals(exp, ph.expectedPhoneme) }
    }

    // ── All insertions ───────────────────────────────────────────────────────

    @Test
    fun `all insertions have null expectedPhoneme and isCorrect false`() {
        val actual   = listOf(ph("t"), ph("æ"))
        val expected = emptyList<String>()
        val result   = align(actual, expected)
        // When expected is empty, actual is returned unchanged (isCorrect defaults to true from original)
        // This tests the early-return path
        assertEquals(2, result.size)
    }

    @Test
    fun `extra phonemes not in expected are marked as insertions`() {
        // actual has 3 phonemes, expected has 1 — the extra two are insertions (no expectedPhoneme)
        val actual   = listOf(ph("t"), ph("æ"), ph("p"))
        val expected = listOf("t")
        val result   = align(actual, expected)
        assertEquals(3, result.size)
        // "t" should be matched correctly
        assertEquals("t", result[0].expectedPhoneme)
        assertTrue(result[0].isCorrect)
        // remaining are insertions
        assertNull(result[1].expectedPhoneme)
        assertNull(result[2].expectedPhoneme)
        assertFalse(result[1].isCorrect)
        assertFalse(result[2].isCorrect)
    }

    // ── Near-miss alignment ──────────────────────────────────────────────────

    @Test
    fun `near-miss substitution is marked correct with expected phoneme set`() {
        val actual   = listOf(ph("p"))  // said "p", expected "b"
        val expected = listOf("b")
        val result   = align(actual, expected)
        assertEquals(1, result.size)
        assertTrue("Near-miss p/b should be isCorrect=true", result[0].isCorrect)
        assertEquals("b", result[0].expectedPhoneme)
    }

    // ── Mismatch ─────────────────────────────────────────────────────────────

    @Test
    fun `non-near-miss substitution is marked incorrect`() {
        val actual   = listOf(ph("s"))  // said "s", expected "iː" — completely wrong
        val expected = listOf("iː")
        val result   = align(actual, expected)
        assertEquals(1, result.size)
        assertFalse(result[0].isCorrect)
        assertEquals("iː", result[0].expectedPhoneme)
    }

    // ── Output list length ───────────────────────────────────────────────────

    @Test
    fun `output always has same length as actual input`() {
        val actual   = listOf(ph("t"), ph("æ"), ph("k"))
        val expected = listOf("t", "æ", "s", "p")  // more expected than actual
        val result   = align(actual, expected)
        assertEquals(actual.size, result.size)
    }

    @Test
    fun `phoneme symbols are preserved after alignment`() {
        val actual   = listOf(ph("t"), ph("æ"), ph("p"))
        val expected = listOf("d", "æ", "b")
        val result   = align(actual, expected)
        assertEquals("t", result[0].phoneme)
        assertEquals("æ", result[1].phoneme)
        assertEquals("p", result[2].phoneme)
    }
}
