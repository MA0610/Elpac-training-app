package com.example.vosk_elpac

import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.ArrowForward
import androidx.compose.material.icons.filled.CheckCircle
import androidx.compose.material.icons.filled.Info
import androidx.compose.material.icons.filled.Warning
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.*
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextDecoration
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.example.vosk_elpac.model.*
import kotlin.math.roundToInt

// ── Quality tier ──────────────────────────────────────────────────────────────

enum class WordQuality { GOOD, PARTIAL, POOR, UNMATCHED }

data class WordFeedback(
    val word: String,
    val quality: WordQuality,
    val phonemes: List<PhonemeResult>,
    val expectedPhonemes: List<String>,
    val avgScore: Float,
    val tip: String?
)

// ── Build per-word feedback ────────────────────────────────────────────────────

fun buildWordFeedback(
    targetPhrase: TargetPhrase,
    phonemes: List<PhonemeResult>,
    comparison: PhonemeComparison?
): List<WordFeedback> {
    val words        = targetPhrase.text.trim().split("\\s+".toRegex())
    val expectedAll  = comparison?.expectedPhonemes ?: emptyList()
    val actualAll    = phonemes.filter { it.phoneme != "∅" }
    if (words.isEmpty() || expectedAll.isEmpty()) return emptyList()

    val totalExpected = expectedAll.size
    val totalWords    = words.size
    var expCursor     = 0
    var actCursor     = 0

    return words.mapIndexed { wordIdx, word ->
        val isLast       = wordIdx == words.lastIndex
        val wordExpCount = if (isLast) {
            (totalExpected - expCursor).coerceAtLeast(1)
        } else {
            ((totalExpected.toFloat() / totalWords) + 0.5f).toInt().coerceAtLeast(1)
        }

        val wordExpPhonemes = expectedAll.drop(expCursor).take(wordExpCount)
        val wordActPhonemes = actualAll.drop(actCursor).take(wordExpCount)
        expCursor += wordExpCount
        actCursor += wordExpCount

        val avgScore = if (wordActPhonemes.isEmpty()) 0f
        else wordActPhonemes.map { it.score }.average().toFloat()

        val quality = when {
            wordActPhonemes.isEmpty()            -> WordQuality.UNMATCHED
            wordActPhonemes.all { it.isCorrect } -> WordQuality.GOOD
            avgScore >= 65f                      -> WordQuality.PARTIAL
            else                                 -> WordQuality.POOR
        }

        WordFeedback(
            word             = word,
            quality          = quality,
            phonemes         = wordActPhonemes,
            expectedPhonemes = wordExpPhonemes,
            avgScore         = avgScore,
            tip              = buildTip(word, wordActPhonemes, wordExpPhonemes, quality)
        )
    }
}

// ── Tips ──────────────────────────────────────────────────────────────────────

private fun buildTip(
    word: String,
    actual: List<PhonemeResult>,
    expected: List<String>,
    quality: WordQuality
): String? {
    if (quality == WordQuality.GOOD) return null
    if (actual.isEmpty()) return "Try to include \"$word\" — it wasn't detected."
    val mismatches = actual.zip(expected).filter { (act, exp) ->
        act.phoneme != exp && act.score < 75f
    }
    if (mismatches.isEmpty()) {
        return if (quality == WordQuality.PARTIAL)
            "\"$word\" is mostly correct — keep practicing for clarity."
        else null
    }
    val (worstAct, worstExp) = mismatches.minByOrNull { it.first.score }!!
    return phonemeTip(worstAct.phoneme, worstExp, word)
}

private fun phonemeTip(actual: String, expected: String, word: String): String {
    val key = "$actual→$expected"
    val specific = mapOf(
        "s→θ"  to "For the \"th\" in \"$word\", place your tongue between your teeth and breathe out.",
        "z→ð"  to "For the voiced \"th\" in \"$word\", touch your tongue to your upper teeth and use your voice.",
        "t→θ"  to "\"th\" in \"$word\" needs your tongue between your teeth — not a \"t\" sound.",
        "d→ð"  to "\"th\" in \"$word\" is made with the tongue touching the upper teeth with voice.",
        "f→θ"  to "For \"th\" in \"$word\", use your tongue between teeth, not lip-teeth like \"f\".",
        "v→ð"  to "For the voiced \"th\" in \"$word\", use tongue-teeth, not lip-teeth.",
        "ɪ→iː" to "Hold the vowel in \"$word\" a bit longer — it's a long \"ee\" sound.",
        "ɛ→æ"  to "The vowel in \"$word\" should be wider — open your mouth more, like in \"cat\".",
        "ʌ→ɑ"  to "The vowel in \"$word\" needs to be further back — like the \"a\" in \"father\".",
        "s→ʃ"  to "In \"$word\", use a \"sh\" sound — round your lips slightly.",
        "ʃ→s"  to "In \"$word\", use a plain \"s\" — don't round your lips.",
        "p→f"  to "In \"$word\", use \"f\" — touch your upper teeth to your lower lip.",
        "b→v"  to "In \"$word\", use \"v\" — touch your upper teeth to your lower lip with voice.",
        "w→v"  to "In \"$word\", use \"w\" — round your lips, don't touch your teeth.",
        "ɹ→l"  to "In \"$word\", curl your tongue back slightly for the English \"r\" sound.",
        "l→ɹ"  to "In \"$word\", keep your tongue tip at the top of your mouth for \"l\".",
    )[key]
    return specific
        ?: "In \"$word\", try /${expected}/ instead of /${actual}/. Listen to a native speaker and mimic the sound."
}

// ── Colors ────────────────────────────────────────────────────────────────────

private fun qualityBg(q: WordQuality): Color = when (q) {
    WordQuality.GOOD      -> Color(0xFF0D2A0D)
    WordQuality.PARTIAL   -> Color(0xFF2A200A)
    WordQuality.POOR      -> Color(0xFF2A0A0A)
    WordQuality.UNMATCHED -> Color(0xFF1A1A2E)
}

private fun qualityBorder(q: WordQuality): Color = when (q) {
    WordQuality.GOOD      -> Color(0xFF4CAF50)
    WordQuality.PARTIAL   -> Color(0xFFFFB300)
    WordQuality.POOR      -> Color(0xFFEF5350)
    WordQuality.UNMATCHED -> Color(0xFF424242)
}

private fun qualityText(q: WordQuality): Color = when (q) {
    WordQuality.GOOD      -> Color(0xFF81C784)
    WordQuality.PARTIAL   -> Color(0xFFFFD54F)
    WordQuality.POOR      -> Color(0xFFEF9A9A)
    WordQuality.UNMATCHED -> Color(0xFF757575)
}

// ── Main composable ───────────────────────────────────────────────────────────

@OptIn(ExperimentalLayoutApi::class)
@Composable
fun TranscriptFeedbackSection(
    targetPhrase: TargetPhrase,
    phonemes: List<PhonemeResult>,
    comparison: PhonemeComparison?,
    modifier: Modifier = Modifier
) {
    val wordFeedbacks = remember(targetPhrase, phonemes, comparison) {
        buildWordFeedback(targetPhrase, phonemes, comparison)
    }

    var selectedWord by remember { mutableStateOf<WordFeedback?>(null) }

    Column(
        modifier = modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(16.dp))
            .background(Color(0xFF12122A))
            .padding(16.dp),
        verticalArrangement = Arrangement.spacedBy(14.dp)
    ) {

        // ── Header ────────────────────────────────────────────────────────
        Text(
            "What you said",
            style      = MaterialTheme.typography.titleSmall,
            color      = Color(0xFF9E9E9E),
            fontWeight = FontWeight.SemiBold
        )

        // ── Highlighted sentence ──────────────────────────────────────────
        // Renders the full target phrase as continuous text.
        // Green = correct, yellow = needs work, red = incorrect.
        val highlightedText = buildAnnotatedString {
            wordFeedbacks.forEachIndexed { idx, wf ->
                withStyle(
                    SpanStyle(
                        color      = qualityText(wf.quality),
                        fontWeight = FontWeight.Medium,
                        fontSize   = 22.sp
                    )
                ) {
                    append(wf.word)
                }
                if (idx < wordFeedbacks.lastIndex) {
                    withStyle(SpanStyle(color = Color(0xFF9E9E9E), fontSize = 22.sp)) {
                        append(" ")
                    }
                }
            }
        }

        Box(
            modifier = Modifier
                .fillMaxWidth()
                .clip(RoundedCornerShape(10.dp))
                .background(Color(0xFF1A1A2E))
                .padding(horizontal = 16.dp, vertical = 14.dp)
        ) {
            Text(text = highlightedText, lineHeight = 32.sp)
        }

        // ── Legend ────────────────────────────────────────────────────────
        Row(horizontalArrangement = Arrangement.spacedBy(16.dp)) {
            LegendDot(Color(0xFF4CAF50), "Good")
            LegendDot(Color(0xFFFFB300), "Needs work")
            LegendDot(Color(0xFFEF5350), "Incorrect")
        }

        // ── Tappable word chips ───────────────────────────────────────────
        Text("Tap a word for details", fontSize = 11.sp, color = Color(0xFF424242))

        FlowRow(
            horizontalArrangement = Arrangement.spacedBy(8.dp),
            verticalArrangement   = Arrangement.spacedBy(8.dp),
            modifier              = Modifier.fillMaxWidth()
        ) {
            wordFeedbacks.forEach { wf ->
                WordChip(
                    feedback   = wf,
                    isSelected = wf == selectedWord,
                    onClick    = { selectedWord = if (selectedWord == wf) null else wf }
                )
            }
        }

        // ── Selected word detail ──────────────────────────────────────────
        selectedWord?.let { wf ->
            WordDetailPanel(feedback = wf)
        }

        // ── Areas to improve ──────────────────────────────────────────────
        val wordsWithTips = wordFeedbacks.filter { it.tip != null }
        if (wordsWithTips.isNotEmpty()) {
            Divider(color = Color(0xFF2A2A4A))
            Text(
                "Areas to improve",
                style      = MaterialTheme.typography.labelMedium,
                color      = Color(0xFF9E9E9E),
                fontWeight = FontWeight.SemiBold
            )
            wordsWithTips.forEach { wf ->
                TipRow(feedback = wf, onClick = { selectedWord = wf })
            }
        } else if (wordFeedbacks.isNotEmpty()) {
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .clip(RoundedCornerShape(8.dp))
                    .background(Color(0xFF0D2A0D))
                    .padding(12.dp),
                horizontalArrangement = Arrangement.spacedBy(8.dp),
                verticalAlignment     = Alignment.CenterVertically
            ) {
                Icon(
                    Icons.Default.CheckCircle,
                    contentDescription = null,
                    tint     = Color(0xFF4CAF50),
                    modifier = Modifier.size(16.dp)
                )
                Text(
                    "Great job! All words pronounced correctly.",
                    color    = Color(0xFF81C784),
                    fontSize = 13.sp
                )
            }
        }
    }
}

// ── Word chip ─────────────────────────────────────────────────────────────────

@Composable
private fun WordChip(
    feedback: WordFeedback,
    isSelected: Boolean,
    onClick: () -> Unit
) {
    Column(
        modifier = Modifier
            .clip(RoundedCornerShape(8.dp))
            .background(qualityBg(feedback.quality))
            .border(
                width = if (isSelected) 2.dp else 1.dp,
                color = if (isSelected) Color.White else qualityBorder(feedback.quality),
                shape = RoundedCornerShape(8.dp)
            )
            .clickable(onClick = onClick)
            .padding(horizontal = 10.dp, vertical = 7.dp),
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        Text(
            text       = feedback.word,
            color      = qualityText(feedback.quality),
            fontSize   = 15.sp,
            fontWeight = FontWeight.Medium
        )
        Text(
            text     = "${feedback.avgScore.roundToInt()}",
            color    = qualityText(feedback.quality).copy(alpha = 0.7f),
            fontSize = 11.sp
        )
    }
}

// ── Word detail panel ─────────────────────────────────────────────────────────

@Composable
private fun WordDetailPanel(feedback: WordFeedback) {
    Column(
        modifier = Modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(12.dp))
            .background(Color(0xFF1A1A3A))
            .padding(14.dp),
        verticalArrangement = Arrangement.spacedBy(10.dp)
    ) {
        Row(
            modifier              = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment     = Alignment.CenterVertically
        ) {
            Text(
                "\"${feedback.word}\"",
                color      = Color.White,
                fontSize   = 18.sp,
                fontWeight = FontWeight.SemiBold
            )
            QualityBadge(feedback.quality, feedback.avgScore)
        }

        if (feedback.phonemes.isNotEmpty() || feedback.expectedPhonemes.isNotEmpty()) {
            Text(
                "Phoneme breakdown",
                color      = Color(0xFF757575),
                fontSize   = 12.sp,
                fontWeight = FontWeight.SemiBold
            )
            Row(
                horizontalArrangement = Arrangement.spacedBy(6.dp),
                modifier              = Modifier.fillMaxWidth()
            ) {
                val maxLen = maxOf(feedback.phonemes.size, feedback.expectedPhonemes.size)
                for (i in 0 until maxLen) {
                    MiniPhonemeCell(
                        actual   = feedback.phonemes.getOrNull(i),
                        expected = feedback.expectedPhonemes.getOrNull(i)
                    )
                }
            }
        }

        if (feedback.tip != null) {
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .clip(RoundedCornerShape(8.dp))
                    .background(Color(0xFF0D1A2A))
                    .padding(10.dp),
                horizontalArrangement = Arrangement.spacedBy(10.dp),
                verticalAlignment     = Alignment.Top
            ) {
                Icon(
                    Icons.Default.Info,
                    contentDescription = null,
                    tint     = Color(0xFF4FC3F7),
                    modifier = Modifier.size(16.dp)
                )
                Text(
                    feedback.tip,
                    color      = Color(0xFFB0BEC5),
                    fontSize   = 13.sp,
                    lineHeight = 19.sp
                )
            }
        } else {
            Row(
                horizontalArrangement = Arrangement.spacedBy(8.dp),
                verticalAlignment     = Alignment.CenterVertically
            ) {
                Icon(
                    Icons.Default.CheckCircle,
                    contentDescription = null,
                    tint     = Color(0xFF4CAF50),
                    modifier = Modifier.size(16.dp)
                )
                Text(
                    "Great pronunciation on this word!",
                    color    = Color(0xFF81C784),
                    fontSize = 13.sp
                )
            }
        }
    }
}

// ── Mini phoneme cell ─────────────────────────────────────────────────────────

@Composable
private fun MiniPhonemeCell(actual: PhonemeResult?, expected: String?) {
    val isCorrect   = actual != null && (actual.isCorrect || actual.phoneme == expected)
    val bgColor     = when {
        actual == null -> Color(0xFF1A1A2E)
        isCorrect      -> Color(0xFF0D2A0D)
        else           -> Color(0xFF2A0A0A)
    }
    val borderColor = when {
        actual == null -> Color(0xFF424242)
        isCorrect      -> Color(0xFF4CAF50)
        else           -> Color(0xFFEF5350)
    }

    Column(
        modifier = Modifier
            .clip(RoundedCornerShape(6.dp))
            .background(bgColor)
            .border(1.dp, borderColor, RoundedCornerShape(6.dp))
            .padding(horizontal = 6.dp, vertical = 4.dp),
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        Text(
            text       = actual?.phoneme ?: "—",
            fontSize   = 14.sp,
            fontFamily = FontFamily.Serif,
            color      = if (isCorrect) Color(0xFF81C784) else Color(0xFFEF9A9A),
            fontWeight = FontWeight.Bold
        )
        if (expected != null && actual?.phoneme != expected) {
            Text(
                text           = expected,
                fontSize       = 10.sp,
                fontFamily     = FontFamily.Serif,
                color          = Color(0xFF757575),
                textDecoration = TextDecoration.None
            )
        }
    }
}

// ── Quality badge ─────────────────────────────────────────────────────────────

@Composable
private fun QualityBadge(quality: WordQuality, score: Float) {
    val label = when (quality) {
        WordQuality.GOOD      -> "Good"
        WordQuality.PARTIAL   -> "Needs work"
        WordQuality.POOR      -> "Incorrect"
        WordQuality.UNMATCHED -> "Not found"
    }
    Row(
        modifier = Modifier
            .clip(RoundedCornerShape(20.dp))
            .background(qualityBg(quality))
            .border(1.dp, qualityBorder(quality), RoundedCornerShape(20.dp))
            .padding(horizontal = 10.dp, vertical = 4.dp),
        horizontalArrangement = Arrangement.spacedBy(6.dp),
        verticalAlignment     = Alignment.CenterVertically
    ) {
        Text(label, color = qualityText(quality), fontSize = 12.sp, fontWeight = FontWeight.SemiBold)
        Text("${score.roundToInt()}", color = qualityText(quality).copy(alpha = 0.8f), fontSize = 11.sp)
    }
}

// ── Tip row ───────────────────────────────────────────────────────────────────

@Composable
private fun TipRow(feedback: WordFeedback, onClick: () -> Unit) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(10.dp))
            .background(Color(0xFF1A1A2E))
            .clickable(onClick = onClick)
            .padding(horizontal = 12.dp, vertical = 10.dp),
        verticalAlignment     = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.spacedBy(10.dp)
    ) {
        Icon(
            if (feedback.quality == WordQuality.POOR) Icons.Default.Warning else Icons.Default.Info,
            contentDescription = null,
            tint     = if (feedback.quality == WordQuality.POOR) Color(0xFFEF5350) else Color(0xFFFFB300),
            modifier = Modifier.size(16.dp)
        )
        Column(modifier = Modifier.weight(1f)) {
            Text(
                "\"${feedback.word}\"",
                color      = Color.White,
                fontSize   = 13.sp,
                fontWeight = FontWeight.Medium
            )
            Text(
                feedback.tip ?: "",
                color      = Color(0xFF9E9E9E),
                fontSize   = 12.sp,
                lineHeight = 17.sp,
                maxLines   = 2
            )
        }
        Icon(
            Icons.Default.ArrowForward,
            contentDescription = null,
            tint     = Color(0xFF424242),
            modifier = Modifier.size(14.dp)
        )
    }
}

// ── Legend dot ────────────────────────────────────────────────────────────────

@Composable
private fun LegendDot(color: Color, label: String) {
    Row(
        verticalAlignment     = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.spacedBy(5.dp)
    ) {
        Box(
            modifier = Modifier
                .size(8.dp)
                .clip(RoundedCornerShape(4.dp))
                .background(color)
        )
        Text(label, fontSize = 11.sp, color = Color(0xFF757575))
    }
}