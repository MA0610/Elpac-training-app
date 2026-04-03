package com.example.phoneme_trainer

import androidx.compose.animation.core.*
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.gestures.detectTapGestures
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Rect
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.*
import androidx.compose.ui.graphics.drawscope.DrawScope
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.text.*
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.example.phoneme_trainer.model.PhonemeCategory
import com.example.phoneme_trainer.model.PhonemeInventory
import com.example.phoneme_trainer.model.PhonemeResult
import com.example.phoneme_trainer.model.WaveformPoint

// ─── Color scheme per phoneme category ───────────────────────────────────────

private val categoryColors = mapOf(
    PhonemeCategory.VOWEL to Color(0xFF4FC3F7),
    PhonemeCategory.CONSONANT_STOP to Color(0xFFEF9A9A),
    PhonemeCategory.CONSONANT_FRICATIVE to Color(0xFFA5D6A7),
    PhonemeCategory.CONSONANT_NASAL to Color(0xFFCE93D8),
    PhonemeCategory.CONSONANT_LIQUID to Color(0xFFFFCC02),
    PhonemeCategory.CONSONANT_AFFRICATE to Color(0xFFFFAB91),
    PhonemeCategory.SILENCE to Color(0xFF616161)
)

private fun phonemeColor(ipa: String): Color {
    val info = PhonemeInventory.PHONEME_INFO[ipa]
    return categoryColors[info?.category] ?: Color(0xFF90A4AE)
}

private fun scoreColor(score: Float): Color = when {
    score >= 80f -> Color(0xFF66BB6A)
    score >= 60f -> Color(0xFFFFCA28)
    score >= 40f -> Color(0xFFFFA726)
    else -> Color(0xFFEF5350)
}

// ─── Waveform display ─────────────────────────────────────────────────────────

@Composable
fun WaveformCanvas(
    waveformPoints: List<WaveformPoint>,
    modifier: Modifier = Modifier,
    isLive: Boolean = false,
    color: Color = Color(0xFF4FC3F7)
) {
    val animatedAlpha by rememberInfiniteTransition(label = "wave").animateFloat(
        initialValue = 0.6f, targetValue = 1f,
        animationSpec = infiniteRepeatable(tween(800), RepeatMode.Reverse),
        label = "alpha"
    )

    Canvas(modifier = modifier.fillMaxSize()) {
        if (waveformPoints.isEmpty()) {
            drawLine(color.copy(alpha = 0.3f), Offset(0f, size.height / 2), Offset(size.width, size.height / 2), 1.5f)
            return@Canvas
        }

        val midY = size.height / 2f
        val alpha = if (isLive) animatedAlpha else 1f
        val path = Path()
        val points = waveformPoints

        points.forEachIndexed { i, pt ->
            val x = i.toFloat() / (points.size - 1).coerceAtLeast(1) * size.width
            val y = midY - (pt.amplitude.coerceIn(-1f, 1f) * midY * 0.85f)
            if (i == 0) path.moveTo(x, y) else path.lineTo(x, y)
        }

        // Gradient fill under waveform
        val fillPath = Path().apply {
            addPath(path)
            lineTo(size.width, midY)
            lineTo(0f, midY)
            close()
        }
        drawPath(fillPath, Brush.verticalGradient(
            listOf(color.copy(alpha = alpha * 0.3f), Color.Transparent)
        ))
        drawPath(path, color.copy(alpha = alpha), style = Stroke(2.dp.toPx(), cap = StrokeCap.Round))
    }
}

// ─── Phoneme timeline ─────────────────────────────────────────────────────────

@Composable
fun PhonemeTimeline(
    phonemes: List<PhonemeResult>,
    totalDurationMs: Long,
    selectedPhoneme: PhonemeResult?,
    onPhonemeClick: (PhonemeResult) -> Unit,
    modifier: Modifier = Modifier
) {
    val textMeasurer = rememberTextMeasurer()

    Box(modifier = modifier) {
        Canvas(
            modifier = Modifier
                .fillMaxSize()
                .pointerInput(phonemes) {
                    detectTapGestures { offset ->
                        if (phonemes.isEmpty() || totalDurationMs == 0L) return@detectTapGestures
                        val timeRatio = offset.x / size.width
                        val tappedTimeMs = (timeRatio * totalDurationMs).toLong()
                        val hit = phonemes.firstOrNull { it.startTimeMs <= tappedTimeMs && tappedTimeMs <= it.endTimeMs }
                        hit?.let { onPhonemeClick(it) }
                    }
                }
        ) {
            if (phonemes.isEmpty() || totalDurationMs == 0L) return@Canvas
            drawPhonemeTimeline(phonemes, totalDurationMs, selectedPhoneme, textMeasurer)
        }
    }
}

private fun DrawScope.drawPhonemeTimeline(
    phonemes: List<PhonemeResult>,
    totalDurationMs: Long,
    selected: PhonemeResult?,
    textMeasurer: TextMeasurer
) {
    val trackHeight = size.height * 0.6f
    val trackTop = size.height * 0.1f
    val cornerRadius = 6.dp.toPx()

    // Background track
    drawRoundRect(
        Color(0xFF1A1A2E),
        topLeft = Offset(0f, trackTop),
        size = Size(size.width, trackHeight),
        cornerRadius = androidx.compose.ui.geometry.CornerRadius(cornerRadius)
    )

    phonemes.forEach { ph ->
        val x1 = ph.startTimeMs.toFloat() / totalDurationMs * size.width
        val x2 = ph.endTimeMs.toFloat() / totalDurationMs * size.width
        val w = (x2 - x1).coerceAtLeast(4.dp.toPx())
        val isSelected = ph == selected
        val baseColor = phonemeColor(ph.phoneme)
        val alpha = if (ph.phoneme == "∅") 0.2f else 1f

        // Phoneme block
        drawRoundRect(
            baseColor.copy(alpha = alpha),
            topLeft = Offset(x1, trackTop + 2.dp.toPx()),
            size = Size(w - 2.dp.toPx(), trackHeight - 4.dp.toPx()),
            cornerRadius = androidx.compose.ui.geometry.CornerRadius(4.dp.toPx())
        )

        // Selection highlight
        if (isSelected) {
            drawRoundRect(
                Color.White,
                topLeft = Offset(x1, trackTop + 2.dp.toPx()),
                size = Size(w - 2.dp.toPx(), trackHeight - 4.dp.toPx()),
                cornerRadius = androidx.compose.ui.geometry.CornerRadius(4.dp.toPx()),
                style = Stroke(2.dp.toPx())
            )
        }

        // Score dot at bottom
        val scoreDotY = trackTop + trackHeight + 8.dp.toPx()
        val dotCx = x1 + w / 2
        drawCircle(scoreColor(ph.score), 4.dp.toPx(), Offset(dotCx, scoreDotY))

        // IPA label (only if block wide enough)
        if (w > 20.dp.toPx() && ph.phoneme != "∅") {
            val measuredText = textMeasurer.measure(
                AnnotatedString(ph.phoneme),
                style = TextStyle(
                    fontSize = 11.sp,
                    fontFamily = FontFamily.Serif,
                    color = Color.Black,
                    fontWeight = FontWeight.Bold
                )
            )
            val labelX = (x1 + (w - measuredText.size.width) / 2).coerceAtLeast(0f)
            val labelY = trackTop + (trackHeight - measuredText.size.height) / 2
            drawText(measuredText, topLeft = Offset(labelX, labelY))
        }
    }
}

// ─── Score ring ───────────────────────────────────────────────────────────────

@Composable
fun ScoreRing(
    score: Float,
    label: String,
    modifier: Modifier = Modifier
) {
    val animScore by animateFloatAsState(
        targetValue = score,
        animationSpec = tween(1200, easing = EaseOutCubic),
        label = "score"
    )
    val color = scoreColor(score)

    Box(modifier = modifier, contentAlignment = Alignment.Center) {
        Canvas(modifier = Modifier.fillMaxSize()) {
            val stroke = 8.dp.toPx()
            val inset = stroke / 2
            val rect = Rect(Offset(inset, inset), Size(size.width - stroke, size.height - stroke))

            // Background arc
            drawArc(Color(0xFF2A2A3A), -90f, 360f, false, rect.topLeft, rect.size, style = Stroke(stroke))

            // Score arc
            drawArc(color, -90f, animScore / 100f * 360f, false, rect.topLeft, rect.size,
                style = Stroke(stroke, cap = StrokeCap.Round))
        }
        Column(horizontalAlignment = Alignment.CenterHorizontally) {
            Text(
                text = "${animScore.toInt()}",
                style = MaterialTheme.typography.headlineMedium,
                fontWeight = FontWeight.Bold,
                color = color
            )
            Text(text = label, style = MaterialTheme.typography.labelSmall, color = Color(0xFF9E9E9E))
        }
    }
}

// ─── Phoneme detail card ──────────────────────────────────────────────────────

@Composable
fun PhonemeDetailCard(phoneme: PhonemeResult, modifier: Modifier = Modifier) {
    val info = PhonemeInventory.PHONEME_INFO[phoneme.phoneme]
    val color = phonemeColor(phoneme.phoneme)

    Row(
        modifier = modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(12.dp))
            .background(Color(0xFF1E1E2E))
            .padding(16.dp),
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.spacedBy(16.dp)
    ) {
        Box(
            modifier = Modifier
                .size(56.dp)
                .clip(RoundedCornerShape(8.dp))
                .background(color.copy(alpha = 0.2f)),
            contentAlignment = Alignment.Center
        ) {
            Text(
                text = phoneme.phoneme,
                fontSize = 28.sp,
                fontFamily = FontFamily.Serif,
                color = color,
                fontWeight = FontWeight.Bold
            )
        }

        Column(modifier = Modifier.weight(1f)) {
            Row(horizontalArrangement = Arrangement.spacedBy(8.dp), verticalAlignment = Alignment.CenterVertically) {
                Text("/${phoneme.phoneme}/", style = MaterialTheme.typography.titleMedium,
                    fontFamily = FontFamily.Serif, color = Color.White)
                if (info != null) {
                    Text(
                        "\"${info.exampleWord}\"",
                        style = MaterialTheme.typography.bodyMedium,
                        color = Color(0xFF9E9E9E)
                    )
                }
            }
            info?.let {
                Text(
                    it.category.name.replace('_', ' ').lowercase().replaceFirstChar(Char::uppercase),
                    style = MaterialTheme.typography.labelSmall,
                    color = color.copy(alpha = 0.8f)
                )
            }
            Text(
                "${phoneme.startTimeMs}ms – ${phoneme.endTimeMs}ms  (${phoneme.durationMs}ms)",
                style = MaterialTheme.typography.labelSmall,
                color = Color(0xFF616161)
            )
        }

        Column(horizontalAlignment = Alignment.CenterHorizontally) {
            Text(
                "${phoneme.score.toInt()}",
                style = MaterialTheme.typography.titleLarge,
                color = scoreColor(phoneme.score),
                fontWeight = FontWeight.Bold
            )
            Text("score", style = MaterialTheme.typography.labelSmall, color = Color(0xFF616161))
        }
    }
}

// ─── Live mic level meter ──────────────────────────────────────────────────────

@Composable
fun MicLevelMeter(level: Float, modifier: Modifier = Modifier) {
    val bars = 20
    val activeColor = Color(0xFF4FC3F7)
    val inactiveColor = Color(0xFF1A1A2E)

    Row(
        modifier = modifier,
        horizontalArrangement = Arrangement.spacedBy(2.dp),
        verticalAlignment = Alignment.CenterVertically
    ) {
        repeat(bars) { i ->
            val threshold = i.toFloat() / bars
            val active = level > threshold
            val barHeight = (0.4f + 0.6f * (i.toFloat() / bars))
            Box(
                modifier = Modifier
                    .width(4.dp)
                    .fillMaxHeight(barHeight)
                    .clip(RoundedCornerShape(2.dp))
                    .background(if (active) activeColor.copy(alpha = 0.7f + 0.3f * barHeight) else inactiveColor)
            )
        }
    }
}
