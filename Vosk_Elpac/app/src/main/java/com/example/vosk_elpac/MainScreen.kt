package com.example.vosk_elpac

import androidx.compose.animation.*
import androidx.compose.animation.core.*
import androidx.compose.foundation.*
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.LazyRow
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.KeyboardActions
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.*
import androidx.compose.material.icons.rounded.Mic
import androidx.compose.material.icons.rounded.Stop
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.scale
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.ImeAction
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.window.Dialog
import com.example.vosk_elpac.model.*
import kotlin.math.roundToInt

@OptIn(ExperimentalLayoutApi::class)
@Composable
fun MainScreen(viewModel: MainViewModel) {
    val uiState by viewModel.uiState.collectAsState()

    Box(
        modifier = Modifier
            .fillMaxSize()
            .background(Color(0xFF0D0D1A))
    ) {
        Column(
            modifier = Modifier
                .fillMaxSize()
                .verticalScroll(rememberScrollState())
                .padding(horizontal = 20.dp, vertical = 16.dp),
            horizontalAlignment = Alignment.CenterHorizontally,
            verticalArrangement = Arrangement.spacedBy(16.dp)
        ) {
            AppHeader()

            // ── Model download progress ────────────────────────────────────
            if (uiState.modelStatus == ModelStatus.CHECKING ||
                uiState.modelStatus == ModelStatus.DOWNLOADING) {
                ModelDownloadCard(
                    status   = uiState.modelStatus,
                    progress = uiState.modelDownloadProgress
                )
            }
            if (uiState.modelStatus == ModelStatus.FAILED) {
                Card(
                    colors = CardDefaults.cardColors(containerColor = Color(0xFF3D1A1A)),
                    modifier = Modifier.fillMaxWidth()
                ) {
                    Text(
                        "Model download failed. Check internet connection and restart the app.",
                        color = Color(0xFFFF6B6B),
                        modifier = Modifier.padding(16.dp),
                        textAlign = TextAlign.Center
                    )
                }
            }

            // ── Target Phrase Section ──────────────────────────────────────
            TargetPhraseSection(
                targetPhrase = uiState.targetPhrase,
                customText = uiState.customPhraseText,
                onSelectPhrase = viewModel::showPhraseSelector,
                onCustomTextChange = viewModel::setCustomPhrase,
                onConfirmCustom = viewModel::confirmCustomPhrase,
                onClear = viewModel::clearPhrase
            )

            // ── Waveform ───────────────────────────────────────────────────
            WaveformSection(uiState)

            // ── Record button ──────────────────────────────────────────────
            RecordButton(
                state = uiState.recordingState,
                onStart = viewModel::startRecording,
                onStop = viewModel::stopRecording
            )

            // ── Live level meter ───────────────────────────────────────────
            AnimatedVisibility(uiState.recordingState == RecordingState.RECORDING) {
                MicLevelMeter(
                    level = uiState.liveLevel,
                    modifier = Modifier
                        .fillMaxWidth()
                        .height(32.dp)
                        .padding(horizontal = 16.dp)
                )
            }

            // ── Processing indicator ───────────────────────────────────────
            AnimatedVisibility(uiState.recordingState == RecordingState.PROCESSING) {
                Row(
                    horizontalArrangement = Arrangement.spacedBy(10.dp),
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    CircularProgressIndicator(
                        modifier = Modifier.size(20.dp),
                        color = Color(0xFF4FC3F7),
                        strokeWidth = 2.dp
                    )
                    Text("Analyzing phonemes…", color = Color(0xFF9E9E9E), fontSize = 14.sp)
                }
            }

            // ── Results ────────────────────────────────────────────────────
            uiState.session?.let { session ->
                AnimatedVisibility(
                    uiState.recordingState == RecordingState.DONE,
                    enter = fadeIn() + expandVertically()
                ) {
                    Column(verticalArrangement = Arrangement.spacedBy(16.dp)) {

                        // Comparison section (if target phrase was set)
                        session.comparison?.let { cmp ->
                            ComparisonSection(
                                comparison = cmp,
                                targetPhrase = session.targetPhrase
                            )
                        }

                        // ── ELPAC level badge ──────────────────────────────
                        uiState.elpacLevel?.let { level ->
                            Row(
                                modifier = Modifier
                                    .fillMaxWidth()
                                    .clip(RoundedCornerShape(12.dp))
                                    .background(Color(0xFF12122A))
                                    .padding(horizontal = 16.dp, vertical = 12.dp),
                                verticalAlignment = Alignment.CenterVertically,
                                horizontalArrangement = Arrangement.spacedBy(10.dp)
                            ) {
                                Text(
                                    "ELPAC level",
                                    color = Color(0xFF9E9E9E),
                                    fontSize = 13.sp
                                )
                                Box(
                                    modifier = Modifier
                                        .clip(RoundedCornerShape(20.dp))
                                        .background(
                                            when {
                                                level.startsWith("Level 4") -> Color(0xFF0D2A0D)
                                                level.startsWith("Level 3") -> Color(0xFF2A200A)
                                                level.startsWith("Level 2") -> Color(0xFF2A1A0A)
                                                else -> Color(0xFF2A0A0A)
                                            }
                                        )
                                        .padding(horizontal = 12.dp, vertical = 5.dp)
                                ) {
                                    Text(
                                        level,
                                        color = when {
                                            level.startsWith("Level 4") -> Color(0xFF81C784)
                                            level.startsWith("Level 3") -> Color(0xFFFFD54F)
                                            level.startsWith("Level 2") -> Color(0xFFFFB74D)
                                            else -> Color(0xFFEF9A9A)
                                        },
                                        fontSize = 13.sp,
                                        fontWeight = FontWeight.SemiBold
                                    )
                                }
                            }
                        }

                        // ── Transcript feedback (colored word highlights) ───
                        session.targetPhrase?.let { phrase ->
                            TranscriptFeedbackSection(
                                targetPhrase = phrase,
                                phonemes     = session.phonemes,
                                comparison   = session.comparison,
                                wordTimings  = session.wordTimings
                            )
                        }

                        // Score rings
                        ScoreSection(session.score)

                        // Phoneme timeline
                        SectionCard(title = "Phoneme Timeline") {
                            Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                                Text(
                                    "Tap a phoneme to inspect it",
                                    style = MaterialTheme.typography.labelSmall,
                                    color = Color(0xFF616161)
                                )
                                val totalMs = session.phonemes.lastOrNull()?.endTimeMs ?: 1000L
                                PhonemeTimeline(
                                    phonemes = session.phonemes,
                                    totalDurationMs = totalMs,
                                    selectedPhoneme = uiState.selectedPhoneme,
                                    onPhonemeClick = viewModel::selectPhoneme,
                                    modifier = Modifier
                                        .fillMaxWidth()
                                        .height(80.dp)
                                )
                                TimelineKey()
                            }
                        }

                        // Selected phoneme detail
                        AnimatedVisibility(uiState.selectedPhoneme != null) {
                            uiState.selectedPhoneme?.let { ph ->
                                SectionCard(title = "Phoneme Detail") {
                                    PhonemeDetailCard(
                                        phoneme = ph,
                                        modifier = Modifier.fillMaxWidth()
                                    )
                                }
                            }
                        }

                        // Expected vs Actual side-by-side chips
                        session.comparison?.let { cmp ->
                            SectionCard(title = "Expected vs Actual") {
                                ExpectedVsActualRow(comparison = cmp)
                            }
                        }

                        // All phoneme chips
                        SectionCard(title = "All Phonemes  (${session.phonemes.filter { it.phoneme != "∅" }.size})") {
                            LazyRow(
                                horizontalArrangement = Arrangement.spacedBy(8.dp),
                                contentPadding = PaddingValues(vertical = 4.dp)
                            ) {
                                items(session.phonemes.filter { it.phoneme != "∅" }) { ph ->
                                    PhonemeChip(
                                        phoneme = ph,
                                        isSelected = ph == uiState.selectedPhoneme,
                                        onClick = { viewModel.selectPhoneme(ph) }
                                    )
                                }
                            }
                        }

                        // Try again button
                        OutlinedButton(
                            onClick = viewModel::reset,
                            modifier = Modifier.fillMaxWidth(),
                            colors = ButtonDefaults.outlinedButtonColors(contentColor = Color(0xFF4FC3F7)),
                            border = BorderStroke(1.dp, Color(0xFF4FC3F7).copy(alpha = 0.5f))
                        ) {
                            Icon(Icons.Default.Refresh, null, Modifier.size(18.dp))
                            Spacer(Modifier.width(8.dp))
                            Text("Try Again")
                        }
                    }
                }
            }

            // ── Error ──────────────────────────────────────────────────────
            uiState.errorMessage?.let { msg ->
                ErrorCard(msg, onDismiss = viewModel::dismissError)
            }

            Spacer(Modifier.height(32.dp))
        }
    }

    // ── Phrase selector dialog ─────────────────────────────────────────────
    if (uiState.showPhraseSelector) {
        PhraseSelectorDialog(
            customText = uiState.customPhraseText,
            onCustomTextChange = viewModel::setCustomPhrase,
            onConfirmCustom = viewModel::confirmCustomPhrase,
            onSelectPreset = viewModel::selectPresetPhrase,
            onDismiss = viewModel::hidePhraseSelector
        )
    }
}

// ── Target Phrase Section ──────────────────────────────────────────────────────

@Composable
private fun TargetPhraseSection(
    targetPhrase: TargetPhrase?,
    customText: String,
    onSelectPhrase: () -> Unit,
    onCustomTextChange: (String) -> Unit,
    onConfirmCustom: () -> Unit,
    onClear: () -> Unit
) {
    Column(
        modifier = Modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(16.dp))
            .background(Color(0xFF12122A))
            .padding(16.dp),
        verticalArrangement = Arrangement.spacedBy(10.dp)
    ) {
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically
        ) {
            Text(
                "Target Phrase",
                style = MaterialTheme.typography.titleSmall,
                color = Color(0xFF9E9E9E),
                fontWeight = FontWeight.SemiBold
            )
            TextButton(onClick = onSelectPhrase) {
                Icon(
                    Icons.Default.List,
                    contentDescription = null,
                    modifier = Modifier.size(16.dp),
                    tint = Color(0xFF4FC3F7)
                )
                Spacer(Modifier.width(4.dp))
                Text("Browse", color = Color(0xFF4FC3F7), fontSize = 13.sp)
            }
        }

        if (targetPhrase != null) {
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .clip(RoundedCornerShape(10.dp))
                    .background(Color(0xFF1A1A3A))
                    .padding(12.dp),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Column(modifier = Modifier.weight(1f)) {
                    Text(
                        "\"${targetPhrase.text}\"",
                        color = Color.White,
                        fontSize = 18.sp,
                        fontWeight = FontWeight.Medium
                    )
                    Text(
                        targetPhrase.category,
                        color = Color(0xFF4FC3F7),
                        fontSize = 11.sp
                    )
                }
                IconButton(onClick = onClear) {
                    Icon(Icons.Default.Close, null, tint = Color(0xFF616161))
                }
            }
            Text(
                "Read the phrase above aloud, then press the mic",
                style = MaterialTheme.typography.bodySmall,
                color = Color(0xFF616161)
            )
        } else {
            OutlinedTextField(
                value = customText,
                onValueChange = onCustomTextChange,
                modifier = Modifier.fillMaxWidth(),
                placeholder = {
                    Text("Type a phrase or browse presets…", color = Color(0xFF424242), fontSize = 14.sp)
                },
                colors = OutlinedTextFieldDefaults.colors(
                    focusedBorderColor = Color(0xFF4FC3F7),
                    unfocusedBorderColor = Color(0xFF2A2A4A),
                    focusedTextColor = Color.White,
                    unfocusedTextColor = Color.White,
                    cursorColor = Color(0xFF4FC3F7)
                ),
                singleLine = true,
                keyboardOptions = KeyboardOptions(imeAction = ImeAction.Done),
                keyboardActions = KeyboardActions(onDone = { onConfirmCustom() }),
                trailingIcon = {
                    if (customText.isNotBlank()) {
                        IconButton(onClick = onConfirmCustom) {
                            Icon(Icons.Default.Check, null, tint = Color(0xFF4FC3F7))
                        }
                    }
                },
                shape = RoundedCornerShape(10.dp)
            )
        }
    }
}

// ── Phrase Selector Dialog ─────────────────────────────────────────────────────

@Composable
private fun PhraseSelectorDialog(
    customText: String,
    onCustomTextChange: (String) -> Unit,
    onConfirmCustom: () -> Unit,
    onSelectPreset: (TargetPhrase) -> Unit,
    onDismiss: () -> Unit
) {
    Dialog(onDismissRequest = onDismiss) {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .clip(RoundedCornerShape(20.dp))
                .background(Color(0xFF12122A))
                .padding(20.dp),
            verticalArrangement = Arrangement.spacedBy(12.dp)
        ) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Text(
                    "Choose a Phrase",
                    style = MaterialTheme.typography.titleMedium,
                    color = Color.White,
                    fontWeight = FontWeight.Bold
                )
                IconButton(onClick = onDismiss) {
                    Icon(Icons.Default.Close, null, tint = Color(0xFF9E9E9E))
                }
            }

            OutlinedTextField(
                value = customText,
                onValueChange = onCustomTextChange,
                modifier = Modifier.fillMaxWidth(),
                placeholder = { Text("Type a custom phrase…", color = Color(0xFF424242)) },
                colors = OutlinedTextFieldDefaults.colors(
                    focusedBorderColor = Color(0xFF4FC3F7),
                    unfocusedBorderColor = Color(0xFF2A2A4A),
                    focusedTextColor = Color.White,
                    unfocusedTextColor = Color.White,
                    cursorColor = Color(0xFF4FC3F7)
                ),
                singleLine = true,
                keyboardOptions = KeyboardOptions(imeAction = ImeAction.Done),
                keyboardActions = KeyboardActions(onDone = { onConfirmCustom() }),
                trailingIcon = {
                    if (customText.isNotBlank()) {
                        IconButton(onClick = onConfirmCustom) {
                            Icon(Icons.Default.Check, null, tint = Color(0xFF4FC3F7))
                        }
                    }
                },
                shape = RoundedCornerShape(10.dp),
                label = { Text("Custom phrase", color = Color(0xFF616161)) }
            )

            Divider(color = Color(0xFF2A2A4A))

            Text(
                "ELPAC Phrases",
                color = Color(0xFF9E9E9E),
                fontSize = 12.sp,
                fontWeight = FontWeight.SemiBold
            )

            LazyColumn(
                modifier = Modifier.heightIn(max = 360.dp),
                verticalArrangement = Arrangement.spacedBy(8.dp)
            ) {
                ElpacPhrases.CATEGORIES.forEach { category ->
                    item {
                        Text(
                            category,
                            color = Color(0xFF4FC3F7),
                            fontSize = 11.sp,
                            fontWeight = FontWeight.SemiBold,
                            modifier = Modifier.padding(top = 4.dp)
                        )
                    }
                    items(ElpacPhrases.ALL.filter { it.category == category }) { phrase ->
                        Row(
                            modifier = Modifier
                                .fillMaxWidth()
                                .clip(RoundedCornerShape(8.dp))
                                .background(Color(0xFF1A1A3A))
                                .clickable { onSelectPreset(phrase) }
                                .padding(horizontal = 14.dp, vertical = 10.dp),
                            verticalAlignment = Alignment.CenterVertically
                        ) {
                            Icon(
                                Icons.Default.PlayArrow,
                                null,
                                tint = Color(0xFF4FC3F7),
                                modifier = Modifier.size(16.dp)
                            )
                            Spacer(Modifier.width(10.dp))
                            Text(phrase.text, color = Color.White, fontSize = 15.sp)
                        }
                    }
                }
            }
        }
    }
}

// ── Comparison Section ─────────────────────────────────────────────────────────

@Composable
private fun ComparisonSection(
    comparison: PhonemeComparison,
    targetPhrase: TargetPhrase?
) {
    Column(
        modifier = Modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(16.dp))
            .background(Color(0xFF12122A))
            .padding(16.dp),
        verticalArrangement = Arrangement.spacedBy(12.dp)
    ) {
        Text(
            "Pronunciation Match",
            style = MaterialTheme.typography.titleSmall,
            color = Color(0xFF9E9E9E),
            fontWeight = FontWeight.SemiBold
        )

        val color = scoreColor(comparison.accuracyPct)
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.Center,
            verticalAlignment = Alignment.CenterVertically
        ) {
            Text(
                "${comparison.accuracyPct.roundToInt()}%",
                fontSize = 52.sp,
                fontWeight = FontWeight.Bold,
                color = color
            )
            Spacer(Modifier.width(12.dp))
            Column {
                Text("${comparison.matchedCount} / ${comparison.totalExpected}", color = Color.White, fontSize = 16.sp)
                Text("phonemes matched", color = Color(0xFF9E9E9E), fontSize = 12.sp)
            }
        }

        val feedback = when {
            comparison.accuracyPct >= 90f -> "Excellent! Your pronunciation is very accurate."
            comparison.accuracyPct >= 75f -> "Good job! A few phonemes need work."
            comparison.accuracyPct >= 55f -> "Keep practicing — focus on the red phonemes."
            else -> "Try again slowly, focusing on each sound."
        }
        Text(
            feedback,
            color = Color(0xFF9E9E9E),
            fontSize = 13.sp,
            textAlign = TextAlign.Center,
            modifier = Modifier.fillMaxWidth()
        )
    }
}

// ── Expected vs Actual Row ─────────────────────────────────────────────────────

@Composable
private fun ExpectedVsActualRow(comparison: PhonemeComparison) {
    Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
        Row(modifier = Modifier.fillMaxWidth()) {
            Text(
                "Expected", color = Color(0xFF4FC3F7), fontSize = 11.sp,
                fontWeight = FontWeight.SemiBold, modifier = Modifier.weight(1f)
            )
            Text(
                "You said", color = Color(0xFF9E9E9E), fontSize = 11.sp,
                fontWeight = FontWeight.SemiBold, modifier = Modifier.weight(1f),
                textAlign = TextAlign.End
            )
        }

        val pairs = comparison.expectedPhonemes.zip(
            comparison.actualPhonemes.map { it.phoneme } +
                    List((comparison.expectedPhonemes.size - comparison.actualPhonemes.size)
                        .coerceAtLeast(0)) { "—" }
        )

        LazyRow(
            horizontalArrangement = Arrangement.spacedBy(6.dp),
            contentPadding = PaddingValues(vertical = 4.dp)
        ) {
            items(pairs) { (expected, actual) ->
                val isMatch = expected == actual || phonemesMatchUi(expected, actual)
                Column(
                    modifier = Modifier
                        .clip(RoundedCornerShape(8.dp))
                        .background(if (isMatch) Color(0xFF0D2A1A) else Color(0xFF2A0D0D))
                        .border(
                            1.dp,
                            if (isMatch) Color(0xFF66BB6A) else Color(0xFFEF5350),
                            RoundedCornerShape(8.dp)
                        )
                        .padding(horizontal = 10.dp, vertical = 8.dp),
                    horizontalAlignment = Alignment.CenterHorizontally
                ) {
                    Text(
                        expected,
                        fontSize = 18.sp,
                        fontFamily = FontFamily.Serif,
                        color = Color(0xFF4FC3F7),
                        fontWeight = FontWeight.Bold
                    )
                    Spacer(Modifier.height(4.dp))
                    Text(
                        actual,
                        fontSize = 18.sp,
                        fontFamily = FontFamily.Serif,
                        color = if (isMatch) Color(0xFF66BB6A) else Color(0xFFEF5350),
                        fontWeight = FontWeight.Bold
                    )
                    Icon(
                        if (isMatch) Icons.Default.Check else Icons.Default.Close,
                        null,
                        modifier = Modifier.size(12.dp),
                        tint = if (isMatch) Color(0xFF66BB6A) else Color(0xFFEF5350)
                    )
                }
            }
        }

        Row(
            horizontalArrangement = Arrangement.spacedBy(16.dp),
            modifier = Modifier.padding(top = 4.dp)
        ) {
            Row(
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.spacedBy(4.dp)
            ) {
                Box(Modifier.size(8.dp).clip(CircleShape).background(Color(0xFF66BB6A)))
                Text("Correct", fontSize = 10.sp, color = Color(0xFF9E9E9E))
            }
            Row(
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.spacedBy(4.dp)
            ) {
                Box(Modifier.size(8.dp).clip(CircleShape).background(Color(0xFFEF5350)))
                Text("Needs work", fontSize = 10.sp, color = Color(0xFF9E9E9E))
            }
        }
    }
}

private fun phonemesMatchUi(a: String, b: String): Boolean {
    if (a == b) return true
    val similar = setOf(
        setOf("ɪ", "iː"), setOf("ʌ", "ɑ"), setOf("ɛ", "æ"),
        setOf("ʊ", "uː"), setOf("ɔ", "oʊ"), setOf("ð", "θ"),
        setOf("s", "z"),  setOf("f", "v"),   setOf("p", "b"),
        setOf("t", "d"),  setOf("k", "ɡ"),   setOf("ʃ", "ʒ")
    )
    return similar.any { it.contains(a) && it.contains(b) }
}

// ── Existing composables (unchanged) ──────────────────────────────────────────

@Composable
private fun AppHeader() {
    Column(horizontalAlignment = Alignment.CenterHorizontally) {
        Text(
            "PhonemeCoach",
            style = MaterialTheme.typography.headlineMedium,
            fontWeight = FontWeight.Bold,
            color = Color.White
        )
        Text(
            "Real-time pronunciation analysis",
            style = MaterialTheme.typography.bodySmall,
            color = Color(0xFF9E9E9E)
        )
    }
}

@Composable
private fun WaveformSection(uiState: MainUiState) {
    Box(
        modifier = Modifier
            .fillMaxWidth()
            .height(100.dp)
            .clip(RoundedCornerShape(12.dp))
            .background(Color(0xFF111122))
            .padding(horizontal = 8.dp, vertical = 4.dp)
    ) {
        when {
            uiState.recordingState == RecordingState.DONE && uiState.session != null ->
                WaveformCanvas(waveformPoints = uiState.session.waveform, isLive = false, color = Color(0xFF4FC3F7))
            uiState.recordingState == RecordingState.RECORDING ->
                WaveformCanvas(waveformPoints = uiState.liveWaveform, isLive = true, color = Color(0xFF80CBC4))
            else ->
                Box(Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
                    Text("Press mic to start", color = Color(0xFF424242), fontSize = 13.sp)
                }
        }
        if (uiState.recordingState == RecordingState.RECORDING && uiState.recordingDurationMs > 0) {
            Text(
                formatDuration(uiState.recordingDurationMs),
                modifier = Modifier
                    .align(Alignment.TopEnd)
                    .padding(6.dp)
                    .clip(RoundedCornerShape(4.dp))
                    .background(Color(0x88000000))
                    .padding(horizontal = 6.dp, vertical = 2.dp),
                color = Color(0xFF80CBC4),
                fontSize = 11.sp
            )
        }
    }
}

@Composable
private fun RecordButton(state: RecordingState, onStart: () -> Unit, onStop: () -> Unit) {
    val pulseScale by rememberInfiniteTransition(label = "pulse").animateFloat(
        initialValue = 1f, targetValue = 1.15f,
        animationSpec = infiniteRepeatable(tween(700, easing = FastOutSlowInEasing), RepeatMode.Reverse),
        label = "scale"
    )
    val isRecording = state == RecordingState.RECORDING
    val bgColor = if (isRecording) Color(0xFFEF5350) else Color(0xFF4FC3F7)

    Box(
        modifier = Modifier
            .size(80.dp)
            .clip(CircleShape)
            .scale(if (isRecording) pulseScale else 1f)
            .background(bgColor)
            .clickable(enabled = state != RecordingState.PROCESSING) {
                if (isRecording) onStop() else onStart()
            },
        contentAlignment = Alignment.Center
    ) {
        Icon(
            if (isRecording) Icons.Rounded.Stop else Icons.Rounded.Mic,
            contentDescription = if (isRecording) "Stop" else "Record",
            modifier = Modifier.size(36.dp),
            tint = Color.White
        )
    }
}

@Composable
private fun ScoreSection(score: PronunciationScore) {
    SectionCard(title = "Pronunciation Score") {
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceEvenly
        ) {
            ScoreRing(score.overallScore, "Overall", Modifier.size(90.dp))
            ScoreRing(score.accuracyScore, "Accuracy", Modifier.size(70.dp))
            ScoreRing(score.fluencyScore, "Fluency", Modifier.size(70.dp))
            ScoreRing(score.completenessScore, "Complete", Modifier.size(70.dp))
        }
    }
}

@Composable
private fun PhonemeChip(phoneme: PhonemeResult, isSelected: Boolean, onClick: () -> Unit) {
    val isCorrect = phoneme.isCorrect
    val bg = when {
        isSelected -> Color(0xFF1E3A5F)
        !isCorrect -> Color(0xFF2A0D0D)
        else       -> Color(0xFF1A1A2E)
    }
    val border = when {
        isSelected -> Color(0xFF4FC3F7)
        !isCorrect -> Color(0xFFEF5350)
        else       -> Color.Transparent
    }
    Column(
        modifier = Modifier
            .clip(RoundedCornerShape(8.dp))
            .background(bg)
            .border(1.dp, border, RoundedCornerShape(8.dp))
            .clickable(onClick = onClick)
            .padding(horizontal = 10.dp, vertical = 8.dp),
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        Text(
            phoneme.phoneme, fontSize = 20.sp, fontFamily = FontFamily.Serif,
            color = Color.White, fontWeight = FontWeight.Bold
        )
        phoneme.expectedPhoneme?.let { exp ->
            if (exp != phoneme.phoneme) {
                Text("→$exp", fontSize = 9.sp, color = Color(0xFFEF5350))
            }
        }
        Text("${phoneme.score.roundToInt()}", fontSize = 11.sp, color = scoreColor(phoneme.score))
        Text("${phoneme.durationMs}ms", fontSize = 9.sp, color = Color(0xFF616161))
    }
}

@Composable
private fun TimelineKey() {
    LazyRow(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
        val categories = mapOf(
            "Vowel"     to Color(0xFF4FC3F7),
            "Stop"      to Color(0xFFEF9A9A),
            "Fricative" to Color(0xFFA5D6A7),
            "Nasal"     to Color(0xFFCE93D8),
            "Liquid"    to Color(0xFFFFCC02)
        )
        items(categories.entries.toList()) { (name, color) ->
            Row(
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.spacedBy(4.dp)
            ) {
                Box(Modifier.size(8.dp).clip(RoundedCornerShape(2.dp)).background(color))
                Text(name, fontSize = 10.sp, color = Color(0xFF9E9E9E))
            }
        }
    }
}

@Composable
private fun SectionCard(title: String, content: @Composable ColumnScope.() -> Unit) {
    Column(
        modifier = Modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(16.dp))
            .background(Color(0xFF12122A))
            .padding(16.dp),
        verticalArrangement = Arrangement.spacedBy(12.dp)
    ) {
        Text(
            title, style = MaterialTheme.typography.titleSmall,
            color = Color(0xFF9E9E9E), fontWeight = FontWeight.SemiBold
        )
        content()
    }
}

@Composable
private fun ErrorCard(message: String, onDismiss: () -> Unit) {
    Card(
        modifier = Modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(containerColor = Color(0xFF3A1A1A)),
        shape = RoundedCornerShape(12.dp)
    ) {
        Row(
            modifier = Modifier.padding(16.dp),
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.spacedBy(12.dp)
        ) {
            Icon(Icons.Default.Warning, null, tint = Color(0xFFEF5350))
            Text(message, Modifier.weight(1f), color = Color.White, fontSize = 14.sp)
            IconButton(onClick = onDismiss) {
                Icon(Icons.Default.Close, null, tint = Color(0xFF9E9E9E))
            }
        }
    }
}

@Composable
private fun ModelDownloadCard(status: ModelStatus, progress: Float) {
    Card(
        colors = CardDefaults.cardColors(containerColor = Color(0xFF1A1A2E)),
        modifier = Modifier.fillMaxWidth()
    ) {
        Column(
            modifier = Modifier.padding(20.dp),
            horizontalAlignment = Alignment.CenterHorizontally,
            verticalArrangement = Arrangement.spacedBy(12.dp)
        ) {
            if (status == ModelStatus.CHECKING) {
                CircularProgressIndicator(color = Color(0xFF7C6AF7))
                Text("Loading phoneme model…", color = Color.White, fontSize = 14.sp)
            } else {
                Text(
                    "Downloading phoneme model",
                    color = Color.White,
                    fontWeight = FontWeight.SemiBold,
                    fontSize = 15.sp
                )
                LinearProgressIndicator(
                    progress = progress,
                    modifier = Modifier.fillMaxWidth(),
                    color = Color(0xFF7C6AF7),
                    trackColor = Color(0xFF2D2D44)
                )
                Text(
                    "${(progress * 100).roundToInt()}%  —  first launch only",
                    color = Color(0xFF9090B0),
                    fontSize = 12.sp
                )
            }
        }
    }
}

private fun formatDuration(ms: Long): String {
    val s = ms / 1000; val r = ms % 1000 / 100; return "$s.${r}s"
}

private fun scoreColor(score: Float): Color = when {
    score >= 80f -> Color(0xFF66BB6A)
    score >= 60f -> Color(0xFFFFCA28)
    else         -> Color(0xFFEF5350)
}