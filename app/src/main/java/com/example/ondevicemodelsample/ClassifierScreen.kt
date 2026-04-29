package com.example.ondevicemodelsample

import android.Manifest
import android.content.pm.PackageManager
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.aspectRatio
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyRow
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.remember
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.core.content.ContextCompat
import androidx.core.content.FileProvider
import androidx.lifecycle.viewmodel.compose.viewModel
import coil.compose.AsyncImage
import com.example.ondevicemodelsample.data.FeedbackStats
import com.example.ondevicemodelsample.ml.Classification
import com.example.ondevicemodelsample.ml.ClassificationResult
import com.example.ondevicemodelsample.ml.ModelPerformance
import com.example.ondevicemodelsample.ml.Verdict
import com.example.ondevicemodelsample.util.BitmapUtils
import com.example.ondevicemodelsample.util.DeviceInfo
import java.io.File

@Composable
fun ClassifierScreen(
    contentPadding: PaddingValues,
    viewModel: ClassifierViewModel = viewModel(),
) {
    val context = LocalContext.current
    val uiState by viewModel.uiState.collectAsState()
    val feedbackStats by viewModel.feedbackStats.collectAsState()
    val deviceInfo = remember { DeviceInfo.collect(context) }

    val pendingCaptureFile = remember { Holder<File>() }
    val pendingCaptureUri = remember { Holder<android.net.Uri>() }

    val galleryLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.PickVisualMedia()
    ) { uri ->
        if (uri != null) viewModel.classify(uri)
    }

    val cameraLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.TakePicture()
    ) { success ->
        val uri = pendingCaptureUri.value
        if (success && uri != null) {
            viewModel.classify(uri)
        }
    }

    fun launchCamera() {
        val file = BitmapUtils.createCaptureFile(context)
        val uri = FileProvider.getUriForFile(
            context,
            "${context.packageName}.fileprovider",
            file
        )
        pendingCaptureFile.value = file
        pendingCaptureUri.value = uri
        cameraLauncher.launch(uri)
    }

    val cameraPermissionLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.RequestPermission()
    ) { granted ->
        if (granted) launchCamera()
    }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(contentPadding)
            .padding(horizontal = 16.dp)
            .verticalScroll(rememberScrollState()),
        horizontalAlignment = Alignment.CenterHorizontally,
    ) {
        Text(
            text = "Plant & Disease Classifier",
            style = MaterialTheme.typography.titleLarge,
            modifier = Modifier.padding(top = 8.dp, bottom = 4.dp),
        )
        Text(
            text = "PlantVillage MobileNetV2 + ImageNet gate · TensorFlow Lite",
            style = MaterialTheme.typography.bodySmall,
            modifier = Modifier.padding(bottom = 16.dp),
        )

        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.spacedBy(12.dp),
        ) {
            Button(
                onClick = {
                    galleryLauncher.launch(
                        androidx.activity.result.PickVisualMediaRequest(
                            ActivityResultContracts.PickVisualMedia.ImageOnly
                        )
                    )
                },
                modifier = Modifier.weight(1f),
            ) { Text("Gallery") }

            OutlinedButton(
                onClick = {
                    val granted = ContextCompat.checkSelfPermission(
                        context, Manifest.permission.CAMERA
                    ) == PackageManager.PERMISSION_GRANTED
                    if (granted) launchCamera()
                    else cameraPermissionLauncher.launch(Manifest.permission.CAMERA)
                },
                modifier = Modifier.weight(1f),
            ) { Text("Camera") }
        }

        Spacer(Modifier.height(12.dp))
        SampleRow { asset -> viewModel.classifyAsset(asset) }
        Spacer(Modifier.height(16.dp))

        Card(modifier = Modifier.fillMaxWidth()) {
            Box(
                modifier = Modifier
                    .fillMaxWidth()
                    .aspectRatio(1f),
                contentAlignment = Alignment.Center,
            ) {
                val uri = uiState.imageUri
                if (uri != null) {
                    AsyncImage(
                        model = uri,
                        contentDescription = "Selected image",
                        contentScale = ContentScale.Crop,
                        modifier = Modifier.fillMaxSize(),
                    )
                } else {
                    Text(
                        text = "Pick a photo or capture one",
                        style = MaterialTheme.typography.bodyMedium,
                    )
                }
                if (uiState.isRunning) {
                    Surface(
                        tonalElevation = 2.dp,
                        modifier = Modifier.padding(16.dp),
                    ) {
                        Row(
                            modifier = Modifier.padding(horizontal = 12.dp, vertical = 8.dp),
                            verticalAlignment = Alignment.CenterVertically,
                        ) {
                            CircularProgressIndicator(
                                modifier = Modifier.height(18.dp).aspectRatio(1f),
                                strokeWidth = 2.dp,
                            )
                            Spacer(Modifier.height(0.dp))
                            Text(
                                "  Classifying…",
                                style = MaterialTheme.typography.bodyMedium,
                            )
                        }
                    }
                }
            }
        }

        Spacer(Modifier.height(16.dp))

        uiState.error?.let {
            Text(
                text = "Error: $it",
                color = MaterialTheme.colorScheme.error,
                style = MaterialTheme.typography.bodyMedium,
            )
            Spacer(Modifier.height(8.dp))
        }

        uiState.result?.let { result ->
            VerdictCard(result)
            Spacer(Modifier.height(12.dp))
            FeedbackCard(
                feedbackGiven = uiState.feedbackGiven,
                onCorrect = { viewModel.submitFeedback(true) },
                onIncorrect = { viewModel.submitFeedback(false) },
            )
            Spacer(Modifier.height(12.dp))
            PerformanceCard(result.performance, deviceInfo.cores)
            Spacer(Modifier.height(12.dp))
            DetailCard(result)
            Spacer(Modifier.height(12.dp))
        }

        FeedbackStatsCard(feedbackStats)
        Spacer(Modifier.height(12.dp))

        DeviceCard(deviceInfo)

        Spacer(Modifier.height(24.dp))
    }
}

@Composable
private fun VerdictCard(result: ClassificationResult) {
    val (label, color) = when (result.verdict) {
        Verdict.PLANT_RECOGNIZED -> "Plant identified" to Color(0xFF2E7D32)
        Verdict.PLANT_UNCERTAIN -> "Plant detected" to Color(0xFF9A6E00)
        Verdict.NOT_PLANT -> "Not a plant" to Color(0xFFB00020)
    }
    Card(
        modifier = Modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(containerColor = color.copy(alpha = 0.10f)),
    ) {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .padding(16.dp),
        ) {
            Text(label, fontWeight = FontWeight.SemiBold, color = color)
            Spacer(Modifier.height(6.dp))
            Text(result.summary, style = MaterialTheme.typography.bodyLarge)
            if (result.verdict != Verdict.NOT_PLANT) {
                Spacer(Modifier.height(8.dp))
                Text(
                    "Confidence: %.1f%%".format(result.confidence * 100f),
                    style = MaterialTheme.typography.bodySmall,
                )
            }
        }
    }
}

@Composable
private fun FeedbackCard(
    feedbackGiven: Boolean,
    onCorrect: () -> Unit,
    onIncorrect: () -> Unit,
) {
    Card(modifier = Modifier.fillMaxWidth()) {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .padding(16.dp),
        ) {
            Text("Was this classification correct?", fontWeight = FontWeight.SemiBold)
            Spacer(Modifier.height(12.dp))
            if (feedbackGiven) {
                Text(
                    "Thanks — feedback recorded.",
                    style = MaterialTheme.typography.bodyMedium,
                    color = MaterialTheme.colorScheme.primary,
                )
            } else {
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.spacedBy(12.dp),
                ) {
                    Button(
                        onClick = onCorrect,
                        modifier = Modifier.weight(1f),
                    ) { Text("Correct") }
                    OutlinedButton(
                        onClick = onIncorrect,
                        modifier = Modifier.weight(1f),
                    ) { Text("Incorrect") }
                }
            }
        }
    }
}

@Composable
private fun FeedbackStatsCard(stats: FeedbackStats) {
    Card(modifier = Modifier.fillMaxWidth()) {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .padding(16.dp),
        ) {
            Text("Feedback stats", fontWeight = FontWeight.SemiBold)
            Spacer(Modifier.height(8.dp))
            HorizontalDivider()
            Spacer(Modifier.height(8.dp))
            if (stats.total == 0) {
                Text(
                    "No feedback yet. Classify an image and tell the app whether it got it right.",
                    style = MaterialTheme.typography.bodySmall,
                )
            } else {
                MetricRow("Total reviews", stats.total.toString())
                MetricRow("Correct", stats.correct.toString())
                MetricRow("Incorrect", stats.incorrect.toString())
                MetricRow("Accuracy", "%.1f%%".format(stats.accuracyPercent))
            }
        }
    }
}

@Composable
private fun PerformanceCard(perf: ModelPerformance, cores: Int) {
    Card(modifier = Modifier.fillMaxWidth()) {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .padding(16.dp),
        ) {
            Text("Performance", fontWeight = FontWeight.SemiBold)
            Spacer(Modifier.height(8.dp))
            HorizontalDivider()
            Spacer(Modifier.height(8.dp))
            MetricRow("Inference (wall)", "${perf.inferenceTimeMs} ms")
            MetricRow("CPU time", "${perf.cpuTimeMs} ms")
            MetricRow(
                "CPU utilization",
                "%.0f%% of %d cores".format(perf.cpuUtilizationPercent, cores),
            )
            MetricRow(
                "Java heap",
                "%.1f MB (Δ %+.2f MB)".format(perf.heapUsedMb, perf.heapDeltaMb),
            )
            MetricRow(
                "Native heap",
                "%.1f MB (Δ %+.2f MB)".format(perf.nativeUsedMb, perf.nativeDeltaMb),
            )
        }
    }
}

@Composable
private fun DeviceCard(info: DeviceInfo) {
    Card(modifier = Modifier.fillMaxWidth()) {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .padding(16.dp),
        ) {
            Text("Device", fontWeight = FontWeight.SemiBold)
            Spacer(Modifier.height(8.dp))
            HorizontalDivider()
            Spacer(Modifier.height(8.dp))
            MetricRow("Model", info.displayName)
            MetricRow("Android", "${info.androidVersion} (API ${info.apiLevel})")
            MetricRow("ABI", info.primaryAbi)
            MetricRow("CPU cores", info.cores.toString())
            MetricRow("System RAM", "${info.totalRamMb} MB")
            MetricRow("App max heap", "${info.maxHeapMb} MB")
        }
    }
}

@Composable
private fun MetricRow(label: String, value: String) {
    Row(
        modifier = Modifier.fillMaxWidth().padding(vertical = 4.dp),
        horizontalArrangement = Arrangement.SpaceBetween,
    ) {
        Text(label, style = MaterialTheme.typography.bodySmall)
        Text(value, style = MaterialTheme.typography.bodySmall, fontWeight = FontWeight.Medium)
    }
}

@Composable
private fun DetailCard(result: ClassificationResult) {
    if (result.plantPredictions.isEmpty() && result.gatePredictions.isEmpty()) return
    Card(modifier = Modifier.fillMaxWidth()) {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .padding(16.dp),
        ) {
            if (result.gatePredictions.isNotEmpty()) {
                Text("Gate model (ImageNet)", fontWeight = FontWeight.SemiBold)
                Spacer(Modifier.height(8.dp))
                HorizontalDivider()
                ScoreList(result.gatePredictions)
            }

            if (result.plantPredictions.isNotEmpty()) {
                if (result.gatePredictions.isNotEmpty()) Spacer(Modifier.height(16.dp))
                Text("Disease model (PlantVillage)", fontWeight = FontWeight.SemiBold)
                Spacer(Modifier.height(8.dp))
                HorizontalDivider()
                ScoreList(result.plantPredictions)
            }
        }
    }
}

@Composable
private fun ScoreList(items: List<Classification>) {
    items.forEachIndexed { i, r ->
        Spacer(Modifier.height(12.dp))
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween,
        ) {
            Text(
                "${i + 1}. ${r.label}",
                fontWeight = if (i == 0) FontWeight.SemiBold else FontWeight.Normal,
            )
            Text("%.1f%%".format(r.confidence * 100f))
        }
        Spacer(Modifier.height(4.dp))
        LinearProgressIndicator(
            progress = { r.confidence.coerceIn(0f, 1f) },
            modifier = Modifier.fillMaxWidth(),
        )
    }
}

private data class Sample(val title: String, val asset: String)

private val SAMPLES = listOf(
    Sample("Tomato · Healthy", "samples/tomato_healthy.jpg"),
    Sample("Tomato · Late blight", "samples/tomato_late_blight.jpg"),
    Sample("Apple · Healthy", "samples/apple_healthy.jpg"),
    Sample("Potato · Early blight", "samples/potato_early_blight.jpg"),
    Sample("Corn · Healthy", "samples/corn_healthy.jpg"),
    Sample("Grape · Black rot", "samples/grape_black_rot.jpg"),
)

@Composable
private fun SampleRow(onPick: (String) -> Unit) {
    Column(modifier = Modifier.fillMaxWidth()) {
        Text(
            "Sample plants (PlantVillage · CC0)",
            style = MaterialTheme.typography.bodySmall,
            modifier = Modifier.padding(bottom = 6.dp),
        )
        LazyRow(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            items(SAMPLES) { sample ->
                Column(
                    modifier = Modifier
                        .width(96.dp)
                        .clickable { onPick(sample.asset) },
                ) {
                    AsyncImage(
                        model = "file:///android_asset/${sample.asset}",
                        contentDescription = sample.title,
                        contentScale = ContentScale.Crop,
                        modifier = Modifier
                            .size(96.dp)
                            .clip(RoundedCornerShape(8.dp)),
                    )
                    Spacer(Modifier.height(4.dp))
                    Text(
                        sample.title,
                        style = MaterialTheme.typography.labelSmall,
                        maxLines = 2,
                    )
                }
            }
        }
    }
}

private class Holder<T> {
    var value: T? = null
}