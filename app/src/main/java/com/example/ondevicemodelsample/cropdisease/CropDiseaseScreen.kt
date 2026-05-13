package com.example.ondevicemodelsample.cropdisease

import android.Manifest
import android.content.pm.PackageManager
import android.net.Uri
import android.os.Build
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts
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
import androidx.compose.foundation.rememberScrollState
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
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.remember
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.core.content.ContextCompat
import androidx.lifecycle.viewmodel.compose.viewModel
import coil.compose.AsyncImage
import com.example.ondevicemodelsample.util.MediaStoreUtils

@Composable
fun CropDiseaseScreen(
    contentPadding: PaddingValues,
    onBack: () -> Unit,
    viewModel: CropDiseaseViewModel = viewModel(),
) {
    val context = LocalContext.current
    val uiState by viewModel.uiState.collectAsState()

    val pendingCaptureUri = remember { UriHolder() }

    val galleryLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.PickVisualMedia()
    ) { uri ->
        if (uri != null) viewModel.classify(uri)
    }

    val cameraLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.TakePicture()
    ) { success ->
        val uri = pendingCaptureUri.value
        if (uri != null) {
            if (success) {
                MediaStoreUtils.finalizeGalleryImage(context, uri)
                viewModel.classify(uri)
            } else {
                MediaStoreUtils.discardGalleryImage(context, uri)
            }
            pendingCaptureUri.value = null
        }
    }

    fun launchCamera() {
        val uri = MediaStoreUtils.createGalleryImageUri(context) ?: return
        pendingCaptureUri.value = uri
        cameraLauncher.launch(uri)
    }

    val cameraPermissionLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.RequestMultiplePermissions()
    ) { results ->
        val cameraGranted = results[Manifest.permission.CAMERA] == true
        val storageOk = Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q ||
            results[Manifest.permission.WRITE_EXTERNAL_STORAGE] == true
        if (cameraGranted && storageOk) launchCamera()
    }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(contentPadding)
            .padding(horizontal = 16.dp)
            .verticalScroll(rememberScrollState()),
        horizontalAlignment = Alignment.CenterHorizontally,
    ) {
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(top = 8.dp, bottom = 4.dp),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            TextButton(onClick = onBack) { Text("← Back") }
        }

        Text(
            text = "Crop Disease Classifier",
            style = MaterialTheme.typography.titleLarge,
        )
        Text(
            text = "EfficientNet-B3 · 17 classes · Corn / Potato / Rice / Wheat / Sugarcane",
            style = MaterialTheme.typography.bodySmall,
            modifier = Modifier.padding(bottom = 16.dp),
        )

        if (uiState.modelMissing) {
            MissingModelCard()
            Spacer(Modifier.height(16.dp))
        }

        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.spacedBy(12.dp),
        ) {
            Button(
                onClick = {
                    galleryLauncher.launch(
                        PickVisualMediaRequest(
                            ActivityResultContracts.PickVisualMedia.ImageOnly
                        )
                    )
                },
                modifier = Modifier.weight(1f),
                enabled = !uiState.modelMissing,
            ) { Text("Gallery") }

            OutlinedButton(
                onClick = {
                    val needed = buildList {
                        if (ContextCompat.checkSelfPermission(
                                context, Manifest.permission.CAMERA
                            ) != PackageManager.PERMISSION_GRANTED
                        ) add(Manifest.permission.CAMERA)
                        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.Q &&
                            ContextCompat.checkSelfPermission(
                                context, Manifest.permission.WRITE_EXTERNAL_STORAGE
                            ) != PackageManager.PERMISSION_GRANTED
                        ) add(Manifest.permission.WRITE_EXTERNAL_STORAGE)
                    }
                    if (needed.isEmpty()) launchCamera()
                    else cameraPermissionLauncher.launch(needed.toTypedArray())
                },
                modifier = Modifier.weight(1f),
                enabled = !uiState.modelMissing,
            ) { Text("Camera") }
        }

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
                                modifier = Modifier
                                    .height(18.dp)
                                    .aspectRatio(1f),
                                strokeWidth = 2.dp,
                            )
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
            PerformanceCard(result)
            Spacer(Modifier.height(12.dp))
            TopKCard(result)
            Spacer(Modifier.height(24.dp))
        }
    }
}

@Composable
private fun MissingModelCard() {
    Card(
        modifier = Modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(
            containerColor = MaterialTheme.colorScheme.errorContainer,
        ),
    ) {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .padding(16.dp),
        ) {
            Text(
                "Model not bundled",
                fontWeight = FontWeight.SemiBold,
                color = MaterialTheme.colorScheme.onErrorContainer,
            )
            Spacer(Modifier.height(6.dp))
            Text(
                "Run the conversion script from the project root to download the " +
                        "PyTorch weights, export to TFLite, and drop the asset into " +
                        "app/src/main/assets/, then rebuild:",
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onErrorContainer,
            )
            Spacer(Modifier.height(8.dp))
            Text(
                "python tools/convert_crop_disease_to_tflite.py",
                style = MaterialTheme.typography.bodySmall,
                fontWeight = FontWeight.Medium,
                color = MaterialTheme.colorScheme.onErrorContainer,
            )
        }
    }
}

@Composable
private fun VerdictCard(result: CropDiseaseResult) {
    val top = result.top
    val isHealthy = top.condition.equals("Healthy", ignoreCase = true)
    val color = if (isHealthy) Color(0xFF2E7D32) else Color(0xFFB00020)

    Card(
        modifier = Modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(containerColor = color.copy(alpha = 0.10f)),
    ) {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .padding(16.dp),
        ) {
            Text(
                if (isHealthy) "Healthy" else "Disease detected",
                fontWeight = FontWeight.SemiBold,
                color = color,
            )
            Spacer(Modifier.height(6.dp))
            Text(
                "${top.crop} · ${top.condition}",
                style = MaterialTheme.typography.bodyLarge,
            )
            Spacer(Modifier.height(8.dp))
            Text(
                "Confidence: %.1f%%".format(top.probability * 100f),
                style = MaterialTheme.typography.bodySmall,
            )
        }
    }
}

@Composable
private fun PerformanceCard(result: CropDiseaseResult) {
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
            MetricRow("Inference (wall)", "${result.inferenceTimeMs} ms")
            MetricRow("CPU time", "${result.cpuTimeMs} ms")
        }
    }
}

@Composable
private fun TopKCard(result: CropDiseaseResult) {
    if (result.topK.isEmpty()) return
    Card(modifier = Modifier.fillMaxWidth()) {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .padding(16.dp),
        ) {
            Text("Top ${result.topK.size}", fontWeight = FontWeight.SemiBold)
            Spacer(Modifier.height(8.dp))
            HorizontalDivider()
            result.topK.forEachIndexed { i, p ->
                Spacer(Modifier.height(12.dp))
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.SpaceBetween,
                ) {
                    Text(
                        "${i + 1}. ${p.crop} · ${p.condition}",
                        fontWeight = if (i == 0) FontWeight.SemiBold else FontWeight.Normal,
                    )
                    Text("%.1f%%".format(p.probability * 100f))
                }
                Spacer(Modifier.height(4.dp))
                LinearProgressIndicator(
                    progress = { p.probability.coerceIn(0f, 1f) },
                    modifier = Modifier.fillMaxWidth(),
                )
            }
        }
    }
}

@Composable
private fun MetricRow(label: String, value: String) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .padding(vertical = 4.dp),
        horizontalArrangement = Arrangement.SpaceBetween,
    ) {
        Text(label, style = MaterialTheme.typography.bodySmall)
        Text(value, style = MaterialTheme.typography.bodySmall, fontWeight = FontWeight.Medium)
    }
}

private class UriHolder {
    var value: Uri? = null
}