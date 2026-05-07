package com.example.ondevicemodelsample.mlkit

import android.app.Application
import android.graphics.Bitmap
import android.net.Uri
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.example.ondevicemodelsample.util.BitmapUtils
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.label.ImageLabel
import com.google.mlkit.vision.label.ImageLabeler
import com.google.mlkit.vision.label.ImageLabeling
import com.google.mlkit.vision.label.defaults.ImageLabelerOptions
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import kotlinx.coroutines.tasks.await
import kotlinx.coroutines.withContext

enum class MlKitVerdict { PLANT, NOT_PLANT }

data class MlKitLabel(val text: String, val confidence: Float)

data class MlKitResult(
    val verdict: MlKitVerdict,
    val matchedLabel: MlKitLabel?,
    val allLabels: List<MlKitLabel>,
    val inferenceTimeMs: Long,
)

data class MlKitPlantUiState(
    val imageUri: Uri? = null,
    val result: MlKitResult? = null,
    val isRunning: Boolean = false,
    val error: String? = null,
)

class MlKitPlantViewModel(application: Application) : AndroidViewModel(application) {

    private val _uiState = MutableStateFlow(MlKitPlantUiState())
    val uiState: StateFlow<MlKitPlantUiState> = _uiState.asStateFlow()

    private val labeler: ImageLabeler = ImageLabeling.getClient(
        ImageLabelerOptions.Builder()
            .setConfidenceThreshold(MIN_CONFIDENCE)
            .build()
    )

    fun classify(uri: Uri) {
        _uiState.value = MlKitPlantUiState(imageUri = uri, isRunning = true)
        viewModelScope.launch {
            runCatching {
                val bitmap: Bitmap = withContext(Dispatchers.Default) {
                    BitmapUtils.decodeSampled(getApplication(), uri)
                        ?: error("Unable to decode image")
                }
                runLabeler(bitmap)
            }.onSuccess { result ->
                _uiState.value = _uiState.value.copy(result = result, isRunning = false)
            }.onFailure { t ->
                _uiState.value = _uiState.value.copy(
                    isRunning = false,
                    error = t.message ?: "Classification failed",
                )
            }
        }
    }

    private suspend fun runLabeler(bitmap: Bitmap): MlKitResult {
        val image = InputImage.fromBitmap(bitmap, 0)
        val start = System.currentTimeMillis()
        val labels: List<ImageLabel> = labeler.process(image).await()
        val elapsed = System.currentTimeMillis() - start

        val all = labels.map { MlKitLabel(it.text, it.confidence) }
            .sortedByDescending { it.confidence }

        val matched = all.firstOrNull { it.text in PLANT_LABELS }
        return MlKitResult(
            verdict = if (matched != null) MlKitVerdict.PLANT else MlKitVerdict.NOT_PLANT,
            matchedLabel = matched,
            allLabels = all,
            inferenceTimeMs = elapsed,
        )
    }

    override fun onCleared() {
        labeler.close()
        super.onCleared()
    }

    companion object {
        private const val MIN_CONFIDENCE = 0.5f

        // Labels in ML Kit's default on-device model that imply "plant".
        // See https://developers.google.com/ml-kit/vision/image-labeling/label-map
        private val PLANT_LABELS: Set<String> = setOf(
            "Plant",
            "Flower",
            "Tree",
            "Leaf",
            "Houseplant",
            "Vegetable",
            "Fruit",
            "Petal",
            "Grass",
            "Flora",
            "Garden",
            "Bonsai",
            "Cactus",
            "Bouquet",
            "Branch",
            "Forest",
        )
    }
}