package com.example.ondevicemodelsample

import android.app.Application
import android.graphics.Bitmap
import android.net.Uri
import androidx.core.net.toUri
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.example.ondevicemodelsample.data.FeedbackDatabase
import com.example.ondevicemodelsample.data.FeedbackEntity
import com.example.ondevicemodelsample.data.FeedbackStats
import com.example.ondevicemodelsample.ml.ClassificationResult
import com.example.ondevicemodelsample.ml.ImageClassifier
import com.example.ondevicemodelsample.util.BitmapUtils
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.SharingStarted
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.stateIn
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

data class ClassifierUiState(
    val imageUri: Uri? = null,
    val result: ClassificationResult? = null,
    val isRunning: Boolean = false,
    val error: String? = null,
    val feedbackGiven: Boolean = false,
)

class ClassifierViewModel(application: Application) : AndroidViewModel(application) {

    private val _uiState = MutableStateFlow(ClassifierUiState())
    val uiState: StateFlow<ClassifierUiState> = _uiState.asStateFlow()

    private val feedbackDao = FeedbackDatabase.get(application).feedbackDao()

    val feedbackStats: StateFlow<FeedbackStats> = feedbackDao.observeStats()
        .stateIn(viewModelScope, SharingStarted.Eagerly, FeedbackStats())

    private var classifier: ImageClassifier? = null

    private fun getOrCreateClassifier(): ImageClassifier {
        return classifier ?: ImageClassifier.create(getApplication()).also { classifier = it }
    }

    fun classifyAsset(assetPath: String) {
        val uri = "file:///android_asset/$assetPath".toUri()
        _uiState.value = _uiState.value.copy(
            imageUri = uri,
            result = null,
            isRunning = true,
            error = null,
            feedbackGiven = false,
        )
        viewModelScope.launch {
            runCatching {
                withContext(Dispatchers.Default) {
                    val bitmap: Bitmap = BitmapUtils.decodeAsset(getApplication(), assetPath)
                        ?: error("Unable to decode asset $assetPath")
                    getOrCreateClassifier().classify(bitmap)
                }
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

    fun classify(uri: Uri) {
        _uiState.value = _uiState.value.copy(
            imageUri = uri,
            result = null,
            isRunning = true,
            error = null,
            feedbackGiven = false,
        )
        viewModelScope.launch {
            runCatching {
                withContext(Dispatchers.Default) {
                    val bitmap: Bitmap = BitmapUtils.decodeSampled(getApplication(), uri)
                        ?: error("Unable to decode image")
                    getOrCreateClassifier().classify(bitmap)
                }
            }.onSuccess { result ->
                _uiState.value = _uiState.value.copy(
                    result = result,
                    isRunning = false,
                )
            }.onFailure { t ->
                _uiState.value = _uiState.value.copy(
                    isRunning = false,
                    error = t.message ?: "Classification failed",
                )
            }
        }
    }

    fun submitFeedback(isCorrect: Boolean) {
        val state = _uiState.value
        val result = state.result ?: return
        if (state.feedbackGiven) return
        _uiState.value = state.copy(feedbackGiven = true)
        viewModelScope.launch {
            feedbackDao.insert(
                FeedbackEntity(
                    verdict = result.verdict.name,
                    predictedLabel = result.summary,
                    confidence = result.confidence,
                    isCorrect = isCorrect,
                    timestamp = System.currentTimeMillis(),
                )
            )
        }
    }

    override fun onCleared() {
        classifier?.close()
        classifier = null
        super.onCleared()
    }
}