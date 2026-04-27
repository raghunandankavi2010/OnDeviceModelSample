package com.example.ondevicemodelsample

import android.app.Application
import android.graphics.Bitmap
import android.net.Uri
import androidx.core.net.toUri
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.example.ondevicemodelsample.ml.ClassificationResult
import com.example.ondevicemodelsample.ml.ImageClassifier
import com.example.ondevicemodelsample.util.BitmapUtils
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

data class ClassifierUiState(
    val imageUri: Uri? = null,
    val result: ClassificationResult? = null,
    val isRunning: Boolean = false,
    val error: String? = null,
)

class ClassifierViewModel(application: Application) : AndroidViewModel(application) {

    private val _uiState = MutableStateFlow(ClassifierUiState())
    val uiState: StateFlow<ClassifierUiState> = _uiState.asStateFlow()

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

    override fun onCleared() {
        classifier?.close()
        classifier = null
        super.onCleared()
    }
}