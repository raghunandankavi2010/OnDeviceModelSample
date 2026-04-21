package com.example.ondevicemodelsample

import android.app.Application
import android.graphics.Bitmap
import android.net.Uri
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.example.ondevicemodelsample.ml.Classification
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
    val results: List<Classification> = emptyList(),
    val isRunning: Boolean = false,
    val error: String? = null,
    val inferenceMs: Long? = null,
)

class ClassifierViewModel(application: Application) : AndroidViewModel(application) {

    private val _uiState = MutableStateFlow(ClassifierUiState())
    val uiState: StateFlow<ClassifierUiState> = _uiState.asStateFlow()

    private var classifier: ImageClassifier? = null

    private fun getOrCreateClassifier(): ImageClassifier {
        return classifier ?: ImageClassifier.create(getApplication()).also { classifier = it }
    }

    fun classify(uri: Uri) {
        _uiState.value = _uiState.value.copy(
            imageUri = uri,
            results = emptyList(),
            isRunning = true,
            error = null,
            inferenceMs = null,
        )
        viewModelScope.launch {
            runCatching {
                withContext(Dispatchers.Default) {
                    val bitmap: Bitmap = BitmapUtils.decodeSampled(getApplication(), uri)
                        ?: error("Unable to decode image")
                    val engine = getOrCreateClassifier()
                    val start = System.currentTimeMillis()
                    val results = engine.classify(bitmap, topK = 3)
                    val elapsed = System.currentTimeMillis() - start
                    results to elapsed
                }
            }.onSuccess { (results, elapsed) ->
                _uiState.value = _uiState.value.copy(
                    results = results,
                    isRunning = false,
                    inferenceMs = elapsed,
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