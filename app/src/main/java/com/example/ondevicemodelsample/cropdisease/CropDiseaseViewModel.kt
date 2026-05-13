package com.example.ondevicemodelsample.cropdisease

import android.app.Application
import android.graphics.Bitmap
import android.net.Uri
import android.util.Log
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.example.ondevicemodelsample.util.BitmapUtils
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

data class CropDiseaseUiState(
    val imageUri: Uri? = null,
    val result: CropDiseaseResult? = null,
    val isRunning: Boolean = false,
    val error: String? = null,
    val modelMissing: Boolean = false,
)

class CropDiseaseViewModel(application: Application) : AndroidViewModel(application) {

    private val _uiState = MutableStateFlow(CropDiseaseUiState())
    val uiState: StateFlow<CropDiseaseUiState> = _uiState.asStateFlow()

    private val classifier: CropDiseaseClassifier? =
        CropDiseaseClassifier.createOrNull(application)

    init {
        if (classifier == null) {
            _uiState.value = _uiState.value.copy(modelMissing = true)
        }
    }

    fun classify(uri: Uri) {
        val runner = classifier
        if (runner == null) {
            _uiState.value = _uiState.value.copy(
                imageUri = uri,
                modelMissing = true,
            )
            return
        }

        _uiState.value = CropDiseaseUiState(imageUri = uri, isRunning = true)

        viewModelScope.launch {
            runCatching {
                val bitmap: Bitmap = withContext(Dispatchers.Default) {
                    BitmapUtils.decodeSampled(getApplication(), uri)
                        ?: error("Unable to decode image")
                }
                withContext(Dispatchers.Default) { runner.classify(bitmap) }
            }.onSuccess { result ->
                _uiState.value = _uiState.value.copy(result = result, isRunning = false)
            }.onFailure { t ->
                Log.e(TAG, "Classification failed", t)
                _uiState.value = _uiState.value.copy(
                    isRunning = false,
                    error = t.message ?: "Classification failed",
                )
            }
        }
    }

    override fun onCleared() {
        classifier?.close()
        super.onCleared()
    }

    companion object {
        private const val TAG = "CropDiseaseViewModel"
    }
}