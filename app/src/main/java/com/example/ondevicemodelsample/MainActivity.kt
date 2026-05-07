package com.example.ondevicemodelsample

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.TopAppBar
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import com.example.ondevicemodelsample.mlkit.MlKitPlantScreen
import com.example.ondevicemodelsample.ui.theme.OnDeviceModelSampleTheme

private enum class Screen { Classifier, MlKit }

class MainActivity : ComponentActivity() {

    @OptIn(ExperimentalMaterial3Api::class)
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContent {
            OnDeviceModelSampleTheme {
                var screen by remember { mutableStateOf(Screen.Classifier) }
                Scaffold(
                    modifier = Modifier.fillMaxSize(),
                    topBar = {
                        if (screen == Screen.Classifier) {
                            TopAppBar(
                                title = { Text("On-device Classifier") },
                                actions = {
                                    TextButton(onClick = { screen = Screen.MlKit }) {
                                        Text("ML Kit")
                                    }
                                },
                            )
                        }
                    },
                ) { innerPadding ->
                    when (screen) {
                        Screen.Classifier -> ClassifierScreen(contentPadding = innerPadding)
                        Screen.MlKit -> MlKitPlantScreen(
                            contentPadding = innerPadding,
                            onBack = { screen = Screen.Classifier },
                        )
                    }
                }
            }
        }
    }
}