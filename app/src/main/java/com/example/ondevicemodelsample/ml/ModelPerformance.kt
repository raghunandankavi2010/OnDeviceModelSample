package com.example.ondevicemodelsample.ml

data class ModelPerformance(
    val inferenceTimeMs: Long,
    val memoryUsageMb: Double,
    val cpuTimeNs: Long = 0L
)
