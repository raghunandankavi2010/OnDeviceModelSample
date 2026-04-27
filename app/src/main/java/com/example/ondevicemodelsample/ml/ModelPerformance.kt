package com.example.ondevicemodelsample.ml

data class ModelPerformance(
    val inferenceTimeMs: Long,
    val cpuTimeMs: Long,
    val cpuUtilizationPercent: Double,
    val heapUsedMb: Double,
    val heapDeltaMb: Double,
    val nativeUsedMb: Double,
    val nativeDeltaMb: Double,
)