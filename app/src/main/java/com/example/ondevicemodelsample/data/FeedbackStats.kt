package com.example.ondevicemodelsample.data

data class FeedbackStats(
    val total: Int = 0,
    val correct: Int = 0,
) {
    val incorrect: Int get() = total - correct
    val accuracyPercent: Float
        get() = if (total == 0) 0f else (correct.toFloat() / total) * 100f
}