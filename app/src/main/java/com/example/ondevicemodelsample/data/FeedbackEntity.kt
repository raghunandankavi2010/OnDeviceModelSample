package com.example.ondevicemodelsample.data

import androidx.room.Entity
import androidx.room.PrimaryKey

@Entity(tableName = "feedback")
data class FeedbackEntity(
    @PrimaryKey(autoGenerate = true) val id: Long = 0,
    val verdict: String,
    val predictedLabel: String,
    val confidence: Float,
    val isCorrect: Boolean,
    val timestamp: Long,
)