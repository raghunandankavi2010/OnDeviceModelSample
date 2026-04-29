package com.example.ondevicemodelsample.data

import androidx.room.Dao
import androidx.room.Insert
import androidx.room.Query
import kotlinx.coroutines.flow.Flow

@Dao
interface FeedbackDao {

    @Insert
    suspend fun insert(entry: FeedbackEntity): Long

    @Query(
        "SELECT COUNT(*) AS total, " +
            "COALESCE(SUM(CASE WHEN isCorrect THEN 1 ELSE 0 END), 0) AS correct " +
            "FROM feedback"
    )
    fun observeStats(): Flow<FeedbackStats>
}