package com.example.ondevicemodelsample.data

import android.content.Context
import androidx.room.Database
import androidx.room.Room
import androidx.room.RoomDatabase

@Database(entities = [FeedbackEntity::class], version = 1, exportSchema = false)
abstract class FeedbackDatabase : RoomDatabase() {
    abstract fun feedbackDao(): FeedbackDao

    companion object {
        @Volatile
        private var INSTANCE: FeedbackDatabase? = null

        fun get(context: Context): FeedbackDatabase {
            return INSTANCE ?: synchronized(this) {
                INSTANCE ?: Room.databaseBuilder(
                    context.applicationContext,
                    FeedbackDatabase::class.java,
                    "feedback.db",
                ).build().also { INSTANCE = it }
            }
        }
    }
}