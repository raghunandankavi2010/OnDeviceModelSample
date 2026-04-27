package com.example.ondevicemodelsample.util

import android.app.ActivityManager
import android.content.Context
import android.os.Build

data class DeviceInfo(
    val manufacturer: String,
    val model: String,
    val androidVersion: String,
    val apiLevel: Int,
    val primaryAbi: String,
    val cores: Int,
    val totalRamMb: Long,
    val maxHeapMb: Long,
) {
    val displayName: String get() = "$manufacturer $model"

    companion object {
        fun collect(context: Context): DeviceInfo {
            val am = context.getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager
            val memInfo = ActivityManager.MemoryInfo()
            am.getMemoryInfo(memInfo)
            return DeviceInfo(
                manufacturer = Build.MANUFACTURER,
                model = Build.MODEL,
                androidVersion = Build.VERSION.RELEASE,
                apiLevel = Build.VERSION.SDK_INT,
                primaryAbi = Build.SUPPORTED_ABIS.firstOrNull() ?: "unknown",
                cores = Runtime.getRuntime().availableProcessors(),
                totalRamMb = memInfo.totalMem / BYTES_PER_MB,
                maxHeapMb = Runtime.getRuntime().maxMemory() / BYTES_PER_MB,
            )
        }

        private const val BYTES_PER_MB = 1024L * 1024L
    }
}