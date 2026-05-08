package com.example.ondevicemodelsample.util

import android.content.ContentValues
import android.content.Context
import android.net.Uri
import android.os.Build
import android.os.Environment
import android.provider.MediaStore
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

object MediaStoreUtils {

    private const val ALBUM = "OnDeviceModelSample"

    fun createGalleryImageUri(context: Context): Uri? {
        val name = "OnDevice_${SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date())}.jpg"
        val resolver = context.contentResolver
        val values = ContentValues().apply {
            put(MediaStore.MediaColumns.DISPLAY_NAME, name)
            put(MediaStore.MediaColumns.MIME_TYPE, "image/jpeg")
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                put(
                    MediaStore.MediaColumns.RELATIVE_PATH,
                    "${Environment.DIRECTORY_PICTURES}/$ALBUM",
                )
                put(MediaStore.MediaColumns.IS_PENDING, 1)
            } else {
                @Suppress("DEPRECATION")
                val picturesDir = Environment.getExternalStoragePublicDirectory(
                    Environment.DIRECTORY_PICTURES,
                )
                val albumDir = File(picturesDir, ALBUM).apply { mkdirs() }
                @Suppress("DEPRECATION")
                put(MediaStore.MediaColumns.DATA, File(albumDir, name).absolutePath)
            }
        }
        return resolver.insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, values)
    }

    fun finalizeGalleryImage(context: Context, uri: Uri) {
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.Q) return
        val values = ContentValues().apply {
            put(MediaStore.MediaColumns.IS_PENDING, 0)
        }
        runCatching { context.contentResolver.update(uri, values, null, null) }
    }

    fun discardGalleryImage(context: Context, uri: Uri) {
        runCatching { context.contentResolver.delete(uri, null, null) }
    }
}