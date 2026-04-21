package com.example.ondevicemodelsample.ml

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Paint
import android.graphics.Rect
import org.tensorflow.lite.Interpreter
import java.io.Closeable
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

data class Classification(val label: String, val confidence: Float)

class ImageClassifier private constructor(
    private val interpreter: Interpreter,
    private val labels: List<String>,
    private val inputWidth: Int,
    private val inputHeight: Int,
) : Closeable {

    private val inputBuffer: ByteBuffer = ByteBuffer
        .allocateDirect(1 * inputHeight * inputWidth * 3 * 4)
        .order(ByteOrder.nativeOrder())

    private val pixels = IntArray(inputWidth * inputHeight)

    fun classify(bitmap: Bitmap, topK: Int = 3): List<Classification> {
        val rgb = if (bitmap.config == Bitmap.Config.ARGB_8888) bitmap
        else bitmap.copy(Bitmap.Config.ARGB_8888, false)

        val resized = centerCropAndResize(rgb, inputWidth, inputHeight)
        resized.getPixels(pixels, 0, inputWidth, 0, 0, inputWidth, inputHeight)

        inputBuffer.rewind()
        for (pixel in pixels) {
            val r = (pixel shr 16) and 0xFF
            val g = (pixel shr 8) and 0xFF
            val b = pixel and 0xFF
            inputBuffer.putFloat((r - MEAN) / STD)
            inputBuffer.putFloat((g - MEAN) / STD)
            inputBuffer.putFloat((b - MEAN) / STD)
        }

        val outputShape = interpreter.getOutputTensor(0).shape()
        val numClasses = outputShape[1]
        val output = Array(1) { FloatArray(numClasses) }

        inputBuffer.rewind()
        interpreter.run(inputBuffer, output)

        val scores = output[0]
        val limit = minOf(scores.size, labels.size)
        return (0 until limit)
            .sortedByDescending { scores[it] }
            .take(topK)
            .map { Classification(labels[it], scores[it]) }
    }

    private fun centerCropAndResize(src: Bitmap, dstW: Int, dstH: Int): Bitmap {
        val side = minOf(src.width, src.height)
        val left = (src.width - side) / 2
        val top = (src.height - side) / 2
        val out = Bitmap.createBitmap(dstW, dstH, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(out)
        val srcRect = Rect(left, top, left + side, top + side)
        val dstRect = Rect(0, 0, dstW, dstH)
        canvas.drawBitmap(src, srcRect, dstRect, bilinearPaint)
        return out
    }

    override fun close() {
        interpreter.close()
    }

    companion object {
        private const val MEAN = 127.5f
        private const val STD = 127.5f
        private const val MODEL_FILE = "model.tflite"
        private const val LABELS_FILE = "labels.txt"

        private val bilinearPaint = Paint(Paint.FILTER_BITMAP_FLAG or Paint.ANTI_ALIAS_FLAG)

        fun create(context: Context): ImageClassifier {
            val modelBuffer = loadModelFile(context)
            val options = Interpreter.Options().apply { setNumThreads(4) }
            val interpreter = Interpreter(modelBuffer, options)

            val inputShape = interpreter.getInputTensor(0).shape()
            val height = inputShape[1]
            val width = inputShape[2]

            val labels = context.assets.open(LABELS_FILE).bufferedReader()
                .useLines { it.toList() }

            return ImageClassifier(interpreter, labels, width, height)
        }

        private fun loadModelFile(context: Context): MappedByteBuffer {
            val fd = context.assets.openFd(MODEL_FILE)
            fd.createInputStream().channel.use { channel ->
                return channel.map(FileChannel.MapMode.READ_ONLY, fd.startOffset, fd.declaredLength)
            }
        }
    }
}