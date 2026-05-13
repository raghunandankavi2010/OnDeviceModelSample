package com.example.ondevicemodelsample.cropdisease

import android.content.Context
import android.content.res.AssetManager
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Paint
import android.graphics.Rect
import android.os.Process
import android.util.Log
import org.tensorflow.lite.Interpreter
import java.io.Closeable
import java.io.FileNotFoundException
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import kotlin.math.exp
import kotlin.math.max

data class CropDiseasePrediction(
    val rawLabel: String,
    val crop: String,
    val condition: String,
    val probability: Float,
)

data class CropDiseaseResult(
    val top: CropDiseasePrediction,
    val topK: List<CropDiseasePrediction>,
    val inferenceTimeMs: Long,
    val cpuTimeMs: Long,
)

class CropDiseaseModelMissingException :
    RuntimeException(
        "crop_disease_model.tflite is not bundled. Run " +
                "`python tools/convert_crop_disease_to_tflite.py` and rebuild."
    )

class CropDiseaseClassifier private constructor(
    private val interpreter: Interpreter,
    private val labels: List<String>,
    private val inputHeight: Int,
    private val inputWidth: Int,
    private val isNchw: Boolean,
    private val numClasses: Int,
) : Closeable {

    private val pixels = IntArray(inputWidth * inputHeight)
    private val inputBuffer: ByteBuffer = ByteBuffer
        .allocateDirect(1 * 3 * inputHeight * inputWidth * 4)
        .order(ByteOrder.nativeOrder())

    fun classify(bitmap: Bitmap, topK: Int = DEFAULT_TOP_K): CropDiseaseResult {
        val startWall = System.currentTimeMillis()
        val startCpu = Process.getElapsedCpuTime()

        val rgb = if (bitmap.config == Bitmap.Config.ARGB_8888) bitmap
        else bitmap.copy(Bitmap.Config.ARGB_8888, false)
        val resized = centerCropAndResize(rgb, inputWidth, inputHeight)
        writeInput(resized)

        val output = Array(1) { FloatArray(numClasses) }
        interpreter.run(inputBuffer, output)
        val probs = softmax(output[0])
        val ranked = rank(probs, topK)

        val wall = (System.currentTimeMillis() - startWall).coerceAtLeast(0L)
        val cpu = (Process.getElapsedCpuTime() - startCpu).coerceAtLeast(0L)

        Log.d(
            TAG,
            "top=${ranked.first().rawLabel}@${"%.3f".format(ranked.first().probability)} " +
                    "wall=${wall}ms cpu=${cpu}ms",
        )

        return CropDiseaseResult(
            top = ranked.first(),
            topK = ranked,
            inferenceTimeMs = wall,
            cpuTimeMs = cpu,
        )
    }

    private fun writeInput(bitmap: Bitmap) {
        bitmap.getPixels(pixels, 0, inputWidth, 0, 0, inputWidth, inputHeight)
        inputBuffer.rewind()

        if (isNchw) {
            // [1, 3, H, W] — write all R, then all G, then all B.
            for (channel in 0 until 3) {
                val mean = IMAGENET_MEAN[channel]
                val std = IMAGENET_STD[channel]
                for (px in pixels) {
                    val raw = when (channel) {
                        0 -> (px shr 16) and 0xFF
                        1 -> (px shr 8) and 0xFF
                        else -> px and 0xFF
                    }
                    inputBuffer.putFloat((raw / 255f - mean) / std)
                }
            }
        } else {
            // [1, H, W, 3] — interleaved RGB.
            for (px in pixels) {
                val r = (px shr 16) and 0xFF
                val g = (px shr 8) and 0xFF
                val b = px and 0xFF
                inputBuffer.putFloat((r / 255f - IMAGENET_MEAN[0]) / IMAGENET_STD[0])
                inputBuffer.putFloat((g / 255f - IMAGENET_MEAN[1]) / IMAGENET_STD[1])
                inputBuffer.putFloat((b / 255f - IMAGENET_MEAN[2]) / IMAGENET_STD[2])
            }
        }
        inputBuffer.rewind()
    }

    private fun softmax(logits: FloatArray): FloatArray {
        var maxLogit = Float.NEGATIVE_INFINITY
        for (v in logits) maxLogit = max(maxLogit, v)
        var sum = 0.0
        val expd = DoubleArray(logits.size)
        for (i in logits.indices) {
            val e = exp((logits[i] - maxLogit).toDouble())
            expd[i] = e
            sum += e
        }
        val out = FloatArray(logits.size)
        for (i in logits.indices) out[i] = (expd[i] / sum).toFloat()
        return out
    }

    private fun rank(probs: FloatArray, topK: Int): List<CropDiseasePrediction> {
        val limit = minOf(probs.size, labels.size)
        return (0 until limit)
            .sortedByDescending { probs[it] }
            .take(topK)
            .map { idx ->
                val (crop, condition) = splitLabel(labels[idx])
                CropDiseasePrediction(
                    rawLabel = labels[idx],
                    crop = crop,
                    condition = condition,
                    probability = probs[idx],
                )
            }
    }

    private fun centerCropAndResize(src: Bitmap, dstW: Int, dstH: Int): Bitmap {
        val side = minOf(src.width, src.height)
        val left = (src.width - side) / 2
        val top = (src.height - side) / 2
        val out = Bitmap.createBitmap(dstW, dstH, Bitmap.Config.ARGB_8888)
        Canvas(out).drawBitmap(
            src,
            Rect(left, top, left + side, top + side),
            Rect(0, 0, dstW, dstH),
            bilinearPaint,
        )
        return out
    }

    override fun close() {
        interpreter.close()
    }

    companion object {
        private const val TAG = "CropDiseaseClassifier"
        private const val DEFAULT_TOP_K = 5
        private const val MODEL_ASSET = "crop_disease_model.tflite"
        private const val LABELS_ASSET = "crop_disease_labels.txt"

        private val IMAGENET_MEAN = floatArrayOf(0.485f, 0.456f, 0.406f)
        private val IMAGENET_STD = floatArrayOf(0.229f, 0.224f, 0.225f)

        private val bilinearPaint = Paint(Paint.FILTER_BITMAP_FLAG or Paint.ANTI_ALIAS_FLAG)

        /**
         * Returns null if the model asset isn't bundled — callers can show a
         * friendly "run the converter" message instead of crashing.
         */
        fun createOrNull(context: Context): CropDiseaseClassifier? = try {
            create(context)
        } catch (_: CropDiseaseModelMissingException) {
            null
        }

        fun create(context: Context): CropDiseaseClassifier {
            val assets = context.assets
            val buffer = loadModelAsset(assets, MODEL_ASSET)
                ?: throw CropDiseaseModelMissingException()

            val options = Interpreter.Options().apply { setNumThreads(4) }
            val interpreter = Interpreter(buffer, options)

            val inputShape = interpreter.getInputTensor(0).shape() // [N, ?, ?, ?]
            val (isNchw, h, w) = parseInputShape(inputShape)
            val outputShape = interpreter.getOutputTensor(0).shape()
            val numClasses = outputShape.last()

            val labels = loadLabels(assets, LABELS_ASSET, numClasses)
            return CropDiseaseClassifier(interpreter, labels, h, w, isNchw, numClasses)
        }

        private fun parseInputShape(shape: IntArray): Triple<Boolean, Int, Int> {
            // Possible layouts: [1, 3, H, W] (NCHW) or [1, H, W, 3] (NHWC).
            return when {
                shape.size == 4 && shape[1] == 3 -> Triple(true, shape[2], shape[3])
                shape.size == 4 && shape[3] == 3 -> Triple(false, shape[1], shape[2])
                else -> error("Unexpected input shape ${shape.contentToString()}")
            }
        }

        private fun loadModelAsset(assets: AssetManager, name: String): MappedByteBuffer? {
            return try {
                val fd = assets.openFd(name)
                fd.createInputStream().channel.use { ch ->
                    ch.map(FileChannel.MapMode.READ_ONLY, fd.startOffset, fd.declaredLength)
                }
            } catch (_: FileNotFoundException) {
                null
            }
        }

        private fun loadLabels(
            assets: AssetManager,
            name: String,
            expected: Int,
        ): List<String> {
            val labels = try {
                assets.open(name).bufferedReader().useLines { lines ->
                    lines.map { it.trim() }.filter { it.isNotEmpty() }.toList()
                }
            } catch (_: FileNotFoundException) {
                FALLBACK_LABELS
            }
            if (labels.size != expected) {
                Log.w(
                    TAG,
                    "Labels count ${labels.size} doesn't match model output $expected",
                )
            }
            return labels
        }

        // Mirrors tools/convert_crop_disease_to_tflite.py CLASS_LABELS so the app
        // still names predictions if the labels asset is missing.
        private val FALLBACK_LABELS = listOf(
            "Corn___Common_Rust",
            "Corn___Gray_Leaf_Spot",
            "Corn___Northern_Leaf_Blight",
            "Corn___Healthy",
            "Potato___Early_Blight",
            "Potato___Late_Blight",
            "Potato___Healthy",
            "Rice___Brown_Spot",
            "Rice___Leaf_Blast",
            "Rice___Neck_Blast",
            "Rice___Healthy",
            "Wheat___Yellow_Rust",
            "Wheat___Brown_Rust",
            "Wheat___Healthy",
            "Sugarcane___Red_Rot",
            "Sugarcane___Bacterial_Blight",
            "Sugarcane___Healthy",
        )
    }
}

internal fun splitLabel(raw: String): Pair<String, String> {
    val parts = raw.split("___", limit = 2)
    val crop = parts.getOrNull(0)?.replace('_', ' ')?.trim().orEmpty()
    val condition = parts.getOrNull(1)?.replace('_', ' ')?.trim().orEmpty()
    return crop to (condition.ifEmpty { "Unknown" })
}