package com.example.ondevicemodelsample.ml

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Paint
import android.graphics.Rect
import android.os.Debug
import android.os.Process
import android.util.Log
import org.tensorflow.lite.Interpreter
import java.io.Closeable
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

data class Classification(val label: String, val confidence: Float)

enum class Verdict { PLANT_RECOGNIZED, PLANT_UNCERTAIN, NOT_PLANT }

data class ClassificationResult(
    val verdict: Verdict,
    val plantType: String?,
    val condition: String?,
    val confidence: Float,
    val summary: String,
    val plantPredictions: List<Classification>,
    val gatePredictions: List<Classification>,
    val performance: ModelPerformance,
)

private class TfliteRunner(
    private val interpreter: Interpreter,
    val labels: List<String>,
    val inputWidth: Int,
    val inputHeight: Int,
    private val mean: Float,
    private val std: Float,
) : Closeable {

    private val inputBuffer: ByteBuffer = ByteBuffer
        .allocateDirect(1 * inputHeight * inputWidth * 3 * 4)
        .order(ByteOrder.nativeOrder())

    private val pixels = IntArray(inputWidth * inputHeight)

    fun run(bitmap: Bitmap): FloatArray {
        val rgb = if (bitmap.config == Bitmap.Config.ARGB_8888) bitmap
        else bitmap.copy(Bitmap.Config.ARGB_8888, false)
        val resized = centerCropAndResize(rgb, inputWidth, inputHeight)
        resized.getPixels(pixels, 0, inputWidth, 0, 0, inputWidth, inputHeight)

        inputBuffer.rewind()
        for (pixel in pixels) {
            val r = (pixel shr 16) and 0xFF
            val g = (pixel shr 8) and 0xFF
            val b = pixel and 0xFF
            inputBuffer.putFloat((r - mean) / std)
            inputBuffer.putFloat((g - mean) / std)
            inputBuffer.putFloat((b - mean) / std)
        }

        val outShape = interpreter.getOutputTensor(0).shape()
        val numClasses = outShape[1]
        val output = Array(1) { FloatArray(numClasses) }
        inputBuffer.rewind()
        interpreter.run(inputBuffer, output)
        return output[0]
    }

    fun topK(scores: FloatArray, k: Int): List<Classification> {
        val limit = minOf(scores.size, labels.size)
        return (0 until limit)
            .sortedByDescending { scores[it] }
            .take(k)
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
        private val bilinearPaint = Paint(Paint.FILTER_BITMAP_FLAG or Paint.ANTI_ALIAS_FLAG)
    }
}

class ImageClassifier private constructor(
    private val gate: TfliteRunner,
    private val plant: TfliteRunner,
) : Closeable {

    fun classify(bitmap: Bitmap): ClassificationResult {
        val startTime = System.currentTimeMillis()
        val startCpu = Process.getElapsedCpuTime()
        val startHeap = usedHeapBytes()
        val startNative = Debug.getNativeHeapAllocatedSize()

        // Stage 1: ImageNet gate runs first. It's a 1001-class model that fires confidently
        // on dogs/cars/people, which lets us reject non-plants before they reach the
        // PlantVillage softmax (which is OOD for anything non-leaf and would otherwise
        // hallucinate a disease label with high confidence).
        val gateScores = gate.run(bitmap)
        val gateTop3 = gate.topK(gateScores, 3)
        val gateTop = gateTop3.first()
        val gateLabel = gateTop.label.trim().lowercase()
        val gateIsPlantLike = gateLabel in PLANT_ALLOWLIST

        val verdict: Verdict
        val plantType: String?
        val condition: String?
        val confidence: Float
        val summary: String
        var plantTop3: List<Classification> = emptyList()

        if (!gateIsPlantLike && gateTop.confidence >= GATE_REJECT_THRESHOLD) {
            verdict = Verdict.NOT_PLANT
            plantType = null
            condition = null
            confidence = gateTop.confidence
            summary = "Doesn't look like a plant. Top guess: ${gateTop.label}."
        } else {
            val plantScores = plant.run(bitmap)
            plantTop3 = plant.topK(plantScores, 3)
            val plantTop = plantTop3.first()

            if (gateIsPlantLike && gateTop.confidence >= GATE_PLANT_THRESHOLD) {
                if (plantTop.confidence >= PLANT_ACCEPT_THRESHOLD) {
                    val (crop, cond) = prettyLabel(plantTop.label)
                    verdict = Verdict.PLANT_RECOGNIZED
                    plantType = crop
                    condition = cond
                    confidence = plantTop.confidence
                    summary = "$crop · $cond"
                } else {
                    val gateGuess = gateTop.label.replaceFirstChar { it.uppercase() }
                    verdict = Verdict.PLANT_UNCERTAIN
                    plantType = gateGuess
                    condition = null
                    confidence = gateTop.confidence
                    summary = "Looks like a plant ($gateGuess) — no confident disease match."
                }
            } else {
                // Gate is ambiguous (low-confidence, neither clearly plant nor non-plant —
                // typical for close-up leaf shots that don't match any ImageNet class).
                // Require a much higher PlantVillage threshold here.
                if (plantTop.confidence >= STRICT_PLANT_THRESHOLD) {
                    val (crop, cond) = prettyLabel(plantTop.label)
                    verdict = Verdict.PLANT_RECOGNIZED
                    plantType = crop
                    condition = cond
                    confidence = plantTop.confidence
                    summary = "$crop · $cond"
                } else {
                    verdict = Verdict.NOT_PLANT
                    plantType = null
                    condition = null
                    confidence = maxOf(gateTop.confidence, plantTop.confidence)
                    summary = "Doesn't look like a plant."
                }
            }
        }

        val endTime = System.currentTimeMillis()
        val endCpu = Process.getElapsedCpuTime()
        val endHeap = usedHeapBytes()
        val endNative = Debug.getNativeHeapAllocatedSize()

        val wallMs = (endTime - startTime).coerceAtLeast(0L)
        val cpuMs = (endCpu - startCpu).coerceAtLeast(0L)
        val cores = Runtime.getRuntime().availableProcessors().coerceAtLeast(1)
        val utilizationPct = if (wallMs > 0) {
            (cpuMs.toDouble() / (wallMs.toDouble() * cores)) * 100.0
        } else 0.0
        val performance = ModelPerformance(
            inferenceTimeMs = wallMs,
            cpuTimeMs = cpuMs,
            cpuUtilizationPercent = utilizationPct,
            heapUsedMb = endHeap / BYTES_PER_MB,
            heapDeltaMb = (endHeap - startHeap) / BYTES_PER_MB,
            nativeUsedMb = endNative / BYTES_PER_MB,
            nativeDeltaMb = (endNative - startNative) / BYTES_PER_MB,
        )

        Log.d(
            TAG,
            "verdict=$verdict plant=$plantType cond=$condition conf=$confidence " +
                    "gateTop=${gateTop.label}@${gateTop.confidence} " +
                    "wall=${performance.inferenceTimeMs}ms cpu=${performance.cpuTimeMs}ms " +
                    "util=${"%.0f".format(performance.cpuUtilizationPercent)}%/" +
                    "${cores}c heap=${"%.1f".format(performance.heapUsedMb)}MB " +
                    "(Δ${"%+.2f".format(performance.heapDeltaMb)}) " +
                    "native=${"%.1f".format(performance.nativeUsedMb)}MB " +
                    "(Δ${"%+.2f".format(performance.nativeDeltaMb)})",
        )

        return ClassificationResult(
            verdict = verdict,
            plantType = plantType,
            condition = condition,
            confidence = confidence,
            summary = summary,
            plantPredictions = plantTop3,
            gatePredictions = gateTop3,
            performance = performance,
        )
    }

    override fun close() {
        gate.close()
        plant.close()
    }

    private fun usedHeapBytes(): Long {
        val r = Runtime.getRuntime()
        return r.totalMemory() - r.freeMemory()
    }

    companion object {
        private const val TAG = "ImageClassifier"
        private const val BYTES_PER_MB = 1024.0 * 1024.0

        private const val PLANT_MODEL = "plant_disease_model.tflite"
        private const val PLANT_LABELS = "plant_labels.txt"
        private const val GATE_MODEL = "model.tflite"
        private const val GATE_LABELS = "labels.txt"

        // PlantVillage MobileNetV2 was trained with pixel/255 normalization
        private const val PLANT_MEAN = 0f
        private const val PLANT_STD = 255f

        // ImageNet MobileNetV2 expects [-1, 1]
        private const val GATE_MEAN = 127.5f
        private const val GATE_STD = 127.5f

        // PlantVillage softmax must clear this when the gate has confirmed plant-like
        private const val PLANT_ACCEPT_THRESHOLD = 0.55f

        // When the gate is ambiguous (low-conf or non-allowlist), require this from
        // PlantVillage. PlantVillage's softmax always sums to 1 even on OOD input, so
        // this needs to be above where dogs/faces tend to peak (~0.5-0.6).
        private const val STRICT_PLANT_THRESHOLD = 0.75f

        // Gate top-1 must clear this for an allowlist plant class to count as "plant"
        private const val GATE_PLANT_THRESHOLD = 0.30f

        // If the gate's top-1 is non-plant with at least this confidence, we reject
        // immediately. Raised from 0.35 → 0.55 so wide-scene camera shots (where the
        // gate's top-1 is something random like "menu" or "table" at ~0.4) can still
        // fall through to PlantVillage.
        private const val GATE_REJECT_THRESHOLD = 0.55f

        private val PLANT_ALLOWLIST = setOf(
            "head cabbage", "broccoli", "cauliflower", "zucchini",
            "spaghetti squash", "acorn squash", "butternut squash",
            "cucumber", "artichoke", "bell pepper", "cardoon",
            "strawberry", "orange", "lemon", "fig", "pineapple",
            "banana", "jackfruit", "custard apple", "pomegranate",
            "rapeseed", "daisy", "yellow lady's slipper", "corn",
            "acorn", "hip", "buckeye",
            "mushroom", "coral fungus", "agaric", "gyromitra",
            "stinkhorn", "earthstar", "hen-of-the-woods", "bolete",
        )

        fun create(context: Context): ImageClassifier {
            val plant = buildRunner(context, PLANT_MODEL, PLANT_LABELS, PLANT_MEAN, PLANT_STD)
            val gate = buildRunner(context, GATE_MODEL, GATE_LABELS, GATE_MEAN, GATE_STD)
            return ImageClassifier(gate, plant)
        }

        private fun buildRunner(
            context: Context,
            modelAsset: String,
            labelsAsset: String,
            mean: Float,
            std: Float,
        ): TfliteRunner {
            val buffer = loadAsset(context, modelAsset)
            val opts = Interpreter.Options().apply { setNumThreads(4) }
            val interpreter = Interpreter(buffer, opts)
            val shape = interpreter.getInputTensor(0).shape() // [1, H, W, 3]
            val h = shape[1]
            val w = shape[2]
            val labels = context.assets.open(labelsAsset).bufferedReader()
                .useLines { it.toList() }
            return TfliteRunner(interpreter, labels, w, h, mean, std)
        }

        private fun loadAsset(context: Context, name: String): MappedByteBuffer {
            val fd = context.assets.openFd(name)
            fd.createInputStream().channel.use { ch ->
                return ch.map(FileChannel.MapMode.READ_ONLY, fd.startOffset, fd.declaredLength)
            }
        }
    }
}

private val CROP_PREFIXES = listOf(
    "cherry including sour" to "Cherry",
    "pepper bell" to "Bell Pepper",
    "corn maize" to "Corn (Maize)",
    "apple" to "Apple",
    "blueberry" to "Blueberry",
    "grape" to "Grape",
    "orange" to "Orange",
    "peach" to "Peach",
    "potato" to "Potato",
    "raspberry" to "Raspberry",
    "soybean" to "Soybean",
    "squash" to "Squash",
    "strawberry" to "Strawberry",
    "tomato" to "Tomato",
)

private fun prettyLabel(raw: String): Pair<String, String> {
    val low = raw.trim().lowercase()
    val match = CROP_PREFIXES.firstOrNull { low.startsWith(it.first) }
        ?: return "Plant" to raw.replaceFirstChar { it.uppercase() }
    val rest = low.removePrefix(match.first).trim()
    val condition = when {
        rest.isEmpty() || rest == "healthy" -> "Healthy"
        else -> rest.replaceFirstChar { it.uppercase() }
    }
    return match.second to condition
}