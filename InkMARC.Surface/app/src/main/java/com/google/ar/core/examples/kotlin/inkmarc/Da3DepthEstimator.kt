package com.inkmarc

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OnnxJavaType
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import ai.onnxruntime.TensorInfo
import android.media.Image
import android.util.Log
import androidx.lifecycle.DefaultLifecycleObserver
import androidx.lifecycle.LifecycleOwner
import com.google.ar.core.Frame
import com.google.ar.core.exceptions.NotYetAvailableException
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import kotlin.math.sqrt
import kotlin.math.abs
import java.util.Collections
import java.util.EnumSet
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean

class Da3DepthEstimator(private val activity: InkMarcActivity) : DefaultLifecycleObserver {
  companion object {
    private const val TAG = "Da3DepthEstimator"
    private const val FRAME_INTERVAL_NS = 200_000_000L
    private const val MIN_RANGE_EPSILON = 1e-8f
    private const val DA3_PATCH_SIZE = 14
    private val IMAGENET_MEAN = floatArrayOf(0.485f, 0.456f, 0.406f)
    private val IMAGENET_STD = floatArrayOf(0.229f, 0.224f, 0.225f)
  }

  private enum class Da3InputMode {
    LEGACY_UINT8_NHWC,
    PREPROCESSED_FLOAT_NCHW,
    PREPROCESSED_FLOAT_NNCHW,
  }

  data class Da3DepthResult(
    val timestampNs: Long,
    val rawDepth: FloatArray,
    val normalizedDepth: FloatArray,
    val width: Int,
    val height: Int,
    val modelInputWidth: Int,
    val modelInputHeight: Int,
    val inferenceMillis: Double,
  )

  private data class PendingRequest(
    val frameData: CameraFrameData,
    val timestampNs: Long,
  )

  private data class Da3ModelSpec(
    val modelPath: String,
    val inputName: String,
    val outputName: String,
    val inputShape: LongArray,
    val inputMode: Da3InputMode,
    val inputWidth: Int,
    val inputHeight: Int,
    val processRes: Int,
  )

  private val executor: ExecutorService = Executors.newSingleThreadExecutor()
  private val inferenceInFlight = AtomicBoolean(false)
  private val pendingRequestLock = Any()

  @Volatile private var latestResult: Da3DepthResult? = null
  @Volatile private var ortEnvironment: OrtEnvironment? = null
  @Volatile private var ortSession: OrtSession? = null
  @Volatile private var modelSpec: Da3ModelSpec? = null
  @Volatile private var lastAcceptedTimestampNs = Long.MIN_VALUE
  @Volatile private var hasLoggedMissingModel = false
  private var pendingRequest: PendingRequest? = null

  override fun onCreate(owner: LifecycleOwner) {
    executor.execute {
      try {
        initializeSessionIfNeeded()
      } catch (e: Exception) {
        Log.e(TAG, "DA3 session initialization failed", e)
      }
    }
  }

  override fun onDestroy(owner: LifecycleOwner) {
    latestResult = null
    ortSession?.close()
    ortSession = null
    ortEnvironment = null
    executor.shutdown()
  }

  fun submitFrame(frame: Frame) {
    if (!canSubmit(frame.timestamp)) return
    val image =
      try {
        frame.acquireCameraImage()
      } catch (_: NotYetAvailableException) {
        return
      } catch (e: Exception) {
        Log.w(TAG, "Skipping DA3 frame: ${e.message}")
        return
      }

    image.use {
      submitCameraImage(it, frame.timestamp)
    }
  }

  fun submitCameraImage(
    image: Image,
    timestampNs: Long,
  ) {
    if (!canSubmit(timestampNs)) return
    val nextRequest =
      PendingRequest(
        frameData = CameraFrameData.fromImage(image),
        timestampNs = timestampNs,
      )
    val accepted =
      synchronized(pendingRequestLock) {
        if (!shouldAcceptTimestampLocked(timestampNs)) {
          false
        } else {
          pendingRequest = nextRequest
          lastAcceptedTimestampNs = timestampNs
          true
        }
      }
    if (!accepted) return

    Log.i(TAG, "submitCameraImage accepted timestamp=$timestampNs")
    scheduleInferenceIfNeeded()
  }

  fun getLatestDepthResult(): Da3DepthResult? = latestResult

  fun canSubmit(timestampNs: Long): Boolean =
    synchronized(pendingRequestLock) { shouldAcceptTimestampLocked(timestampNs) }

  private fun scheduleInferenceIfNeeded() {
    if (!inferenceInFlight.compareAndSet(false, true)) return
    executor.execute {
      try {
        initializeSessionIfNeeded() ?: return@execute
        while (true) {
          val request =
            synchronized(pendingRequestLock) {
              pendingRequest.also { pendingRequest = null }
            } ?: return@execute
          try {
            Log.i(TAG, "Starting per-frame DA3 inference timestamp=${request.timestampNs}")
            val da3Result = runInference(request.frameData, request.timestampNs) ?: continue
            latestResult = da3Result
            Log.i(
              TAG,
              "Completed per-frame DA3 inference timestamp=${request.timestampNs} depth=${da3Result.width}x${da3Result.height} ms=${da3Result.inferenceMillis}"
            )
          } catch (e: Exception) {
            Log.e(TAG, "DA3 inference failed", e)
          }
        }
      } finally {
        inferenceInFlight.set(false)
        val hasPending =
          synchronized(pendingRequestLock) {
            pendingRequest != null
          }
        if (hasPending) {
          scheduleInferenceIfNeeded()
        }
      }
    }
  }

  private fun shouldAcceptTimestampLocked(timestampNs: Long): Boolean {
    if (
      lastAcceptedTimestampNs != Long.MIN_VALUE &&
        timestampNs - lastAcceptedTimestampNs < FRAME_INTERVAL_NS
    ) {
      return false
    }
    if (ortSession != null) return true

    val modelPath = findModelAssetPath()
    if (modelPath == null && !hasLoggedMissingModel) {
      hasLoggedMissingModel = true
      Log.w(TAG, "No DA3 Small ONNX asset found in assets/models.")
    }
    return modelPath != null
  }

  private fun initializeSessionIfNeeded(): OrtSession? {
    ortSession?.let { return it }
    val modelPath = findModelAssetPath() ?: return null

    return synchronized(this) {
      ortSession?.let { return@synchronized it }

      val env = ortEnvironment ?: OrtEnvironment.getEnvironment().also { ortEnvironment = it }
      val modelBytes = activity.assets.open(modelPath).use { it.readBytes() }
      val session = createDa3Session(env, modelBytes)

      val resolvedInputName =
        session.inputNames.firstOrNull()
          ?: throw IllegalStateException("DA3 ONNX model has no inputs.")
      val resolvedOutputName =
        session.outputNames.firstOrNull { it.contains("depth", ignoreCase = true) }
          ?: session.outputNames.firstOrNull()
          ?: throw IllegalStateException("DA3 ONNX model has no outputs.")
      val inputInfo =
        session.inputInfo[resolvedInputName]?.info as? TensorInfo
          ?: throw IllegalStateException("DA3 ONNX input metadata is unavailable.")
      val inputShape = inputInfo.shape
      val inputMode =
        when {
          inputInfo.type == OnnxJavaType.UINT8 && inputShape.size == 4 -> Da3InputMode.LEGACY_UINT8_NHWC
          inputInfo.type == OnnxJavaType.FLOAT && inputShape.size == 4 -> Da3InputMode.PREPROCESSED_FLOAT_NCHW
          inputInfo.type == OnnxJavaType.FLOAT && inputShape.size == 5 -> Da3InputMode.PREPROCESSED_FLOAT_NNCHW
          else ->
            throw IllegalStateException(
              "Unsupported DA3 input contract type=${inputInfo.type} shape=${inputShape.contentToString()}"
            )
        }
      val inputHeight =
        when (inputMode) {
          Da3InputMode.LEGACY_UINT8_NHWC -> inputShape[1].toInt()
          Da3InputMode.PREPROCESSED_FLOAT_NCHW -> inputShape[2].toInt()
          Da3InputMode.PREPROCESSED_FLOAT_NNCHW -> inputShape[3].toInt()
        }
      val inputWidth =
        when (inputMode) {
          Da3InputMode.LEGACY_UINT8_NHWC -> inputShape[2].toInt()
          Da3InputMode.PREPROCESSED_FLOAT_NCHW -> inputShape[3].toInt()
          Da3InputMode.PREPROCESSED_FLOAT_NNCHW -> inputShape[4].toInt()
        }
      val processRes = minOf(inputHeight, inputWidth)

      modelSpec =
        Da3ModelSpec(
          modelPath = modelPath,
          inputName = resolvedInputName,
          outputName = resolvedOutputName,
          inputShape = inputShape.copyOf(),
          inputMode = inputMode,
          inputWidth = inputWidth,
          inputHeight = inputHeight,
          processRes = processRes,
        )
      ortSession = session
      Log.i(
        TAG,
        "Initialized DA3 session from $modelPath input=$resolvedInputName shape=${inputInfo.shape.contentToString()} type=${inputInfo.type} output=$resolvedOutputName mode=$inputMode",
      )
      session
    }
  }

  private fun createDa3Session(env: OrtEnvironment, modelBytes: ByteArray): OrtSession {
    val qnnOptions = OrtSession.SessionOptions()
    qnnOptions.setIntraOpNumThreads(1)
    qnnOptions.setInterOpNumThreads(1)
    qnnOptions.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
    return try {
      qnnOptions.addQnn(mapOf("backend_type" to "htp"))
      Log.i(TAG, "DA3 attempting QNN HTP execution provider with CPU fallback enabled")
      env.createSession(modelBytes, qnnOptions)
    } catch (e: Exception) {
      Log.w(TAG, "DA3 QNN HTP initialization failed, falling back to CPU", e)
      qnnOptions.close()
      val cpuOptions = OrtSession.SessionOptions()
      cpuOptions.setIntraOpNumThreads(1)
      cpuOptions.setInterOpNumThreads(1)
      cpuOptions.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
      Log.i(TAG, "DA3 using CPU execution provider")
      env.createSession(modelBytes, cpuOptions)
    }
  }

  private fun findModelAssetPath(): String? {
    val modelFiles = activity.assets.list("models") ?: return null
    val preferredOrder =
      listOf(
        "depth_proxy_504x322_qnn_qdq_u16a_u8w.onnx",
        "da3_small_depth_504x322_int8_matmul.onnx",
        "da3_small_depth_504x322.onnx",
        "da3_small_depth_504x322_qnn_qdq_u16a_u8w_conv_matmul.onnx",
        "da3_small.onnx",
      )
    return preferredOrder.firstOrNull { candidate -> modelFiles.any { it.equals(candidate, ignoreCase = true) } }
      ?.let { "models/$it" }
      ?: modelFiles.firstOrNull {
        it.endsWith(".onnx", ignoreCase = true) &&
          it.contains("da3", ignoreCase = true) &&
          it.contains("small", ignoreCase = true)
      }?.let { "models/$it" }
  }

  private fun runInference(frameData: CameraFrameData, timestampNs: Long): Da3DepthResult? {
    val session = ortSession ?: return null
    val env = ortEnvironment ?: return null
    val spec = modelSpec ?: return null

    val startedAtNs = System.nanoTime()
    val rgbMat = frameData.toRgbMat()
    try {
      return when (spec.inputMode) {
        Da3InputMode.LEGACY_UINT8_NHWC ->
          runLegacyUint8Inference(session, env, spec, rgbMat, timestampNs, startedAtNs)
        Da3InputMode.PREPROCESSED_FLOAT_NCHW ->
          runPreprocessedFloatInference(session, env, spec, rgbMat, timestampNs, startedAtNs)
        Da3InputMode.PREPROCESSED_FLOAT_NNCHW ->
          runPreprocessedFloatInference(session, env, spec, rgbMat, timestampNs, startedAtNs)
      }
    } finally {
      rgbMat.release()
    }
  }

  private fun runLegacyUint8Inference(
    session: OrtSession,
    env: OrtEnvironment,
    spec: Da3ModelSpec,
    rgbMat: Mat,
    timestampNs: Long,
    startedAtNs: Long,
  ): Da3DepthResult? {
    val modelInput = Mat()
    try {
      Imgproc.resize(
        rgbMat,
        modelInput,
        Size(spec.inputWidth.toDouble(), spec.inputHeight.toDouble()),
        0.0,
        0.0,
        Imgproc.INTER_LINEAR,
      )
      val inputBytes = ByteArray(spec.inputWidth * spec.inputHeight * 3)
      modelInput.get(0, 0, inputBytes)
      val inputBuffer =
        ByteBuffer.allocateDirect(inputBytes.size)
          .order(ByteOrder.nativeOrder())
          .put(inputBytes)
      inputBuffer.rewind()
      OnnxTensor.createTensor(
        env,
        inputBuffer,
        longArrayOf(1, spec.inputHeight.toLong(), spec.inputWidth.toLong(), 3),
        OnnxJavaType.UINT8,
      ).use { inputTensor ->
        return runSessionAndDecode(
          session = session,
          inputName = spec.inputName,
          outputName = spec.outputName,
          inputTensor = inputTensor,
          timestampNs = timestampNs,
          modelInputWidth = spec.inputWidth,
          modelInputHeight = spec.inputHeight,
          startedAtNs = startedAtNs,
        )
      }
    } finally {
      modelInput.release()
    }
  }

  private fun runPreprocessedFloatInference(
    session: OrtSession,
    env: OrtEnvironment,
    spec: Da3ModelSpec,
    rgbMat: Mat,
    timestampNs: Long,
    startedAtNs: Long,
  ): Da3DepthResult? {
    val preprocessed = preprocessDa3FloatInput(rgbMat, spec) ?: return null
    OnnxTensor.createTensor(env, FloatBuffer.wrap(preprocessed), spec.inputShape).use { inputTensor ->
      return runSessionAndDecode(
        session = session,
        inputName = spec.inputName,
        outputName = spec.outputName,
        inputTensor = inputTensor,
        timestampNs = timestampNs,
        modelInputWidth = spec.inputWidth,
        modelInputHeight = spec.inputHeight,
        startedAtNs = startedAtNs,
      )
    }
  }

  private fun runSessionAndDecode(
    session: OrtSession,
    inputName: String,
    outputName: String,
    inputTensor: OnnxTensor,
    timestampNs: Long,
    modelInputWidth: Int,
    modelInputHeight: Int,
    startedAtNs: Long,
  ): Da3DepthResult? {
    session.run(Collections.singletonMap(inputName, inputTensor)).use { outputs ->
      val outputTensor = outputs[0] as? OnnxTensor
      if (outputTensor == null) {
        Log.w(TAG, "DA3 output '$outputName' was not a tensor in session results.")
        return null
      }
      outputTensor.use {
        val outputInfo = outputTensor.info as? TensorInfo ?: return null
        val extractedDepth = extractDepth(outputTensor.floatBuffer, outputInfo.shape)
        if (extractedDepth == null) {
          Log.w(
            TAG,
            "Unable to decode DA3 output shape=${outputInfo.shape.contentToString()} type=${outputInfo.type}",
          )
          return null
        }
        val depth = extractedDepth.values
        val height = extractedDepth.height
        val width = extractedDepth.width
        val normalizedDepth = percentileNormalize(depth)
        return Da3DepthResult(
          timestampNs = timestampNs,
          rawDepth = depth,
          normalizedDepth = normalizedDepth,
          width = width,
          height = height,
          modelInputWidth = modelInputWidth,
          modelInputHeight = modelInputHeight,
          inferenceMillis = (System.nanoTime() - startedAtNs) / 1_000_000.0,
        )
      }
    }
  }

  private fun preprocessDa3FloatInput(rgbMat: Mat, spec: Da3ModelSpec): FloatArray? {
    val resized = resizeLowerBound(rgbMat, spec.processRes)
    val divisible = makeDivisibleByResize(resized, DA3_PATCH_SIZE)
    try {
      val finalMat =
        if (divisible.rows() == spec.inputHeight && divisible.cols() == spec.inputWidth) {
          divisible
        } else {
          val exact = Mat()
          Imgproc.resize(
            divisible,
            exact,
            Size(spec.inputWidth.toDouble(), spec.inputHeight.toDouble()),
            0.0,
            0.0,
            if (spec.inputWidth > divisible.cols() || spec.inputHeight > divisible.rows()) {
              Imgproc.INTER_CUBIC
            } else {
              Imgproc.INTER_AREA
            },
          )
          Log.w(
            TAG,
            "DA3 preprocess produced ${divisible.cols()}x${divisible.rows()}, resizing exactly to ${spec.inputWidth}x${spec.inputHeight} for model compatibility.",
          )
          exact
        }
      try {
        return matToDa3FloatInput(finalMat, spec.inputMode)
      } finally {
        if (finalMat !== divisible) {
          finalMat.release()
        }
      }
    } finally {
      resized.release()
      divisible.release()
    }
  }

  private fun resizeLowerBound(rgbMat: Mat, processRes: Int): Mat {
    val width = rgbMat.cols()
    val height = rgbMat.rows()
    val shortest = minOf(width, height)
    if (shortest <= 0) return rgbMat.clone()
    val scale = processRes.toDouble() / shortest.toDouble()
    if (abs(scale - 1.0) < 1e-6) return rgbMat.clone()
    val newWidth = maxOf(1, kotlin.math.round(width * scale).toInt())
    val newHeight = maxOf(1, kotlin.math.round(height * scale).toInt())
    val resized = Mat()
    Imgproc.resize(
      rgbMat,
      resized,
      Size(newWidth.toDouble(), newHeight.toDouble()),
      0.0,
      0.0,
      if (scale > 1.0) Imgproc.INTER_CUBIC else Imgproc.INTER_AREA,
    )
    return resized
  }

  private fun makeDivisibleByResize(rgbMat: Mat, patchSize: Int): Mat {
    fun nearestMultiple(value: Int): Int {
      val down = (value / patchSize) * patchSize
      val up = down + patchSize
      return if (abs(up - value) <= abs(value - down)) up else down
    }

    val newWidth = maxOf(1, nearestMultiple(rgbMat.cols()))
    val newHeight = maxOf(1, nearestMultiple(rgbMat.rows()))
    if (newWidth == rgbMat.cols() && newHeight == rgbMat.rows()) {
      return rgbMat.clone()
    }
    val resized = Mat()
    Imgproc.resize(
      rgbMat,
      resized,
      Size(newWidth.toDouble(), newHeight.toDouble()),
      0.0,
      0.0,
      if (newWidth > rgbMat.cols() || newHeight > rgbMat.rows()) Imgproc.INTER_CUBIC else Imgproc.INTER_AREA,
    )
    return resized
  }

  private fun matToDa3FloatInput(rgbMat: Mat): FloatArray {
    return matToDa3FloatNnchwInput(rgbMat)
  }

  private fun matToDa3FloatInput(rgbMat: Mat, inputMode: Da3InputMode): FloatArray =
    when (inputMode) {
      Da3InputMode.PREPROCESSED_FLOAT_NCHW -> matToDa3FloatNchwInput(rgbMat)
      Da3InputMode.PREPROCESSED_FLOAT_NNCHW -> matToDa3FloatNnchwInput(rgbMat)
      else -> error("Float DA3 preprocessing requested for non-float mode $inputMode")
    }

  private fun matToDa3FloatNnchwInput(rgbMat: Mat): FloatArray {
    val width = rgbMat.cols()
    val height = rgbMat.rows()
    val rgbBytes = ByteArray(width * height * 3)
    rgbMat.get(0, 0, rgbBytes)
    val planeSize = width * height
    val input = FloatArray(1 * 1 * 3 * planeSize)
    for (index in 0 until planeSize) {
      val base = index * 3
      val r = (rgbBytes[base].toInt() and 0xFF) / 255f
      val g = (rgbBytes[base + 1].toInt() and 0xFF) / 255f
      val b = (rgbBytes[base + 2].toInt() and 0xFF) / 255f
      input[index] = (r - IMAGENET_MEAN[0]) / IMAGENET_STD[0]
      input[planeSize + index] = (g - IMAGENET_MEAN[1]) / IMAGENET_STD[1]
      input[2 * planeSize + index] = (b - IMAGENET_MEAN[2]) / IMAGENET_STD[2]
    }
    return input
  }

  private fun matToDa3FloatNchwInput(rgbMat: Mat): FloatArray {
    val width = rgbMat.cols()
    val height = rgbMat.rows()
    val rgbBytes = ByteArray(width * height * 3)
    rgbMat.get(0, 0, rgbBytes)
    val planeSize = width * height
    val input = FloatArray(3 * planeSize)
    for (index in 0 until planeSize) {
      val base = index * 3
      val r = (rgbBytes[base].toInt() and 0xFF) / 255f
      val g = (rgbBytes[base + 1].toInt() and 0xFF) / 255f
      val b = (rgbBytes[base + 2].toInt() and 0xFF) / 255f
      input[index] = (r - IMAGENET_MEAN[0]) / IMAGENET_STD[0]
      input[planeSize + index] = (g - IMAGENET_MEAN[1]) / IMAGENET_STD[1]
      input[2 * planeSize + index] = (b - IMAGENET_MEAN[2]) / IMAGENET_STD[2]
    }
    return input
  }

  private data class ExtractedDepth(
    val values: FloatArray,
    val width: Int,
    val height: Int,
  )

  private fun extractDepth(buffer: java.nio.FloatBuffer, shape: LongArray): ExtractedDepth? {
    val output = FloatArray(buffer.remaining())
    buffer.duplicate().get(output)
    if (output.isEmpty()) return null

    val resolvedHeight = resolveOutputHeight(shape, output.size)
    val resolvedWidth = resolveOutputWidth(shape, output.size)
    if (resolvedHeight <= 0 || resolvedWidth <= 0) return null

    val count = resolvedHeight * resolvedWidth
    if (output.size < count) return null
    return ExtractedDepth(
      values = output.copyOfRange(0, count),
      width = resolvedWidth,
      height = resolvedHeight,
    )
  }

  private fun resolveOutputHeight(shape: LongArray, outputCount: Int): Int {
    val candidate = shape.getOrNull(shape.size - 2)?.toInt() ?: -1
    if (candidate > 0) return candidate
    return inferSquareOutputSize(outputCount)
  }

  private fun resolveOutputWidth(shape: LongArray, outputCount: Int): Int {
    val candidate = shape.getOrNull(shape.size - 1)?.toInt() ?: -1
    if (candidate > 0) return candidate
    return inferSquareOutputSize(outputCount)
  }

  private fun inferSquareOutputSize(outputCount: Int): Int {
    modelSpec?.let { spec ->
      if (outputCount == spec.inputWidth * spec.inputHeight) {
        return spec.inputWidth
      }
    }
    val side = sqrt(outputCount.toDouble()).toInt()
    return if (side * side == outputCount) side else -1
  }

  private fun percentileNormalize(depthValues: FloatArray): FloatArray {
    if (depthValues.isEmpty()) return depthValues
    val sorted = depthValues.copyOf()
    sorted.sort()
    val p02 = percentileFromSorted(sorted, 0.02f)
    val p98 = percentileFromSorted(sorted, 0.98f)
    val denominator = kotlin.math.max(p98 - p02, MIN_RANGE_EPSILON)
    return FloatArray(depthValues.size) { index ->
      ((depthValues[index] - p02) / denominator).coerceIn(0f, 1f)
    }
  }

  private fun percentileFromSorted(sortedValues: FloatArray, percentile: Float): Float {
    val lastIndex = sortedValues.lastIndex
    if (lastIndex <= 0) return sortedValues.first()
    val position = percentile.coerceIn(0f, 1f) * lastIndex
    val lowerIndex = position.toInt()
    val upperIndex = minOf(lastIndex, lowerIndex + 1)
    if (lowerIndex == upperIndex) return sortedValues[lowerIndex]
    val weight = position - lowerIndex
    return sortedValues[lowerIndex] * (1f - weight) + sortedValues[upperIndex] * weight
  }
}
