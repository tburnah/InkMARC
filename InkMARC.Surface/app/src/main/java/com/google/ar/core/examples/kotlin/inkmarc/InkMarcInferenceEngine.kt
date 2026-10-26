package com.inkmarc

import android.media.Image
import android.util.Log
import androidx.lifecycle.DefaultLifecycleObserver
import androidx.lifecycle.LifecycleOwner
import com.google.ar.core.Coordinates2d
import com.google.ar.core.Frame
import com.google.ar.core.exceptions.NotYetAvailableException
import org.opencv.core.Mat
import org.opencv.core.Rect
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean
import kotlin.math.exp
import kotlin.math.max
import kotlin.math.roundToInt

class InkMarcInferenceEngine(private val activity: InkMarcActivity) : DefaultLifecycleObserver {
  companion object {
    private const val TAG = "InkMarcInference"
    private const val FRAME_INTERVAL_NS = 200_000_000L
    private const val OUTPUT_WIDTH = 432
    private const val OUTPUT_HEIGHT = 672
    private const val PRESENCE_THRESHOLD = 0.36f
    private const val TOUCH_RAW_THRESHOLD = 0.28f
    private const val TOUCH_HYSTERESIS_ON = 0.60f
    private const val TOUCH_HYSTERESIS_OFF = 0.10f
    private const val TOUCH_MIN_ON_FRAMES = 1
    private const val TOUCH_MIN_OFF_FRAMES = 1
  }

  data class InkMarcResult(
    val presenceProbability: Float,
    val presentRaw: Boolean,
    val touchProbability: Float,
    val touchingRaw: Boolean,
    val touchingHysteresis: Boolean,
    val xPx: Float,
    val yPx: Float,
    val xyNormalized: FloatArray,
  )

  private data class PendingRequest(
    val frameData: CameraFrameData,
    val timestampNs: Long,
    val surfaceCornersImage: FloatArray?,
  )

  private data class RgbPreparation(
    val red: FloatArray,
    val green: FloatArray,
    val blue: FloatArray,
    val transform: ResizeCropTransform,
  )

  private data class ResizeCropTransform(
    val scale: Float,
    val scaledWidth: Int,
    val scaledHeight: Int,
    val cropLeft: Int,
    val cropTop: Int,
  )

  private val runner = InkMarcRunner(activity.assets)
  private val surfaceGeometryBuilder = SurfaceGeometryBuilder()
  private val executor: ExecutorService = Executors.newSingleThreadExecutor()
  private val inferenceInFlight = AtomicBoolean(false)
  private val pendingRequestLock = Any()

  @Volatile private var latestResult: InkMarcResult? = null
  @Volatile private var lastAcceptedTimestampNs = Long.MIN_VALUE
  @Volatile private var hasLoggedMissingModel = false
  private var pendingRequest: PendingRequest? = null
  private var touchingHysteresisState = false
  private var aboveOnThresholdFrames = 0
  private var belowOffThresholdFrames = 0

  override fun onCreate(owner: LifecycleOwner) {
    executor.execute {
      if (!runner.hasModelAsset()) {
        logMissingModel()
      }
    }
  }

  override fun onDestroy(owner: LifecycleOwner) {
    latestResult = null
    pendingRequest = null
    runner.close()
    executor.shutdown()
  }

  fun submitFrame(frame: Frame, surfaceCornersView: FloatArray? = null) {
    if (!canSubmit(frame.timestamp)) return
    val image =
      try {
        frame.acquireCameraImage()
      } catch (_: NotYetAvailableException) {
        return
      } catch (e: Exception) {
        Log.w(TAG, "Skipping inference frame: ${e.message}")
        return
      }

    image.use {
      submitCameraImage(
        image = it,
        timestampNs = frame.timestamp,
        frame = frame,
        surfaceCornersView = surfaceCornersView,
      )
    }
  }

  fun submitCameraImage(
    image: Image,
    timestampNs: Long,
    frame: Frame? = null,
    surfaceCornersView: FloatArray? = null,
    surfaceCornersImage: FloatArray? = null,
  ) {
    if (!canSubmit(timestampNs)) return
    val nextRequest =
      PendingRequest(
        frameData = CameraFrameData.fromImage(image),
        timestampNs = timestampNs,
        surfaceCornersImage =
          surfaceCornersImage ?: projectSurfaceCornersToImage(frame, surfaceCornersView),
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

  fun canSubmit(timestampNs: Long): Boolean =
    synchronized(pendingRequestLock) { shouldAcceptTimestampLocked(timestampNs) }

  private fun scheduleInferenceIfNeeded() {
    if (!inferenceInFlight.compareAndSet(false, true)) return
    executor.execute {
      try {
        while (true) {
          val request =
            synchronized(pendingRequestLock) {
              pendingRequest.also { pendingRequest = null }
            } ?: return@execute
          try {
            runInference(
              cameraFrameData = request.frameData,
              surfaceCornersImage = request.surfaceCornersImage,
            )
            activity.updateInferenceCursor()
          } catch (e: Exception) {
            Log.e(TAG, "InkTrack inference failed timestamp=${request.timestampNs}", e)
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

  private fun runInference(
    cameraFrameData: CameraFrameData,
    surfaceCornersImage: FloatArray?,
  ): InkMarcResult? {
    Log.i(TAG, "InkMarcInference started")
    val rgb = prepareRgb(cameraFrameData) ?: return null
    val surfaceCornersOutput = mapCornersToOutput(surfaceCornersImage, rgb.transform)
    val geometry = surfaceGeometryBuilder.build(surfaceCornersOutput)

    logMinMax("rgb_r", rgb.red)
    logMinMax("rgb_g", rgb.green)
    logMinMax("rgb_b", rgb.blue)
    logMinMax("surface_mask", geometry.mask)
    logMinMax("surface_u", geometry.u)
    logMinMax("surface_v", geometry.v)

    val inputTensor = packInput(rgb.red, rgb.green, rgb.blue, geometry)
    logTensorChannels(rgb.red, rgb.green, rgb.blue, geometry.mask, geometry.u, geometry.v)

    val outputs = runner.run(inputTensor) ?: return null
    Log.i(TAG, "InkTrack runner returned outputs")
    val presenceProbability = softmaxSecond(outputs.stateLogits)
    val presentRaw = presenceProbability >= PRESENCE_THRESHOLD
    val touchLogit = outputs.touchLogits.firstOrNull() ?: 0f
    val touchProbability = sigmoid(touchLogit)
    val touchingRaw = presentRaw && touchProbability >= TOUCH_RAW_THRESHOLD
    val touchingHysteresis = updateTouchHysteresis(presentRaw, touchProbability)

    val xyNormalized = outputs.xy.copyOf()
    val xPx = xyNormalized.getOrElse(0) { 0f } * OUTPUT_WIDTH
    val yPx = xyNormalized.getOrElse(1) { 0f } * OUTPUT_HEIGHT

    Log.d(
      TAG,
      "presence logits=${outputs.stateLogits.joinToString()} prob=$presenceProbability present=$presentRaw",
    )
    Log.d(TAG, "touch logit=$touchLogit prob=$touchProbability raw=$touchingRaw hyst=$touchingHysteresis")
    Log.d(TAG, "xy normalized=${xyNormalized.joinToString()} px=($xPx,$yPx)")

    return InkMarcResult(
      presenceProbability = presenceProbability,
      presentRaw = presentRaw,
      touchProbability = touchProbability,
      touchingRaw = touchingRaw,
      touchingHysteresis = touchingHysteresis,
      xPx = xPx,
      yPx = yPx,
      xyNormalized = xyNormalized,
    ).also {
      latestResult = it
      Log.i(
        TAG,
        "InkTrack inference finished present=$presentRaw touch=$touchingHysteresis xy=($xPx,$yPx)"
      )
    }
  }

  fun getLatestResult(): InkMarcResult? = latestResult

  private fun prepareRgb(cameraFrameData: CameraFrameData): RgbPreparation? {
    val rgbMat = cameraFrameData.toRgbMat()
    try {
      val sourceWidth = rgbMat.cols()
      val sourceHeight = rgbMat.rows()
      if (sourceWidth <= 0 || sourceHeight <= 0) return null

      val scale =
        max(
          OUTPUT_WIDTH.toFloat() / sourceWidth.toFloat(),
          OUTPUT_HEIGHT.toFloat() / sourceHeight.toFloat()
        )
      val scaledWidth = max(1, (sourceWidth * scale).roundToInt())
      val scaledHeight = max(1, (sourceHeight * scale).roundToInt())
      val cropLeft = max(0, (scaledWidth - OUTPUT_WIDTH) / 2)
      val cropTop = max(0, (scaledHeight - OUTPUT_HEIGHT) / 2)

      val resized = Mat()
      try {
        Imgproc.resize(
          rgbMat,
          resized,
          Size(scaledWidth.toDouble(), scaledHeight.toDouble()),
          0.0,
          0.0,
          if (scale >= 1f) Imgproc.INTER_LINEAR else Imgproc.INTER_AREA,
        )
        val cropped = resized.submat(Rect(cropLeft, cropTop, OUTPUT_WIDTH, OUTPUT_HEIGHT)).clone()
        try {
          val rgbBytes = ByteArray(OUTPUT_WIDTH * OUTPUT_HEIGHT * 3)
          cropped.get(0, 0, rgbBytes)
          val red = FloatArray(OUTPUT_WIDTH * OUTPUT_HEIGHT)
          val green = FloatArray(red.size)
          val blue = FloatArray(red.size)
          for (index in red.indices) {
            val base = index * 3
            red[index] = (rgbBytes[base].toInt() and 0xFF) / 255f
            green[index] = (rgbBytes[base + 1].toInt() and 0xFF) / 255f
            blue[index] = (rgbBytes[base + 2].toInt() and 0xFF) / 255f
          }
          return RgbPreparation(
            red = red,
            green = green,
            blue = blue,
            transform =
              ResizeCropTransform(
                scale = scale,
                scaledWidth = scaledWidth,
                scaledHeight = scaledHeight,
                cropLeft = cropLeft,
                cropTop = cropTop,
              )
          )
        } finally {
          cropped.release()
        }
      } finally {
        resized.release()
      }
    } finally {
      rgbMat.release()
    }
  }

  private fun mapCornersToOutput(
    surfaceCornersImage: FloatArray?,
    transform: ResizeCropTransform,
  ): FloatArray? {
    if (surfaceCornersImage == null || surfaceCornersImage.size < 8) return null
    return FloatArray(8).also { mapped ->
      for (index in 0 until 4) {
        val x = surfaceCornersImage[index * 2] * transform.scale - transform.cropLeft
        val y = surfaceCornersImage[index * 2 + 1] * transform.scale - transform.cropTop
        mapped[index * 2] = x.coerceIn(0f, (OUTPUT_WIDTH - 1).toFloat())
        mapped[index * 2 + 1] = y.coerceIn(0f, (OUTPUT_HEIGHT - 1).toFloat())
      }
    }
  }

  private fun packInput(
    red: FloatArray,
    green: FloatArray,
    blue: FloatArray,
    geometry: SurfaceGeometryBuilder.SurfaceGeometry,
  ): FloatArray {
    val planeSize = OUTPUT_WIDTH * OUTPUT_HEIGHT
    val tensor = FloatArray(6 * planeSize)
    red.copyInto(tensor, 0 * planeSize)
    green.copyInto(tensor, 1 * planeSize)
    blue.copyInto(tensor, 2 * planeSize)
    geometry.mask.copyInto(tensor, 3 * planeSize)
    geometry.u.copyInto(tensor, 4 * planeSize)
    geometry.v.copyInto(tensor, 5 * planeSize)
    return tensor
  }

  private fun softmaxSecond(logits: FloatArray): Float {
    if (logits.size < 2) return 0f
    val maxLogit = max(logits[0], logits[1])
    val exp0 = exp((logits[0] - maxLogit).toDouble()).toFloat()
    val exp1 = exp((logits[1] - maxLogit).toDouble()).toFloat()
    return exp1 / (exp0 + exp1)
  }

  private fun sigmoid(value: Float): Float = (1.0 / (1.0 + exp((-value).toDouble()))).toFloat()

  private fun updateTouchHysteresis(present: Boolean, touchProbability: Float): Boolean {
    if (!present) {
      touchingHysteresisState = false
      aboveOnThresholdFrames = 0
      belowOffThresholdFrames = 0
      return false
    }

    if (!touchingHysteresisState) {
      if (touchProbability >= TOUCH_HYSTERESIS_ON) {
        aboveOnThresholdFrames += 1
        if (aboveOnThresholdFrames >= TOUCH_MIN_ON_FRAMES) {
          touchingHysteresisState = true
          aboveOnThresholdFrames = 0
        }
      } else {
        aboveOnThresholdFrames = 0
      }
      belowOffThresholdFrames = 0
      return touchingHysteresisState
    }

    if (touchProbability <= TOUCH_HYSTERESIS_OFF) {
      belowOffThresholdFrames += 1
      if (belowOffThresholdFrames >= TOUCH_MIN_OFF_FRAMES) {
        touchingHysteresisState = false
        belowOffThresholdFrames = 0
      }
    } else {
      belowOffThresholdFrames = 0
    }
    aboveOnThresholdFrames = 0
    return touchingHysteresisState
  }

  private fun logTensorChannels(
    red: FloatArray,
    green: FloatArray,
    blue: FloatArray,
    mask: FloatArray,
    u: FloatArray,
    v: FloatArray,
  ) {
    logMinMax("tensor_ch0_r", red)
    logMinMax("tensor_ch1_g", green)
    logMinMax("tensor_ch2_b", blue)
    logMinMax("tensor_ch3_mask", mask)
    logMinMax("tensor_ch4_u", u)
    logMinMax("tensor_ch5_v", v)
  }

  private fun logMinMax(label: String, values: FloatArray) {
    if (values.isEmpty()) {
      Log.d(TAG, "$label min=nan max=nan")
      return
    }
    var minValue = Float.POSITIVE_INFINITY
    var maxValue = Float.NEGATIVE_INFINITY
    for (value in values) {
      if (value < minValue) minValue = value
      if (value > maxValue) maxValue = value
    }
    Log.d(TAG, "$label min=$minValue max=$maxValue")
  }

  private fun shouldAcceptTimestampLocked(timestampNs: Long): Boolean {
    if (
      lastAcceptedTimestampNs != Long.MIN_VALUE &&
        timestampNs - lastAcceptedTimestampNs < FRAME_INTERVAL_NS
    ) {
      return false
    }
    if (runner.hasModelAsset()) return true
    logMissingModel()
    return false
  }

  private fun logMissingModel() {
    if (hasLoggedMissingModel) return
    hasLoggedMissingModel = true
    Log.w(TAG, "No InkTrack ONNX asset found at assets/models/inktrack_rgb_geometry.onnx.")
  }

  private fun projectSurfaceCornersToImage(frame: Frame?, surfaceCornersView: FloatArray?): FloatArray? {
    if (frame == null || surfaceCornersView == null || surfaceCornersView.size < 8) return null
    val imageCoords = FloatArray(surfaceCornersView.size)
    frame.transformCoordinates2d(
      Coordinates2d.VIEW,
      surfaceCornersView,
      Coordinates2d.IMAGE_PIXELS,
      imageCoords,
    )
    return imageCoords
  }
}
