package com.inkmarc

import android.graphics.Bitmap
import android.media.Image
import android.util.Log
import androidx.lifecycle.DefaultLifecycleObserver
import androidx.lifecycle.LifecycleOwner
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.framework.image.MPImage
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarker
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarkerResult
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt
import kotlin.math.roundToInt
import org.opencv.android.Utils
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.MatOfInt
import org.opencv.core.MatOfPoint
import org.opencv.core.Point
import org.opencv.core.Scalar
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc

/**
 * Runs MediaPipe Hand Landmarker and exposes a recent image-space exclusion mask.
 *
 * The mask is deliberately larger than the landmark hull. SurfaceTrackerCore treats every non-zero
 * pixel as unavailable for seeding, optical-flow tracking, and feature replenishment. Keeping the
 * latest two hulls covers detector latency and a modest amount of between-frame hand motion.
 */
class MediaPipeHandOcclusionMasker(private val activity: InkMarcActivity) :
  DefaultLifecycleObserver {
  companion object {
    private const val TAG = "HandOcclusionMasker"
    private const val MODEL_ASSET_PATH = "models/hand_landmarker.task"
    private const val SUBMISSION_INTERVAL_NS = 100_000_000L
    private const val MASK_MAX_AGE_NS = 650_000_000L
    private const val MASK_PADDING_FRACTION = 0.075f
    private const val MIN_MASK_PADDING_PX = 28
    private const val MAX_MASK_PADDING_PX = 140
    private const val FOREARM_MAX_LENGTH_FRACTION = 0.45
    private const val FOREARM_WIDTH_FRACTION = 0.65
  }

  data class MaskSnapshot(
    val mask: Mat,
    val ageMs: Double,
    val areaRatio: Double,
  )

  private data class TimedMask(
    val mask: Mat,
    val timestampNs: Long,
    val width: Int,
    val height: Int,
  )

  private val executor: ExecutorService = Executors.newSingleThreadExecutor()
  private val inferenceInFlight = AtomicBoolean(false)
  private val maskLock = Any()

  @Volatile private var handLandmarker: HandLandmarker? = null
  @Volatile private var destroyed = false
  @Volatile private var lastSubmittedTimestampNs = Long.MIN_VALUE
  private var latestMask: TimedMask? = null
  private var previousMask: TimedMask? = null

  override fun onCreate(owner: LifecycleOwner) {
    executor.execute(::setupHandLandmarker)
  }

  override fun onPause(owner: LifecycleOwner) {
    clearMasks()
  }

  override fun onDestroy(owner: LifecycleOwner) {
    destroyed = true
    clearMasks()
    executor.execute {
      handLandmarker?.close()
      handLandmarker = null
    }
    executor.shutdown()
  }

  /** Copies a throttled camera frame and submits it without blocking the AR render thread. */
  fun submitFrame(image: Image, timestampNs: Long) {
    if (destroyed || handLandmarker == null) return
    if (
      lastSubmittedTimestampNs != Long.MIN_VALUE &&
        timestampNs - lastSubmittedTimestampNs < SUBMISSION_INTERVAL_NS
    ) return
    if (!inferenceInFlight.compareAndSet(false, true)) return

    val frameData =
      try {
        CameraFrameData.fromImage(image)
      } catch (e: Exception) {
        inferenceInFlight.set(false)
        Log.w(TAG, "Unable to copy camera frame for hand detection", e)
        return
      }
    lastSubmittedTimestampNs = timestampNs

    executor.execute {
      var mpImage: MPImage? = null
      try {
        if (destroyed) return@execute
        val rgb = frameData.toRgbMat()
        val rgba = Mat()
        try {
          Imgproc.cvtColor(rgb, rgba, Imgproc.COLOR_RGB2RGBA)
          val bitmap = Bitmap.createBitmap(frameData.width, frameData.height, Bitmap.Config.ARGB_8888)
          Utils.matToBitmap(rgba, bitmap)
          mpImage = BitmapImageBuilder(bitmap).build()
        } finally {
          rgb.release()
          rgba.release()
        }

        val detector = handLandmarker
        if (detector == null) {
          mpImage?.close()
          mpImage = null
          inferenceInFlight.set(false)
          return@execute
        }
        detector.detectAsync(mpImage!!, timestampNs / 1_000_000L)
        // The result/error callback owns mpImage after a successful submission.
        mpImage = null
      } catch (e: Exception) {
        Log.w(TAG, "MediaPipe hand detection submission failed", e)
        mpImage?.close()
        inferenceInFlight.set(false)
      }
    }
  }

  /** Returns a caller-owned mask in camera IMAGE_PIXELS coordinates, or null when none is recent. */
  fun getExclusionMask(width: Int, height: Int, timestampNs: Long): Mat? =
    getExclusionMaskSnapshot(width, height, timestampNs)?.mask

  /** Includes mask age and coverage so recordings can prove that hand rejection was active. */
  fun getExclusionMaskSnapshot(width: Int, height: Int, timestampNs: Long): MaskSnapshot? =
    synchronized(maskLock) {
      val masks =
        listOfNotNull(previousMask, latestMask).filter {
          it.width == width &&
            it.height == height &&
            timestampNs >= it.timestampNs &&
            timestampNs - it.timestampNs <= MASK_MAX_AGE_NS
        }
      if (masks.isEmpty()) return@synchronized null
      val combined =
        if (masks.size == 1) {
          masks.first().mask.clone()
        } else {
          Mat.zeros(height, width, CvType.CV_8UC1).also { output ->
            for (timedMask in masks) {
              Core.bitwise_or(output, timedMask.mask, output)
            }
          }
      }
      val newestTimestampNs = masks.maxOf { it.timestampNs }
      MaskSnapshot(
        mask = combined,
        ageMs = (timestampNs - newestTimestampNs).toDouble() / 1_000_000.0,
        areaRatio = Core.countNonZero(combined).toDouble() / (width.toDouble() * height),
      )
    }

  private fun setupHandLandmarker() {
    if (destroyed) return
    try {
      val baseOptions = BaseOptions.builder().setModelAssetPath(MODEL_ASSET_PATH).build()
      val options =
        HandLandmarker.HandLandmarkerOptions.builder()
          .setBaseOptions(baseOptions)
          .setRunningMode(RunningMode.LIVE_STREAM)
          .setNumHands(1)
          .setMinHandDetectionConfidence(0.5f)
          .setMinHandPresenceConfidence(0.5f)
          .setMinTrackingConfidence(0.5f)
          .setResultListener(::onHandResult)
          .setErrorListener(::onHandError)
          .build()
      handLandmarker = HandLandmarker.createFromOptions(activity, options)
      Log.i(TAG, "MediaPipe Hand Landmarker initialized")
    } catch (e: Exception) {
      Log.e(TAG, "MediaPipe Hand Landmarker initialization failed", e)
    }
  }

  private fun onHandResult(result: HandLandmarkerResult, input: MPImage) {
    try {
      if (destroyed) return
      val mask = buildMask(result, input.width, input.height) ?: return
      val timestampNs = result.timestampMs() * 1_000_000L
      synchronized(maskLock) {
        previousMask?.mask?.release()
        previousMask = latestMask
        latestMask = TimedMask(mask, timestampNs, input.width, input.height)
      }
    } catch (e: Exception) {
      Log.w(TAG, "Unable to build hand exclusion mask", e)
    } finally {
      input.close()
      inferenceInFlight.set(false)
    }
  }

  private fun onHandError(error: RuntimeException) {
    Log.w(TAG, "MediaPipe Hand Landmarker error", error)
    inferenceInFlight.set(false)
  }

  private fun buildMask(result: HandLandmarkerResult, width: Int, height: Int): Mat? {
    val hands = result.landmarks()
    if (hands.isEmpty() || width <= 0 || height <= 0) return null

    val mask = Mat.zeros(height, width, CvType.CV_8UC1)
    for (hand in hands) {
      if (hand.size < 3) continue
      val points =
        hand.map { landmark ->
          Point(
            (landmark.x() * (width - 1)).coerceIn(0f, (width - 1).toFloat()).toDouble(),
            (landmark.y() * (height - 1)).coerceIn(0f, (height - 1).toFloat()).toDouble(),
          )
        }
      fillConvexHull(mask, points)
      extendMaskPastWrist(mask, points, width, height)
    }

    if (Core.countNonZero(mask) == 0) {
      mask.release()
      return null
    }

    val padding =
      (min(width, height) * MASK_PADDING_FRACTION)
        .roundToInt()
        .coerceIn(MIN_MASK_PADDING_PX, MAX_MASK_PADDING_PX)
    val kernelSize = max(3, padding * 2 + 1)
    val kernel =
      Imgproc.getStructuringElement(
        Imgproc.MORPH_ELLIPSE,
        Size(kernelSize.toDouble(), kernelSize.toDouble()),
      )
    try {
      Imgproc.dilate(mask, mask, kernel)
    } finally {
      kernel.release()
    }
    return mask
  }

  /**
   * Hand landmarks end at the wrist even though the forearm is another moving foreground object.
   * Extend a conservative corridor from the wrist away from the palm, capped to avoid masking an
   * excessive part of the surface when the complete hand is visible in the middle of the frame.
   */
  private fun extendMaskPastWrist(mask: Mat, points: List<Point>, width: Int, height: Int) {
    if (points.size < 18) return
    val wrist = points[0]
    val palmIndices = intArrayOf(5, 9, 13, 17)
    val palmCenter =
      Point(
        palmIndices.sumOf { points[it].x } / palmIndices.size,
        palmIndices.sumOf { points[it].y } / palmIndices.size,
      )
    val dx = wrist.x - palmCenter.x
    val dy = wrist.y - palmCenter.y
    val length = sqrt(dx * dx + dy * dy)
    if (!length.isFinite() || length < 1.0) return

    val unitX = dx / length
    val unitY = dy / length
    val distanceToBoundary = rayDistanceToImageBoundary(wrist, unitX, unitY, width, height)
    val extensionLength =
      min(distanceToBoundary, min(width, height) * FOREARM_MAX_LENGTH_FRACTION)
    if (!extensionLength.isFinite() || extensionLength <= 1.0) return

    val palmWidth =
      sqrt(
        (points[5].x - points[17].x) * (points[5].x - points[17].x) +
          (points[5].y - points[17].y) * (points[5].y - points[17].y),
      )
    val thickness =
      max(MIN_MASK_PADDING_PX * 2.0, palmWidth * FOREARM_WIDTH_FRACTION)
        .roundToInt()
        .coerceAtMost(MAX_MASK_PADDING_PX * 2)
    val endpoint =
      Point(
        (wrist.x + unitX * extensionLength).coerceIn(0.0, (width - 1).toDouble()),
        (wrist.y + unitY * extensionLength).coerceIn(0.0, (height - 1).toDouble()),
      )
    Imgproc.line(mask, wrist, endpoint, Scalar(255.0), thickness, Imgproc.LINE_AA)
  }

  private fun rayDistanceToImageBoundary(
    origin: Point,
    unitX: Double,
    unitY: Double,
    width: Int,
    height: Int,
  ): Double {
    val distances = ArrayList<Double>(4)
    if (unitX > 1e-6) distances += ((width - 1.0) - origin.x) / unitX
    if (unitX < -1e-6) distances += (0.0 - origin.x) / unitX
    if (unitY > 1e-6) distances += ((height - 1.0) - origin.y) / unitY
    if (unitY < -1e-6) distances += (0.0 - origin.y) / unitY
    return distances.filter { it >= 0.0 && it.isFinite() }.minOrNull() ?: 0.0
  }

  private fun fillConvexHull(mask: Mat, points: List<Point>) {
    val pointMat = MatOfPoint(*points.toTypedArray())
    val hullIndices = MatOfInt()
    try {
      Imgproc.convexHull(pointMat, hullIndices)
      val source = pointMat.toArray()
      val hull = ArrayList<Point>()
      for (index in hullIndices.toArray()) {
        source.getOrNull(index)?.let(hull::add)
      }
      if (hull.size >= 3) {
        val hullMat = MatOfPoint(*hull.toTypedArray())
        try {
          Imgproc.fillConvexPoly(mask, hullMat, Scalar(255.0))
        } finally {
          hullMat.release()
        }
      }
    } finally {
      pointMat.release()
      hullIndices.release()
    }
  }

  private fun clearMasks() {
    synchronized(maskLock) {
      previousMask?.mask?.release()
      latestMask?.mask?.release()
      previousMask = null
      latestMask = null
    }
  }
}
