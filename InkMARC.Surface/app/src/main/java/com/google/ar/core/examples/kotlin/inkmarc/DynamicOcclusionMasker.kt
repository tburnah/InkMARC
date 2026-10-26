package com.inkmarc

import com.inkmarc.fallback.SurfaceTrackerCore
import kotlin.math.max
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.MatOfPoint
import org.opencv.core.Scalar
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc

/**
 * Detects coherent scene changes after compensating the previous frame with the IMU plane motion.
 *
 * This complements the hand landmark mask: it catches a clipped hand and its moving cast shadow.
 * Small residuals from table grain, sensor noise, or an imperfect warp are removed by morphology
 * and connected-component filtering. A decaying temporal union bridges slow shadow motion.
 */
class DynamicOcclusionMasker : AutoCloseable {
  companion object {
    private const val DIFFERENCE_THRESHOLD = 18.0
    private const val MIN_COMPONENT_AREA_RATIO = 0.0025
    private const val MAX_FRAME_MASK_RATIO = 0.52
    private const val MAX_GLOBAL_BRIGHTNESS_COMPENSATION = 32.0
    private const val PROCESSING_SCALE = 0.5
    private val BLUR_SIZE = Size(7.0, 7.0)
    private val OPEN_SIZE = Size(3.0, 3.0)
    private val CLOSE_SIZE = Size(7.0, 7.0)
    private val DILATE_SIZE = Size(7.0, 7.0)
    private val DECAY_SIZE = Size(3.0, 3.0)
  }

  data class MaskSnapshot(
    val mask: Mat?,
    val areaRatio: Double,
  )

  private var previousGray: Mat? = null
  private var persistentMask: Mat? = null

  fun update(
    grayFrame: Mat,
    previousToCurrent: SurfaceTrackerCore.SurfaceTransform?,
  ): MaskSnapshot {
    val scaledGray = Mat()
    Imgproc.resize(
      grayFrame,
      scaledGray,
      Size(),
      PROCESSING_SCALE,
      PROCESSING_SCALE,
      Imgproc.INTER_AREA,
    )
    val previous = previousGray
    if (previous == null || previous.size() != scaledGray.size() || previousToCurrent == null) {
      cacheGray(scaledGray)
      scaledGray.release()
      clearPersistentMask()
      return MaskSnapshot(null, 0.0)
    }

    val warpedPrevious = Mat()
    val valid = Mat()
    val sourceValid = Mat(previous.rows(), previous.cols(), CvType.CV_8UC1, Scalar(255.0))
    val warpedPersistent = Mat()
    val previousBlurred = Mat()
    val currentBlurred = Mat()
    val brightnessAdjustedPrevious = Mat()
    val difference = Mat()
    val rawMask = Mat()
    val filteredMask = Mat.zeros(scaledGray.rows(), scaledGray.cols(), CvType.CV_8UC1)
    val contourInput = Mat()
    val hierarchy = Mat()
    val openKernel = Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, OPEN_SIZE)
    val closeKernel = Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, CLOSE_SIZE)
    val dilateKernel = Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, DILATE_SIZE)
    val decayKernel = Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, DECAY_SIZE)
    var output: Mat? = null
    try {
      warp(previous, warpedPrevious, previousToCurrent, Imgproc.INTER_LINEAR)
      warp(sourceValid, valid, previousToCurrent, Imgproc.INTER_NEAREST)
      Imgproc.threshold(valid, valid, 250.0, 255.0, Imgproc.THRESH_BINARY)

      Imgproc.GaussianBlur(warpedPrevious, previousBlurred, BLUR_SIZE, 0.0)
      Imgproc.GaussianBlur(scaledGray, currentBlurred, BLUR_SIZE, 0.0)
      val brightnessDelta =
        (Core.mean(currentBlurred, valid).`val`[0] - Core.mean(previousBlurred, valid).`val`[0])
          .coerceIn(-MAX_GLOBAL_BRIGHTNESS_COMPENSATION, MAX_GLOBAL_BRIGHTNESS_COMPENSATION)
      previousBlurred.convertTo(
        brightnessAdjustedPrevious,
        CvType.CV_8UC1,
        1.0,
        brightnessDelta,
      )
      Core.absdiff(currentBlurred, brightnessAdjustedPrevious, difference)
      Imgproc.threshold(
        difference,
        rawMask,
        DIFFERENCE_THRESHOLD,
        255.0,
        Imgproc.THRESH_BINARY,
      )
      Core.bitwise_and(rawMask, valid, rawMask)
      Imgproc.morphologyEx(rawMask, rawMask, Imgproc.MORPH_OPEN, openKernel)
      Imgproc.morphologyEx(rawMask, rawMask, Imgproc.MORPH_CLOSE, closeKernel)

      rawMask.copyTo(contourInput)
      val contours = ArrayList<MatOfPoint>()
      Imgproc.findContours(
        contourInput,
        contours,
        hierarchy,
        Imgproc.RETR_EXTERNAL,
        Imgproc.CHAIN_APPROX_SIMPLE,
      )
      val minimumArea = scaledGray.rows().toDouble() * scaledGray.cols() * MIN_COMPONENT_AREA_RATIO
      try {
        for (contour in contours) {
          if (Imgproc.contourArea(contour) >= minimumArea) {
            Imgproc.drawContours(filteredMask, listOf(contour), -1, Scalar(255.0), -1)
          }
        }
      } finally {
        contours.forEach(MatOfPoint::release)
      }
      Imgproc.dilate(filteredMask, filteredMask, dilateKernel)

      val prior = persistentMask
      if (prior != null && prior.size() == scaledGray.size()) {
        warp(prior, warpedPersistent, previousToCurrent, Imgproc.INTER_NEAREST)
        Imgproc.erode(warpedPersistent, warpedPersistent, decayKernel)
        Core.bitwise_or(filteredMask, warpedPersistent, filteredMask)
      }
      Core.bitwise_and(filteredMask, valid, filteredMask)

      val areaRatio = maskAreaRatio(filteredMask)
      if (areaRatio in 0.0..MAX_FRAME_MASK_RATIO && Core.countNonZero(filteredMask) > 0) {
        output = Mat()
        Imgproc.resize(
          filteredMask,
          output,
          grayFrame.size(),
          0.0,
          0.0,
          Imgproc.INTER_NEAREST,
        )
        persistentMask?.release()
        persistentMask = filteredMask.clone()
        return MaskSnapshot(output, areaRatio)
      }

      clearPersistentMask()
      return MaskSnapshot(null, 0.0)
    } finally {
      cacheGray(scaledGray)
      scaledGray.release()
      warpedPrevious.release()
      valid.release()
      sourceValid.release()
      warpedPersistent.release()
      previousBlurred.release()
      currentBlurred.release()
      brightnessAdjustedPrevious.release()
      difference.release()
      rawMask.release()
      filteredMask.release()
      contourInput.release()
      hierarchy.release()
      openKernel.release()
      closeKernel.release()
      dilateKernel.release()
      decayKernel.release()
    }
  }

  fun reset(grayFrame: Mat? = null) {
    previousGray?.release()
    previousGray =
      grayFrame?.let { source ->
        Mat().also { scaled ->
          Imgproc.resize(
            source,
            scaled,
            Size(),
            PROCESSING_SCALE,
            PROCESSING_SCALE,
            Imgproc.INTER_AREA,
          )
        }
      }
    clearPersistentMask()
  }

  override fun close() {
    reset()
  }

  private fun cacheGray(grayFrame: Mat) {
    val cached = previousGray
    if (cached == null || cached.size() != grayFrame.size() || cached.type() != grayFrame.type()) {
      cached?.release()
      previousGray = grayFrame.clone()
    } else {
      grayFrame.copyTo(cached)
    }
  }

  private fun clearPersistentMask() {
    persistentMask?.release()
    persistentMask = null
  }

  private fun warp(
    source: Mat,
    destination: Mat,
    transform: SurfaceTrackerCore.SurfaceTransform,
    interpolation: Int,
  ) {
    when (transform) {
      is SurfaceTrackerCore.HomographyTransform -> {
        val matrix = Mat(3, 3, CvType.CV_64F)
        try {
          val coefficients = transform.coefficientsCopy()
          coefficients[2] *= PROCESSING_SCALE
          coefficients[5] *= PROCESSING_SCALE
          coefficients[6] /= PROCESSING_SCALE
          coefficients[7] /= PROCESSING_SCALE
          matrix.put(0, 0, *coefficients)
          Imgproc.warpPerspective(
            source,
            destination,
            matrix,
            source.size(),
            interpolation,
            Core.BORDER_CONSTANT,
            Scalar(0.0),
          )
        } finally {
          matrix.release()
        }
      }
      is SurfaceTrackerCore.AffineTransform -> {
        val matrix = Mat(2, 3, CvType.CV_64F)
        try {
          val coefficients = transform.coefficientsCopy()
          coefficients[2] *= PROCESSING_SCALE
          coefficients[5] *= PROCESSING_SCALE
          matrix.put(0, 0, *coefficients)
          Imgproc.warpAffine(
            source,
            destination,
            matrix,
            source.size(),
            interpolation,
            Core.BORDER_CONSTANT,
            Scalar(0.0),
          )
        } finally {
          matrix.release()
        }
      }
      is SurfaceTrackerCore.TranslationTransform -> {
        val matrix = Mat(2, 3, CvType.CV_64F)
        try {
          matrix.put(
            0,
            0,
            1.0,
            0.0,
            transform.dx * PROCESSING_SCALE,
            0.0,
            1.0,
            transform.dy * PROCESSING_SCALE,
          )
          Imgproc.warpAffine(
            source,
            destination,
            matrix,
            source.size(),
            interpolation,
            Core.BORDER_CONSTANT,
            Scalar(0.0),
          )
        } finally {
          matrix.release()
        }
      }
    }
  }

  private fun maskAreaRatio(mask: Mat): Double =
    Core.countNonZero(mask).toDouble() / max(1.0, mask.rows().toDouble() * mask.cols())
}
