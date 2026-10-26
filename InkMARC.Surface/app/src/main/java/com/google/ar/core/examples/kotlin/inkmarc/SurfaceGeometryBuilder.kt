package com.inkmarc

import android.util.Log
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.MatOfPoint2f
import org.opencv.core.Point
import org.opencv.core.MatOfPoint
import org.opencv.core.Scalar
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc

class SurfaceGeometryBuilder {
  companion object {
    private const val TAG = "SurfaceGeometry"
    const val OUTPUT_WIDTH = 432
    const val OUTPUT_HEIGHT = 672
    private const val CANONICAL_WIDTH = 480
    private const val CANONICAL_HEIGHT = 320
  }

  data class SurfaceGeometry(
    val mask: FloatArray,
    val u: FloatArray,
    val v: FloatArray,
  )

  fun mapNormalizedPointToSurface(xNormalized: Float, yNormalized: Float, surfaceCorners: FloatArray?): FloatArray? {
    if (surfaceCorners == null || surfaceCorners.size < 8) return null
    val ordered = orderCorners(surfaceCorners)
    val src =
      MatOfPoint2f(
        Point(0.0, 0.0),
        Point((CANONICAL_WIDTH - 1).toDouble(), 0.0),
        Point((CANONICAL_WIDTH - 1).toDouble(), (CANONICAL_HEIGHT - 1).toDouble()),
        Point(0.0, (CANONICAL_HEIGHT - 1).toDouble()),
      )
    val dst =
      MatOfPoint2f(
        Point(ordered[0].toDouble(), ordered[1].toDouble()),
        Point(ordered[2].toDouble(), ordered[3].toDouble()),
        Point(ordered[4].toDouble(), ordered[5].toDouble()),
        Point(ordered[6].toDouble(), ordered[7].toDouble()),
      )
    val inputPoint =
      MatOfPoint2f(
        Point(
          xNormalized.coerceIn(0f, 1f) * (CANONICAL_WIDTH - 1).toFloat().toDouble(),
          yNormalized.coerceIn(0f, 1f) * (CANONICAL_HEIGHT - 1).toFloat().toDouble(),
        )
      )
    val outputPoint = MatOfPoint2f()

    try {
      val homography = Imgproc.getPerspectiveTransform(src, dst)
      try {
        Core.perspectiveTransform(inputPoint, outputPoint, homography)
        val mapped = outputPoint.toArray().firstOrNull() ?: return null
        return floatArrayOf(mapped.x.toFloat(), mapped.y.toFloat())
      } finally {
        homography.release()
      }
    } finally {
      src.release()
      dst.release()
      inputPoint.release()
      outputPoint.release()
    }
  }

  fun build(surfaceCorners: FloatArray?): SurfaceGeometry {
    if (surfaceCorners == null || surfaceCorners.size < 8) {
      val zeros = FloatArray(OUTPUT_WIDTH * OUTPUT_HEIGHT)
      return SurfaceGeometry(mask = zeros.copyOf(), u = zeros.copyOf(), v = zeros.copyOf())
    }

    val ordered = orderCorners(surfaceCorners)
    val src =
      MatOfPoint2f(
        Point(0.0, 0.0),
        Point((CANONICAL_WIDTH - 1).toDouble(), 0.0),
        Point((CANONICAL_WIDTH - 1).toDouble(), (CANONICAL_HEIGHT - 1).toDouble()),
        Point(0.0, (CANONICAL_HEIGHT - 1).toDouble()),
      )
    val dst =
      MatOfPoint2f(
        Point(ordered[0].toDouble(), ordered[1].toDouble()),
        Point(ordered[2].toDouble(), ordered[3].toDouble()),
        Point(ordered[4].toDouble(), ordered[5].toDouble()),
        Point(ordered[6].toDouble(), ordered[7].toDouble()),
      )

    val homography = Imgproc.getPerspectiveTransform(src, dst)
    src.release()
    dst.release()

    val maskSource = Mat(CANONICAL_HEIGHT, CANONICAL_WIDTH, CvType.CV_32FC1, Scalar(1.0))
    val uSource = Mat(CANONICAL_HEIGHT, CANONICAL_WIDTH, CvType.CV_32FC1)
    val vSource = Mat(CANONICAL_HEIGHT, CANONICAL_WIDTH, CvType.CV_32FC1)
    val outputSize = Size(OUTPUT_WIDTH.toDouble(), OUTPUT_HEIGHT.toDouble())

    try {
      fillCoordinateRamps(uSource, vSource)
      val warpedMask = warp(maskSource, homography, outputSize, Imgproc.INTER_NEAREST)
      val warpedU = warp(uSource, homography, outputSize, Imgproc.INTER_LINEAR)
      val warpedV = warp(vSource, homography, outputSize, Imgproc.INTER_LINEAR)
      try {
        val mask = FloatArray(OUTPUT_WIDTH * OUTPUT_HEIGHT)
        val u = FloatArray(mask.size)
        val v = FloatArray(mask.size)
        warpedMask.get(0, 0, mask)
        warpedU.get(0, 0, u)
        warpedV.get(0, 0, v)
        for (index in mask.indices) {
          val binMask = if (mask[index] > 0.5f) 1f else 0f
          mask[index] = binMask
          u[index] *= binMask
          v[index] *= binMask
        }
        return SurfaceGeometry(mask = mask, u = u, v = v)
      } finally {
        warpedMask.release()
        warpedU.release()
        warpedV.release()
      }
    } catch (e: Exception) {
      Log.e(TAG, "Failed to build surface geometry", e)
      val zeros = FloatArray(OUTPUT_WIDTH * OUTPUT_HEIGHT)
      return SurfaceGeometry(mask = zeros.copyOf(), u = zeros.copyOf(), v = zeros.copyOf())
    } finally {
      homography.release()
      maskSource.release()
      uSource.release()
      vSource.release()
    }
  }

  private fun fillCoordinateRamps(uMat: Mat, vMat: Mat) {
    val uRow = FloatArray(CANONICAL_WIDTH) { column ->
      if (CANONICAL_WIDTH == 1) 0f else column.toFloat() / (CANONICAL_WIDTH - 1).toFloat()
    }
    for (row in 0 until CANONICAL_HEIGHT) {
      uMat.put(row, 0, uRow)
    }

    for (row in 0 until CANONICAL_HEIGHT) {
      val value =
        if (CANONICAL_HEIGHT == 1) 0f else row.toFloat() / (CANONICAL_HEIGHT - 1).toFloat()
      val vRow = FloatArray(CANONICAL_WIDTH) { value }
      vMat.put(row, 0, vRow)
    }
  }

  private fun warp(source: Mat, homography: Mat, outputSize: Size, interpolation: Int): Mat =
    Mat().also { output ->
      Imgproc.warpPerspective(
        source,
        output,
        homography,
        outputSize,
        interpolation,
        Core.BORDER_CONSTANT,
        Scalar(0.0),
      )
    }

  private fun orderCorners(corners: FloatArray): FloatArray {
    val points =
      listOf(
        Point(corners[0].toDouble(), corners[1].toDouble()),
        Point(corners[2].toDouble(), corners[3].toDouble()),
        Point(corners[4].toDouble(), corners[5].toDouble()),
        Point(corners[6].toDouble(), corners[7].toDouble()),
      )
    val sortedByY = points.sortedWith(compareBy<Point> { it.y }.thenBy { it.x })
    val top = sortedByY.take(2).sortedBy { it.x }
    val bottom = sortedByY.takeLast(2).sortedBy { it.x }
    return floatArrayOf(
      top[0].x.toFloat(),
      top[0].y.toFloat(),
      top[1].x.toFloat(),
      top[1].y.toFloat(),
      bottom[1].x.toFloat(),
      bottom[1].y.toFloat(),
      bottom[0].x.toFloat(),
      bottom[0].y.toFloat(),
    )
  }
}
