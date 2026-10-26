package com.inkmarc

import android.media.Image
import org.opencv.core.CvType
import org.opencv.core.Mat
import java.nio.ByteBuffer
import kotlin.math.roundToInt

data class CameraFrameData(
  val width: Int,
  val height: Int,
  val yBytes: ByteArray,
  val uBytes: ByteArray,
  val vBytes: ByteArray,
  val yRowStride: Int,
  val uRowStride: Int,
  val vRowStride: Int,
  val yPixelStride: Int,
  val uPixelStride: Int,
  val vPixelStride: Int,
) {
  fun toRgbMat(): Mat {
    val rgbBytes = ByteArray(width * height * 3)
    var outputIndex = 0
    for (row in 0 until height) {
      val uvRow = row / 2
      for (col in 0 until width) {
        val uvCol = col / 2
        val y = yBytes[row * yRowStride + col * yPixelStride].toInt() and 0xFF
        val u = (uBytes[uvRow * uRowStride + uvCol * uPixelStride].toInt() and 0xFF) - 128
        val v = (vBytes[uvRow * vRowStride + uvCol * vPixelStride].toInt() and 0xFF) - 128

        val r = (y + 1.402f * v).roundToInt().coerceIn(0, 255)
        val g = (y - 0.344136f * u - 0.714136f * v).roundToInt().coerceIn(0, 255)
        val b = (y + 1.772f * u).roundToInt().coerceIn(0, 255)

        rgbBytes[outputIndex++] = r.toByte()
        rgbBytes[outputIndex++] = g.toByte()
        rgbBytes[outputIndex++] = b.toByte()
      }
    }

    return Mat(height, width, CvType.CV_8UC3).also { mat ->
      mat.put(0, 0, rgbBytes)
    }
  }

  companion object {
    fun fromImage(image: Image): CameraFrameData {
      require(image.planes.size >= 3) { "Expected a YUV_420_888 camera image." }
      val yPlane = image.planes[0]
      val uPlane = image.planes[1]
      val vPlane = image.planes[2]
      return CameraFrameData(
        width = image.width,
        height = image.height,
        yBytes = yPlane.buffer.duplicate().rewindCopy(),
        uBytes = uPlane.buffer.duplicate().rewindCopy(),
        vBytes = vPlane.buffer.duplicate().rewindCopy(),
        yRowStride = yPlane.rowStride,
        uRowStride = uPlane.rowStride,
        vRowStride = vPlane.rowStride,
        yPixelStride = yPlane.pixelStride,
        uPixelStride = uPlane.pixelStride,
        vPixelStride = vPlane.pixelStride,
      )
    }
  }
}

private fun ByteBuffer.rewindCopy(): ByteArray {
  rewind()
  return ByteArray(remaining()).also { get(it) }
}
