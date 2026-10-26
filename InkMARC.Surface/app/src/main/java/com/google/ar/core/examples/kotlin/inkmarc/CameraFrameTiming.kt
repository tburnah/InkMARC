package com.inkmarc

import kotlin.math.roundToLong

/** Exposure timing for one raw, sensor-oriented Camera2 image. */
data class CameraFrameTiming(
  val sourceTimestampNs: Long,
  val firstRowExposureStartElapsedRealtimeNs: Long,
  val exposureTimeNs: Long,
  val rollingShutterSkewNs: Long,
  val imageHeight: Int,
) {
  val centerExposureTimestampNs: Long
    get() = timestampForRowFraction(0.5)

  val hasRollingShutterTiming: Boolean
    get() = rollingShutterSkewNs > 0L && imageHeight > 1

  fun timestampForImageRow(row: Double): Long {
    val fraction =
      if (imageHeight <= 1) 0.5
      else (row / (imageHeight - 1).toDouble()).coerceIn(0.0, 1.0)
    return timestampForRowFraction(fraction)
  }

  fun timestampForRowFraction(fraction: Double): Long {
    val rowStartOffsetNs = rollingShutterSkewNs * fraction.coerceIn(0.0, 1.0)
    return firstRowExposureStartElapsedRealtimeNs +
      rowStartOffsetNs.roundToLong() +
      exposureTimeNs.coerceAtLeast(0L) / 2L
  }

  companion object {
    fun instantaneous(timestampNs: Long, sourceTimestampNs: Long = timestampNs, imageHeight: Int = 1) =
      CameraFrameTiming(
        sourceTimestampNs = sourceTimestampNs,
        firstRowExposureStartElapsedRealtimeNs = timestampNs,
        exposureTimeNs = 0L,
        rollingShutterSkewNs = 0L,
        imageHeight = imageHeight,
      )
  }
}
