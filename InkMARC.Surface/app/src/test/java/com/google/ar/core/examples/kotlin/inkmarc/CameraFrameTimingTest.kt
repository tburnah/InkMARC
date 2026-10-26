package com.inkmarc

import org.junit.Assert.assertEquals
import org.junit.Test

class CameraFrameTimingTest {
  @Test
  fun reportsExposureMidpointAtEachImageRow() {
    val timing =
      CameraFrameTiming(
        sourceTimestampNs = 1_000_000_000L,
        firstRowExposureStartElapsedRealtimeNs = 2_000_000_000L,
        exposureTimeNs = 10_000_000L,
        rollingShutterSkewNs = 20_000_000L,
        imageHeight = 101,
      )

    assertEquals(2_005_000_000L, timing.timestampForImageRow(0.0))
    assertEquals(2_015_000_000L, timing.timestampForImageRow(50.0))
    assertEquals(2_025_000_000L, timing.timestampForImageRow(100.0))
    assertEquals(2_015_000_000L, timing.centerExposureTimestampNs)
  }
}
