package com.inkmarc

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class CameraImuTimeOffsetEstimatorTest {
  @Test
  fun calibratesOnlyAfterSustainedObservableMotion() {
    val estimator = CameraImuTimeOffsetEstimator(evaluationIntervalFrames = 3)
    repeat(2) {
      assertFalse(estimator.shouldEvaluate(0.8, 30, 50, 2.0))
    }
    assertTrue(estimator.shouldEvaluate(0.8, 30, 50, 2.0))
  }

  @Test
  fun rejectsOffsetChangeWithoutMeaningfulResidualGain() {
    val estimator = CameraImuTimeOffsetEstimator()
    val changed =
      estimator.update(
        listOf(
          CameraImuTimeOffsetEstimator.Score(-2_000_000L, 2.00, 30, 50),
          CameraImuTimeOffsetEstimator.Score(0L, 2.05, 30, 50),
          CameraImuTimeOffsetEstimator.Score(2_000_000L, 2.02, 30, 50),
        ),
      )
    assertFalse(changed)
    assertEquals(0L, estimator.offsetNs)
  }

  @Test
  fun movesTowardClearlyBetterNeighboringOffset() {
    val estimator = CameraImuTimeOffsetEstimator()
    val scores =
      listOf(
        CameraImuTimeOffsetEstimator.Score(-2_000_000L, 2.40, 31, 50),
        CameraImuTimeOffsetEstimator.Score(0L, 2.10, 31, 50),
        CameraImuTimeOffsetEstimator.Score(2_000_000L, 1.75, 31, 50),
      )
    assertFalse(estimator.update(scores))
    assertTrue(estimator.update(scores))
    assertEquals(2_000_000L, estimator.offsetNs)
  }
}
