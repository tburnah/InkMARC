package com.inkmarc

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Test

class PlaneRelockBufferTest {
  @Test
  fun proposesRelockAfterFiveConsistentMovingEstimates() {
    val buffer = PlaneRelockBuffer()
    val startNs = 1_000_000_000L
    var proposal: PlaneRelockBuffer.Proposal? = null

    repeat(5) { index ->
      val seconds = index * 0.033
      proposal =
        buffer.addRejected(
          timestampNs = startNs + index * 33_000_000L,
          planeTerm =
            doubleArrayOf(
              0.10 + 0.12 * seconds,
              -0.04 + 0.08 * seconds,
              0.16 - 0.10 * seconds,
            ),
          supportRatio = 0.48,
          inlierCount = 27,
          fitRmsePx = 3.2,
        )
      if (index < 4) assertNull(proposal)
    }

    assertNotNull(proposal)
    val accepted = requireNotNull(proposal)
    assertEquals(5, accepted.candidateCount)
    assertEquals(0.12, accepted.velocity[0], 1e-6)
    assertEquals(0.08, accepted.velocity[1], 1e-6)
    assertEquals(-0.10, accepted.velocity[2], 1e-6)
  }

  @Test
  fun refusesInconsistentDepthHypotheses() {
    val buffer = PlaneRelockBuffer()
    val depth = doubleArrayOf(0.10, 0.17, 0.09, 0.18, 0.08)
    var proposal: PlaneRelockBuffer.Proposal? = null
    repeat(5) { index ->
      proposal =
        buffer.addRejected(
          timestampNs = 2_000_000_000L + index * 33_000_000L,
          planeTerm = doubleArrayOf(0.08, -0.03, depth[index]),
          supportRatio = 0.50,
          inlierCount = 30,
          fitRmsePx = 2.5,
        )
    }
    assertNull(proposal)
  }

  @Test
  fun timeGapRestartsCandidateSequence() {
    val buffer = PlaneRelockBuffer()
    repeat(4) { index ->
      buffer.addRejected(
        timestampNs = 3_000_000_000L + index * 33_000_000L,
        planeTerm = doubleArrayOf(0.10, -0.04, 0.12),
        supportRatio = 0.45,
        inlierCount = 24,
        fitRmsePx = 3.0,
      )
    }
    val proposal =
      buffer.addRejected(
        timestampNs = 3_500_000_000L,
        planeTerm = doubleArrayOf(0.10, -0.04, 0.12),
        supportRatio = 0.45,
        inlierCount = 24,
        fitRmsePx = 3.0,
      )

    assertNull(proposal)
    assertEquals(1, buffer.candidateCount)
  }
}
