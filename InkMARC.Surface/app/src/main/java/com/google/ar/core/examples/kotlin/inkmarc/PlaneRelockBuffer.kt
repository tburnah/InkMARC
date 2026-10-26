package com.inkmarc

import java.util.ArrayDeque
import kotlin.math.hypot
import kotlin.math.sqrt

/**
 * Recognizes a sustained, smooth visual plane estimate after the normal innovation gate rejects
 * it. A single outlier can never move the plane state; a re-lock proposal needs a short sequence
 * whose translation-over-height terms are well explained by constant velocity.
 */
internal class PlaneRelockBuffer(
  private val requiredCandidates: Int = 5,
  private val minSupportRatio: Double = 0.35,
  private val minInlierCount: Int = 18,
  private val maxFitRmsePx: Double = 8.0,
  private val maxCandidateGapNs: Long = 150_000_000L,
  private val maxXyLineResidual: Double = 0.030,
  private val maxDepthLineResidual: Double = 0.040,
  private val maxVelocityPerSecond: Double = 4.0,
) {
  data class Proposal(
    val timestampNs: Long,
    val targetTerm: DoubleArray,
    val velocity: DoubleArray,
    val candidateCount: Int,
  )

  private data class Candidate(
    val timestampNs: Long,
    val planeTerm: DoubleArray,
  )

  private val candidates = ArrayDeque<Candidate>()

  val candidateCount: Int
    get() = candidates.size

  fun reset() = candidates.clear()

  fun addRejected(
    timestampNs: Long,
    planeTerm: DoubleArray,
    supportRatio: Double,
    inlierCount: Int,
    fitRmsePx: Double,
  ): Proposal? {
    if (
      planeTerm.size != 3 ||
        planeTerm.any { !it.isFinite() } ||
        supportRatio < minSupportRatio ||
        inlierCount < minInlierCount ||
        !fitRmsePx.isFinite() ||
        fitRmsePx > maxFitRmsePx
    ) {
      reset()
      return null
    }

    val previousTimestampNs = candidates.peekLast()?.timestampNs
    if (
      previousTimestampNs != null &&
        (timestampNs <= previousTimestampNs || timestampNs - previousTimestampNs > maxCandidateGapNs)
    ) {
      reset()
    }
    candidates.addLast(Candidate(timestampNs, planeTerm.copyOf()))
    while (candidates.size > requiredCandidates) candidates.removeFirst()
    if (candidates.size < requiredCandidates) return null

    val window = candidates.toList()
    val originTimestampNs = window.first().timestampNs
    val timesSeconds =
      DoubleArray(window.size) { index -> (window[index].timestampNs - originTimestampNs) / 1e9 }
    val meanTime = timesSeconds.average()
    val timeVariance = timesSeconds.sumOf { time -> square(time - meanTime) }
    if (timeVariance <= 1e-9) return null

    val intercepts = DoubleArray(3)
    val velocities = DoubleArray(3)
    for (component in 0..2) {
      val meanValue = window.map { it.planeTerm[component] }.average()
      val covariance =
        window.indices.sumOf { index ->
          (timesSeconds[index] - meanTime) * (window[index].planeTerm[component] - meanValue)
        }
      val velocity = covariance / timeVariance
      if (!velocity.isFinite() || kotlin.math.abs(velocity) > maxVelocityPerSecond) return null
      velocities[component] = velocity
      intercepts[component] = meanValue - velocity * meanTime
    }

    var maxXyResidual = 0.0
    var maxDepthResidual = 0.0
    for (index in window.indices) {
      val residualX =
        window[index].planeTerm[0] - (intercepts[0] + velocities[0] * timesSeconds[index])
      val residualY =
        window[index].planeTerm[1] - (intercepts[1] + velocities[1] * timesSeconds[index])
      val residualZ =
        window[index].planeTerm[2] - (intercepts[2] + velocities[2] * timesSeconds[index])
      maxXyResidual = maxOf(maxXyResidual, hypot(residualX, residualY))
      maxDepthResidual = maxOf(maxDepthResidual, kotlin.math.abs(residualZ))
    }
    if (maxXyResidual > maxXyLineResidual || maxDepthResidual > maxDepthLineResidual) return null

    val finalTime = timesSeconds.last()
    val target = DoubleArray(3) { component -> intercepts[component] + velocities[component] * finalTime }
    if (!target.all(Double::isFinite) || vectorNorm(target) > 4.0) return null
    return Proposal(timestampNs, target, velocities, window.size)
  }

  private fun square(value: Double): Double = value * value

  private fun vectorNorm(vector: DoubleArray): Double =
    sqrt(vector.sumOf { value -> value * value })
}
