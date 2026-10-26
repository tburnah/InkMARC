package com.inkmarc

/**
 * Slow, bounded online calibration of the camera exposure clock against the IMU orientation.
 * Calibration is attempted only during observable rotation and compares neighboring offsets using
 * the same robust plane residual as the tracker.
 */
internal class CameraImuTimeOffsetEstimator(
  private val evaluationIntervalFrames: Int = 12,
  private val stepNs: Long = 2_000_000L,
  private val maxAbsOffsetNs: Long = 30_000_000L,
  private val minAngularSpeedRadPerSecond: Double = 0.25,
  private val minInlierCount: Int = 18,
  private val minSupportRatio: Double = 0.40,
  private val maxRmsePx: Double = 8.0,
  private val minObjectiveImprovementPx: Double = 0.12,
) {
  data class Score(
    val offsetNs: Long,
    val rmsePx: Double,
    val inlierCount: Int,
    val sampleCount: Int,
  )

  var offsetNs: Long = 0L
    private set

  private var eligibleFramesSinceEvaluation = 0
  private var pendingDirection = 0
  private var pendingDirectionCount = 0

  fun shouldEvaluate(
    angularSpeedRadPerSecond: Double?,
    inlierCount: Int,
    sampleCount: Int,
    rmsePx: Double,
  ): Boolean {
    val supportRatio =
      if (sampleCount > 0) inlierCount.toDouble() / sampleCount.toDouble() else 0.0
    val eligible =
      angularSpeedRadPerSecond != null &&
        angularSpeedRadPerSecond >= minAngularSpeedRadPerSecond &&
        inlierCount >= minInlierCount &&
        supportRatio >= minSupportRatio &&
        rmsePx.isFinite() &&
        rmsePx <= maxRmsePx
    if (!eligible) {
      eligibleFramesSinceEvaluation = 0
      return false
    }
    eligibleFramesSinceEvaluation += 1
    if (eligibleFramesSinceEvaluation < evaluationIntervalFrames) return false
    eligibleFramesSinceEvaluation = 0
    return true
  }

  fun candidateOffsetsNs(): LongArray =
    longArrayOf(
        (offsetNs - stepNs).coerceAtLeast(-maxAbsOffsetNs),
        offsetNs,
        (offsetNs + stepNs).coerceAtMost(maxAbsOffsetNs),
      )
      .distinct()
      .toLongArray()

  fun update(scores: List<Score>): Boolean {
    val valid =
      scores.filter {
        it.sampleCount > 0 &&
          it.inlierCount >= minInlierCount &&
          it.rmsePx.isFinite() &&
          it.rmsePx <= maxRmsePx
      }
    val center = valid.firstOrNull { it.offsetNs == offsetNs } ?: return false
    val maxSupport = valid.maxOfOrNull(Score::inlierCount) ?: return false
    fun objective(score: Score): Double =
      score.rmsePx + (maxSupport - score.inlierCount).coerceAtLeast(0) * 0.08

    val best = valid.minByOrNull(::objective) ?: return false
    if (
      best.offsetNs == offsetNs ||
        objective(center) - objective(best) < minObjectiveImprovementPx
    ) {
      pendingDirection = 0
      pendingDirectionCount = 0
      return false
    }
    val direction = if (best.offsetNs > offsetNs) 1 else -1
    if (direction == pendingDirection) {
      pendingDirectionCount += 1
    } else {
      pendingDirection = direction
      pendingDirectionCount = 1
    }
    if (pendingDirectionCount < 2) return false
    offsetNs = best.offsetNs.coerceIn(-maxAbsOffsetNs, maxAbsOffsetNs)
    pendingDirection = 0
    pendingDirectionCount = 0
    return true
  }
}
