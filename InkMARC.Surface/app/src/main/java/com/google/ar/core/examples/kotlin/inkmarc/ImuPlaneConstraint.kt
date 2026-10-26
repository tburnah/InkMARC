package com.inkmarc

import com.inkmarc.fallback.SurfaceTrackerCore
import org.opencv.core.Point
import kotlin.math.abs
import kotlin.math.hypot
import kotlin.math.sqrt

/**
 * Constrains visual motion to a calibrated horizontal plane.
 *
 * The fused IMU orientation fixes camera rotation and the plane normal. Visual inlier pairs then
 * solve only the three translation-over-plane-distance parameters in
 * H = K (R + u n^T) K^-1. This retains the projective motion caused by moving over the table while
 * avoiding a free eight-degree-of-freedom homography.
 */
class ImuPlaneConstraint(
  private val orientationProvider: GravityDirectionProvider,
) {
  companion object {
    private const val MIN_PLANE_FIT_INLIERS = 4
    private const val MIN_EFFECTIVE_PLANE_SUPPORT = 2.75
    private const val MIN_WEIGHTED_PLANE_COVERAGE = 0.20
    private const val MAX_PLANE_FIT_RMSE_PX = 12.0
    private const val MAX_PLANE_TERM_NORM = 4.0
    private const val MAX_PREDICTION_SECONDS = 0.14
    private const val MAX_PLANE_TERM_PREDICTION = 0.20
    private const val MAX_PLANE_TERM_VELOCITY_PER_SECOND = 4.0
    private const val VELOCITY_FILTER_ALPHA = 0.55
    private const val MIN_PLANE_TRACK_OBSERVATIONS = 2
    private const val INITIAL_PLANE_TRACK_CONFIDENCE = 0.50
    private const val PLANE_TRACK_AGREEMENT_PX = 8.0
    private const val PLANE_TRACK_CONFIDENCE_GAIN = 0.25
    private const val PLANE_TRACK_CONFIDENCE_DECAY = 0.50
    private const val RELOCK_BLEND_FRAMES = 5
    private const val RELOCK_MEASUREMENT_MIN_SUPPORT_RATIO = 0.35
    private const val RELOCK_MEASUREMENT_MIN_INLIERS = 18
    private const val RELOCK_MEASUREMENT_MAX_RMSE_PX = 8.0
    private const val RELOCK_TARGET_XY_TOLERANCE = 0.10
    private const val RELOCK_TARGET_DEPTH_TOLERANCE = 0.10
    private const val RELOCK_TARGET_MEASUREMENT_ALPHA = 0.35
    private const val MAX_RELOCK_TARGET_ADVANCE_SECONDS = 0.15
    private const val REFERENCE_ROW_ORIENTATION_BINS = 33
  }

  private data class PlaneTrackState(
    var confidence: Double = INITIAL_PLANE_TRACK_CONFIDENCE,
    var observations: Int = 0,
  )

  private data class InnovationLimits(
    val xy: Double,
    val depth: Double,
  )

  private data class RelockStep(
    val planeTerm: DoubleArray,
    val alpha: Double,
    val framesRemaining: Int,
    val completionVelocity: DoubleArray?,
  )

  private data class TimedPlaneEstimate(
    val currentOrientation: DoubleArray,
    val relativeRotation: DoubleArray,
    val perPointRelativeRotations: List<DoubleArray>?,
    val estimate: PlaneMotionGeometry.Estimate,
  )

  private var referenceQuad: List<Point>? = null
  private var referenceDeviceToWorld: DoubleArray? = null
  private var referenceCameraToDevice: DoubleArray? = null
  private var referencePlaneNormalCamera: DoubleArray? = null
  private var referenceFrameTiming: CameraFrameTiming? = null
  private var referenceRowOrientations: List<DoubleArray>? = null
  private var lastPlaneTerm: DoubleArray? = null
  private var lastPlaneTermTimestampNs: Long? = null
  private val planeTermVelocity = DoubleArray(3)
  private var hasVelocityEstimate = false
  private val planeTrackStates = HashMap<Int, PlaneTrackState>()
  private val relockBuffer = PlaneRelockBuffer()
  private var relockTarget: DoubleArray? = null
  private var relockTargetTimestampNs: Long? = null
  private var relockVelocity: DoubleArray? = null
  private var relockFramesRemaining = 0
  private var relockDisagreementCount = 0
  private val cameraImuTimeOffsetEstimator = CameraImuTimeOffsetEstimator()

  @Synchronized
  fun seed(
    quad: List<Point>,
    timestampNs: Long,
    coordinateMapper: CameraCoordinateMapper,
    preserveMotion: Boolean = false,
    frameTiming: CameraFrameTiming? = null,
  ) {
    val imuTimestampNs = cameraImuTimestamp(timestampNs, frameTiming)
    val deviceToWorld = orientationProvider.getDeviceToWorldRotation(imuTimestampNs)
    val cameraToDevice = cameraToDeviceRotation(coordinateMapper.rotationDegrees)
    referenceQuad = quad.map(::copyPoint)
    referenceDeviceToWorld = deviceToWorld
    referenceCameraToDevice = cameraToDevice
    referencePlaneNormalCamera =
      deviceToWorld?.let { horizontalPlaneNormalInReferenceCamera(it, cameraToDevice) }
    referenceFrameTiming = frameTiming
    referenceRowOrientations =
      if (frameTiming?.hasRollingShutterTiming == true) {
        List(REFERENCE_ROW_ORIENTATION_BINS) { index ->
          val fraction = index.toDouble() / (REFERENCE_ROW_ORIENTATION_BINS - 1).toDouble()
          orientationProvider.getDeviceToWorldRotation(
            frameTiming.timestampForRowFraction(fraction) + cameraImuTimeOffsetEstimator.offsetNs,
          ) ?: deviceToWorld ?: identity3x3()
        }
      } else {
        null
      }
    lastPlaneTerm = doubleArrayOf(0.0, 0.0, 0.0)
    lastPlaneTermTimestampNs = timestampNs
    planeTrackStates.clear()
    clearRelockState()
    if (!preserveMotion) {
      planeTermVelocity.fill(0.0)
      hasVelocityEstimate = false
    }
  }

  @Synchronized
  fun reset() {
    referenceQuad = null
    referenceDeviceToWorld = null
    referenceCameraToDevice = null
    referencePlaneNormalCamera = null
    referenceFrameTiming = null
    referenceRowOrientations = null
    lastPlaneTerm = null
    lastPlaneTermTimestampNs = null
    planeTermVelocity.fill(0.0)
    hasVelocityEstimate = false
    planeTrackStates.clear()
    clearRelockState()
  }

  /** IMU/plane prediction used as LK's initial flow from the previous camera frame. */
  @Synchronized
  fun frameToFrameTransform(
    previousTimestampNs: Long,
    currentTimestampNs: Long,
    coordinateMapper: CameraCoordinateMapper,
    previousFrameTiming: CameraFrameTiming? = null,
    currentFrameTiming: CameraFrameTiming? = null,
  ): SurfaceTrackerCore.SurfaceTransform? {
    val previousHomography =
      referenceHomographyAt(
        previousTimestampNs,
        coordinateMapper,
        cameraImuTimestamp(previousTimestampNs, previousFrameTiming),
      )
    val currentHomography =
      referenceHomographyAt(
        currentTimestampNs,
        coordinateMapper,
        cameraImuTimestamp(currentTimestampNs, currentFrameTiming),
      )
    if (previousHomography != null && currentHomography != null) {
      val previousInverse = PlaneMotionGeometry.invert3x3(previousHomography)
      if (previousInverse != null) {
        return SurfaceTrackerCore.HomographyTransform(
          PlaneMotionGeometry.multiply3x3(currentHomography, previousInverse),
        )
      }
    }

    // Orientation is still useful before the first reliable visual translation estimate.
    val previousOrientation =
      orientationProvider.getDeviceToWorldRotation(
        cameraImuTimestamp(previousTimestampNs, previousFrameTiming),
      ) ?: return null
    val currentOrientation =
      orientationProvider.getDeviceToWorldRotation(
        cameraImuTimestamp(currentTimestampNs, currentFrameTiming),
      ) ?: return null
    val cameraToDevice = cameraToDeviceRotation(coordinateMapper.rotationDegrees)
    val relativeRotation =
      relativeCameraRotation(previousOrientation, currentOrientation, cameraToDevice)
    val rotationHomography =
      PlaneMotionGeometry.planeHomography(
        relativeRotation,
        doubleArrayOf(0.0, 0.0, 1.0),
        doubleArrayOf(0.0, 0.0, 0.0),
        coordinateMapper.intrinsics,
      ) ?: return null
    return SurfaceTrackerCore.HomographyTransform(rotationHomography)
  }

  /** Best inertial/constant-velocity plane projection when visual tracking is unavailable. */
  @Synchronized
  fun predictQuad(
    timestampNs: Long,
    coordinateMapper: CameraCoordinateMapper,
  ): List<Point>? {
    val seedQuad = referenceQuad ?: return null
    val homography = referenceHomographyAt(timestampNs, coordinateMapper) ?: return null
    return PlaneMotionGeometry.mapPoints(seedQuad, homography)?.takeIf(::isValidQuad)
  }

  @Synchronized
  fun constrain(
    result: SurfaceTrackerCore.TrackerResult,
    frameTimestampNs: Long,
    outputTimestampNs: Long,
    coordinateMapper: CameraCoordinateMapper,
    frameTiming: CameraFrameTiming? = null,
  ): SurfaceTrackerCore.TrackerResult {
    val visualQuad = result.mappedQuad
    if (
      visualQuad == null ||
        result.heldLastGoodTransform ||
        result.state != SurfaceTrackerCore.State.TRACKING
    ) {
      return predictedResult(result, outputTimestampNs, coordinateMapper, "predicted_only")
    }

    val referenceOrientation = referenceDeviceToWorld
      ?: return result.withImuReason("reference_orientation_missing")
    val cameraToDevice = referenceCameraToDevice
      ?: return result.withImuReason("camera_rotation_missing")
    val planeNormal = referencePlaneNormalCamera
      ?: return result.withImuReason("plane_normal_missing")

    val hasFeaturePairs =
      result.referenceInlierPoints.size >= MIN_PLANE_FIT_INLIERS &&
        result.referenceInlierPoints.size == result.currentInlierPoints.size
    val allReferencePoints =
      if (hasFeaturePairs) result.referenceInlierPoints else referenceQuad.orEmpty()
    val allCurrentPoints =
      if (hasFeaturePairs) result.currentInlierPoints else visualQuad
    val hasTrackIds =
      hasFeaturePairs && result.inlierTrackIds.size == allReferencePoints.size
    val pairIndices = allReferencePoints.indices.toList()
    val referencePoints = pairIndices.map(allReferencePoints::get)
    val currentPoints = pairIndices.map(allCurrentPoints::get)
    val pointWeights =
      if (hasTrackIds) {
        buildPlanePointWeights(result.inlierTrackIds, result.inlierTrackWeights)
      } else {
        List(referencePoints.size) { 1.0 }
      }
    val priorPlaneTerm = predictedPlaneTermAt(frameTimestampNs)
    var timedEstimate =
      fitTimedPlaneEstimate(
        referencePoints,
        currentPoints,
        referenceOrientation,
        cameraToDevice,
        planeNormal,
        coordinateMapper.intrinsics,
        priorPlaneTerm,
        frameTimestampNs,
        frameTiming,
        cameraImuTimeOffsetEstimator.offsetNs,
        pointWeights,
      ) ?: return predictedResult(result, outputTimestampNs, coordinateMapper, "fit_failed")
    var estimate = timedEstimate.estimate
    var timingOffsetAdjusted = false
    val currentImuTimestampNs = cameraImuTimestamp(frameTimestampNs, frameTiming)
    val angularSpeed = orientationProvider.getAngularSpeedRadPerSecond(currentImuTimestampNs)
    if (
      cameraImuTimeOffsetEstimator.shouldEvaluate(
        angularSpeed,
        estimate.inlierCount,
        estimate.sampleCount,
        estimate.rmsePx,
      )
    ) {
      val currentOffsetNs = cameraImuTimeOffsetEstimator.offsetNs
      val candidateFits = mutableListOf<Pair<Long, TimedPlaneEstimate>>()
      for (candidateOffsetNs in cameraImuTimeOffsetEstimator.candidateOffsetsNs()) {
        val fit =
          if (candidateOffsetNs == currentOffsetNs) {
            timedEstimate
          } else {
            fitTimedPlaneEstimate(
              referencePoints,
              currentPoints,
              referenceOrientation,
              cameraToDevice,
              planeNormal,
              coordinateMapper.intrinsics,
              priorPlaneTerm,
              frameTimestampNs,
              frameTiming,
              candidateOffsetNs,
              pointWeights,
            )
          }
        if (fit != null) {
          candidateFits.add(candidateOffsetNs to fit)
        }
      }
      timingOffsetAdjusted =
        cameraImuTimeOffsetEstimator.update(
          candidateFits.map { (offsetNs, fit) ->
            CameraImuTimeOffsetEstimator.Score(
              offsetNs,
              fit.estimate.rmsePx,
              fit.estimate.inlierCount,
              fit.estimate.sampleCount,
            )
          },
        )
      if (timingOffsetAdjusted) {
        candidateFits
          .firstOrNull { it.first == cameraImuTimeOffsetEstimator.offsetNs }
          ?.second
          ?.let { calibratedFit ->
            timedEstimate = calibratedFit
            estimate = calibratedFit.estimate
          }
      }
    }
    val relativeRotation = timedEstimate.relativeRotation
    val allPointRelativeRotations =
      buildRowRelativeRotations(
        allReferencePoints,
        allCurrentPoints,
        frameTiming,
        timedEstimate.currentOrientation,
        referenceOrientation,
        cameraToDevice,
        cameraImuTimeOffsetEstimator.offsetNs,
      )
    if (estimate.inlierCount < MIN_PLANE_FIT_INLIERS) {
      return predictedResult(
        result,
        outputTimestampNs,
        coordinateMapper,
        "insufficient_plane_support:${estimate.inlierCount}",
      )
    }
    if (estimate.effectiveInlierCount < MIN_EFFECTIVE_PLANE_SUPPORT) {
      return predictedResult(
        result,
        outputTimestampNs,
        coordinateMapper,
        "insufficient_effective_plane_support:${format(estimate.effectiveInlierCount)}",
      )
    }
    val weightedCoverage =
      weightedPlaneCoverage(
        referencePoints,
        pointWeights,
        estimate.inlierIndices,
        coordinateMapper.intrinsics,
      )
    if (weightedCoverage < MIN_WEIGHTED_PLANE_COVERAGE) {
      return predictedResult(
        result,
        outputTimestampNs,
        coordinateMapper,
        "insufficient_weighted_plane_coverage:${format(weightedCoverage)}",
      )
    }
    if (estimate.rmsePx > MAX_PLANE_FIT_RMSE_PX) {
      return predictedResult(
        result,
        outputTimestampNs,
        coordinateMapper,
        "fit_rmse_rejected:${format(estimate.rmsePx)}",
      )
    }
    val termNorm = vectorNorm(estimate.planeTerm)
    if (!termNorm.isFinite() || termNorm > MAX_PLANE_TERM_NORM) {
      return predictedResult(
        result,
        outputTimestampNs,
        coordinateMapper,
        "plane_term_rejected:${format(termNorm)}",
      )
    }

    // Keep the gate aware of how much of the complete visual population supports the plane. A
    // small, internally consistent subset must not masquerade as high-confidence global support.
    val supportRatio =
      if (allReferencePoints.isNotEmpty()) {
        estimate.effectiveInlierCount / allReferencePoints.size.toDouble()
      } else {
        0.0
      }
    val innovation =
      priorPlaneTerm?.let { prior -> DoubleArray(3) { index -> estimate.planeTerm[index] - prior[index] } }
    val limits = innovationLimits(supportRatio, estimate.inlierCount, frameTimestampNs)
    val innovationRejected =
      innovation != null &&
        (hypot(innovation[0], innovation[1]) > limits.xy || abs(innovation[2]) > limits.depth)
    var relockStep =
      if (relockFramesRemaining > 0 && priorPlaneTerm != null) {
        advanceRelock(
          priorPlaneTerm,
          estimate.planeTerm,
          supportRatio,
          estimate.inlierCount,
          estimate.rmsePx,
          frameTimestampNs,
        )
      } else {
        null
      }

    if (relockStep == null && innovationRejected) {
      val proposal =
        relockBuffer.addRejected(
          frameTimestampNs,
          estimate.planeTerm,
          supportRatio,
          estimate.inlierCount,
          estimate.rmsePx,
        )
      if (proposal == null) {
        return predictedResult(
          result,
          outputTimestampNs,
          coordinateMapper,
          "innovation_rejected:${format(innovation?.get(0) ?: 0.0)}:" +
            "${format(innovation?.get(1) ?: 0.0)}:${format(innovation?.get(2) ?: 0.0)}:" +
            "limit=${format(limits.xy)}:${format(limits.depth)}:" +
            "support=${estimate.inlierCount}/${estimate.sampleCount}:plane_tracks=" +
            "${pairIndices.size}/${allReferencePoints.size}:" +
            "relock_candidates=${relockBuffer.candidateCount}/5",
        )
      }
      beginRelock(proposal)
      relockStep =
        advanceRelock(
          priorPlaneTerm,
          estimate.planeTerm,
          supportRatio,
          estimate.inlierCount,
          estimate.rmsePx,
          frameTimestampNs,
        )
    }

    if (relockStep == null && innovationRejected) {
      return predictedResult(
        result,
        outputTimestampNs,
        coordinateMapper,
        "innovation_rejected:relock_cancelled:relock_candidates=" +
          "${relockBuffer.candidateCount}/5",
      )
    }

    val filterAlpha = relockStep?.alpha ?: planeMeasurementAlpha(supportRatio, estimate.inlierCount)
    val acceptedPlaneTerm =
      relockStep?.planeTerm
        ?: if (priorPlaneTerm != null) {
          DoubleArray(3) { index ->
            priorPlaneTerm[index] + filterAlpha * (estimate.planeTerm[index] - priorPlaneTerm[index])
          }
        } else {
          estimate.planeTerm.copyOf()
        }
    if (relockStep == null) relockBuffer.reset()

    val frameHomography =
      PlaneMotionGeometry.planeHomography(
        relativeRotation,
        planeNormal,
        acceptedPlaneTerm,
        coordinateMapper.intrinsics,
      ) ?: return predictedResult(result, outputTimestampNs, coordinateMapper, "homography_failed")
    val frameQuad =
      referenceQuad?.let { PlaneMotionGeometry.mapPoints(it, frameHomography) }
        ?.takeIf(::isValidQuad)
        ?: return predictedResult(result, outputTimestampNs, coordinateMapper, "quad_rejected")

    if (hasTrackIds) {
      updatePlaneTrackStates(
        result.inlierTrackIds,
        PlaneMotionGeometry.planeReprojectionErrors(
          allReferencePoints,
          allCurrentPoints,
          relativeRotation,
          planeNormal,
          acceptedPlaneTerm,
          coordinateMapper.intrinsics,
          allPointRelativeRotations,
        ),
      )
    }
    acceptPlaneTerm(
      acceptedPlaneTerm,
      frameTimestampNs,
      updateVelocity = relockStep == null,
    )
    relockStep?.completionVelocity?.let(::completeRelock)
    val constrained = predictQuad(outputTimestampNs, coordinateMapper) ?: frameQuad
    val predictionMs = (outputTimestampNs - frameTimestampNs).coerceAtLeast(0L) / 1e6
    val reason =
      combineReasons(
        result.rejectionReason,
          "imu_plane:fit_rmse_px=${format(estimate.rmsePx)}:" +
          "support=${estimate.inlierCount}/${estimate.sampleCount}:" +
          "effective_support=${format(estimate.effectiveInlierCount)}/" +
          "${format(estimate.effectiveSampleCount)}:" +
          "weighted_coverage=${format(weightedCoverage)}:" +
          "overall_support=${format(supportRatio)}:" +
          "plane_tracks=${pairIndices.size}/${allReferencePoints.size}:" +
          "raw_translation_over_height=${format(estimate.planeTerm[0])}:" +
          "${format(estimate.planeTerm[1])}:${format(estimate.planeTerm[2])}:" +
          "translation_over_height=${format(acceptedPlaneTerm[0])}:" +
          "${format(acceptedPlaneTerm[1])}:${format(acceptedPlaneTerm[2])}:" +
          "innovation=${format(innovation?.get(0) ?: 0.0)}:" +
          "${format(innovation?.get(1) ?: 0.0)}:${format(innovation?.get(2) ?: 0.0)}:" +
          "filter_alpha=${format(filterAlpha)}:" +
          "relock=${relockStep?.let { "blend:${it.framesRemaining}" } ?: "none"}:" +
          "camera_imu_offset_ms=${format(cameraImuTimeOffsetEstimator.offsetNs / 1e6)}:" +
          "timing_adjusted=$timingOffsetAdjusted:" +
          "exposure_ms=${format((frameTiming?.exposureTimeNs ?: 0L) / 1e6)}:" +
          "rolling_shutter_ms=${format((frameTiming?.rollingShutterSkewNs ?: 0L) / 1e6)}:" +
          "prediction_ms=${format(predictionMs)}:" +
          "term_velocity=${format(planeTermVelocity[0])}:" +
          "${format(planeTermVelocity[1])}:${format(planeTermVelocity[2])}",
      )
    return result.copy(
      mappedQuad = constrained,
      rejectionReason = reason,
      effectivePlaneSupport = estimate.effectiveInlierCount.toFloat(),
      weightedPlaneCoverage = weightedCoverage.toFloat(),
    )
  }

  private fun fitTimedPlaneEstimate(
    referencePoints: List<Point>,
    currentPoints: List<Point>,
    referenceOrientation: DoubleArray,
    cameraToDevice: DoubleArray,
    planeNormal: DoubleArray,
    intrinsics: CameraIntrinsics,
    priorPlaneTerm: DoubleArray?,
    frameTimestampNs: Long,
    frameTiming: CameraFrameTiming?,
    cameraImuOffsetNs: Long,
    pointWeights: List<Double>,
  ): TimedPlaneEstimate? {
    val currentCenterTimestampNs =
      if (frameTiming != null) {
        frameTiming.centerExposureTimestampNs + cameraImuOffsetNs
      } else {
        frameTimestampNs
      }
    val currentOrientation =
      orientationProvider.getDeviceToWorldRotation(currentCenterTimestampNs) ?: return null
    val centerRelativeRotation =
      relativeCameraRotation(referenceOrientation, currentOrientation, cameraToDevice)
    val rowRelativeRotations =
      buildRowRelativeRotations(
        referencePoints,
        currentPoints,
        frameTiming,
        currentOrientation,
        referenceOrientation,
        cameraToDevice,
        cameraImuOffsetNs,
      )
    val estimate =
      PlaneMotionGeometry.fitPlaneTerm(
        referencePoints,
        currentPoints,
        centerRelativeRotation,
        planeNormal,
        intrinsics,
        priorPlaneTerm,
        rowRelativeRotations,
        pointWeights,
      ) ?: return null
    return TimedPlaneEstimate(
      currentOrientation,
      centerRelativeRotation,
      rowRelativeRotations,
      estimate,
    )
  }

  private fun buildRowRelativeRotations(
    referencePoints: List<Point>,
    currentPoints: List<Point>,
    frameTiming: CameraFrameTiming?,
    currentCenterOrientation: DoubleArray,
    referenceCenterOrientation: DoubleArray,
    cameraToDevice: DoubleArray,
    cameraImuOffsetNs: Long,
  ): List<DoubleArray>? {
    if (
      referencePoints.size != currentPoints.size ||
        (frameTiming?.hasRollingShutterTiming != true && referenceRowOrientations == null)
    ) {
      return null
    }
    return referencePoints.indices.map { index ->
      val referenceRowOrientation =
        referenceOrientationForImageRow(referencePoints[index].y) ?: referenceCenterOrientation
      val currentRowOrientation =
        if (frameTiming?.hasRollingShutterTiming == true) {
          orientationProvider.getDeviceToWorldRotation(
            frameTiming.timestampForImageRow(currentPoints[index].y) + cameraImuOffsetNs,
          ) ?: currentCenterOrientation
        } else {
          currentCenterOrientation
        }
      relativeCameraRotation(referenceRowOrientation, currentRowOrientation, cameraToDevice)
    }
  }

  private fun referenceOrientationForImageRow(imageRow: Double): DoubleArray? {
    val timing = referenceFrameTiming ?: return null
    val orientations = referenceRowOrientations ?: return null
    if (orientations.isEmpty() || timing.imageHeight <= 1) return null
    val fraction =
      (imageRow / (timing.imageHeight - 1).toDouble()).coerceIn(0.0, 1.0)
    val index =
      (fraction * (orientations.size - 1).toDouble()).toInt().coerceIn(0, orientations.lastIndex)
    return orientations[index]
  }

  private fun cameraImuTimestamp(
    fallbackTimestampNs: Long,
    frameTiming: CameraFrameTiming?,
  ): Long =
    if (frameTiming != null) {
      frameTiming.centerExposureTimestampNs + cameraImuTimeOffsetEstimator.offsetNs
    } else {
      fallbackTimestampNs
    }

  private fun beginRelock(proposal: PlaneRelockBuffer.Proposal) {
    relockTarget = proposal.targetTerm.copyOf()
    relockTargetTimestampNs = proposal.timestampNs
    relockVelocity = proposal.velocity.copyOf()
    relockFramesRemaining = RELOCK_BLEND_FRAMES
    relockDisagreementCount = 0
    relockBuffer.reset()

    // The old velocity belongs to the stale hypothesis. Do not extrapolate it while transitioning
    // to the consistently observed visual hypothesis.
    planeTermVelocity.fill(0.0)
    hasVelocityEstimate = false
  }

  private fun advanceRelock(
    priorPlaneTerm: DoubleArray,
    measuredPlaneTerm: DoubleArray,
    supportRatio: Double,
    inlierCount: Int,
    fitRmsePx: Double,
    timestampNs: Long,
  ): RelockStep? {
    val target = relockTarget ?: return null
    val targetTimestampNs = relockTargetTimestampNs ?: return null
    val velocity = relockVelocity ?: return null
    if (timestampNs < targetTimestampNs || relockFramesRemaining <= 0) {
      cancelRelock()
      return null
    }

    val advanceSeconds =
      ((timestampNs - targetTimestampNs) / 1e9)
        .coerceIn(0.0, MAX_RELOCK_TARGET_ADVANCE_SECONDS)
    for (index in 0..2) target[index] += velocity[index] * advanceSeconds
    relockTargetTimestampNs = timestampNs

    val measurementIsUsable =
      supportRatio >= RELOCK_MEASUREMENT_MIN_SUPPORT_RATIO &&
        inlierCount >= RELOCK_MEASUREMENT_MIN_INLIERS &&
        fitRmsePx.isFinite() &&
        fitRmsePx <= RELOCK_MEASUREMENT_MAX_RMSE_PX
    val targetDeltaX = measuredPlaneTerm[0] - target[0]
    val targetDeltaY = measuredPlaneTerm[1] - target[1]
    val targetDeltaDepth = measuredPlaneTerm[2] - target[2]
    val measurementAgrees =
      measurementIsUsable &&
        hypot(targetDeltaX, targetDeltaY) <= RELOCK_TARGET_XY_TOLERANCE &&
        abs(targetDeltaDepth) <= RELOCK_TARGET_DEPTH_TOLERANCE
    if (measurementAgrees) {
      relockDisagreementCount = 0
      for (index in 0..2) {
        target[index] +=
          RELOCK_TARGET_MEASUREMENT_ALPHA * (measuredPlaneTerm[index] - target[index])
      }
    } else {
      relockDisagreementCount += 1
      if (relockDisagreementCount >= 2) {
        cancelRelock()
        return null
      }
    }

    val remainingBeforeStep = relockFramesRemaining
    val alpha = 1.0 / remainingBeforeStep.toDouble()
    val blended =
      DoubleArray(3) { index -> priorPlaneTerm[index] + alpha * (target[index] - priorPlaneTerm[index]) }
    relockFramesRemaining -= 1
    val completionVelocity =
      if (relockFramesRemaining == 0) velocity.copyOf() else null
    return RelockStep(blended, alpha, relockFramesRemaining, completionVelocity)
  }

  private fun completeRelock(velocity: DoubleArray) {
    for (index in 0..2) {
      planeTermVelocity[index] =
        velocity[index]
          .coerceIn(
            -MAX_PLANE_TERM_VELOCITY_PER_SECOND,
            MAX_PLANE_TERM_VELOCITY_PER_SECOND,
          )
    }
    hasVelocityEstimate = true
    clearRelockTransition()
  }

  private fun cancelRelock() {
    clearRelockTransition()
    relockBuffer.reset()
  }

  private fun clearRelockState() {
    clearRelockTransition()
    relockBuffer.reset()
  }

  private fun clearRelockTransition() {
    relockTarget = null
    relockTargetTimestampNs = null
    relockVelocity = null
    relockFramesRemaining = 0
    relockDisagreementCount = 0
  }

  private fun buildPlanePointWeights(
    trackIds: List<Int>,
    visualWeights: List<Float>,
  ): List<Double> {
    return trackIds.indices.map { index ->
      val visual = visualWeights.getOrNull(index)?.toDouble()?.coerceIn(0.02, 1.0) ?: 0.65
      val state = planeTrackStates[trackIds[index]]
      val temporal =
        if (state != null && state.observations >= MIN_PLANE_TRACK_OBSERVATIONS) {
          state.confidence.coerceIn(0.02, 1.0)
        } else {
          INITIAL_PLANE_TRACK_CONFIDENCE
        }
      sqrt(visual * temporal).coerceIn(0.02, 1.0)
    }
  }

  private fun weightedPlaneCoverage(
    points: List<Point>,
    weights: List<Double>,
    inlierIndices: List<Int>,
    intrinsics: CameraIntrinsics,
  ): Double {
    if (points.isEmpty() || points.size != weights.size || inlierIndices.isEmpty()) return 0.0
    val width = (intrinsics.cx * 2.0).coerceAtLeast(1.0)
    val height = (intrinsics.cy * 2.0).coerceAtLeast(1.0)
    val columns = 4
    val rows = 3
    val cellWeights = DoubleArray(columns * rows)
    for (index in inlierIndices) {
      val point = points.getOrNull(index) ?: continue
      val weight = weights.getOrNull(index) ?: continue
      if (point.x !in 0.0..width || point.y !in 0.0..height) continue
      val column = ((point.x / width) * columns).toInt().coerceIn(0, columns - 1)
      val row = ((point.y / height) * rows).toInt().coerceIn(0, rows - 1)
      cellWeights[row * columns + column] += weight
    }
    return cellWeights.count { it >= 0.30 }.toDouble() / cellWeights.size.toDouble()
  }

  private fun updatePlaneTrackStates(trackIds: List<Int>, errorsPx: List<Double>) {
    if (trackIds.size != errorsPx.size) return
    val activeIds = trackIds.toHashSet()
    planeTrackStates.keys.retainAll(activeIds)
    for (index in trackIds.indices) {
      val state = planeTrackStates.getOrPut(trackIds[index]) { PlaneTrackState() }
      val error = errorsPx[index]
      state.confidence =
        if (error.isFinite() && error <= PLANE_TRACK_AGREEMENT_PX) {
          state.confidence + PLANE_TRACK_CONFIDENCE_GAIN * (1.0 - state.confidence)
        } else {
          state.confidence * PLANE_TRACK_CONFIDENCE_DECAY
        }
      state.observations += 1
    }
  }

  private fun innovationLimits(
    supportRatio: Double,
    inlierCount: Int,
    timestampNs: Long,
  ): InnovationLimits {
    val base =
      when {
        supportRatio >= 0.65 && inlierCount >= 20 -> InnovationLimits(xy = 0.18, depth = 0.12)
        supportRatio >= 0.45 && inlierCount >= 10 -> InnovationLimits(xy = 0.10, depth = 0.07)
        else -> InnovationLimits(xy = 0.055, depth = 0.035)
      }
    val previousTimestampNs = lastPlaneTermTimestampNs
    val elapsedScale =
      if (previousTimestampNs == null) {
        1.0
      } else {
        val elapsedSeconds = ((timestampNs - previousTimestampNs).coerceAtLeast(0L) / 1e9)
        (1.0 + elapsedSeconds * 2.0).coerceIn(1.0, 1.4)
      }
    val startupScale = if (hasVelocityEstimate) 1.0 else 1.5
    return InnovationLimits(
      xy = base.xy * elapsedScale * startupScale,
      depth = base.depth * elapsedScale * startupScale,
    )
  }

  private fun planeMeasurementAlpha(supportRatio: Double, inlierCount: Int): Double =
    when {
      supportRatio >= 0.70 && inlierCount >= 20 -> 0.90
      supportRatio >= 0.50 && inlierCount >= 10 -> 0.70
      else -> 0.35
    }

  private fun predictedResult(
    result: SurfaceTrackerCore.TrackerResult,
    timestampNs: Long,
    coordinateMapper: CameraCoordinateMapper,
    reason: String,
  ): SurfaceTrackerCore.TrackerResult {
    val predicted = predictQuad(timestampNs, coordinateMapper)
      ?: return result.withImuReason(reason)
    return result.copy(
      mappedQuad = predicted,
      heldLastGoodTransform = true,
      rejectionReason = combineReasons(result.rejectionReason, "imu_plane:$reason"),
    )
  }

  private fun acceptPlaneTerm(
    planeTerm: DoubleArray,
    timestampNs: Long,
    updateVelocity: Boolean = true,
  ) {
    val previous = lastPlaneTerm
    val previousTimestampNs = lastPlaneTermTimestampNs
    if (updateVelocity && previous != null && previousTimestampNs != null) {
      val dtSeconds = (timestampNs - previousTimestampNs) / 1e9
      if (dtSeconds in 0.015..0.20) {
        for (index in 0..2) {
          val measuredVelocity =
            ((planeTerm[index] - previous[index]) / dtSeconds)
              .coerceIn(
                -MAX_PLANE_TERM_VELOCITY_PER_SECOND,
                MAX_PLANE_TERM_VELOCITY_PER_SECOND,
              )
          planeTermVelocity[index] =
            if (hasVelocityEstimate) {
              blend(planeTermVelocity[index], measuredVelocity)
            } else {
              measuredVelocity
            }
        }
        hasVelocityEstimate = true
      }
    }
    lastPlaneTerm = planeTerm.copyOf()
    lastPlaneTermTimestampNs = timestampNs
  }

  private fun predictedPlaneTermAt(timestampNs: Long): DoubleArray? {
    val planeTerm = lastPlaneTerm?.copyOf() ?: return null
    val correctionTimestampNs = lastPlaneTermTimestampNs ?: return planeTerm
    if (!hasVelocityEstimate || timestampNs <= correctionTimestampNs) return planeTerm
    val predictionSeconds =
      ((timestampNs - correctionTimestampNs) / 1e9).coerceIn(0.0, MAX_PREDICTION_SECONDS)
    for (index in 0..2) {
      val prediction =
        (planeTermVelocity[index] * predictionSeconds)
          .coerceIn(-MAX_PLANE_TERM_PREDICTION, MAX_PLANE_TERM_PREDICTION)
      planeTerm[index] += prediction
    }
    return planeTerm
  }

  private fun referenceHomographyAt(
    timestampNs: Long,
    coordinateMapper: CameraCoordinateMapper,
    orientationTimestampNs: Long = timestampNs,
  ): DoubleArray? {
    val referenceOrientation = referenceDeviceToWorld ?: return null
    val cameraToDevice = referenceCameraToDevice ?: return null
    val planeNormal = referencePlaneNormalCamera ?: return null
    val currentOrientation =
      orientationProvider.getDeviceToWorldRotation(orientationTimestampNs) ?: return null
    val planeTerm = predictedPlaneTermAt(timestampNs) ?: return null
    val relativeRotation =
      relativeCameraRotation(referenceOrientation, currentOrientation, cameraToDevice)
    return PlaneMotionGeometry.planeHomography(
      relativeRotation,
      planeNormal,
      planeTerm,
      coordinateMapper.intrinsics,
    )
  }

  private fun relativeCameraRotation(
    referenceDeviceToWorld: DoubleArray,
    currentDeviceToWorld: DoubleArray,
    cameraToDevice: DoubleArray,
  ): DoubleArray =
    PlaneMotionGeometry.multiply3x3(
      transpose3x3(cameraToDevice),
      PlaneMotionGeometry.multiply3x3(
        transpose3x3(currentDeviceToWorld),
        PlaneMotionGeometry.multiply3x3(referenceDeviceToWorld, cameraToDevice),
      ),
    )

  private fun horizontalPlaneNormalInReferenceCamera(
    referenceDeviceToWorld: DoubleArray,
    cameraToDevice: DoubleArray,
  ): DoubleArray? {
    val referenceCameraToWorld =
      PlaneMotionGeometry.multiply3x3(referenceDeviceToWorld, cameraToDevice)
    // Android's rotation-vector world frame has +Z opposite gravity.
    val normal =
      multiply3x3Vector(
        transpose3x3(referenceCameraToWorld),
        doubleArrayOf(0.0, 0.0, 1.0),
      )
    val length = vectorNorm(normal)
    if (!length.isFinite() || length <= 1e-8) return null
    return DoubleArray(3) { normal[it] / length }
  }

  private fun cameraToDeviceRotation(rotationDegrees: Int): DoubleArray {
    val rawToDisplay =
      when ((rotationDegrees % 360 + 360) % 360) {
        90 -> doubleArrayOf(0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0)
        180 -> doubleArrayOf(-1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0)
        270 -> doubleArrayOf(0.0, 1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 1.0)
        else -> identity3x3()
      }
    val displayCameraToDevice =
      doubleArrayOf(
        1.0, 0.0, 0.0,
        0.0, -1.0, 0.0,
        0.0, 0.0, -1.0,
      )
    return PlaneMotionGeometry.multiply3x3(displayCameraToDevice, rawToDisplay)
  }

  private fun multiply3x3Vector(matrix: DoubleArray, vector: DoubleArray): DoubleArray =
    doubleArrayOf(
      matrix[0] * vector[0] + matrix[1] * vector[1] + matrix[2] * vector[2],
      matrix[3] * vector[0] + matrix[4] * vector[1] + matrix[5] * vector[2],
      matrix[6] * vector[0] + matrix[7] * vector[1] + matrix[8] * vector[2],
    )

  private fun transpose3x3(matrix: DoubleArray): DoubleArray =
    doubleArrayOf(
      matrix[0], matrix[3], matrix[6],
      matrix[1], matrix[4], matrix[7],
      matrix[2], matrix[5], matrix[8],
    )

  private fun identity3x3(): DoubleArray =
    doubleArrayOf(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)

  private fun vectorNorm(vector: DoubleArray): Double =
    sqrt(vector.sumOf { value -> value * value })

  private fun blend(previous: Double, measured: Double): Double =
    previous * (1.0 - VELOCITY_FILTER_ALPHA) + measured * VELOCITY_FILTER_ALPHA

  private fun isValidQuad(quad: List<Point>): Boolean {
    if (quad.size != 4 || quad.any { !it.x.isFinite() || !it.y.isFinite() }) return false
    var direction = 0
    var twiceArea = 0.0
    for (index in quad.indices) {
      val a = quad[index]
      val b = quad[(index + 1) % quad.size]
      val c = quad[(index + 2) % quad.size]
      twiceArea += a.x * b.y - b.x * a.y
      val cross = (b.x - a.x) * (c.y - b.y) - (b.y - a.y) * (c.x - b.x)
      if (kotlin.math.abs(cross) > 1e-6) {
        val nextDirection = if (cross > 0.0) 1 else -1
        if (direction != 0 && direction != nextDirection) return false
        direction = nextDirection
      }
    }
    return direction != 0 && kotlin.math.abs(twiceArea) > 16.0
  }

  private fun SurfaceTrackerCore.TrackerResult.withImuReason(
    reason: String,
  ): SurfaceTrackerCore.TrackerResult =
    copy(rejectionReason = combineReasons(rejectionReason, "imu_plane:$reason"))

  private fun combineReasons(first: String?, second: String): String =
    if (first.isNullOrBlank()) second else "$first|$second"

  private fun copyPoint(point: Point): Point = Point(point.x, point.y)

  private fun format(value: Double): String =
    String.format(java.util.Locale.US, "%.3f", value)
}
