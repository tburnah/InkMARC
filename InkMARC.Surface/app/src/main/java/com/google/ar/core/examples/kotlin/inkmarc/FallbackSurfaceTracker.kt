package com.inkmarc

import android.media.Image
import android.opengl.Matrix
import android.os.SystemClock
import android.util.Log
import android.view.Choreographer
import androidx.lifecycle.DefaultLifecycleObserver
import androidx.lifecycle.LifecycleOwner
import com.google.ar.core.Camera
import com.google.ar.core.Coordinates2d
import com.google.ar.core.Frame
import com.google.ar.core.Plane
import com.google.ar.core.Pose
import com.google.ar.core.TrackingState
import com.google.ar.core.examples.java.common.samplerender.arcore.PlaneRenderer
import com.inkmarc.fallback.SurfaceTrackerCore
import com.google.ar.core.exceptions.NotYetAvailableException
import org.opencv.android.OpenCVLoader
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Point

data class SurfaceTrackingTelemetry(
  val state: String,
  val motionModel: String?,
  val heldLastGoodTransform: Boolean,
  val activeTrackCount: Int,
  val candidateTrackCount: Int,
  val stableTrackCount: Int,
  val inlierCount: Int,
  val inlierRatio: Float,
  val meanReprojectionErrorPx: Double?,
  val referenceFeatureCoverage: Float,
  val currentFeatureCoverage: Float,
  val transformCondition: Double?,
  val confidence: Float,
  val rejectionReason: String?,
  val handMaskAgeMs: Double? = null,
  val handMaskAreaRatio: Double = 0.0,
  val motionMaskAreaRatio: Double = 0.0,
  val combinedMaskAreaRatio: Double = 0.0,
  val maskRejectedTrackCount: Int = 0,
  val reacquiredTrackCount: Int = 0,
  val effectivePlaneSupport: Float? = null,
  val weightedPlaneCoverage: Float? = null,
)

data class SurfaceTrackingSnapshot(
  val viewQuad: FloatArray?,
  val telemetry: SurfaceTrackingTelemetry?,
)

class FallbackSurfaceTracker(private val activity: InkMarcActivity) : DefaultLifecycleObserver {
  companion object {
    private const val TAG = "FallbackSurfaceTracker"
    private const val RESEED_INTERVAL_FRAMES = 15
    private const val MAX_INERTIAL_RECOVERY_FRAMES = 30
    // The 183433 replay reacquired only two tracks and slightly reduced valid-frame count. Keep the
    // implementation available for controlled experiments without paying its live search cost yet.
    private const val ENABLE_OCCLUSION_TRACK_MEMORY = false
  }

  private val tracker =
    if (OpenCVLoader.initDebug()) {
      // Keep descriptor matching out of the camera callback. SIFT gallery attempts previously
      // blocked this path for 200-600 ms and rebased the IMU plane whenever they promoted a
      // keyframe. Sparse tracking plus the controlled visual re-lock in ImuPlaneConstraint is the
      // live path; a gallery can be restored later as an asynchronous, recovery-only component.
      SurfaceTrackerCore(
        SurfaceTrackerCore.Config(
          maxFeatures = 90,
          maxDetectedCorners = 360,
          useFullFrameFeatureSupport = true,
          allowOffscreenMappedQuad = true,
          enableOcclusionTrackMemory = ENABLE_OCCLUSION_TRACK_MEMORY,
        ),
      )
    } else {
      null
    }
  private val imuPlaneConstraint = ImuPlaneConstraint(activity.gravityDirectionProvider)
  private val dynamicOcclusionMasker = DynamicOcclusionMasker()
  private val projectionMatrix = FloatArray(16)
  private val viewMatrix = FloatArray(16)
  private val clipPoint = FloatArray(4)
  private val cameraPoint = FloatArray(4)
  private val worldPoint = FloatArray(4)
  private var grayMat: Mat? = null
  private var directRowBuffer = ByteArray(0)
  private var stridedRowBuffer = ByteArray(0)
  private var contiguousFrameBuffer = ByteArray(0)
  @Volatile private var hasSeed = false
  private var inertialRecoveryFrameCount = 0
  private var lastTrackerFrameTiming: CameraFrameTiming? = null
  private var frameCounter = 0L
  private var lastSeedPlane: Plane? = null
  private var lastSeedFrame = Long.MIN_VALUE
  private var lastStatus: String? = null
  private var pendingPlacementWorldQuad: List<FloatArray>? = null
  @Volatile private var latestCoordinateMapper: CameraCoordinateMapper? = null
  private var predictionLoopActive = false
  @Volatile private var latestTrackedViewQuad: FloatArray? = null
  @Volatile private var latestTrackingTelemetry: SurfaceTrackingTelemetry? = null
  private var latestOcclusionTelemetry = OcclusionTelemetry()

  private data class OcclusionTelemetry(
    val handMaskAgeMs: Double? = null,
    val handMaskAreaRatio: Double = 0.0,
    val motionMaskAreaRatio: Double = 0.0,
    val combinedMaskAreaRatio: Double = 0.0,
  )

  override fun onResume(owner: LifecycleOwner) {
    if (tracker == null) {
      publishStatus(activity.getString(R.string.fallback_status_disabled))
    } else {
      publishStatus(activity.getString(R.string.fallback_status_waiting_seed))
    }
    predictionLoopActive = true
    Choreographer.getInstance().postFrameCallback(predictionFrameCallback)
  }

  override fun onPause(owner: LifecycleOwner) {
    predictionLoopActive = false
    Choreographer.getInstance().removeFrameCallback(predictionFrameCallback)
    dynamicOcclusionMasker.reset()
    clearOverlay()
  }

  override fun onDestroy(owner: LifecycleOwner) {
    clearOverlay()
    tracker?.close()
    imuPlaneConstraint.reset()
    dynamicOcclusionMasker.close()
    grayMat?.release()
    grayMat = null
    latestCoordinateMapper = null
  }

  private val predictionFrameCallback =
    object : Choreographer.FrameCallback {
      override fun doFrame(frameTimeNanos: Long) {
        if (!predictionLoopActive) return
        val coordinateMapper = latestCoordinateMapper
        if (hasSeed && coordinateMapper != null) {
          imuPlaneConstraint
            .predictQuad(SystemClock.elapsedRealtimeNanos(), coordinateMapper)
            ?.let(coordinateMapper::imageQuadToView)
            ?.let(::updateOverlay)
        }
        Choreographer.getInstance().postFrameCallback(this)
      }
    }

  fun requestSeedFromPlacementPose(pose: Pose) {
    // A one-to-one comparison must track the same physical rectangle as ARCore. The larger seed
    // remains useful for the standalone fallback UI, but would make corner agreement meaningless.
    val seedScale = if (InkMarcActivity.ENABLE_ARCORE_COMPARISON) 1f else PlacementRectangleSpec.FALLBACK_SEED_SCALE
    pendingPlacementWorldQuad =
      buildWorldQuadCorners(
        pose,
        PlacementRectangleSpec.WIDTH_METERS * seedScale,
        PlacementRectangleSpec.HEIGHT_METERS * seedScale
      )
    hasSeed = false
    inertialRecoveryFrameCount = 0
    lastTrackerFrameTiming = null
    imuPlaneConstraint.reset()
    dynamicOcclusionMasker.reset()
    lastSeedPlane = null
    latestTrackedViewQuad = null
    latestTrackingTelemetry = waitingTelemetry("placement_requested")
  }

  fun getLatestTrackedViewQuad(): FloatArray? = latestTrackedViewQuad?.clone()

  fun getLatestTrackingSnapshot(): SurfaceTrackingSnapshot =
    SurfaceTrackingSnapshot(
      viewQuad = latestTrackedViewQuad?.clone(),
      telemetry = latestTrackingTelemetry,
    )

  /** Processes Camera2 frames without creating or touching an ARCore session. */
  fun processStandaloneFrame(
    image: Image,
    frameTiming: CameraFrameTiming,
    coordinateMapper: CameraCoordinateMapper,
  ) {
    val tracker = tracker ?: return
    val timestampNs = frameTiming.centerExposureTimestampNs
    latestCoordinateMapper = coordinateMapper
    frameCounter += 1

    val currentViewQuad = getLatestTrackedViewQuad()
    val currentImageQuad = currentViewQuad?.let(coordinateMapper::viewQuadToImage)
    val inferenceEngine = activity.inkMarcInferenceEngine
    if (inferenceEngine?.canSubmit(timestampNs) == true) {
      inferenceEngine.submitCameraImage(
        image = image,
        timestampNs = timestampNs,
        surfaceCornersImage = currentImageQuad,
      )
    }
    if (activity.da3DepthEstimator?.canSubmit(timestampNs) == true) {
      activity.da3DepthEstimator?.submitCameraImage(image, timestampNs)
    }
    activity.handOcclusionMasker?.submitFrame(image, timestampNs)

    val grayFrame = acquireGrayMat(image)
    val initialFlowTransform =
      if (hasSeed) {
        lastTrackerFrameTiming?.let { previousFrameTiming ->
          imuPlaneConstraint.frameToFrameTransform(
            previousFrameTiming.centerExposureTimestampNs,
            timestampNs,
            coordinateMapper,
            previousFrameTiming,
            frameTiming,
          )
        }
      } else {
        null
      }
    val handMaskSnapshot =
      activity.handOcclusionMasker?.getExclusionMaskSnapshot(
        image.width,
        image.height,
        timestampNs,
      )
    val motionMaskSnapshot = dynamicOcclusionMasker.update(grayFrame, initialFlowTransform)
    val exclusionMask =
      combineExclusionMasks(
        grayFrame.rows(),
        grayFrame.cols(),
        handMaskSnapshot?.mask,
        motionMaskSnapshot.mask,
      )
    latestOcclusionTelemetry =
      OcclusionTelemetry(
        handMaskAgeMs = handMaskSnapshot?.ageMs,
        handMaskAreaRatio = handMaskSnapshot?.areaRatio ?: 0.0,
        motionMaskAreaRatio = motionMaskSnapshot.areaRatio,
        combinedMaskAreaRatio = exclusionMask?.let(::maskAreaRatio) ?: 0.0,
      )
    try {
      val tap = activity.view.touchHelper.pollTap()
      if (tap != null) {
        try {
          val center = coordinateMapper.viewToImage(tap.x, tap.y)
          seedStandaloneTracker(grayFrame, center, exclusionMask, coordinateMapper, frameTiming)
        } finally {
          tap.recycle()
        }
        return
      }

      if (!hasSeed) {
        latestTrackingTelemetry = waitingTelemetry("waiting_for_seed")
        clearOverlay()
        publishStatus(activity.getString(R.string.fallback_status_waiting_seed))
        return
      }

      if (inertialRecoveryFrameCount > 0) {
        recoverStandaloneTracker(grayFrame, exclusionMask, coordinateMapper, frameTiming)
        return
      }

      val rawResult = tracker.update(grayFrame, exclusionMask, initialFlowTransform)
      lastTrackerFrameTiming = frameTiming
      val outputTimestampNs = SystemClock.elapsedRealtimeNanos()
      val result =
        imuPlaneConstraint.constrain(
          rawResult,
          timestampNs,
          outputTimestampNs,
          coordinateMapper,
          frameTiming,
        )
      val viewQuad = result.mappedQuad?.let(coordinateMapper::imageQuadToView)
      cacheTrackerResult(result, viewQuad)
      when (result.state) {
        SurfaceTrackerCore.State.TRACKING -> {
          inertialRecoveryFrameCount = 0
          updateOverlay(viewQuad)
          activity.updateInferenceCursor()
          publishStatus(
            activity.getString(
              R.string.fallback_status_tracking,
              result.activeTrackCount,
              result.inlierCount,
            ),
          )
        }
        SurfaceTrackerCore.State.INSUFFICIENT_FEATURES -> {
          if (result.heldLastGoodTransform && viewQuad != null) {
            updateOverlay(viewQuad)
          } else {
            clearOverlay()
          }
          publishStatus(activity.getString(R.string.fallback_status_insufficient))
        }
        SurfaceTrackerCore.State.LOST -> {
          if (viewQuad != null) {
            inertialRecoveryFrameCount = 1
            updateOverlay(viewQuad)
            publishStatus(activity.getString(R.string.fallback_status_insufficient))
          } else {
            hasSeed = false
            lastTrackerFrameTiming = null
            clearOverlay()
            publishStatus(activity.getString(R.string.fallback_status_lost))
          }
        }
        SurfaceTrackerCore.State.IDLE -> {
          hasSeed = false
          lastTrackerFrameTiming = null
          clearOverlay()
          publishStatus(activity.getString(R.string.fallback_status_waiting_seed))
        }
      }
    } finally {
      exclusionMask?.release()
      handMaskSnapshot?.mask?.release()
      motionMaskSnapshot.mask?.release()
    }
  }

  private fun seedStandaloneTracker(
    grayFrame: Mat,
    center: FloatArray,
    exclusionMask: Mat?,
    coordinateMapper: CameraCoordinateMapper,
    frameTiming: CameraFrameTiming,
  ) {
    val timestampNs = frameTiming.centerExposureTimestampNs
    val halfWidth = grayFrame.cols() * 0.22
    val halfHeight = grayFrame.rows() * 0.18
    val minX = (center[0] - halfWidth).coerceIn(0.0, grayFrame.cols() - 1.0)
    val maxX = (center[0] + halfWidth).coerceIn(0.0, grayFrame.cols() - 1.0)
    val minY = (center[1] - halfHeight).coerceIn(0.0, grayFrame.rows() - 1.0)
    val maxY = (center[1] + halfHeight).coerceIn(0.0, grayFrame.rows() - 1.0)
    val imageQuad =
      listOf(
        Point(minX, minY),
        Point(maxX, minY),
        Point(maxX, maxY),
        Point(minX, maxY),
      )
    val seedResult = tracker?.seedReference(grayFrame, imageQuad, exclusionMask) ?: return
    hasSeed = seedResult.success
    lastSeedPlane = null
    lastSeedFrame = frameCounter
    if (seedResult.success) {
      imuPlaneConstraint.seed(imageQuad, timestampNs, coordinateMapper, frameTiming = frameTiming)
      dynamicOcclusionMasker.reset(grayFrame)
      inertialRecoveryFrameCount = 0
      lastTrackerFrameTiming = frameTiming
      latestTrackedViewQuad = coordinateMapper.imageQuadToView(imageQuad)
      latestTrackingTelemetry = seededTelemetry(seedResult.trackCount)
      updateOverlay(latestTrackedViewQuad)
      publishStatus(activity.getString(R.string.fallback_status_seeded))
    } else {
      inertialRecoveryFrameCount = 0
      lastTrackerFrameTiming = null
      latestTrackedViewQuad = null
      latestTrackingTelemetry = waitingTelemetry("seed_failed:${seedResult.trackCount}")
      clearOverlay()
      publishStatus(activity.getString(R.string.fallback_status_seed_failed, seedResult.trackCount))
    }
  }

  private fun recoverStandaloneTracker(
    grayFrame: Mat,
    exclusionMask: Mat?,
    coordinateMapper: CameraCoordinateMapper,
    frameTiming: CameraFrameTiming,
  ) {
    val timestampNs = frameTiming.centerExposureTimestampNs
    val predictedImageQuad =
      imuPlaneConstraint.predictQuad(timestampNs, coordinateMapper)
        ?: run {
          hasSeed = false
          inertialRecoveryFrameCount = 0
          lastTrackerFrameTiming = null
          clearOverlay()
          publishStatus(activity.getString(R.string.fallback_status_lost))
          return
        }
    val seedResult = tracker?.seedReference(grayFrame, predictedImageQuad, exclusionMask) ?: return
    if (seedResult.success) {
      imuPlaneConstraint.seed(
        predictedImageQuad,
        timestampNs,
        coordinateMapper,
        frameTiming = frameTiming,
      )
      dynamicOcclusionMasker.reset(grayFrame)
      inertialRecoveryFrameCount = 0
      lastTrackerFrameTiming = frameTiming
      val outputQuad =
        imuPlaneConstraint.predictQuad(SystemClock.elapsedRealtimeNanos(), coordinateMapper)
          ?: predictedImageQuad
      val viewQuad = coordinateMapper.imageQuadToView(outputQuad)
      latestTrackedViewQuad = viewQuad
      latestTrackingTelemetry = seededTelemetry(seedResult.trackCount, "auto_recovered")
      updateOverlay(viewQuad)
      publishStatus(activity.getString(R.string.fallback_status_seeded))
      return
    }

    inertialRecoveryFrameCount += 1
    if (inertialRecoveryFrameCount > MAX_INERTIAL_RECOVERY_FRAMES) {
      hasSeed = false
      inertialRecoveryFrameCount = 0
      lastTrackerFrameTiming = null
      clearOverlay()
      latestTrackingTelemetry = waitingTelemetry("auto_recovery_timeout")
      publishStatus(activity.getString(R.string.fallback_status_lost))
      return
    }

    val outputQuad =
      imuPlaneConstraint.predictQuad(SystemClock.elapsedRealtimeNanos(), coordinateMapper)
        ?: predictedImageQuad
    val viewQuad = coordinateMapper.imageQuadToView(outputQuad)
    latestTrackedViewQuad = viewQuad
    latestTrackingTelemetry = recoveryTelemetry(seedResult.trackCount)
    updateOverlay(viewQuad)
    publishStatus(activity.getString(R.string.fallback_status_insufficient))
  }

  /**
   * Processes an ARCore camera frame. Comparison mode deliberately does not consume ARCore's
   * plane results or camera-tracking state: the shared tap supplies the initial surface only.
   */
  fun processFrame(
    frame: Frame,
    camera: Camera,
    allPlanes: Collection<Plane>,
    comparisonMode: Boolean = false,
  ) {
    val tracker = tracker ?: return
    val image =
      try {
        frame.acquireCameraImage()
      } catch (_: NotYetAvailableException) {
        return
      } catch (e: Exception) {
        Log.w(TAG, "Skipping fallback frame: ${e.message}")
        return
      }

    try {
      frameCounter += 1
      val inferenceEngine = activity.inkMarcInferenceEngine
      if (inferenceEngine?.canSubmit(frame.timestamp) == true) {
        inferenceEngine.submitCameraImage(
          image = image,
          timestampNs = frame.timestamp,
          frame = frame,
          surfaceCornersView = activity.getActiveSurfaceCornersView(),
        )
      }

      if (!comparisonMode && camera.trackingState != TrackingState.TRACKING) {
        latestTrackedViewQuad = null
        latestTrackingTelemetry = waitingTelemetry("waiting_for_arcore")
        clearOverlay()
        publishStatus(activity.getString(R.string.fallback_status_waiting_arcore))
        return
      }

      val surfaceWidth = activity.view.surfaceView.width
      val surfaceHeight = activity.view.surfaceView.height
      if (surfaceWidth <= 0 || surfaceHeight <= 0) return

      val grayFrame = acquireGrayMat(image)
      activity.handOcclusionMasker?.submitFrame(image, frame.timestamp)
      val exclusionMask =
        activity.handOcclusionMasker?.getExclusionMask(image.width, image.height, frame.timestamp)
      try {
        val trackedPlane = if (comparisonMode) null else selectTrackedPlane(allPlanes, camera)
        if (trackedPlane != null) {
          maybeSeedFromPlane(
            frame,
            camera,
            trackedPlane,
            grayFrame,
            surfaceWidth,
            surfaceHeight,
            exclusionMask,
          )
          if (hasSeed) {
            val result = tracker.update(grayFrame, exclusionMask)
            val viewQuad = result.mappedQuad?.let { mapImageQuadToView(frame, it) }
            cacheTrackerResult(result, viewQuad)
            activity.updateInferenceCursor()
            publishStatus(activity.getString(R.string.fallback_status_seeded))
          }
          clearOverlay()
          return
        }

        maybeSeedFromPlacementQuad(
          frame,
          camera,
          grayFrame,
          surfaceWidth,
          surfaceHeight,
          exclusionMask,
        )

        if (!hasSeed) {
          latestTrackedViewQuad = null
          latestTrackingTelemetry = waitingTelemetry("waiting_for_seed")
          clearOverlay()
          publishStatus(activity.getString(R.string.fallback_status_waiting_seed))
          return
        }

        val result = tracker.update(grayFrame, exclusionMask)
        val viewQuad = result.mappedQuad?.let { mapImageQuadToView(frame, it) }
        cacheTrackerResult(result, viewQuad)
        when (result.state) {
          SurfaceTrackerCore.State.TRACKING -> {
            updateOverlay(viewQuad)
            activity.updateInferenceCursor()
            publishStatus(
              activity.getString(
                R.string.fallback_status_tracking,
                result.activeTrackCount,
                result.inlierCount
              )
            )
          }
          SurfaceTrackerCore.State.INSUFFICIENT_FEATURES -> {
            if (result.heldLastGoodTransform && viewQuad != null) {
              updateOverlay(viewQuad)
            } else {
              clearOverlay()
            }
            publishStatus(activity.getString(R.string.fallback_status_insufficient))
          }
          SurfaceTrackerCore.State.LOST -> {
            hasSeed = false
            lastSeedPlane = null
            clearOverlay()
            publishStatus(activity.getString(R.string.fallback_status_lost))
          }
          SurfaceTrackerCore.State.IDLE -> {
            clearOverlay()
            publishStatus(activity.getString(R.string.fallback_status_waiting_seed))
          }
        }
      } finally {
        exclusionMask?.release()
      }
    } finally {
      image.close()
    }
  }

  private fun maybeSeedFromPlane(
    frame: Frame,
    camera: Camera,
    plane: Plane,
    grayFrame: Mat,
    surfaceWidth: Int,
    surfaceHeight: Int,
    exclusionMask: Mat?,
  ) {
    if (!shouldReseed(plane)) return
    val imageQuad = projectPlaneQuadToImage(frame, camera, plane, surfaceWidth, surfaceHeight) ?: return
    val seedResult = tracker?.seedReference(grayFrame, imageQuad, exclusionMask) ?: return
    hasSeed = seedResult.success
    if (seedResult.success) {
      lastSeedPlane = plane
      lastSeedFrame = frameCounter
      latestTrackingTelemetry = seededTelemetry(seedResult.trackCount)
      publishStatus(activity.getString(R.string.fallback_status_seeded))
    } else {
      latestTrackingTelemetry = waitingTelemetry("seed_failed:${seedResult.trackCount}")
      publishStatus(activity.getString(R.string.fallback_status_seed_failed, seedResult.trackCount))
    }
  }

  private fun shouldReseed(plane: Plane): Boolean {
    if (!hasSeed) return true
    if (plane !== lastSeedPlane) return true
    return frameCounter - lastSeedFrame >= RESEED_INTERVAL_FRAMES
  }

  private fun maybeSeedFromPlacementQuad(
    frame: Frame,
    camera: Camera,
    grayFrame: Mat,
    surfaceWidth: Int,
    surfaceHeight: Int,
    exclusionMask: Mat?,
  ) {
    val worldQuad = pendingPlacementWorldQuad ?: return
    val imageQuad =
      projectWorldQuadToImage(frame, camera, worldQuad, surfaceWidth, surfaceHeight) ?: return
    val seedResult = tracker?.seedReference(grayFrame, imageQuad, exclusionMask) ?: return
    hasSeed = seedResult.success
    if (seedResult.success) {
      pendingPlacementWorldQuad = null
      lastSeedPlane = null
      lastSeedFrame = frameCounter
      latestTrackingTelemetry = seededTelemetry(seedResult.trackCount)
      publishStatus(activity.getString(R.string.fallback_status_seeded))
    } else {
      latestTrackingTelemetry = waitingTelemetry("seed_failed:${seedResult.trackCount}")
      publishStatus(activity.getString(R.string.fallback_status_seed_failed, seedResult.trackCount))
    }
  }

  private fun selectTrackedPlane(allPlanes: Collection<Plane>, camera: Camera): Plane? =
    allPlanes
      .asSequence()
      .filter { it.trackingState == TrackingState.TRACKING && it.subsumedBy == null }
      .filter { PlaneRenderer.calculateDistanceToPlane(it.centerPose, camera.pose) > 0f }
      .maxByOrNull { it.extentX * it.extentZ }

  private fun projectPlaneQuadToImage(
    frame: Frame,
    camera: Camera,
    plane: Plane,
    surfaceWidth: Int,
    surfaceHeight: Int
  ): List<Point>? {
    camera.getProjectionMatrix(projectionMatrix, 0, 0.1f, 100f)
    camera.getViewMatrix(viewMatrix, 0)

    val pose = plane.centerPose
    val right = FloatArray(3)
    val forward = FloatArray(3)
    pose.getTransformedAxis(0, 1f, right, 0)
    pose.getTransformedAxis(2, 1f, forward, 0)
    val centerX = pose.tx()
    val centerY = pose.ty()
    val centerZ = pose.tz()
    val halfX = plane.extentX * 0.5f
    val halfZ = plane.extentZ * 0.5f

    val worldCorners =
      listOf(
        floatArrayOf(centerX - right[0] * halfX - forward[0] * halfZ, centerY - right[1] * halfX - forward[1] * halfZ, centerZ - right[2] * halfX - forward[2] * halfZ),
        floatArrayOf(centerX + right[0] * halfX - forward[0] * halfZ, centerY + right[1] * halfX - forward[1] * halfZ, centerZ + right[2] * halfX - forward[2] * halfZ),
        floatArrayOf(centerX + right[0] * halfX + forward[0] * halfZ, centerY + right[1] * halfX + forward[1] * halfZ, centerZ + right[2] * halfX + forward[2] * halfZ),
        floatArrayOf(centerX - right[0] * halfX + forward[0] * halfZ, centerY - right[1] * halfX + forward[1] * halfZ, centerZ - right[2] * halfX + forward[2] * halfZ)
      )

    return projectWorldQuadToImage(frame, camera, worldCorners, surfaceWidth, surfaceHeight)
  }

  private fun projectWorldQuadToImage(
    frame: Frame,
    camera: Camera,
    worldQuad: List<FloatArray>,
    surfaceWidth: Int,
    surfaceHeight: Int
  ): List<Point>? {
    camera.getProjectionMatrix(projectionMatrix, 0, 0.1f, 100f)
    camera.getViewMatrix(viewMatrix, 0)

    val viewCoords = FloatArray(worldQuad.size * 2)
    for ((index, point) in worldQuad.withIndex()) {
      val projected =
        projectWorldPointToView(point[0], point[1], point[2], surfaceWidth, surfaceHeight)
          ?: return null
      viewCoords[index * 2] = projected.first
      viewCoords[index * 2 + 1] = projected.second
    }

    val imageCoords = FloatArray(viewCoords.size)
    frame.transformCoordinates2d(Coordinates2d.VIEW, viewCoords, Coordinates2d.IMAGE_PIXELS, imageCoords)

    val imageWidth = grayMat?.cols() ?: return null
    val imageHeight = grayMat?.rows() ?: return null
    val points = ArrayList<Point>(worldQuad.size)
    for (i in imageCoords.indices step 2) {
      val x = imageCoords[i].toDouble()
      val y = imageCoords[i + 1].toDouble()
      if (x < 0.0 || y < 0.0 || x >= imageWidth || y >= imageHeight) return null
      points.add(Point(x, y))
    }
    return points
  }

  private fun mapImageQuadToView(frame: Frame, imageQuad: List<Point>): FloatArray? {
    if (imageQuad.size < 4) return null
    val imageCoords = FloatArray(imageQuad.size * 2)
    for ((index, point) in imageQuad.withIndex()) {
      imageCoords[index * 2] = point.x.toFloat()
      imageCoords[index * 2 + 1] = point.y.toFloat()
    }
    val viewCoords = FloatArray(imageCoords.size)
    frame.transformCoordinates2d(Coordinates2d.IMAGE_PIXELS, imageCoords, Coordinates2d.VIEW, viewCoords)
    return viewCoords
  }

  private fun projectWorldPointToView(
    x: Float,
    y: Float,
    z: Float,
    surfaceWidth: Int,
    surfaceHeight: Int
  ): Pair<Float, Float>? {
    worldPoint[0] = x
    worldPoint[1] = y
    worldPoint[2] = z
    worldPoint[3] = 1f
    Matrix.multiplyMV(cameraPoint, 0, viewMatrix, 0, worldPoint, 0)
    Matrix.multiplyMV(clipPoint, 0, projectionMatrix, 0, cameraPoint, 0)
    val w = clipPoint[3]
    if (w == 0f) return null
    val ndcX = clipPoint[0] / w
    val ndcY = clipPoint[1] / w
    val ndcZ = clipPoint[2] / w
    if (!ndcX.isFinite() || !ndcY.isFinite() || ndcZ < -1f || ndcZ > 1f) return null
    val viewX = (ndcX * 0.5f + 0.5f) * surfaceWidth
    val viewY = (0.5f - ndcY * 0.5f) * surfaceHeight
    return viewX to viewY
  }

  private fun acquireGrayMat(image: Image): Mat {
    val plane = image.planes[0]
    val width = image.width
    val height = image.height
    val grayMat =
      this.grayMat?.takeIf { it.rows() == height && it.cols() == width }
        ?: Mat(height, width, CvType.CV_8UC1).also {
          this.grayMat?.release()
          this.grayMat = it
        }

    val directRow = ensureCapacity(width, true)
    val stridedRow = ensureCapacity(plane.rowStride, false)
    val buffer = plane.buffer.duplicate()
    val rowStride = plane.rowStride
    val pixelStride = plane.pixelStride
    val contiguousSize = width * height
    if (pixelStride == 1 && rowStride == width && buffer.remaining() >= contiguousSize) {
      if (contiguousFrameBuffer.size < contiguousSize) {
        contiguousFrameBuffer = ByteArray(contiguousSize)
      }
      buffer.get(contiguousFrameBuffer, 0, contiguousSize)
      grayMat.put(0, 0, contiguousFrameBuffer)
      return grayMat
    }
    for (row in 0 until height) {
      buffer.position(row * rowStride)
      if (pixelStride == 1) {
        buffer.get(directRow, 0, width)
        grayMat.put(row, 0, directRow)
      } else {
        val bytesToRead = minOf(rowStride, buffer.remaining())
        buffer.get(stridedRow, 0, bytesToRead)
        for (col in 0 until width) {
          directRow[col] = stridedRow[col * pixelStride]
        }
        grayMat.put(row, 0, directRow)
      }
    }
    return grayMat
  }

  private fun ensureCapacity(size: Int, direct: Boolean): ByteArray {
    if (direct) {
      if (directRowBuffer.size < size) {
        directRowBuffer = ByteArray(size)
      }
      return directRowBuffer
    }
    if (stridedRowBuffer.size < size) {
      stridedRowBuffer = ByteArray(size)
    }
    return stridedRowBuffer
  }

  private fun combineExclusionMasks(rows: Int, cols: Int, vararg masks: Mat?): Mat? {
    val usable = masks.filterNotNull().filter { !it.empty() && it.rows() == rows && it.cols() == cols }
    if (usable.isEmpty()) return null
    val combined = Mat.zeros(rows, cols, CvType.CV_8UC1)
    for (mask in usable) {
      Core.bitwise_or(combined, mask, combined)
    }
    return combined
  }

  private fun maskAreaRatio(mask: Mat): Double =
    Core.countNonZero(mask).toDouble() / (mask.rows().toDouble() * mask.cols())

  private fun cacheTrackerResult(result: SurfaceTrackerCore.TrackerResult, viewQuad: FloatArray?) {
    latestTrackedViewQuad = viewQuad?.clone()
    latestTrackingTelemetry =
      SurfaceTrackingTelemetry(
        state = result.state.name.lowercase(),
        motionModel = result.transform?.model?.name?.lowercase(),
        heldLastGoodTransform = result.heldLastGoodTransform,
        activeTrackCount = result.activeTrackCount,
        candidateTrackCount = result.candidateTrackCount,
        stableTrackCount = result.stableTrackCount,
        inlierCount = result.inlierCount,
        inlierRatio = result.inlierRatio,
        meanReprojectionErrorPx = result.meanReprojectionErrorPx,
        referenceFeatureCoverage = result.referenceFeatureCoverage,
        currentFeatureCoverage = result.currentFeatureCoverage,
        transformCondition = result.transformCondition,
        confidence = result.confidence,
        rejectionReason = result.rejectionReason,
        handMaskAgeMs = latestOcclusionTelemetry.handMaskAgeMs,
        handMaskAreaRatio = latestOcclusionTelemetry.handMaskAreaRatio,
        motionMaskAreaRatio = latestOcclusionTelemetry.motionMaskAreaRatio,
        combinedMaskAreaRatio = latestOcclusionTelemetry.combinedMaskAreaRatio,
        maskRejectedTrackCount = result.maskRejectedTrackCount,
        reacquiredTrackCount = result.reacquiredTrackCount,
        effectivePlaneSupport = result.effectivePlaneSupport,
        weightedPlaneCoverage = result.weightedPlaneCoverage,
      )
  }

  private fun seededTelemetry(
    trackCount: Int,
    reason: String? = null,
  ): SurfaceTrackingTelemetry =
    SurfaceTrackingTelemetry(
      state = "seeded",
      motionModel = null,
      heldLastGoodTransform = false,
      activeTrackCount = trackCount,
      candidateTrackCount = trackCount,
      stableTrackCount = 0,
      inlierCount = 0,
      inlierRatio = 0f,
      meanReprojectionErrorPx = null,
      referenceFeatureCoverage = 0f,
      currentFeatureCoverage = 0f,
      transformCondition = null,
      confidence = 0f,
      rejectionReason = reason,
      handMaskAgeMs = latestOcclusionTelemetry.handMaskAgeMs,
      handMaskAreaRatio = latestOcclusionTelemetry.handMaskAreaRatio,
      motionMaskAreaRatio = latestOcclusionTelemetry.motionMaskAreaRatio,
      combinedMaskAreaRatio = latestOcclusionTelemetry.combinedMaskAreaRatio,
    )

  private fun recoveryTelemetry(trackCount: Int): SurfaceTrackingTelemetry =
    SurfaceTrackingTelemetry(
      state = "recovering",
      motionModel = "imu",
      heldLastGoodTransform = true,
      activeTrackCount = trackCount,
      candidateTrackCount = trackCount,
      stableTrackCount = 0,
      inlierCount = 0,
      inlierRatio = 0f,
      meanReprojectionErrorPx = null,
      referenceFeatureCoverage = 0f,
      currentFeatureCoverage = 0f,
      transformCondition = null,
      confidence = 0f,
      rejectionReason = "auto_reseed_waiting:$trackCount",
      handMaskAgeMs = latestOcclusionTelemetry.handMaskAgeMs,
      handMaskAreaRatio = latestOcclusionTelemetry.handMaskAreaRatio,
      motionMaskAreaRatio = latestOcclusionTelemetry.motionMaskAreaRatio,
      combinedMaskAreaRatio = latestOcclusionTelemetry.combinedMaskAreaRatio,
    )

  private fun waitingTelemetry(reason: String): SurfaceTrackingTelemetry =
    SurfaceTrackingTelemetry(
      state = "waiting",
      motionModel = null,
      heldLastGoodTransform = false,
      activeTrackCount = 0,
      candidateTrackCount = 0,
      stableTrackCount = 0,
      inlierCount = 0,
      inlierRatio = 0f,
      meanReprojectionErrorPx = null,
      referenceFeatureCoverage = 0f,
      currentFeatureCoverage = 0f,
      transformCondition = null,
      confidence = 0f,
      rejectionReason = reason,
      handMaskAgeMs = latestOcclusionTelemetry.handMaskAgeMs,
      handMaskAreaRatio = latestOcclusionTelemetry.handMaskAreaRatio,
      motionMaskAreaRatio = latestOcclusionTelemetry.motionMaskAreaRatio,
      combinedMaskAreaRatio = latestOcclusionTelemetry.combinedMaskAreaRatio,
    )

  private fun publishStatus(status: String) {
    if (status == lastStatus) return
    lastStatus = status
    activity.runOnUiThread {
      activity.view.updateFallbackStatus(status)
    }
  }

  private fun updateOverlay(points: FloatArray?) {
    activity.view.fallbackOverlayView.updateQuad(points)
  }

  private fun clearOverlay() {
    activity.view.fallbackOverlayView.updateQuad(null)
    activity.view.updateInferenceCursor(null)
  }

  private fun buildWorldQuadCorners(
    pose: Pose,
    widthMeters: Float,
    heightMeters: Float
  ): List<FloatArray> {
    val right = FloatArray(3)
    val forward = FloatArray(3)
    pose.getTransformedAxis(0, 1f, right, 0)
    pose.getTransformedAxis(2, 1f, forward, 0)
    val centerX = pose.tx()
    val centerY = pose.ty()
    val centerZ = pose.tz()
    val halfWidth = widthMeters * 0.5f
    val halfHeight = heightMeters * 0.5f
    return listOf(
      floatArrayOf(
        centerX - right[0] * halfWidth - forward[0] * halfHeight,
        centerY - right[1] * halfWidth - forward[1] * halfHeight,
        centerZ - right[2] * halfWidth - forward[2] * halfHeight
      ),
      floatArrayOf(
        centerX + right[0] * halfWidth - forward[0] * halfHeight,
        centerY + right[1] * halfWidth - forward[1] * halfHeight,
        centerZ + right[2] * halfWidth - forward[2] * halfHeight
      ),
      floatArrayOf(
        centerX + right[0] * halfWidth + forward[0] * halfHeight,
        centerY + right[1] * halfWidth + forward[1] * halfHeight,
        centerZ + right[2] * halfWidth + forward[2] * halfHeight
      ),
      floatArrayOf(
        centerX - right[0] * halfWidth + forward[0] * halfHeight,
        centerY - right[1] * halfWidth + forward[1] * halfHeight,
        centerZ - right[2] * halfWidth + forward[2] * halfHeight
      ),
    )
  }
}

