/*
 * Copyright 2021 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package com.inkmarc

import android.opengl.Matrix
import android.util.Log
import androidx.lifecycle.DefaultLifecycleObserver
import androidx.lifecycle.LifecycleOwner
import com.google.ar.core.Anchor
import com.google.ar.core.Camera
import com.google.ar.core.Config
import com.google.ar.core.DepthPoint
import com.google.ar.core.Frame
import com.google.ar.core.HitResult
import com.google.ar.core.Plane
import com.google.ar.core.Pose
import com.google.ar.core.Session
import com.google.ar.core.Trackable
import com.google.ar.core.TrackingFailureReason
import com.google.ar.core.TrackingState
import com.google.ar.core.examples.java.common.helpers.DisplayRotationHelper
import com.google.ar.core.examples.java.common.helpers.TrackingStateHelper
import com.google.ar.core.examples.java.common.samplerender.Framebuffer
import com.google.ar.core.examples.java.common.samplerender.Mesh
import com.google.ar.core.examples.java.common.samplerender.SampleRender
import com.google.ar.core.examples.java.common.samplerender.Shader
import com.google.ar.core.examples.java.common.samplerender.VertexBuffer
import com.google.ar.core.examples.java.common.samplerender.arcore.BackgroundRenderer
import com.google.ar.core.examples.java.common.samplerender.arcore.PlaneRenderer
import com.google.ar.core.exceptions.CameraNotAvailableException
import com.google.ar.core.exceptions.NotYetAvailableException
import java.io.IOException
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer

/** Renders the InkMARC application using our example Renderer. */
class InkMarcRenderer(val activity: InkMarcActivity) :
  SampleRender.Renderer, DefaultLifecycleObserver {
  companion object {
    val TAG = "InkMarcRenderer"

    private const val Z_NEAR = 0.1f
    private const val Z_FAR = 100f
    private const val MIN_RECTANGLE_SCALE = 0.25f
    private const val MAX_RECTANGLE_SCALE = 4.0f
    private const val MAX_GESTURE_TRANSLATION_DELTA_METERS = 0.5f
    private const val DEPTH_REFINEMENT_INTERVAL_NS = 250_000_000L
    private const val DEPTH_REFINEMENT_MIN_DELTA_METERS = 0.01f
    private const val DEPTH_REFINEMENT_MAX_DELTA_METERS = 0.25f
  }

  private lateinit var planeRenderer: PlaneRenderer
  private lateinit var backgroundRenderer: BackgroundRenderer
  private lateinit var virtualSceneFramebuffer: Framebuffer
  private var hasSetTextureNames = false

  // Point Cloud
  private lateinit var pointCloudVertexBuffer: VertexBuffer
  private lateinit var pointCloudMesh: Mesh
  private lateinit var pointCloudShader: Shader

  // Keep track of the last point cloud rendered to avoid updating the VBO if point cloud
  // was not changed. Do this using the timestamp since we can't compare PointCloud objects.
  private var lastPointCloudTimestamp: Long = 0

  // Placed rectangle
  private lateinit var placementRectangleMesh: Mesh
  private lateinit var placementRectangleShader: Shader

  private val wrappedAnchors = mutableListOf<WrappedAnchor>()

  // Temporary matrix allocated here to reduce number of allocations for each frame.
  private val modelMatrix = FloatArray(16)
  private val projectionMatrix = FloatArray(16)
  private val viewMatrix = FloatArray(16)
  private val viewProjectionMatrix = FloatArray(16)
  private val inverseViewProjectionMatrix = FloatArray(16)
  private val modelViewMatrix = FloatArray(16) // view x model
  private val modelViewProjectionMatrix = FloatArray(16) // projection x view x model
  private val clipPoint = FloatArray(4)
  private val cameraPoint = FloatArray(4)
  private val worldPoint = FloatArray(4)
  private val localCornerPoint = FloatArray(4)
  private val transformedCornerPoint = FloatArray(4)
  private var lastPlacementRefinementTimestampNs = Long.MIN_VALUE
  private var latestPlacementScreenCorners: List<FloatArray>? = null

  private val session
    get() = activity.arCoreSessionHelper.session

  private val displayRotationHelper = DisplayRotationHelper(activity)
  private val trackingStateHelper = TrackingStateHelper(activity)

  override fun onResume(owner: LifecycleOwner) {
    if (!InkMarcActivity.ENABLE_ARCORE) return
    displayRotationHelper.onResume()
    hasSetTextureNames = false
  }

  override fun onPause(owner: LifecycleOwner) {
    if (!InkMarcActivity.ENABLE_ARCORE) return
    displayRotationHelper.onPause()
  }

  fun getPlacementScreenCorners(): List<FloatArray>? =
    latestPlacementScreenCorners?.map { it.copyOf() }

  fun getPlacementEvidence(): ArCorePlacementEvidence? =
    wrappedAnchors.firstOrNull()?.let {
      ArCorePlacementEvidence(it.source.name.lowercase(), it.anchor.trackingState.name)
    }

  override fun onSurfaceCreated(render: SampleRender) {
    if (!InkMarcActivity.ENABLE_ARCORE) return
    try {
      planeRenderer = PlaneRenderer(render)
      backgroundRenderer = BackgroundRenderer(render)
      virtualSceneFramebuffer = Framebuffer(render, /* width= */ 1, /* height= */ 1)

      pointCloudShader =
        Shader.createFromAssets(
            render,
            "shaders/point_cloud.vert",
            "shaders/point_cloud.frag",
            /* defines= */ null
          )
          .setVec4("u_Color", floatArrayOf(31.0f / 255.0f, 188.0f / 255.0f, 210.0f / 255.0f, 1.0f))
          .setFloat("u_PointSize", 5.0f)

      pointCloudVertexBuffer =
        VertexBuffer(render, /* numberOfEntriesPerVertex= */ 4, /* entries= */ null)
      pointCloudMesh =
        Mesh(
          render,
          Mesh.PrimitiveMode.POINTS,
          /* indexBuffer= */ null,
          arrayOf(pointCloudVertexBuffer)
        )

      placementRectangleMesh = createPlacementRectangleMesh(render)
      placementRectangleShader =
        Shader(
            render,
            """
            #version 300 es
            layout(location = 0) in vec3 a_Position;
            layout(location = 1) in vec2 a_TexCoord;
            uniform mat4 u_ModelViewProjection;
            out vec2 v_TexCoord;

            void main() {
              v_TexCoord = a_TexCoord;
              gl_Position = u_ModelViewProjection * vec4(a_Position, 1.0);
            }
            """
              .trimIndent(),
            """
            #version 300 es
            precision mediump float;
            in vec2 v_TexCoord;
            uniform vec4 u_FillColor;
            uniform vec4 u_BorderColor;
            uniform float u_BorderThickness;
            out vec4 o_FragColor;

            void main() {
              float edgeDistance = min(
                min(v_TexCoord.x, 1.0 - v_TexCoord.x),
                min(v_TexCoord.y, 1.0 - v_TexCoord.y)
              );
              float borderMask = 1.0 - step(u_BorderThickness, edgeDistance);
              o_FragColor = mix(u_FillColor, u_BorderColor, borderMask);
            }
            """
              .trimIndent(),
            /* defines= */ null
          )
          .setVec4("u_FillColor", floatArrayOf(1f, 1f, 1f, 0.2f))
          .setVec4("u_BorderColor", floatArrayOf(1f, 1f, 1f, 0.85f))
          .setFloat("u_BorderThickness", 0.02f)
          .setBlend(Shader.BlendFactor.SRC_ALPHA, Shader.BlendFactor.ONE_MINUS_SRC_ALPHA)
          .setCullFace(false)
    } catch (e: IOException) {
      Log.e(TAG, "Failed to read a required asset file", e)
      showError("Failed to read a required asset file: $e")
    }
  }

  override fun onSurfaceChanged(render: SampleRender, width: Int, height: Int) {
    if (!InkMarcActivity.ENABLE_ARCORE) return
    displayRotationHelper.onSurfaceChanged(width, height)
    virtualSceneFramebuffer.resize(width, height)
  }

  override fun onDrawFrame(render: SampleRender) {
    if (!InkMarcActivity.ENABLE_ARCORE) return
    val session = session ?: return

    if (!hasSetTextureNames) {
      session.setCameraTextureNames(intArrayOf(backgroundRenderer.cameraColorTexture.textureId))
      hasSetTextureNames = true
    }

    displayRotationHelper.updateSessionIfNeeded(session)

    val frame =
      try {
        session.update()
      } catch (e: CameraNotAvailableException) {
        Log.e(TAG, "Camera not available during onDrawFrame", e)
        showError("Camera not available. Try restarting the app.")
        return
      }

    val camera = frame.camera

    try {
      backgroundRenderer.setUseDepthVisualization(
        render,
        false
      )
      backgroundRenderer.setUseOcclusion(render, false)
    } catch (e: IOException) {
      Log.e(TAG, "Failed to read a required asset file", e)
      showError("Failed to read a required asset file: $e")
      return
    }

    backgroundRenderer.updateDisplayGeometry(frame)
    val shouldGetDepthImage = false
    if (camera.trackingState == TrackingState.TRACKING && shouldGetDepthImage) {
      try {
        frame.acquireDepthImage16Bits().use { depthImage ->
          backgroundRenderer.updateCameraDepthTexture(depthImage)
        }
      } catch (_: NotYetAvailableException) {
        // This normally means that depth data is not available yet.
      }
    }

    val allPlanes = session.getAllTrackables<Plane>(Plane::class.java)
    if (!InkMarcActivity.ENABLE_FALLBACK_TRACKER) {
      activity.inkMarcInferenceEngine?.submitFrame(frame, activity.getActiveSurfaceCornersView())
    }
    if (InkMarcActivity.ENABLE_FALLBACK_TRACKER) {
      activity.fallbackSurfaceTracker?.processFrame(
        frame,
        camera,
        allPlanes,
        comparisonMode = InkMarcActivity.ENABLE_ARCORE_COMPARISON,
      )
    }
    val canUseDepthPlacement = session.isDepthModeSupported(Config.DepthMode.AUTOMATIC)
    val hasTrackedPlane =
      allPlanes.any { plane ->
        plane.trackingState == TrackingState.TRACKING && plane.subsumedBy == null
      }

    trackingStateHelper.updateKeepScreenOnFlag(camera.trackingState)

    val message: String? =
      when {
        camera.trackingState == TrackingState.PAUSED &&
          camera.trackingFailureReason == TrackingFailureReason.NONE ->
          activity.getString(R.string.searching_planes)
        camera.trackingState == TrackingState.PAUSED ->
          TrackingStateHelper.getTrackingFailureReasonString(camera)
        !hasTrackedPlane && canUseDepthPlacement ->
          activity.getString(R.string.waiting_depth_taps)
        !hasTrackedPlane -> activity.getString(R.string.searching_planes)
        wrappedAnchors.isEmpty() -> activity.getString(R.string.waiting_taps)
        else -> null
      }
    if (message == null) {
      activity.view.snackbarHelper.hide(activity)
    } else {
      activity.view.snackbarHelper.showMessage(activity, message)
    }

    if (frame.timestamp != 0L) {
      backgroundRenderer.drawBackground(render)
    }

    if (camera.trackingState == TrackingState.PAUSED) {
      latestPlacementScreenCorners = null
      activity.updateInferenceCursor()
      return
    }

    camera.getProjectionMatrix(projectionMatrix, 0, Z_NEAR, Z_FAR)
    camera.getViewMatrix(viewMatrix, 0)
    Matrix.multiplyMM(viewProjectionMatrix, 0, projectionMatrix, 0, viewMatrix, 0)
    Matrix.invertM(inverseViewProjectionMatrix, 0, viewProjectionMatrix, 0)
    handleTap(frame, camera)
    applyPlacementGestures(frame)
    refinePlacement(frame, camera, hasTrackedPlane)

    var pointCloudPointCount = 0
    frame.acquirePointCloud().use { pointCloud ->
      if (pointCloud.timestamp > lastPointCloudTimestamp) {
        pointCloudVertexBuffer.set(pointCloud.points)
        lastPointCloudTimestamp = pointCloud.timestamp
      }
      Matrix.multiplyMM(modelViewProjectionMatrix, 0, projectionMatrix, 0, viewMatrix, 0)
      pointCloudShader.setMat4("u_ModelViewProjection", modelViewProjectionMatrix)
      render.draw(pointCloudMesh, pointCloudShader)
      pointCloudPointCount = pointCloud.points.limit() / 4
    }

    planeRenderer.drawPlanes(render, allPlanes, camera.displayOrientedPose, projectionMatrix)

    render.clear(virtualSceneFramebuffer, 0f, 0f, 0f, 0f)
    latestPlacementScreenCorners = null
    for (wrappedAnchor in wrappedAnchors) {
      if (wrappedAnchor.anchor.trackingState != TrackingState.TRACKING) {
        continue
      }

      wrappedAnchor.anchor.pose.toMatrix(modelMatrix, 0)
      Matrix.translateM(modelMatrix, 0, 0f, PlacementRectangleSpec.SURFACE_OFFSET_METERS, 0f)
      Matrix.rotateM(modelMatrix, 0, Math.toDegrees(wrappedAnchor.yawRadians.toDouble()).toFloat(), 0f, 1f, 0f)
      Matrix.scaleM(modelMatrix, 0, wrappedAnchor.scale, 1f, wrappedAnchor.scale)
      latestPlacementScreenCorners = projectRectangleCorners(modelMatrix)
      Matrix.multiplyMM(modelViewMatrix, 0, viewMatrix, 0, modelMatrix, 0)
      Matrix.multiplyMM(modelViewProjectionMatrix, 0, projectionMatrix, 0, modelViewMatrix, 0)

      placementRectangleShader.setMat4("u_ModelViewProjection", modelViewProjectionMatrix)
      render.draw(placementRectangleMesh, placementRectangleShader, virtualSceneFramebuffer)
    }

    if (InkMarcActivity.ENABLE_DATA_RECORDING) {
      activity.datasetRecorder?.onFrame(
        session = session,
        frame = frame,
        camera = camera,
        allPlanes = allPlanes,
        pointCloudPointCount = pointCloudPointCount,
        fallbackSnapshot = activity.fallbackSurfaceTracker?.getLatestTrackingSnapshot(),
        placementEvidence = activity.getArCorePlacementEvidence(),
        arCorePlacementCornersView = activity.getArCorePlacementCornersView(),
      )
    }

    backgroundRenderer.drawVirtualScene(render, virtualSceneFramebuffer, Z_NEAR, Z_FAR)
    if (!InkMarcActivity.ENABLE_FALLBACK_TRACKER) {
      activity.updateInferenceCursor()
    }
  }

  private fun handleTap(frame: Frame, camera: Camera) {
    if (camera.trackingState != TrackingState.TRACKING) return
    val tap = activity.view.touchHelper.pollTap() ?: return

    val firstHitResult = frame.hitTest(tap).firstPlacementHit(camera)

    if (firstHitResult != null) {
      val wrappedAnchor = createWrappedAnchor(firstHitResult, frame, camera) ?: return
      replacePlacement(wrappedAnchor, reseedFallback = true)
      activity.runOnUiThread { activity.view.showOcclusionDialogIfNeeded() }
    }
  }

  private fun refinePlacement(frame: Frame, camera: Camera, hasTrackedPlane: Boolean) {
    val currentPlacement = wrappedAnchors.firstOrNull() ?: return
    if (currentPlacement.source != PlacementSource.DEPTH) return

    val trackedQuad = activity.fallbackSurfaceTracker?.getLatestTrackedViewQuad()
    val samplePoint =
      trackedQuad?.quadCenter()
        ?: projectPoseToView(currentPlacement.anchor.pose, activity.view.surfaceView.width, activity.view.surfaceView.height)
        ?: return

    if (hasTrackedPlane) {
      val planeHit = frame.hitTest(samplePoint[0], samplePoint[1]).firstPlaneHit(camera)
      if (planeHit != null) {
        val planePlacement = createWrappedAnchor(planeHit, frame, camera) ?: return
        replacePlacement(planePlacement, reseedFallback = false, preserveTransform = true)
        return
      }
    }

    if (frame.timestamp - lastPlacementRefinementTimestampNs < DEPTH_REFINEMENT_INTERVAL_NS) return
    val depthHit = frame.hitTest(samplePoint[0], samplePoint[1]).firstDepthHit() ?: return
    val refinedPlacement = createWrappedAnchor(depthHit, frame, camera) ?: return
    val delta =
      distanceBetween(currentPlacement.anchor.pose, refinedPlacement.anchor.pose)
    if (delta < DEPTH_REFINEMENT_MIN_DELTA_METERS || delta > DEPTH_REFINEMENT_MAX_DELTA_METERS) {
      refinedPlacement.anchor.detach()
      return
    }

    replacePlacement(refinedPlacement, reseedFallback = true, preserveTransform = true)
    lastPlacementRefinementTimestampNs = frame.timestamp
  }

  private fun applyPlacementGestures(frame: Frame) {
    val currentPlacement = wrappedAnchors.firstOrNull() ?: return
    val delta = activity.view.touchHelper.consumeTransformDelta()
    if (
      delta.scaleFactor == 1f &&
      delta.rotationRadians == 0f &&
      delta.panX == 0f &&
      delta.panY == 0f
    ) return
    currentPlacement.scale =
      (currentPlacement.scale * delta.scaleFactor).coerceIn(MIN_RECTANGLE_SCALE, MAX_RECTANGLE_SCALE)
    currentPlacement.yawRadians += delta.rotationRadians
    if (delta.panX != 0f || delta.panY != 0f) {
      translatePlacementOnHorizontalPlane(currentPlacement, frame, delta.panX, delta.panY)
    }
  }

  private fun translatePlacementOnHorizontalPlane(
    currentPlacement: WrappedAnchor,
    frame: Frame,
    panX: Float,
    panY: Float
  ) {
    val session = session ?: return
    val surfaceWidth = activity.view.surfaceView.width
    val surfaceHeight = activity.view.surfaceView.height
    val screenPoint = projectPoseToView(currentPlacement.anchor.pose, surfaceWidth, surfaceHeight) ?: return
    val targetX = (screenPoint[0] + panX).coerceIn(0f, surfaceWidth.toFloat())
    val targetY = (screenPoint[1] + panY).coerceIn(0f, surfaceHeight.toFloat())
    val worldUp = activity.gravityDirectionProvider.getWorldUp(frame)
    val currentPosition =
      floatArrayOf(
        currentPlacement.anchor.pose.tx(),
        currentPlacement.anchor.pose.ty(),
        currentPlacement.anchor.pose.tz(),
      )
    val ray = screenPointToWorldRay(targetX, targetY, surfaceWidth, surfaceHeight) ?: return
    val intersection =
      intersectRayWithPlane(
        rayOrigin = ray.first,
        rayDirection = ray.second,
        planePoint = currentPosition,
        planeNormal = worldUp
      ) ?: return
    val deltaMeters = distanceBetween(currentPosition, intersection)
    if (deltaMeters > MAX_GESTURE_TRANSLATION_DELTA_METERS) return

    val translatedPose =
      Pose(
        floatArrayOf(intersection[0], intersection[1], intersection[2]),
        floatArrayOf(
          currentPlacement.anchor.pose.qx(),
          currentPlacement.anchor.pose.qy(),
          currentPlacement.anchor.pose.qz(),
          currentPlacement.anchor.pose.qw(),
        )
      )
    val translatedAnchor = session.createAnchor(translatedPose)
    val translatedPlacement =
      WrappedAnchor(translatedAnchor, currentPlacement.trackable, currentPlacement.source).also {
        it.scale = currentPlacement.scale
        it.yawRadians = currentPlacement.yawRadians
      }
    replacePlacement(translatedPlacement, reseedFallback = true, preserveTransform = true)
  }

  private fun createWrappedAnchor(
    hitResult: HitResult,
    frame: Frame,
    camera: Camera
  ): WrappedAnchor? {
    val session = session ?: return null
    val gravityAlignedPose = createGravityAlignedPose(hitResult.hitPose, frame, camera)
    return when (val trackable = hitResult.trackable) {
      is Plane -> WrappedAnchor(session.createAnchor(gravityAlignedPose), trackable, PlacementSource.PLANE)
      is DepthPoint -> WrappedAnchor(session.createAnchor(gravityAlignedPose), trackable, PlacementSource.DEPTH)
      else -> null
    }
  }

  private fun replacePlacement(
    wrappedAnchor: WrappedAnchor,
    reseedFallback: Boolean,
    preserveTransform: Boolean = false
  ) {
    val previousPlacement = wrappedAnchors.firstOrNull()
    wrappedAnchors.forEach { it.anchor.detach() }
    wrappedAnchors.clear()
    if (preserveTransform && previousPlacement != null) {
      wrappedAnchor.scale = previousPlacement.scale
      wrappedAnchor.yawRadians = previousPlacement.yawRadians
    }
    wrappedAnchors.add(wrappedAnchor)
    if (reseedFallback) {
      activity.fallbackSurfaceTracker?.requestSeedFromPlacementPose(wrappedAnchor.anchor.pose)
    }
  }

  private fun createGravityAlignedPose(hitPose: Pose, frame: Frame, camera: Camera): Pose {
    val up = activity.gravityDirectionProvider.getWorldUp(frame)
    val cameraForward = FloatArray(3)
    camera.pose.getTransformedAxis(2, -1f, cameraForward, 0)
    val horizontalForward =
      subtract(cameraForward, scale(up, dot(cameraForward, up))).normalizedOrDefault(floatArrayOf(0f, 0f, -1f))
    val right = cross(horizontalForward, up).normalizedOrDefault(floatArrayOf(1f, 0f, 0f))
    val correctedForward = cross(up, right).normalizedOrDefault(horizontalForward)
    val rotationQuaternion = quaternionFromBasis(right, up, correctedForward)
    return Pose(floatArrayOf(hitPose.tx(), hitPose.ty(), hitPose.tz()), rotationQuaternion)
  }

  private fun projectPoseToView(
    pose: Pose,
    surfaceWidth: Int,
    surfaceHeight: Int
  ): FloatArray? {
    if (surfaceWidth <= 0 || surfaceHeight <= 0) return null
    worldPoint[0] = pose.tx()
    worldPoint[1] = pose.ty()
    worldPoint[2] = pose.tz()
    worldPoint[3] = 1f
    Matrix.multiplyMV(cameraPoint, 0, viewMatrix, 0, worldPoint, 0)
    Matrix.multiplyMV(clipPoint, 0, projectionMatrix, 0, cameraPoint, 0)
    val w = clipPoint[3]
    if (w == 0f) return null
    val ndcX = clipPoint[0] / w
    val ndcY = clipPoint[1] / w
    if (!ndcX.isFinite() || !ndcY.isFinite()) return null
    return floatArrayOf(
      (ndcX * 0.5f + 0.5f) * surfaceWidth,
      (0.5f - ndcY * 0.5f) * surfaceHeight,
    )
  }

  private fun projectRectangleCorners(modelMatrix: FloatArray): List<FloatArray>? {
    val halfWidth = PlacementRectangleSpec.WIDTH_METERS / 2f
    val halfHeight = PlacementRectangleSpec.HEIGHT_METERS / 2f
    val localCorners =
      arrayOf(
        floatArrayOf(-halfWidth, 0f, -halfHeight, 1f),
        floatArrayOf(halfWidth, 0f, -halfHeight, 1f),
        floatArrayOf(halfWidth, 0f, halfHeight, 1f),
        floatArrayOf(-halfWidth, 0f, halfHeight, 1f),
      )
    val surfaceWidth = activity.view.surfaceView.width
    val surfaceHeight = activity.view.surfaceView.height
    if (surfaceWidth <= 0 || surfaceHeight <= 0) return null

    val screenCorners = ArrayList<FloatArray>(localCorners.size)
    for (corner in localCorners) {
      localCornerPoint[0] = corner[0]
      localCornerPoint[1] = corner[1]
      localCornerPoint[2] = corner[2]
      localCornerPoint[3] = corner[3]
      Matrix.multiplyMV(transformedCornerPoint, 0, modelMatrix, 0, localCornerPoint, 0)
      val projected =
        projectWorldPointToView(
          transformedCornerPoint[0],
          transformedCornerPoint[1],
          transformedCornerPoint[2],
          surfaceWidth,
          surfaceHeight
        ) ?: return null
      screenCorners.add(projected)
    }
    return screenCorners
  }

  private fun projectWorldPointToView(
    x: Float,
    y: Float,
    z: Float,
    surfaceWidth: Int,
    surfaceHeight: Int
  ): FloatArray? {
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
    if (!ndcX.isFinite() || !ndcY.isFinite()) return null
    return floatArrayOf(
      (ndcX * 0.5f + 0.5f) * surfaceWidth,
      (0.5f - ndcY * 0.5f) * surfaceHeight,
    )
  }

  private fun screenPointToWorldRay(
    x: Float,
    y: Float,
    surfaceWidth: Int,
    surfaceHeight: Int
  ): Pair<FloatArray, FloatArray>? {
    if (surfaceWidth <= 0 || surfaceHeight <= 0) return null
    val nearPoint = unprojectPoint(x, y, -1f, surfaceWidth, surfaceHeight) ?: return null
    val farPoint = unprojectPoint(x, y, 1f, surfaceWidth, surfaceHeight) ?: return null
    val direction = subtract(farPoint, nearPoint).normalizedOrDefault(floatArrayOf(0f, 0f, -1f))
    return nearPoint to direction
  }

  private fun unprojectPoint(
    x: Float,
    y: Float,
    ndcZ: Float,
    surfaceWidth: Int,
    surfaceHeight: Int
  ): FloatArray? {
    val ndcX = x / surfaceWidth * 2f - 1f
    val ndcY = 1f - y / surfaceHeight * 2f
    val point = floatArrayOf(ndcX, ndcY, ndcZ, 1f)
    val result = FloatArray(4)
    Matrix.multiplyMV(result, 0, inverseViewProjectionMatrix, 0, point, 0)
    val w = result[3]
    if (w == 0f) return null
    return floatArrayOf(result[0] / w, result[1] / w, result[2] / w)
  }

  private fun showError(errorMessage: String) =
    activity.view.snackbarHelper.showError(activity, errorMessage)

  private fun createPlacementRectangleMesh(render: SampleRender): Mesh {
    val halfWidth = PlacementRectangleSpec.WIDTH_METERS / 2f
    val halfHeight = PlacementRectangleSpec.HEIGHT_METERS / 2f
    val vertices =
      floatArrayOf(
        -halfWidth, 0f, -halfHeight,
        halfWidth, 0f, -halfHeight,
        -halfWidth, 0f, halfHeight,
        halfWidth, 0f, halfHeight,
      )
    val uvCoords =
      floatArrayOf(
        0f, 0f,
        1f, 0f,
        0f, 1f,
        1f, 1f,
      )

    val vertexBuffer = VertexBuffer(render, 3, directFloatBuffer(vertices))
    val uvBuffer = VertexBuffer(render, 2, directFloatBuffer(uvCoords))
    return Mesh(
      render,
      Mesh.PrimitiveMode.TRIANGLE_STRIP,
      /* indexBuffer= */ null,
      arrayOf(vertexBuffer, uvBuffer)
    )
  }

  private fun directFloatBuffer(values: FloatArray): FloatBuffer =
    ByteBuffer.allocateDirect(values.size * Float.SIZE_BYTES)
      .order(ByteOrder.nativeOrder())
      .asFloatBuffer()
      .apply {
        put(values)
        rewind()
      }
}

private fun List<com.google.ar.core.HitResult>.firstPlacementHit(camera: Camera) =
  firstPlaneHit(camera) ?: firstDepthHit()

private fun List<com.google.ar.core.HitResult>.firstPlaneHit(camera: Camera) =
  firstOrNull { hit ->
    when (val trackable = hit.trackable) {
      is Plane ->
        trackable.isPoseInPolygon(hit.hitPose) &&
          PlaneRenderer.calculateDistanceToPlane(hit.hitPose, camera.pose) > 0f
      else -> false
    }
  }

private fun List<com.google.ar.core.HitResult>.firstDepthHit() =
  firstOrNull { it.trackable is DepthPoint }

private data class WrappedAnchor(
  val anchor: Anchor,
  val trackable: Trackable,
  val source: PlacementSource,
) {
  var scale: Float = 1f
  var yawRadians: Float = 0f
}

private enum class PlacementSource {
  PLANE,
  DEPTH,
}

data class ArCorePlacementEvidence(
  val source: String,
  val trackingState: String,
)

private fun FloatArray.quadCenter(): FloatArray {
  val pointCount = size / 2
  var sumX = 0f
  var sumY = 0f
  for (i in indices step 2) {
    sumX += this[i]
    sumY += this[i + 1]
  }
  return floatArrayOf(sumX / pointCount, sumY / pointCount)
}

private fun distanceBetween(a: Pose, b: Pose): Float {
  val dx = a.tx() - b.tx()
  val dy = a.ty() - b.ty()
  val dz = a.tz() - b.tz()
  return kotlin.math.sqrt(dx * dx + dy * dy + dz * dz)
}

private fun distanceBetween(a: FloatArray, b: FloatArray): Float {
  val dx = a[0] - b[0]
  val dy = a[1] - b[1]
  val dz = a[2] - b[2]
  return kotlin.math.sqrt(dx * dx + dy * dy + dz * dz)
}

private fun dot(a: FloatArray, b: FloatArray): Float = a[0] * b[0] + a[1] * b[1] + a[2] * b[2]

private fun cross(a: FloatArray, b: FloatArray) =
  floatArrayOf(
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
  )

private fun subtract(a: FloatArray, b: FloatArray) =
  floatArrayOf(a[0] - b[0], a[1] - b[1], a[2] - b[2])

private fun scale(v: FloatArray, scalar: Float) =
  floatArrayOf(v[0] * scalar, v[1] * scalar, v[2] * scalar)

private fun intersectRayWithPlane(
  rayOrigin: FloatArray,
  rayDirection: FloatArray,
  planePoint: FloatArray,
  planeNormal: FloatArray
): FloatArray? {
  val denominator = dot(rayDirection, planeNormal)
  if (kotlin.math.abs(denominator) <= 1e-6f) return null
  val t = dot(subtract(planePoint, rayOrigin), planeNormal) / denominator
  if (t <= 0f) return null
  return floatArrayOf(
    rayOrigin[0] + rayDirection[0] * t,
    rayOrigin[1] + rayDirection[1] * t,
    rayOrigin[2] + rayDirection[2] * t,
  )
}

private fun FloatArray.normalizedOrDefault(defaultValue: FloatArray): FloatArray {
  val length = kotlin.math.sqrt(this[0] * this[0] + this[1] * this[1] + this[2] * this[2])
  if (length <= 1e-6f) return defaultValue
  return floatArrayOf(this[0] / length, this[1] / length, this[2] / length)
}

private fun quaternionFromBasis(
  right: FloatArray,
  up: FloatArray,
  forward: FloatArray
): FloatArray {
  val m00 = right[0]
  val m01 = up[0]
  val m02 = forward[0]
  val m10 = right[1]
  val m11 = up[1]
  val m12 = forward[1]
  val m20 = right[2]
  val m21 = up[2]
  val m22 = forward[2]
  val trace = m00 + m11 + m22

  return if (trace > 0f) {
    val s = kotlin.math.sqrt(trace + 1f) * 2f
    floatArrayOf(
      (m21 - m12) / s,
      (m02 - m20) / s,
      (m10 - m01) / s,
      0.25f * s,
    )
  } else if (m00 > m11 && m00 > m22) {
    val s = kotlin.math.sqrt(1f + m00 - m11 - m22) * 2f
    floatArrayOf(
      0.25f * s,
      (m01 + m10) / s,
      (m02 + m20) / s,
      (m21 - m12) / s,
    )
  } else if (m11 > m22) {
    val s = kotlin.math.sqrt(1f + m11 - m00 - m22) * 2f
    floatArrayOf(
      (m01 + m10) / s,
      0.25f * s,
      (m12 + m21) / s,
      (m02 - m20) / s,
    )
  } else {
    val s = kotlin.math.sqrt(1f + m22 - m00 - m11) * 2f
    floatArrayOf(
      (m02 + m20) / s,
      (m12 + m21) / s,
      0.25f * s,
      (m10 - m01) / s,
    )
  }
}

