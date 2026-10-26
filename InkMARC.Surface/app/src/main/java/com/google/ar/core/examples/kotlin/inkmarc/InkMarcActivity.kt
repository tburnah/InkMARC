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

import android.os.Bundle
import android.util.Log
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import com.google.ar.core.Config
import com.google.ar.core.Session
import com.google.ar.core.examples.java.common.helpers.CameraPermissionHelper
import com.google.ar.core.examples.java.common.helpers.DepthSettings
import com.google.ar.core.examples.java.common.helpers.FullScreenHelper
import com.google.ar.core.examples.java.common.samplerender.SampleRender
import com.google.ar.core.examples.kotlin.common.helpers.ARCoreSessionLifecycleHelper
import com.google.ar.core.exceptions.CameraNotAvailableException
import com.google.ar.core.exceptions.UnavailableApkTooOldException
import com.google.ar.core.exceptions.UnavailableDeviceNotCompatibleException
import com.google.ar.core.exceptions.UnavailableSdkTooOldException
import com.google.ar.core.exceptions.UnavailableUserDeclinedInstallationException
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import org.json.JSONObject

/**
 * This is a simple example that shows how to create an augmented reality (AR) application using the
 * ARCore API. The application displays detected planes and related tracking visualizations.
 */
class InkMarcActivity : AppCompatActivity() {
  companion object {
    private const val TAG = "InkMarcActivity"
    // Comparison mode runs both trackers on the exact same ARCore camera frames. It requires an
    // ARCore-supported device; set this false to return to the Camera2-only fallback path.
    const val ENABLE_ARCORE_COMPARISON = true
    const val ENABLE_ARCORE = ENABLE_ARCORE_COMPARISON
    const val ENABLE_DATA_RECORDING = ENABLE_ARCORE_COMPARISON
    const val ENABLE_FALLBACK_TRACKER = true
    const val ENABLE_SESSION_VIDEO_RECORDING = true
    const val ENABLE_SESSION_SENSOR_RECORDING = true
    const val ENABLE_INK_MARC_INFERENCE = false
    const val ENABLE_HAND_OCCLUSION_MASK = true
    const val ENABLE_DA3_DEPTH = false
    private const val INACTIVE_CURSOR_ALPHA = 0.35f
    private const val TOUCH_LABEL = "Touch"
    private const val NO_TOUCH_LABEL = "No-Touch"
  }

  lateinit var arCoreSessionHelper: ARCoreSessionLifecycleHelper
  lateinit var view: InkMarcView
  lateinit var renderer: InkMarcRenderer
  lateinit var gravityDirectionProvider: GravityDirectionProvider
  var datasetRecorder: TrackingDatasetRecorder? = null
  var fallbackSurfaceTracker: FallbackSurfaceTracker? = null
  var handOcclusionMasker: MediaPipeHandOcclusionMasker? = null
  var da3DepthEstimator: Da3DepthEstimator? = null
  var inkMarcInferenceEngine: InkMarcInferenceEngine? = null
  private var standaloneCameraController: StandaloneCameraController? = null
  private var arCoreFailoverActive = false
  private val surfaceGeometryBuilder = SurfaceGeometryBuilder()

  val depthSettings = DepthSettings()

  override fun onCreate(savedInstanceState: Bundle?) {
    super.onCreate(savedInstanceState)

    // Build the UI first because both camera paths publish lifecycle status to it.
    view = InkMarcView(this)
    lifecycle.addObserver(view)
    setContentView(view.root)

    if (ENABLE_ARCORE || ENABLE_FALLBACK_TRACKER) {
      gravityDirectionProvider = GravityDirectionProvider(this)
      lifecycle.addObserver(gravityDirectionProvider)
    }

    if (ENABLE_DATA_RECORDING) {
      datasetRecorder = TrackingDatasetRecorder(this)
      lifecycle.addObserver(datasetRecorder!!)
      view.showFinishCaptureButton(ENABLE_ARCORE_COMPARISON)
    }

    if (ENABLE_FALLBACK_TRACKER) {
      fallbackSurfaceTracker = FallbackSurfaceTracker(this)
      lifecycle.addObserver(fallbackSurfaceTracker!!)

      if (ENABLE_HAND_OCCLUSION_MASK) {
        handOcclusionMasker = MediaPipeHandOcclusionMasker(this)
        lifecycle.addObserver(handOcclusionMasker!!)
      }
    }

    if (ENABLE_INK_MARC_INFERENCE) {
      inkMarcInferenceEngine = InkMarcInferenceEngine(this)
      lifecycle.addObserver(inkMarcInferenceEngine!!)
    }

    if (ENABLE_DA3_DEPTH) {
      da3DepthEstimator = Da3DepthEstimator(this)
      lifecycle.addObserver(da3DepthEstimator!!)
    }

    if (ENABLE_ARCORE) {
      setupArCore()
      depthSettings.onCreate(this)
    } else {
      view.useFallbackCamera()
      standaloneCameraController = StandaloneCameraController(this)
      lifecycle.addObserver(standaloneCameraController!!)
      Log.i(
        TAG,
        "ARCore and stylus inference disabled; running standalone tracking with hand occlusion=$ENABLE_HAND_OCCLUSION_MASK",
      )
    }
  }

  private fun setupArCore() {
    arCoreSessionHelper = ARCoreSessionLifecycleHelper(this)
    arCoreSessionHelper.exceptionCallback =
      { exception ->
        val message =
          when (exception) {
            is UnavailableUserDeclinedInstallationException ->
              "Please install Google Play Services for AR"
            is UnavailableApkTooOldException -> "Please update ARCore"
            is UnavailableSdkTooOldException -> "Please update this app"
            is UnavailableDeviceNotCompatibleException -> "This device does not support AR"
            is CameraNotAvailableException -> "Camera not available. Try restarting the app."
            else -> "Failed to create AR session: $exception"
          }
        Log.e(TAG, "ARCore threw an exception; switching to Camera2 fallback", exception)
        activateCamera2Failover(exception, message)
      }
    arCoreSessionHelper.beforeSessionResume = ::configureSession
    lifecycle.addObserver(arCoreSessionHelper)

    renderer = InkMarcRenderer(this)
    lifecycle.addObserver(renderer)
    SampleRender(view.surfaceView, renderer, assets)
  }

  /**
   * ARCore owns the camera while it is running, so a failed ARCore startup must release that
   * session before Camera2 can take over. This is a fallback-only capture: it records the reason
   * ARCore was unavailable, but naturally cannot produce a one-to-one ARCore comparison.
   */
  private fun activateCamera2Failover(exception: Exception, message: String) {
    if (arCoreFailoverActive) return
    arCoreFailoverActive = true
    writeArCoreFailoverLog(exception)
    if (::arCoreSessionHelper.isInitialized) {
      arCoreSessionHelper.closeSession()
      lifecycle.removeObserver(arCoreSessionHelper)
    }
    view.hideRecordingStatus()
    view.showFinishCaptureButton(false)
    view.useFallbackCamera()
    view.updateFallbackStatus("ARCore unavailable; Camera2 fallback active")
    view.snackbarHelper.showMessageWithDismiss(this, "$message Falling back to Camera2 tracking.")
    if (standaloneCameraController == null) {
      standaloneCameraController = StandaloneCameraController(this)
      lifecycle.addObserver(standaloneCameraController!!)
    }
    standaloneCameraController?.start()
  }

  /** Finishes the active ARCore comparison while the app remains open and its export is visible. */
  fun finishCurrentCapture() {
    when (datasetRecorder?.finishCapture() ?: TrackingDatasetRecorder.FinishResult.NOT_RECORDING) {
      TrackingDatasetRecorder.FinishResult.EXPORTED -> {
        view.disableFinishCaptureButton()
        Toast.makeText(this, R.string.recording_export_complete, Toast.LENGTH_LONG).show()
      }
      TrackingDatasetRecorder.FinishResult.NOT_RECORDING -> {
        Toast.makeText(this, R.string.recording_not_started, Toast.LENGTH_LONG).show()
      }
      TrackingDatasetRecorder.FinishResult.FAILED -> {
        view.disableFinishCaptureButton()
        Toast.makeText(this, R.string.recording_export_unsuccessful, Toast.LENGTH_LONG).show()
      }
    }
  }

  private fun writeArCoreFailoverLog(exception: Exception) {
    runCatching {
      val directory = getExternalFilesDir("failover") ?: filesDir
      directory.mkdirs()
      val timestamp = SimpleDateFormat("yyyyMMdd_HHmmss_SSS", Locale.US).format(Date())
      val output = File(directory, "arcore_failover_$timestamp.json")
      val payload =
        JSONObject()
          .put("timestampWallClockMs", System.currentTimeMillis())
          .put("mode", "camera2_fallback_only")
          .put("exceptionClass", exception.javaClass.name)
          .put("message", exception.message ?: JSONObject.NULL)
          .put("comparisonAvailable", false)
      output.writeText(payload.toString(2))
      Log.i(TAG, "Wrote ARCore failover log to ${output.absolutePath}")
    }.onFailure { Log.e(TAG, "Unable to write ARCore failover log", it) }
  }

  // Configure the session, using Lighting Estimation, and Depth mode.
  fun configureSession(session: Session) {
    session.configure(
      session.config.apply {
        lightEstimationMode = Config.LightEstimationMode.ENVIRONMENTAL_HDR
        planeFindingMode = Config.PlaneFindingMode.HORIZONTAL

        // Depth API is used if it is configured in InkMARC's settings.
        depthMode =
          if (session.isDepthModeSupported(Config.DepthMode.AUTOMATIC)) {
            Config.DepthMode.AUTOMATIC
          } else {
            Config.DepthMode.DISABLED
          }

        instantPlacementMode = Config.InstantPlacementMode.DISABLED

      }
    )
    datasetRecorder?.onSessionConfigured(session)
  }

  override fun onRequestPermissionsResult(
    requestCode: Int,
    permissions: Array<String>,
    results: IntArray
  ) {
    super.onRequestPermissionsResult(requestCode, permissions, results)
    if (CameraPermissionHelper.hasCameraPermission(this)) {
      standaloneCameraController?.start()
    } else {
      // Use toast instead of snackbar here since the activity will exit.
      Toast.makeText(this, "Camera permission is needed to run this application", Toast.LENGTH_LONG)
        .show()
      if (!CameraPermissionHelper.shouldShowRequestPermissionRationale(this)) {
        // Permission denied with checking "Do not ask again".
        CameraPermissionHelper.launchPermissionSettings(this)
      }
      finish()
    }
  }

  override fun onWindowFocusChanged(hasFocus: Boolean) {
    super.onWindowFocusChanged(hasFocus)
    FullScreenHelper.setFullScreenOnWindowFocusChanged(this, hasFocus)
  }

  fun getActiveSurfaceCornersView(): FloatArray? {
    fallbackSurfaceTracker?.getLatestTrackedViewQuad()?.let { return it }
    return if (ENABLE_ARCORE && ::renderer.isInitialized) {
      renderer.getPlacementScreenCorners()?.toFlatCornerArray()
    } else {
      null
    }
  }

  fun getArCorePlacementEvidence(): ArCorePlacementEvidence? =
    if (ENABLE_ARCORE && ::renderer.isInitialized) renderer.getPlacementEvidence() else null

  fun getArCorePlacementCornersView(): FloatArray? =
    if (ENABLE_ARCORE && ::renderer.isInitialized) {
      renderer.getPlacementScreenCorners()?.toFlatCornerArray()
    } else {
      null
    }

  fun updateInferenceCursor() {
    val result = inkMarcInferenceEngine?.getLatestResult()
    if (result == null) {
      view.updateInferenceCursor(null)
      return
    }

    val screenPoint =
      surfaceGeometryBuilder.mapNormalizedPointToSurface(
        xNormalized = result.xyNormalized.getOrElse(0) { 0f },
        yNormalized = result.xyNormalized.getOrElse(1) { 0f },
        surfaceCorners = getActiveSurfaceCornersView(),
      )
    val alpha = if (result.presentRaw) 1f else INACTIVE_CURSOR_ALPHA
    val label = if (result.touchingHysteresis) TOUCH_LABEL else NO_TOUCH_LABEL
    view.updateInferenceCursor(screenPoint, alpha, label)
  }
}

private fun List<FloatArray>.toFlatCornerArray(): FloatArray {
  val flattened = FloatArray(size * 2)
  forEachIndexed { index, point ->
    flattened[index * 2] = point[0]
    flattened[index * 2 + 1] = point[1]
  }
  return flattened
}

