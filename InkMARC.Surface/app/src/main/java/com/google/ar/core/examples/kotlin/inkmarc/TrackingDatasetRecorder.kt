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

import android.content.ContentValues
import android.content.Context
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.net.Uri
import android.os.Environment
import android.provider.MediaStore
import android.util.Log
import android.view.View
import androidx.lifecycle.DefaultLifecycleObserver
import androidx.lifecycle.LifecycleOwner
import com.google.ar.core.Camera
import com.google.ar.core.Frame
import com.google.ar.core.Plane
import com.google.ar.core.Pose
import com.google.ar.core.RecordingConfig
import com.google.ar.core.RecordingStatus
import com.google.ar.core.Session
import com.google.ar.core.TrackingState
import java.io.BufferedWriter
import java.io.File
import java.io.FileInputStream
import java.io.IOException
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import org.json.JSONArray
import org.json.JSONObject

/**
 * Records an ARCore MP4 dataset and sidecar logs for plane tracking and IMU values.
 *
 * The saved MP4 comes from ARCore session recording, so it contains the camera feed without the
 * plane overlay rendered by the preview.
 */
class TrackingDatasetRecorder(private val activity: InkMarcActivity) : DefaultLifecycleObserver {
  companion object {
    private const val TAG = "TrackingDatasetRecorder"
    private const val FLUSH_FRAME_INTERVAL = 10
    private const val FLUSH_SENSOR_INTERVAL = 120
    private const val SENSOR_RATE_FASTEST_US = 0
    private const val SENSOR_RATE_FALLBACK_US = 5000
    private const val EXPORT_COLLECTION_FOLDER = "InkMARCCaptures"
  }

  private enum class HudState {
    WAITING,
    RECORDING,
    STOPPED,
    ERROR,
  }

  enum class FinishResult {
    EXPORTED,
    NOT_RECORDING,
    FAILED,
  }

  private val sensorManager =
    activity.getSystemService(Context.SENSOR_SERVICE) as SensorManager
  private val accelerometer = sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER)
  private val gyroscope = sensorManager.getDefaultSensor(Sensor.TYPE_GYROSCOPE)
  private val ioLock = Any()

  private var sensorsRegistered = false
  private var captureDirectory: File? = null
  private var captureId: String? = null
  private var manifestFile: File? = null
  private var mp4File: File? = null
  private var planesFile: File? = null
  private var imuFile: File? = null
  private var planeWriter: BufferedWriter? = null
  private var imuWriter: BufferedWriter? = null
  private var recordingInitialized = false
  private var captureFinishedByUser = false
  private var firstRecordedFrameTimestampNs: Long? = null
  private var firstTrackedPlaneFrameTimestampNs: Long? = null
  private var sessionRecordingStartedWallClockMs: Long? = null
  private var lastLoggedFrameTimestampNs: Long = 0
  private var lastObservedFrameTimestampNs: Long = 0
  private var frameRowsSinceFlush = 0
  private var sensorRowsSinceFlush = 0
  private var nextPlaneId = 1
  private val planeIds = mutableMapOf<Plane, Int>()
  private var lastCompletedCapturePath: String? = null
  private var lastCompletedCaptureId: String? = null
  private var lastHudState: HudState? = null
  private var lastHudPath: String? = null
  private var lastHudDetail: String? = null

  private val sensorListener =
    object : SensorEventListener {
      override fun onSensorChanged(event: SensorEvent) {
        val values = event.values
        if (values.size < 3) return

        synchronized(ioLock) {
          val writer = imuWriter ?: return
          if (!recordingInitialized) return

          writer.write(
            buildString(96) {
              append(event.timestamp)
              append(',')
              append(sensorTypeName(event.sensor.type))
              append(',')
              append(values[0])
              append(',')
              append(values[1])
              append(',')
              append(values[2])
              append(',')
              append(event.accuracy)
            }
          )
          writer.newLine()
          sensorRowsSinceFlush += 1
          if (sensorRowsSinceFlush >= FLUSH_SENSOR_INTERVAL) {
            writer.flush()
            sensorRowsSinceFlush = 0
          }
        }
      }

      override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) = Unit
    }

  override fun onResume(owner: LifecycleOwner) {
    registerSensorsIfNeeded()
    postHudState(
      state = HudState.WAITING,
      path = captureBaseDirectory().absolutePath,
      detail = activity.getString(R.string.recording_waiting_for_plane)
    )
  }

  override fun onPause(owner: LifecycleOwner) {
    unregisterSensors()
    stopAndCloseCapture()
  }

  override fun onDestroy(owner: LifecycleOwner) {
    unregisterSensors()
    stopAndCloseCapture()
  }

  fun onSessionConfigured(session: Session) {
    synchronized(ioLock) {
      if (recordingInitialized || captureFinishedByUser) return
      try {
        // Begin before Session.resume() so the dataset includes the no-plane and depth-only
        // portions of a trial, not merely the eventual successful plane state.
        startCaptureLocked(session)
      } catch (e: Exception) {
        Log.e(TAG, "Failed to start full-session dataset capture", e)
        postHudState(
          state = HudState.ERROR,
          path = captureBaseDirectory().absolutePath,
          detail = activity.getString(R.string.recording_failed, e.message ?: "unknown error")
        )
      }
    }
  }

  fun onFrame(
    session: Session,
    frame: Frame,
    camera: Camera,
    allPlanes: Collection<Plane>,
    pointCloudPointCount: Int,
    fallbackSnapshot: SurfaceTrackingSnapshot?,
    placementEvidence: ArCorePlacementEvidence?,
    arCorePlacementCornersView: FloatArray?,
  ) {
    val frameTimestampNs = frame.timestamp
    if (frameTimestampNs == 0L) return

    val lineToWrite: String? =
      synchronized(ioLock) {
        if (frameTimestampNs == lastObservedFrameTimestampNs) {
          return@synchronized null
        }

        if (frameTimestampNs != lastObservedFrameTimestampNs) {
          lastObservedFrameTimestampNs = frameTimestampNs
        }

        if (!recordingInitialized || frameTimestampNs == lastLoggedFrameTimestampNs) {
          return@synchronized null
        }

        if (firstRecordedFrameTimestampNs == null) firstRecordedFrameTimestampNs = frameTimestampNs
        val hasTrackedPlane = allPlanes.any { it.trackingState == TrackingState.TRACKING }
        if (hasTrackedPlane && firstTrackedPlaneFrameTimestampNs == null) {
          firstTrackedPlaneFrameTimestampNs = frameTimestampNs
          writeManifestLocked(status = "recording")
        }
        lastLoggedFrameTimestampNs = frameTimestampNs
        buildFrameSample(
          frame,
          camera,
          allPlanes,
          pointCloudPointCount,
          fallbackSnapshot,
          placementEvidence,
          arCorePlacementCornersView,
        )
      }

    if (lineToWrite == null) return

    synchronized(ioLock) {
      val writer = planeWriter ?: return
      writer.write(lineToWrite)
      writer.newLine()
      frameRowsSinceFlush += 1
      if (frameRowsSinceFlush >= FLUSH_FRAME_INTERVAL) {
        writer.flush()
        frameRowsSinceFlush = 0
      }
    }
  }

  /**
   * Stops the current ARCore recording and exports it before returning. A user-finished capture
   * stays stopped for this activity instance, preventing the next frame from starting a new one.
   */
  fun finishCapture(): FinishResult {
    synchronized(ioLock) {
      if (!recordingInitialized) return FinishResult.NOT_RECORDING
      captureFinishedByUser = true
    }
    return if (stopAndCloseCapture()) FinishResult.EXPORTED else FinishResult.FAILED
  }

  private fun registerSensorsIfNeeded() {
    if (sensorsRegistered) return
    try {
      accelerometer?.let {
        sensorManager.registerListener(sensorListener, it, SENSOR_RATE_FASTEST_US)
      }
      gyroscope?.let {
        sensorManager.registerListener(sensorListener, it, SENSOR_RATE_FASTEST_US)
      }
    } catch (e: SecurityException) {
      Log.w(TAG, "High-rate sensor registration rejected; falling back to 5000us", e)
      sensorManager.unregisterListener(sensorListener)
      accelerometer?.let {
        sensorManager.registerListener(sensorListener, it, SENSOR_RATE_FALLBACK_US)
      }
      gyroscope?.let {
        sensorManager.registerListener(sensorListener, it, SENSOR_RATE_FALLBACK_US)
      }
      activity.runOnUiThread {
        activity.view.snackbarHelper.showMessageForLongDuration(
          activity,
          activity.getString(R.string.recording_sensor_fallback)
        )
      }
    }
    sensorsRegistered = true
  }

  private fun unregisterSensors() {
    if (!sensorsRegistered) return
    sensorManager.unregisterListener(sensorListener)
    sensorsRegistered = false
  }

  private fun startCaptureLocked(session: Session) {
    if (recordingInitialized) return

    val baseDirectory = captureBaseDirectory()
    if (!baseDirectory.exists() && !baseDirectory.mkdirs()) {
      throw IllegalStateException("Failed to create capture directory: ${baseDirectory.absolutePath}")
    }

    val id = createCaptureId()
    val directory = File(baseDirectory, id)
    if (!directory.mkdirs()) {
      throw IllegalStateException("Failed to create capture output directory: ${directory.absolutePath}")
    }

    val manifest = File(directory, "capture_manifest.json")
    val mp4 = File(directory, "session.mp4")
    val planes = File(directory, "planes.ndjson")
    val imu = File(directory, "imu.csv")

    val planeOutput = planes.bufferedWriter()
    val imuOutput = imu.bufferedWriter()
    imuOutput.write("sensor_timestamp_ns,sensor_type,x,y,z,accuracy")
    imuOutput.newLine()

    captureDirectory = directory
      captureId = id
    manifestFile = manifest
    mp4File = mp4
    planesFile = planes
    imuFile = imu
    planeWriter = planeOutput
    imuWriter = imuOutput
    sessionRecordingStartedWallClockMs = System.currentTimeMillis()
    firstRecordedFrameTimestampNs = null
    firstTrackedPlaneFrameTimestampNs = null
    lastLoggedFrameTimestampNs = 0
    frameRowsSinceFlush = 0
    sensorRowsSinceFlush = 0
    planeIds.clear()
    nextPlaneId = 1

    writeManifestLocked(status = "initializing")

    val recordingConfig =
      RecordingConfig(session)
        .setMp4DatasetUri(Uri.fromFile(mp4))
        .setAutoStopOnPause(true)

    try {
      session.startRecording(recordingConfig)
    } catch (e: Exception) {
      closeOutputsLocked()
      directory.deleteRecursively()
      resetStateLocked()
      throw e
    }

    recordingInitialized = true
      writeManifestLocked(status = "recording")
      postHudState(
        state = HudState.RECORDING,
        path = exportRelativePath(id),
        detail = activity.getString(R.string.recording_active)
      )
    Log.i(
      TAG,
      "Started full-session dataset capture $captureId in ${directory.absolutePath}"
    )
  }

  private fun stopAndCloseCapture(): Boolean =
    synchronized(ioLock) {
      if (!recordingInitialized &&
        planeWriter == null &&
        imuWriter == null &&
        manifestFile == null
      ) {
        return@synchronized false
      }

      val session = activity.arCoreSessionHelper.session
      if (session != null && session.recordingStatus == RecordingStatus.OK) {
        try {
          session.stopRecording()
        } catch (e: Exception) {
          Log.e(TAG, "Failed to stop ARCore session recording", e)
        }
      }

      val completedCaptureId = captureId
      val completedCaptureDirectory = captureDirectory
      lastCompletedCaptureId = completedCaptureId
      lastCompletedCapturePath =
        completedCaptureId?.let { exportRelativePath(it) } ?: captureBaseDirectory().absolutePath
      writeManifestLocked(status = "stopped")
      closeOutputsLocked()
      Log.i(TAG, "Dataset capture stopped at ${captureDirectory?.absolutePath}")
      resetStateLocked()
      postHudState(
        state = HudState.STOPPED,
        path = lastCompletedCapturePath ?: captureBaseDirectory().absolutePath,
        detail = activity.getString(R.string.recording_exporting)
      )
      if (completedCaptureDirectory != null && completedCaptureId != null) {
        // A closing activity can be killed before an asynchronous export reaches Downloads. Copy
        // while this lifecycle callback is still executing so a completed capture is durable and
        // visible before the user leaves the app.
        exportFinishedCapture(completedCaptureDirectory, completedCaptureId)
      } else {
        false
      }
    }

  private fun buildFrameSample(
    frame: Frame,
    camera: Camera,
    allPlanes: Collection<Plane>,
    pointCloudPointCount: Int,
    fallbackSnapshot: SurfaceTrackingSnapshot?,
    placementEvidence: ArCorePlacementEvidence?,
    arCorePlacementCornersView: FloatArray?,
  ): String {
    val json = JSONObject()
    json.put("frameTimestampNs", frame.timestamp)
    json.put("androidCameraTimestampNs", frame.androidCameraTimestamp)
    json.put("cameraTrackingState", camera.trackingState.name)
    json.put("cameraTrackingFailureReason", camera.trackingFailureReason.name)
    json.put("cameraPose", poseToJson(camera.pose))
    if (camera.trackingState == TrackingState.TRACKING) {
      json.put("androidSensorPose", poseToJson(frame.androidSensorPose))
    } else {
      json.put("androidSensorPose", JSONObject.NULL)
    }
    json.put("imageIntrinsics", intrinsicsToJson(camera))
    json.put("pointCloudPointCount", pointCloudPointCount)

    val planesArray = JSONArray()
    for (plane in allPlanes) {
      val planeJson = JSONObject()
      planeJson.put("id", getPlaneIdLocked(plane))
      planeJson.put("trackingState", plane.trackingState.name)
      planeJson.put("type", plane.type.name)
      planeJson.put("centerPose", poseToJson(plane.centerPose))
      planeJson.put("extentX", plane.extentX.toDouble())
      planeJson.put("extentZ", plane.extentZ.toDouble())
      planeJson.put("polygon", polygonToJson(plane))
      val subsumedBy = plane.subsumedBy
      if (subsumedBy == null) {
        planeJson.put("subsumedById", JSONObject.NULL)
      } else {
        planeJson.put("subsumedById", getPlaneIdLocked(subsumedBy))
      }
      planesArray.put(planeJson)
    }
    json.put("planes", planesArray)
    val trackedHorizontalPlanes =
      allPlanes.count {
        it.trackingState == TrackingState.TRACKING &&
          it.subsumedBy == null &&
          it.type == Plane.Type.HORIZONTAL_UPWARD_FACING
      }
    json.put("trackedHorizontalPlaneCount", trackedHorizontalPlanes)
    json.put("fallback", fallbackToJson(fallbackSnapshot))
    json.put(
      "placement",
      JSONObject()
        .put("source", placementEvidence?.source ?: JSONObject.NULL)
        .put("trackingState", placementEvidence?.trackingState ?: JSONObject.NULL)
        .put("cornersView", arCorePlacementCornersView?.let(::floatArrayToJson) ?: JSONObject.NULL),
    )
    json.put(
      "comparison",
      comparisonToJson(
        arCoreUsable = camera.trackingState == TrackingState.TRACKING && trackedHorizontalPlanes > 0,
        fallbackSnapshot = fallbackSnapshot,
      ),
    )
    return json.toString()
  }

  private fun fallbackToJson(snapshot: SurfaceTrackingSnapshot?): JSONObject {
    val json = JSONObject()
    val telemetry = snapshot?.telemetry
    json.put("hasQuad", snapshot?.viewQuad != null)
    json.put("quadView", snapshot?.viewQuad?.let(::floatArrayToJson) ?: JSONObject.NULL)
    json.put("state", telemetry?.state ?: JSONObject.NULL)
    json.put("motionModel", telemetry?.motionModel ?: JSONObject.NULL)
    json.put("heldLastGoodTransform", telemetry?.heldLastGoodTransform ?: JSONObject.NULL)
    json.put("activeTrackCount", telemetry?.activeTrackCount ?: JSONObject.NULL)
    json.put("candidateTrackCount", telemetry?.candidateTrackCount ?: JSONObject.NULL)
    json.put("stableTrackCount", telemetry?.stableTrackCount ?: JSONObject.NULL)
    json.put("inlierCount", telemetry?.inlierCount ?: JSONObject.NULL)
    json.put("inlierRatio", telemetry?.inlierRatio ?: JSONObject.NULL)
    json.put("confidence", telemetry?.confidence ?: JSONObject.NULL)
    json.put("rejectionReason", telemetry?.rejectionReason ?: JSONObject.NULL)
    return json
  }

  private fun comparisonToJson(
    arCoreUsable: Boolean,
    fallbackSnapshot: SurfaceTrackingSnapshot?,
  ): JSONObject {
    val fallbackTelemetry = fallbackSnapshot?.telemetry
    val fallbackUsable =
      fallbackSnapshot?.viewQuad != null &&
        fallbackTelemetry?.state in setOf("tracking", "seeded")
    val outcome =
      when {
        arCoreUsable && fallbackUsable -> "both_tracking"
        arCoreUsable -> "arcore_only"
        fallbackUsable -> "fallback_only"
        else -> "neither_tracking"
      }
    return JSONObject()
      .put("arCoreUsable", arCoreUsable)
      .put("fallbackUsable", fallbackUsable)
      .put("outcome", outcome)
  }

  private fun floatArrayToJson(values: FloatArray): JSONArray =
    JSONArray().also { output -> values.forEach { output.put(it.toDouble()) } }

  private fun poseToJson(pose: Pose): JSONObject =
    JSONObject()
      .put("tx", pose.tx().toDouble())
      .put("ty", pose.ty().toDouble())
      .put("tz", pose.tz().toDouble())
      .put("qx", pose.qx().toDouble())
      .put("qy", pose.qy().toDouble())
      .put("qz", pose.qz().toDouble())
      .put("qw", pose.qw().toDouble())

  private fun intrinsicsToJson(camera: Camera): JSONObject {
    val intrinsics = camera.imageIntrinsics
    val focalLength = intrinsics.focalLength
    val principalPoint = intrinsics.principalPoint
    val imageDimensions = intrinsics.imageDimensions
    return JSONObject()
      .put("fx", focalLength[0].toDouble())
      .put("fy", focalLength[1].toDouble())
      .put("cx", principalPoint[0].toDouble())
      .put("cy", principalPoint[1].toDouble())
      .put("width", imageDimensions[0])
      .put("height", imageDimensions[1])
  }

  private fun polygonToJson(plane: Plane): JSONArray {
    val polygon = plane.polygon.duplicate()
    polygon.rewind()
    val polygonArray = JSONArray()
    while (polygon.hasRemaining()) {
      polygonArray.put(polygon.get().toDouble())
    }
    return polygonArray
  }

  private fun getPlaneIdLocked(plane: Plane): Int = planeIds.getOrPut(plane) { nextPlaneId++ }

  private fun writeManifestLocked(status: String) {
    val manifest = manifestFile ?: return
    val json = JSONObject()
    json.put("captureId", captureId)
    json.put("status", status)
    json.put("sessionRecordingStartedWallClockMs", sessionRecordingStartedWallClockMs)
    json.put("firstRecordedFrameTimestampNs", firstRecordedFrameTimestampNs ?: JSONObject.NULL)
    json.put("firstTrackedPlaneFrameTimestampNs", firstTrackedPlaneFrameTimestampNs ?: JSONObject.NULL)
    json.put("mp4Path", mp4File?.absolutePath)
    json.put("planesPath", planesFile?.absolutePath)
    json.put("imuPath", imuFile?.absolutePath)
    json.put("exportRelativePath", captureId?.let { exportRelativePath(it) } ?: JSONObject.NULL)
    json.put("recordsPrePlaneEvidence", true)
    json.put("comparisonMode", InkMarcActivity.ENABLE_ARCORE_COMPARISON)
    json.put("frameLogSchema", 2)
    manifest.writeText(json.toString(2))
  }

  private fun closeOutputsLocked() {
    try {
      planeWriter?.flush()
    } catch (_: Exception) {
    }
    try {
      imuWriter?.flush()
    } catch (_: Exception) {
    }
    try {
      planeWriter?.close()
    } catch (_: Exception) {
    }
    try {
      imuWriter?.close()
    } catch (_: Exception) {
    }
    planeWriter = null
    imuWriter = null
  }

  private fun resetStateLocked() {
    recordingInitialized = false
    firstRecordedFrameTimestampNs = null
    firstTrackedPlaneFrameTimestampNs = null
    sessionRecordingStartedWallClockMs = null
    lastLoggedFrameTimestampNs = 0
    lastObservedFrameTimestampNs = 0
    frameRowsSinceFlush = 0
    sensorRowsSinceFlush = 0
    planeIds.clear()
    nextPlaneId = 1
    captureDirectory = null
    captureId = null
    manifestFile = null
    mp4File = null
    planesFile = null
    imuFile = null
  }

  private fun postHudState(state: HudState, path: String, detail: String) {
    if (state == lastHudState && path == lastHudPath && detail == lastHudDetail) return
    lastHudState = state
    lastHudPath = path
    lastHudDetail = detail
    activity.runOnUiThread {
      // Recording HUD is disabled in InkMARC unless the instrumentation UI is restored.
      activity.view.updateRecordingStatus(
        activity.getString(R.string.recording_status_format, detail, path)
      )
    }
  }

  private fun captureBaseDirectory(): File =
    activity.getExternalFilesDir("captures") ?: File(activity.filesDir, "captures")

  private fun exportFinishedCapture(sourceDirectory: File, captureId: String): Boolean {
    val exportPath = exportRelativePath(captureId)
    try {
      exportDirectoryToDownloads(sourceDirectory, captureId)
      postHudState(
        state = HudState.STOPPED,
        path = exportPath,
        detail = activity.getString(R.string.recording_exported)
      )
      Log.i(TAG, "Exported capture $captureId to $exportPath")
      return true
    } catch (e: Exception) {
      Log.e(TAG, "Failed to export capture $captureId", e)
      postHudState(
        state = HudState.ERROR,
        path = exportPath,
        detail = activity.getString(R.string.recording_export_failed, e.message ?: "unknown error")
      )
      activity.runOnUiThread {
        activity.view.snackbarHelper.showMessageWithDismiss(
          activity,
          activity.getString(R.string.recording_export_failed, e.message ?: "unknown error")
        )
      }
      return false
    }
  }

  private fun exportDirectoryToDownloads(sourceDirectory: File, captureId: String) {
    val exportFiles = listOf("session.mp4", "planes.ndjson", "imu.csv", "capture_manifest.json")
    for (fileName in exportFiles) {
      val sourceFile = File(sourceDirectory, fileName)
      if (!sourceFile.exists()) continue
      exportFileToDownloads(sourceFile, captureId)
    }
  }

  private fun exportFileToDownloads(sourceFile: File, captureId: String) {
    val resolver = activity.contentResolver
    val relativePath = exportRelativePath(captureId)
    val contentValues =
      ContentValues().apply {
        put(MediaStore.MediaColumns.DISPLAY_NAME, sourceFile.name)
        put(MediaStore.MediaColumns.MIME_TYPE, mimeTypeFor(sourceFile.name))
        put(MediaStore.MediaColumns.RELATIVE_PATH, relativePath)
        put(MediaStore.MediaColumns.IS_PENDING, 1)
      }

    val collection = MediaStore.Downloads.EXTERNAL_CONTENT_URI
    val itemUri =
      resolver.insert(collection, contentValues)
        ?: throw IOException("Failed to create MediaStore entry for ${sourceFile.name}")

    try {
      FileInputStream(sourceFile).use { input ->
        resolver.openOutputStream(itemUri, "wt")?.use { output ->
          input.copyTo(output)
        } ?: throw IOException("Failed to open MediaStore output stream for ${sourceFile.name}")
      }
      contentValues.clear()
      contentValues.put(MediaStore.MediaColumns.IS_PENDING, 0)
      resolver.update(itemUri, contentValues, null, null)
    } catch (e: Exception) {
      resolver.delete(itemUri, null, null)
      throw e
    }
  }

  private fun exportRelativePath(captureId: String): String =
    "${Environment.DIRECTORY_DOWNLOADS}/$EXPORT_COLLECTION_FOLDER/$captureId/"

  private fun mimeTypeFor(fileName: String): String =
    when {
      fileName.endsWith(".mp4") -> "video/mp4"
      fileName.endsWith(".csv") -> "text/csv"
      fileName.endsWith(".json") || fileName.endsWith(".ndjson") -> "application/json"
      else -> "application/octet-stream"
    }

  private fun createCaptureId(): String =
    SimpleDateFormat("yyyyMMdd_HHmmss_SSS", Locale.US).format(Date())

  private fun sensorTypeName(sensorType: Int): String =
    when (sensorType) {
      Sensor.TYPE_ACCELEROMETER -> "accelerometer"
      Sensor.TYPE_GYROSCOPE -> "gyroscope"
      else -> "sensor_$sensorType"
    }
}

