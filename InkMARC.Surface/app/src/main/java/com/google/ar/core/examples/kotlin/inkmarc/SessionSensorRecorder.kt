package com.inkmarc

import android.content.ContentValues
import android.content.Context
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.media.MediaScannerConnection
import android.net.Uri
import android.os.Build
import android.os.Environment
import android.os.Handler
import android.os.HandlerThread
import android.provider.MediaStore
import java.io.BufferedWriter
import java.io.File
import java.io.OutputStreamWriter
import java.util.Locale

/** Writes IMU samples and camera/tracker markers on the video's monotonic timebase. */
class SessionSensorRecorder(
  private val activity: InkMarcActivity,
  private val sessionId: String,
  private val videoDisplayName: String,
  private val videoStartElapsedRealtimeNs: Long,
  private val cameraTimestampsUseElapsedRealtime: Boolean,
  private val analysisImageWidth: Int,
  private val analysisImageHeight: Int,
  private val viewWidth: Int,
  private val viewHeight: Int,
  private val analysisRotationDegrees: Int,
  private val videoOrientationHintDegrees: Int,
) {
  companion object {
    private const val SENSOR_PERIOD_US = 5_000
    private const val FLUSH_INTERVAL_ROWS = 256
  }

  private data class Output(
    val displayName: String,
    val uri: Uri? = null,
    val file: File? = null,
  )

  private val sensorManager = activity.getSystemService(Context.SENSOR_SERVICE) as SensorManager
  private val lock = Any()
  private var sensorThread: HandlerThread? = null
  private var writer: BufferedWriter? = null
  private var output: Output? = null
  private var rowCount = 0
  private var closed = false

  private val listener =
    object : SensorEventListener {
      override fun onSensorChanged(event: SensorEvent) {
        writeSensorRow(event)
      }

      override fun onAccuracyChanged(sensor: Sensor, accuracy: Int) {
        writeRow(
          timestampNs = android.os.SystemClock.elapsedRealtimeNanos(),
          event = "accuracy",
          sensorType = sensor.stringType,
          accuracy = accuracy,
          values = floatArrayOf(),
          quad = null,
        )
      }
    }

  fun start() {
    val nextOutput = createOutput()
    var nextWriter: BufferedWriter? = null
    try {
      val stream =
        if (nextOutput.uri != null) {
          activity.contentResolver.openOutputStream(nextOutput.uri, "w")
            ?: error("Unable to open sensor log")
        } else {
          checkNotNull(nextOutput.file).outputStream()
        }
      nextWriter = BufferedWriter(OutputStreamWriter(stream, Charsets.UTF_8), 64 * 1024)
      nextWriter.appendLine("# session_id=$sessionId")
      nextWriter.appendLine("# video_file=$videoDisplayName")
      nextWriter.appendLine("# clock=android_elapsed_realtime_nanoseconds")
      nextWriter.appendLine("# video_start_elapsed_realtime_ns=$videoStartElapsedRealtimeNs")
      nextWriter.appendLine("# analysis_image_width=$analysisImageWidth")
      nextWriter.appendLine("# analysis_image_height=$analysisImageHeight")
      nextWriter.appendLine("# view_width=$viewWidth")
      nextWriter.appendLine("# view_height=$viewHeight")
      nextWriter.appendLine("# analysis_rotation_degrees=$analysisRotationDegrees")
      nextWriter.appendLine("# video_orientation_hint_degrees=$videoOrientationHintDegrees")
      nextWriter.appendLine("# preview_scaling=center_crop")
      nextWriter.appendLine(
        "# camera_timestamp_source=${if (cameraTimestampsUseElapsedRealtime) "elapsed_realtime" else "monotonic_mapped_to_elapsed_realtime"}",
      )
      nextWriter.appendLine("# tracker_uses_imu=true")
      nextWriter.appendLine("# tracker_imu_model=game_rotation_vector+gyro_horizontal_plane_constraint")
      nextWriter.appendLine("# tracker_visual_support=full_frame_horizontal_plane")
      nextWriter.appendLine("# tracker_hand_occlusion=mediapipe_landmarks+wrist_forearm_extension")
      nextWriter.appendLine("# tracker_dynamic_occlusion=imu_compensated_temporal_frame_residual")
      nextWriter.appendLine("# tracker_diagnostics_schema=7")
      nextWriter.appendLine(
        "timestamp_ns,source_timestamp_ns,callback_start_timestamp_ns,arrival_timestamp_ns," +
          "effective_frame_timestamp_ns,exposure_time_ns,rolling_shutter_skew_ns," +
          "camera_delivery_latency_ms,processing_latency_ms,total_latency_ms," +
          "video_time_ns,event,sensor_type,accuracy,v0,v1,v2,v3," +
          "quad_x0,quad_y0,quad_x1,quad_y1,quad_x2,quad_y2,quad_x3,quad_y3," +
          "tracker_state,tracker_motion_model,tracker_held,tracker_active_tracks," +
          "tracker_candidate_tracks,tracker_stable_tracks,tracker_inliers,tracker_inlier_ratio," +
          "tracker_reprojection_error_px,tracker_reference_coverage,tracker_current_coverage," +
          "tracker_transform_condition,tracker_confidence,tracker_hand_mask_age_ms," +
          "tracker_hand_mask_area_ratio,tracker_motion_mask_area_ratio," +
          "tracker_combined_mask_area_ratio,tracker_mask_rejected_tracks," +
          "tracker_reacquired_tracks,tracker_effective_plane_support," +
          "tracker_weighted_plane_coverage," +
          "tracker_rejection_reason",
      )
      nextWriter.flush()

      synchronized(lock) {
        output = nextOutput
        writer = nextWriter
      }
      val thread = HandlerThread("InkMarcSessionSensors").also { it.start() }
      sensorThread = thread
      val handler = Handler(thread.looper)
      listOf(
          Sensor.TYPE_ACCELEROMETER,
          Sensor.TYPE_GYROSCOPE,
          Sensor.TYPE_GRAVITY,
          Sensor.TYPE_LINEAR_ACCELERATION,
          Sensor.TYPE_MAGNETIC_FIELD,
          Sensor.TYPE_ROTATION_VECTOR,
          Sensor.TYPE_GAME_ROTATION_VECTOR,
        )
        .mapNotNull(sensorManager::getDefaultSensor)
        .forEach { sensor ->
          sensorManager.registerListener(listener, sensor, SENSOR_PERIOD_US, 0, handler)
        }
    } catch (e: Exception) {
      sensorManager.unregisterListener(listener)
      sensorThread?.quitSafely()
      sensorThread = null
      runCatching { nextWriter?.close() }
      discardOutput(nextOutput)
      synchronized(lock) {
        writer = null
        output = null
        closed = true
      }
      throw e
    }
  }

  fun recordCameraFrame(
    frameTiming: CameraFrameTiming,
    callbackStartElapsedRealtimeNs: Long,
    arrivalElapsedRealtimeNs: Long,
    trackedViewQuad: FloatArray?,
    trackerTelemetry: SurfaceTrackingTelemetry?,
  ) {
    writeRow(
      timestampNs = frameTiming.centerExposureTimestampNs,
      sourceTimestampNs = frameTiming.sourceTimestampNs,
      callbackStartTimestampNs = callbackStartElapsedRealtimeNs,
      arrivalTimestampNs = arrivalElapsedRealtimeNs,
      effectiveFrameTimestampNs = frameTiming.centerExposureTimestampNs,
      exposureTimeNs = frameTiming.exposureTimeNs,
      rollingShutterSkewNs = frameTiming.rollingShutterSkewNs,
      event = "camera_frame",
      sensorType = "camera",
      accuracy = null,
      values = floatArrayOf(),
      quad = trackedViewQuad,
      trackerTelemetry = trackerTelemetry,
    )
  }

  fun stopAndFinalize(): String? {
    sensorManager.unregisterListener(listener)
    sensorThread?.quitSafely()
    sensorThread = null
    val activeOutput: Output
    synchronized(lock) {
      if (closed) return null
      closed = true
      activeOutput = output ?: return null
      runCatching { writer?.flush() }
      runCatching { writer?.close() }
      writer = null
      output = null
    }

    if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q && activeOutput.uri != null) {
      activity.contentResolver.update(
        activeOutput.uri,
        ContentValues().apply { put(MediaStore.Downloads.IS_PENDING, 0) },
        null,
        null,
      )
    } else {
      activeOutput.file?.let { file ->
        MediaScannerConnection.scanFile(activity, arrayOf(file.absolutePath), arrayOf("text/csv"), null)
      }
    }
    return activeOutput.displayName
  }

  private fun writeSensorRow(event: SensorEvent) {
    writeRow(
      timestampNs = event.timestamp,
      sourceTimestampNs = event.timestamp,
      event = "sensor",
      sensorType = event.sensor.stringType,
      accuracy = event.accuracy,
      values = event.values,
      quad = null,
    )
  }

  private fun writeRow(
    timestampNs: Long,
    sourceTimestampNs: Long = timestampNs,
    callbackStartTimestampNs: Long? = null,
    arrivalTimestampNs: Long? = null,
    effectiveFrameTimestampNs: Long? = null,
    exposureTimeNs: Long? = null,
    rollingShutterSkewNs: Long? = null,
    event: String,
    sensorType: String,
    accuracy: Int?,
    values: FloatArray,
    quad: FloatArray?,
    trackerTelemetry: SurfaceTrackingTelemetry? = null,
  ) {
    synchronized(lock) {
      if (closed) return
      val activeWriter = writer ?: return
      val columns = ArrayList<String>(44)
      columns += timestampNs.toString()
      columns += sourceTimestampNs.toString()
      columns += callbackStartTimestampNs?.toString().orEmpty()
      columns += arrivalTimestampNs?.toString().orEmpty()
      columns += effectiveFrameTimestampNs?.toString().orEmpty()
      columns += exposureTimeNs?.toString().orEmpty()
      columns += rollingShutterSkewNs?.toString().orEmpty()
      val deliveryReferenceTimestampNs = effectiveFrameTimestampNs ?: sourceTimestampNs
      columns +=
        callbackStartTimestampNs
          ?.let { (it - deliveryReferenceTimestampNs).toDouble() / 1_000_000.0 }
          ?.toCsvValue()
          .orEmpty()
      columns +=
        if (callbackStartTimestampNs != null && arrivalTimestampNs != null) {
          ((arrivalTimestampNs - callbackStartTimestampNs).toDouble() / 1_000_000.0).toCsvValue()
        } else {
          ""
        }
      columns +=
        arrivalTimestampNs
          ?.let { (it - deliveryReferenceTimestampNs).toDouble() / 1_000_000.0 }
          ?.toCsvValue()
          .orEmpty()
      columns += (timestampNs - videoStartElapsedRealtimeNs).toString()
      columns += event
      columns += sensorType
      columns += accuracy?.toString().orEmpty()
      for (index in 0 until 4) {
        columns += values.getOrNull(index)?.toCsvValue().orEmpty()
      }
      for (index in 0 until 8) {
        columns += quad?.getOrNull(index)?.toCsvValue().orEmpty()
      }
      columns += trackerTelemetry?.state.orEmpty()
      columns += trackerTelemetry?.motionModel.orEmpty()
      columns += trackerTelemetry?.heldLastGoodTransform?.toString().orEmpty()
      columns += trackerTelemetry?.activeTrackCount?.toString().orEmpty()
      columns += trackerTelemetry?.candidateTrackCount?.toString().orEmpty()
      columns += trackerTelemetry?.stableTrackCount?.toString().orEmpty()
      columns += trackerTelemetry?.inlierCount?.toString().orEmpty()
      columns += trackerTelemetry?.inlierRatio?.toCsvValue().orEmpty()
      columns += trackerTelemetry?.meanReprojectionErrorPx?.toCsvValue().orEmpty()
      columns += trackerTelemetry?.referenceFeatureCoverage?.toCsvValue().orEmpty()
      columns += trackerTelemetry?.currentFeatureCoverage?.toCsvValue().orEmpty()
      columns += trackerTelemetry?.transformCondition?.toCsvValue().orEmpty()
      columns += trackerTelemetry?.confidence?.toCsvValue().orEmpty()
      columns += trackerTelemetry?.handMaskAgeMs?.toCsvValue().orEmpty()
      columns += trackerTelemetry?.handMaskAreaRatio?.toCsvValue().orEmpty()
      columns += trackerTelemetry?.motionMaskAreaRatio?.toCsvValue().orEmpty()
      columns += trackerTelemetry?.combinedMaskAreaRatio?.toCsvValue().orEmpty()
      columns += trackerTelemetry?.maskRejectedTrackCount?.toString().orEmpty()
      columns += trackerTelemetry?.reacquiredTrackCount?.toString().orEmpty()
      columns += trackerTelemetry?.effectivePlaneSupport?.toCsvValue().orEmpty()
      columns += trackerTelemetry?.weightedPlaneCoverage?.toCsvValue().orEmpty()
      columns += trackerTelemetry?.rejectionReason?.toCsvField().orEmpty()
      activeWriter.appendLine(columns.joinToString(","))
      rowCount += 1
      if (rowCount % FLUSH_INTERVAL_ROWS == 0) activeWriter.flush()
    }
  }

  private fun createOutput(): Output {
    val displayName = "${sessionId}_sensors.csv"
    if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
      val values =
        ContentValues().apply {
          put(MediaStore.Downloads.DISPLAY_NAME, displayName)
          put(MediaStore.Downloads.MIME_TYPE, "text/csv")
          put(MediaStore.Downloads.RELATIVE_PATH, "${Environment.DIRECTORY_DOWNLOADS}/InkMARC")
          put(MediaStore.Downloads.IS_PENDING, 1)
        }
      val uri =
        activity.contentResolver.insert(MediaStore.Downloads.EXTERNAL_CONTENT_URI, values)
          ?: error("Unable to create sensor log")
      return Output(displayName = displayName, uri = uri)
    }

    val parent =
      File(
        activity.getExternalFilesDir(Environment.DIRECTORY_DOCUMENTS) ?: activity.filesDir,
        "InkMARC",
      ).apply { mkdirs() }
    return Output(displayName = displayName, file = File(parent, displayName))
  }

  private fun discardOutput(output: Output) {
    runCatching { output.uri?.let { activity.contentResolver.delete(it, null, null) } }
    runCatching { output.file?.delete() }
  }
}

private fun Float.toCsvValue(): String = String.format(Locale.US, "%.9g", this)

private fun Double.toCsvValue(): String = String.format(Locale.US, "%.9g", this)

private fun String.toCsvField(): String =
  if (contains(',') || contains('"') || contains('\n') || contains('\r')) {
    "\"${replace("\"", "\"\"")}\""
  } else {
    this
  }
