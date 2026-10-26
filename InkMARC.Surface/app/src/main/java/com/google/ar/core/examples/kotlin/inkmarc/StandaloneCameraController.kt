package com.inkmarc

import android.Manifest
import android.annotation.SuppressLint
import android.content.Context
import android.content.pm.PackageManager
import android.graphics.ImageFormat
import android.graphics.Matrix
import android.graphics.SurfaceTexture
import android.hardware.camera2.CameraCaptureSession
import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CameraDevice
import android.hardware.camera2.CameraManager
import android.hardware.camera2.CaptureResult
import android.hardware.camera2.CaptureRequest
import android.hardware.camera2.TotalCaptureResult
import android.media.ImageReader
import android.os.Handler
import android.os.HandlerThread
import android.os.SystemClock
import android.util.Log
import android.util.Size
import android.view.Surface
import android.view.TextureView
import android.widget.Toast
import androidx.core.content.ContextCompat
import androidx.lifecycle.DefaultLifecycleObserver
import androidx.lifecycle.LifecycleOwner
import com.google.ar.core.examples.java.common.helpers.CameraPermissionHelper
import kotlin.math.abs
import kotlin.math.max

/** Camera2 source used while ARCore is switched off. */
class StandaloneCameraController(private val activity: InkMarcActivity) :
  DefaultLifecycleObserver, TextureView.SurfaceTextureListener {
  companion object {
    private const val TAG = "StandaloneCamera"
    private const val TARGET_FRAME_AREA = 1280 * 720
    private const val MAX_CAPTURE_TIMING_ENTRIES = 12
  }

  private data class CaptureTimingMetadata(
    val exposureTimeNs: Long,
    val rollingShutterSkewNs: Long,
  )

  private val textureView = activity.view.fallbackCameraView
  private val cameraManager = activity.getSystemService(Context.CAMERA_SERVICE) as CameraManager
  private var cameraThread: HandlerThread? = null
  private var cameraHandler: Handler? = null
  private var cameraDevice: CameraDevice? = null
  private var captureSession: CameraCaptureSession? = null
  private var imageReader: ImageReader? = null
  private var previewSurface: Surface? = null
  private var sessionVideoRecorder: SessionVideoRecorder? = null
  @Volatile private var sessionSensorRecorder: SessionSensorRecorder? = null
  private var cameraId: String? = null
  private var streamSize: Size? = null
  private var sensorOrientationDegrees = 0
  private var cameraTimestampsUseElapsedRealtime = false
  private var cameraToElapsedRealtimeOffsetNs = 0L
  private val captureTimingLock = Any()
  private val captureTimingByTimestamp = LinkedHashMap<Long, CaptureTimingMetadata>()
  private var latestCaptureTimingMetadata: CaptureTimingMetadata? = null
  private var analysisRotationDegrees = 0
  private var videoOrientationHintDegrees = 0
  private var cameraIntrinsics = CameraIntrinsics.defaultFor(1280, 720)
  @Volatile private var mapper: CameraCoordinateMapper? = null
  @Volatile private var resumed = false
  @Volatile private var opening = false

  init {
    textureView.surfaceTextureListener = this
  }

  override fun onResume(owner: LifecycleOwner) {
    resumed = true
    start()
  }

  override fun onPause(owner: LifecycleOwner) {
    resumed = false
    closeCamera()
    stopCameraThread()
  }

  override fun onDestroy(owner: LifecycleOwner) {
    closeCamera()
    stopCameraThread()
  }

  fun start() {
    if (!resumed) return
    startCameraThread()
    if (!textureView.isAvailable) return
    if (ContextCompat.checkSelfPermission(activity, Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
      CameraPermissionHelper.requestCameraPermission(activity)
      return
    }
    openCamera()
  }

  @SuppressLint("MissingPermission")
  private fun openCamera() {
    if (cameraDevice != null || opening || !resumed) return
    try {
      val selectedId = selectBackCamera()
      val selectedSize = streamSize ?: error("No shared preview/YUV camera size")
      val surfaceTexture = textureView.surfaceTexture ?: return
      surfaceTexture.setDefaultBufferSize(selectedSize.width, selectedSize.height)
      updateTransform(textureView.width, textureView.height)

      imageReader =
        ImageReader.newInstance(selectedSize.width, selectedSize.height, ImageFormat.YUV_420_888, 2).apply {
          setOnImageAvailableListener(
            { reader ->
              val image = reader.acquireLatestImage() ?: return@setOnImageAvailableListener
              image.use {
                val callbackStartElapsedRealtimeNs = SystemClock.elapsedRealtimeNanos()
                val coordinateMapper = mapper ?: return@use
                val frameTiming =
                  frameTimingForImage(
                    it.timestamp,
                    it.height,
                    callbackStartElapsedRealtimeNs,
                  )
                try {
                  activity.fallbackSurfaceTracker?.processStandaloneFrame(
                    image = it,
                    frameTiming = frameTiming,
                    coordinateMapper = coordinateMapper,
                  )
                } catch (e: Exception) {
                  Log.e(TAG, "Fallback frame processing failed", e)
                }
                val trackingSnapshot = activity.fallbackSurfaceTracker?.getLatestTrackingSnapshot()
                sessionSensorRecorder?.recordCameraFrame(
                  frameTiming = frameTiming,
                  callbackStartElapsedRealtimeNs = callbackStartElapsedRealtimeNs,
                  arrivalElapsedRealtimeNs = SystemClock.elapsedRealtimeNanos(),
                  trackedViewQuad = trackingSnapshot?.viewQuad,
                  trackerTelemetry = trackingSnapshot?.telemetry,
                )
              }
            },
            cameraHandler,
          )
        }

      opening = true
      cameraManager.openCamera(selectedId, cameraStateCallback, cameraHandler)
    } catch (e: Exception) {
      opening = false
      reportCameraError(e.message ?: e.javaClass.simpleName)
    }
  }

  private val cameraStateCallback =
    object : CameraDevice.StateCallback() {
      override fun onOpened(camera: CameraDevice) {
        opening = false
        if (!resumed) {
          camera.close()
          return
        }
        cameraDevice = camera
        createPreviewSession(camera)
      }

      override fun onDisconnected(camera: CameraDevice) {
        opening = false
        camera.close()
        closeCamera()
      }

      override fun onError(camera: CameraDevice, error: Int) {
        opening = false
        camera.close()
        closeCamera()
        reportCameraError("camera error $error")
      }
    }

  private fun createPreviewSession(camera: CameraDevice) {
    val surfaceTexture = textureView.surfaceTexture ?: return
    val readerSurface = imageReader?.surface ?: return
    val surface = Surface(surfaceTexture)
    previewSurface = surface
    try {
      val recorder =
        if (InkMarcActivity.ENABLE_SESSION_VIDEO_RECORDING) {
          try {
            SessionVideoRecorder(activity).also {
              it.prepare(checkNotNull(streamSize), videoOrientationHintDegrees)
            }
          } catch (e: Exception) {
            reportRecordingError(e.message ?: e.javaClass.simpleName)
            null
          }
        } else {
          null
        }
      sessionVideoRecorder = recorder
      val recordingSurface = recorder?.surface
      val outputSurfaces = mutableListOf(surface, readerSurface)
      recordingSurface?.let(outputSurfaces::add)
      val request =
        camera.createCaptureRequest(
          if (recordingSurface != null) CameraDevice.TEMPLATE_RECORD else CameraDevice.TEMPLATE_PREVIEW,
        ).apply {
          addTarget(surface)
          addTarget(readerSurface)
          recordingSurface?.let(::addTarget)
          set(CaptureRequest.CONTROL_AF_MODE, CaptureRequest.CONTROL_AF_MODE_CONTINUOUS_PICTURE)
          set(CaptureRequest.CONTROL_AE_MODE, CaptureRequest.CONTROL_AE_MODE_ON)
        }
      camera.createCaptureSession(
        outputSurfaces,
        object : CameraCaptureSession.StateCallback() {
          override fun onConfigured(session: CameraCaptureSession) {
            if (cameraDevice !== camera) {
              session.close()
              return
            }
            captureSession = session
            try {
              session.setRepeatingRequest(request.build(), captureCallback, cameraHandler)
              if (recorder != null) {
                val videoStartElapsedRealtimeNs = recorder.start()
                if (InkMarcActivity.ENABLE_SESSION_SENSOR_RECORDING) {
                  try {
                    sessionSensorRecorder =
                      SessionSensorRecorder(
                          activity = activity,
                          sessionId = recorder.sessionId,
                          videoDisplayName = recorder.videoDisplayName,
                          videoStartElapsedRealtimeNs = videoStartElapsedRealtimeNs,
                          cameraTimestampsUseElapsedRealtime = cameraTimestampsUseElapsedRealtime,
                          analysisImageWidth = checkNotNull(streamSize).width,
                          analysisImageHeight = checkNotNull(streamSize).height,
                          viewWidth = textureView.width,
                          viewHeight = textureView.height,
                          analysisRotationDegrees = analysisRotationDegrees,
                          videoOrientationHintDegrees = videoOrientationHintDegrees,
                        )
                        .also { it.start() }
                  } catch (e: Exception) {
                    reportRecordingError("sensor log: ${e.message ?: e.javaClass.simpleName}")
                  }
                }
                reportRecordingStarted()
              }
            } catch (e: Exception) {
              reportRecordingError(e.message ?: e.javaClass.simpleName)
            }
          }

          override fun onConfigureFailed(session: CameraCaptureSession) {
            sessionVideoRecorder?.stopAndFinalize()
            sessionVideoRecorder = null
            reportCameraError("preview configuration failed")
          }
        },
        cameraHandler,
      )
    } catch (e: Exception) {
      sessionVideoRecorder?.stopAndFinalize()
      sessionVideoRecorder = null
      surface.release()
      previewSurface = null
      reportCameraError(e.message ?: e.javaClass.simpleName)
    }
  }

  private fun selectBackCamera(): String {
    cameraId?.let { return it }
    val selected =
      cameraManager.cameraIdList.firstOrNull { id ->
        cameraManager.getCameraCharacteristics(id).get(CameraCharacteristics.LENS_FACING) ==
          CameraCharacteristics.LENS_FACING_BACK
      } ?: cameraManager.cameraIdList.firstOrNull() ?: error("No camera found")
    val characteristics = cameraManager.getCameraCharacteristics(selected)
    sensorOrientationDegrees =
      characteristics.get(CameraCharacteristics.SENSOR_ORIENTATION) ?: 0
    cameraTimestampsUseElapsedRealtime =
      characteristics.get(CameraCharacteristics.SENSOR_INFO_TIMESTAMP_SOURCE) ==
        CameraCharacteristics.SENSOR_INFO_TIMESTAMP_SOURCE_REALTIME
    cameraToElapsedRealtimeOffsetNs =
      if (cameraTimestampsUseElapsedRealtime) {
        0L
      } else {
        // UNKNOWN camera timestamps are commonly CLOCK_MONOTONIC while SensorEvent timestamps use
        // CLOCK_BOOTTIME. Their difference is stable while the camera session is running.
        SystemClock.elapsedRealtimeNanos() - System.nanoTime()
      }
    val configuration =
      characteristics.get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP)
        ?: error("Camera has no stream configuration")
    val previewSizes = configuration.getOutputSizes(SurfaceTexture::class.java)?.toSet().orEmpty()
    val yuvSizes = configuration.getOutputSizes(ImageFormat.YUV_420_888)?.toList().orEmpty()
    val recordingSizes =
      configuration.getOutputSizes(android.media.MediaRecorder::class.java)?.toSet().orEmpty()
    val selectedStreamSize =
      yuvSizes
        .filter { it in previewSizes && (!InkMarcActivity.ENABLE_SESSION_VIDEO_RECORDING || it in recordingSizes) }
        .minByOrNull { size ->
          val area = size.width * size.height
          val aspectPenalty = abs(size.width.toFloat() / size.height - 16f / 9f) * TARGET_FRAME_AREA
          abs(area - TARGET_FRAME_AREA) + aspectPenalty.toInt()
        }
        ?: error("Camera has no size shared by preview and YUV analysis")
    streamSize = selectedStreamSize
    cameraIntrinsics = calculateCameraIntrinsics(characteristics, selectedStreamSize)
    cameraId = selected
    return selected
  }

  private fun updateTransform(viewWidth: Int, viewHeight: Int) {
    val size = streamSize ?: return
    if (viewWidth <= 0 || viewHeight <= 0) return
    val displayOrientationDegrees =
      when (textureView.display?.rotation ?: Surface.ROTATION_0) {
        Surface.ROTATION_90 -> 90
        Surface.ROTATION_180 -> 180
        Surface.ROTATION_270 -> 270
        else -> 0
      }

    // ImageReader YUV frames remain in landscape sensor coordinates even though this device's
    // TextureView preview is already display-oriented. Keep separate transforms so portrait
    // motion and tracker overlays use view axes without rotating the direct preview.
    analysisRotationDegrees =
      (sensorOrientationDegrees - displayOrientationDegrees + 360) % 360
    videoOrientationHintDegrees = analysisRotationDegrees
    mapper =
      CameraCoordinateMapper(
        size.width,
        size.height,
        viewWidth,
        viewHeight,
        analysisRotationDegrees,
        cameraIntrinsics,
      )
    // SurfaceTexture is already upright, but its portrait image is stretched to the view bounds.
    // Correct only the aspect ratio, using a uniform center crop without another rotation.
    val logicalPreviewWidth =
      if (analysisRotationDegrees == 90 || analysisRotationDegrees == 270) size.height else size.width
    val logicalPreviewHeight =
      if (analysisRotationDegrees == 90 || analysisRotationDegrees == 270) size.width else size.height
    val stretchedScaleX = viewWidth.toFloat() / logicalPreviewWidth
    val stretchedScaleY = viewHeight.toFloat() / logicalPreviewHeight
    val uniformFillScale = max(stretchedScaleX, stretchedScaleY)
    // TextureView applies this matrix directly to the displayed content. Expand the compressed
    // axis to the uniform fill scale; using the inverse here makes portrait subjects narrower.
    val displayScaleX = uniformFillScale / stretchedScaleX
    val displayScaleY = uniformFillScale / stretchedScaleY
    textureView.setTransform(
      Matrix().apply {
        setScale(
          displayScaleX,
          displayScaleY,
          viewWidth * 0.5f,
          viewHeight * 0.5f,
        )
      },
    )
  }

  private fun closeCamera() {
    opening = false
    try {
      captureSession?.stopRepeating()
      captureSession?.abortCaptures()
    } catch (_: Exception) {
      // The camera may already be disconnected.
    }
    captureSession?.close()
    captureSession = null
    val savedSensorLog = sessionSensorRecorder?.stopAndFinalize()
    sessionSensorRecorder = null
    if (savedSensorLog != null) reportSensorSaved(savedSensorLog)
    val savedVideo = sessionVideoRecorder?.stopAndFinalize()
    sessionVideoRecorder = null
    if (savedVideo != null) reportRecordingSaved(savedVideo)
    cameraDevice?.close()
    cameraDevice = null
    imageReader?.close()
    imageReader = null
    previewSurface?.release()
    previewSurface = null
    synchronized(captureTimingLock) {
      captureTimingByTimestamp.clear()
      latestCaptureTimingMetadata = null
    }
  }

  private val captureCallback =
    object : CameraCaptureSession.CaptureCallback() {
      override fun onCaptureCompleted(
        session: CameraCaptureSession,
        request: CaptureRequest,
        result: TotalCaptureResult,
      ) {
        val timestampNs = result.get(CaptureResult.SENSOR_TIMESTAMP) ?: return
        val timing =
          CaptureTimingMetadata(
            exposureTimeNs = result.get(CaptureResult.SENSOR_EXPOSURE_TIME) ?: 0L,
            rollingShutterSkewNs = result.get(CaptureResult.SENSOR_ROLLING_SHUTTER_SKEW) ?: 0L,
          )
        synchronized(captureTimingLock) {
          latestCaptureTimingMetadata = timing
          captureTimingByTimestamp[timestampNs] = timing
          while (captureTimingByTimestamp.size > MAX_CAPTURE_TIMING_ENTRIES) {
            val oldest = captureTimingByTimestamp.entries.firstOrNull()?.key ?: break
            captureTimingByTimestamp.remove(oldest)
          }
        }
      }
    }

  private fun frameTimingForImage(
    sourceTimestampNs: Long,
    imageHeight: Int,
    callbackElapsedRealtimeNs: Long,
  ): CameraFrameTiming {
    val metadata =
      synchronized(captureTimingLock) {
        captureTimingByTimestamp.remove(sourceTimestampNs) ?: latestCaptureTimingMetadata
      }
    val elapsedRealtimeTimestampNs =
      if (cameraTimestampsUseElapsedRealtime) {
        sourceTimestampNs
      } else {
        // UNKNOWN is vendor-defined. Select between the two Android camera clock conventions by
        // choosing the candidate that produces a plausible, smallest non-negative delivery delay.
        val candidates =
          longArrayOf(
            sourceTimestampNs,
            sourceTimestampNs + cameraToElapsedRealtimeOffsetNs,
          )
        candidates
          .filter { candidate -> candidate <= callbackElapsedRealtimeNs }
          .minByOrNull { candidate -> callbackElapsedRealtimeNs - candidate }
          ?: sourceTimestampNs + cameraToElapsedRealtimeOffsetNs
      }
    return CameraFrameTiming(
      sourceTimestampNs = sourceTimestampNs,
      firstRowExposureStartElapsedRealtimeNs = elapsedRealtimeTimestampNs,
      exposureTimeNs = metadata?.exposureTimeNs ?: 0L,
      rollingShutterSkewNs = metadata?.rollingShutterSkewNs ?: 0L,
      imageHeight = imageHeight,
    )
  }

  private fun startCameraThread() {
    if (cameraThread != null) return
    cameraThread = HandlerThread("InkMarcFallbackCamera").also { it.start() }
    cameraHandler = Handler(cameraThread!!.looper)
  }

  private fun stopCameraThread() {
    cameraThread?.quitSafely()
    cameraThread = null
    cameraHandler = null
  }

  private fun reportCameraError(message: String) {
    Log.e(TAG, message)
    activity.runOnUiThread {
      activity.view.updateFallbackStatus(
        activity.getString(R.string.fallback_status_camera_error, message),
      )
    }
  }

  private fun calculateCameraIntrinsics(
    characteristics: CameraCharacteristics,
    outputSize: Size,
  ): CameraIntrinsics {
    val activeArray =
      characteristics.get(CameraCharacteristics.SENSOR_INFO_PRE_CORRECTION_ACTIVE_ARRAY_SIZE)
        ?: characteristics.get(CameraCharacteristics.SENSOR_INFO_ACTIVE_ARRAY_SIZE)
        ?: return CameraIntrinsics.defaultFor(outputSize.width, outputSize.height)
    val calibration = characteristics.get(CameraCharacteristics.LENS_INTRINSIC_CALIBRATION)
    val focalLengths = characteristics.get(CameraCharacteristics.LENS_INFO_AVAILABLE_FOCAL_LENGTHS)
    val physicalSize = characteristics.get(CameraCharacteristics.SENSOR_INFO_PHYSICAL_SIZE)
    val activeWidth = activeArray.width().toDouble()
    val activeHeight = activeArray.height().toDouble()
    val outputAspect = outputSize.width.toDouble() / outputSize.height.toDouble()
    val activeAspect = activeWidth / activeHeight
    val cropWidth: Double
    val cropHeight: Double
    val cropLeft: Double
    val cropTop: Double
    if (activeAspect > outputAspect) {
      cropHeight = activeHeight
      cropWidth = cropHeight * outputAspect
      cropLeft = activeArray.left + (activeWidth - cropWidth) * 0.5
      cropTop = activeArray.top.toDouble()
    } else {
      cropWidth = activeWidth
      cropHeight = cropWidth / outputAspect
      cropLeft = activeArray.left.toDouble()
      cropTop = activeArray.top + (activeHeight - cropHeight) * 0.5
    }
    val scaleX = outputSize.width / cropWidth
    val scaleY = outputSize.height / cropHeight

    val activeFx: Double
    val activeFy: Double
    val activeCx: Double
    val activeCy: Double
    if (calibration != null && calibration.size >= 4) {
      activeFx = calibration[0].toDouble()
      activeFy = calibration[1].toDouble()
      activeCx = calibration[2].toDouble()
      activeCy = calibration[3].toDouble()
    } else if (focalLengths != null && focalLengths.isNotEmpty() && physicalSize != null) {
      activeFx = focalLengths[0] / physicalSize.width * activeWidth
      activeFy = focalLengths[0] / physicalSize.height * activeHeight
      activeCx = activeArray.exactCenterX().toDouble()
      activeCy = activeArray.exactCenterY().toDouble()
    } else {
      return CameraIntrinsics.defaultFor(outputSize.width, outputSize.height)
    }
    return CameraIntrinsics(
      fx = activeFx * scaleX,
      fy = activeFy * scaleY,
      cx = (activeCx - cropLeft) * scaleX,
      cy = (activeCy - cropTop) * scaleY,
    )
  }

  private fun reportRecordingStarted() {
    activity.runOnUiThread {
      Toast.makeText(activity, R.string.session_video_recording_started, Toast.LENGTH_SHORT).show()
    }
  }

  private fun reportRecordingSaved(displayName: String) {
    activity.runOnUiThread {
      Toast.makeText(
        activity,
        activity.getString(R.string.session_video_saved, displayName),
        Toast.LENGTH_LONG,
      ).show()
    }
  }

  private fun reportSensorSaved(displayName: String) {
    activity.runOnUiThread {
      Toast.makeText(
        activity,
        activity.getString(R.string.session_sensor_saved, displayName),
        Toast.LENGTH_LONG,
      ).show()
    }
  }

  private fun reportRecordingError(message: String) {
    Log.e(TAG, "Session video recording failed: $message")
    activity.runOnUiThread {
      Toast.makeText(
        activity,
        activity.getString(R.string.session_video_recording_failed, message),
        Toast.LENGTH_LONG,
      ).show()
    }
  }

  override fun onSurfaceTextureAvailable(surface: SurfaceTexture, width: Int, height: Int) {
    start()
  }

  override fun onSurfaceTextureSizeChanged(surface: SurfaceTexture, width: Int, height: Int) {
    updateTransform(width, height)
  }

  override fun onSurfaceTextureDestroyed(surface: SurfaceTexture): Boolean {
    closeCamera()
    return true
  }

  override fun onSurfaceTextureUpdated(surface: SurfaceTexture) = Unit
}

data class CameraIntrinsics(
  val fx: Double,
  val fy: Double,
  val cx: Double,
  val cy: Double,
) {
  companion object {
    fun defaultFor(width: Int, height: Int): CameraIntrinsics {
      val focal = max(width, height) * 0.9
      return CameraIntrinsics(focal, focal, width * 0.5, height * 0.5)
    }
  }
}

/** Maps center-cropped, rotated camera image coordinates to the visible preview and back. */
class CameraCoordinateMapper(
  private val imageWidth: Int,
  private val imageHeight: Int,
  private val viewWidth: Int,
  private val viewHeight: Int,
  val rotationDegrees: Int,
  val intrinsics: CameraIntrinsics,
) {
  private val rotatedWidth = if (rotationDegrees == 90 || rotationDegrees == 270) imageHeight else imageWidth
  private val rotatedHeight = if (rotationDegrees == 90 || rotationDegrees == 270) imageWidth else imageHeight
  private val scale = max(viewWidth.toFloat() / rotatedWidth, viewHeight.toFloat() / rotatedHeight)
  private val offsetX = (viewWidth - rotatedWidth * scale) * 0.5f
  private val offsetY = (viewHeight - rotatedHeight * scale) * 0.5f

  fun imageToView(x: Float, y: Float): FloatArray {
    val rotated =
      when (rotationDegrees) {
        90 -> floatArrayOf(imageHeight - y, x)
        180 -> floatArrayOf(imageWidth - x, imageHeight - y)
        270 -> floatArrayOf(y, imageWidth - x)
        else -> floatArrayOf(x, y)
      }
    return floatArrayOf(rotated[0] * scale + offsetX, rotated[1] * scale + offsetY)
  }

  fun viewToImage(x: Float, y: Float): FloatArray {
    val rotatedX = (x - offsetX) / scale
    val rotatedY = (y - offsetY) / scale
    val image =
      when (rotationDegrees) {
        90 -> floatArrayOf(rotatedY, imageHeight - rotatedX)
        180 -> floatArrayOf(imageWidth - rotatedX, imageHeight - rotatedY)
        270 -> floatArrayOf(imageWidth - rotatedY, rotatedX)
        else -> floatArrayOf(rotatedX, rotatedY)
      }
    image[0] = image[0].coerceIn(0f, imageWidth - 1f)
    image[1] = image[1].coerceIn(0f, imageHeight - 1f)
    return image
  }

  fun imageQuadToView(points: List<org.opencv.core.Point>): FloatArray {
    val result = FloatArray(points.size * 2)
    points.forEachIndexed { index, point ->
      val mapped = imageToView(point.x.toFloat(), point.y.toFloat())
      result[index * 2] = mapped[0]
      result[index * 2 + 1] = mapped[1]
    }
    return result
  }

  fun viewQuadToImage(points: FloatArray): FloatArray {
    val result = FloatArray(points.size)
    for (index in points.indices step 2) {
      val mapped = viewToImage(points[index], points[index + 1])
      result[index] = mapped[0]
      result[index + 1] = mapped[1]
    }
    return result
  }
}
