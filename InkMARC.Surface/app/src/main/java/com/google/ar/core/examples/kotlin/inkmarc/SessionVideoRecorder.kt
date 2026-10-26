package com.inkmarc

import android.content.ContentValues
import android.media.MediaRecorder
import android.media.MediaScannerConnection
import android.net.Uri
import android.os.Build
import android.os.Environment
import android.os.ParcelFileDescriptor
import android.os.SystemClock
import android.provider.MediaStore
import android.util.Size
import android.view.Surface
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

/** Records the raw fallback camera stream to a portrait-oriented H.264 MP4. */
class SessionVideoRecorder(private val activity: InkMarcActivity) {
  private data class Output(
    val displayName: String,
    val uri: Uri? = null,
    val descriptor: ParcelFileDescriptor? = null,
    val file: File? = null,
  )

  private var recorder: MediaRecorder? = null
  private var output: Output? = null
  private var started = false

  val sessionId: String
    get() = checkNotNull(output).displayName.removeSuffix(".mp4")

  val videoDisplayName: String
    get() = checkNotNull(output).displayName

  val surface: Surface
    get() = checkNotNull(recorder).surface

  fun prepare(size: Size, orientationHintDegrees: Int) {
    check(recorder == null) { "Recorder is already prepared" }
    val nextOutput = createOutput()
    val nextRecorder =
      if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
        MediaRecorder(activity)
      } else {
        @Suppress("DEPRECATION")
        MediaRecorder()
      }
    try {
      nextRecorder.setVideoSource(MediaRecorder.VideoSource.SURFACE)
      nextRecorder.setOutputFormat(MediaRecorder.OutputFormat.MPEG_4)
      nextRecorder.setVideoEncoder(MediaRecorder.VideoEncoder.H264)
      nextRecorder.setVideoEncodingBitRate(8_000_000)
      nextRecorder.setVideoFrameRate(30)
      nextRecorder.setVideoSize(size.width, size.height)
      nextRecorder.setOrientationHint(orientationHintDegrees)
      nextOutput.descriptor?.let { nextRecorder.setOutputFile(it.fileDescriptor) }
        ?: nextRecorder.setOutputFile(checkNotNull(nextOutput.file).absolutePath)
      nextRecorder.prepare()
      output = nextOutput
      recorder = nextRecorder
    } catch (e: Exception) {
      nextRecorder.release()
      discardOutput(nextOutput)
      throw e
    }
  }

  /** Starts recording and returns the monotonic timestamp used as the shared session origin. */
  fun start(): Long {
    check(!started) { "Recorder is already running" }
    val videoStartElapsedRealtimeNs = SystemClock.elapsedRealtimeNanos()
    checkNotNull(recorder).start()
    started = true
    return videoStartElapsedRealtimeNs
  }

  /** Stops recording and returns the saved display name, or null if no usable video was written. */
  fun stopAndFinalize(): String? {
    val activeRecorder = recorder ?: return null
    val activeOutput = output
    var success = false
    try {
      if (started) {
        activeRecorder.stop()
        success = true
      }
    } catch (_: RuntimeException) {
      success = false
    } finally {
      started = false
      runCatching { activeRecorder.reset() }
      runCatching { activeRecorder.release() }
      recorder = null
      runCatching { activeOutput?.descriptor?.close() }
      output = null
    }

    if (activeOutput == null) return null
    return if (success) {
      publishOutput(activeOutput)
      activeOutput.displayName
    } else {
      discardOutput(activeOutput)
      null
    }
  }

  private fun createOutput(): Output {
    val displayName =
      "InkMARC_${SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date())}.mp4"
    if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
      val values =
        ContentValues().apply {
          put(MediaStore.Video.Media.DISPLAY_NAME, displayName)
          put(MediaStore.Video.Media.MIME_TYPE, "video/mp4")
          put(MediaStore.Video.Media.RELATIVE_PATH, "${Environment.DIRECTORY_MOVIES}/InkMARC")
          put(MediaStore.Video.Media.IS_PENDING, 1)
        }
      val uri =
        activity.contentResolver.insert(MediaStore.Video.Media.EXTERNAL_CONTENT_URI, values)
          ?: error("Unable to create MediaStore video")
      val descriptor =
        activity.contentResolver.openFileDescriptor(uri, "w")
          ?: run {
            activity.contentResolver.delete(uri, null, null)
            error("Unable to open MediaStore video")
          }
      return Output(displayName = displayName, uri = uri, descriptor = descriptor)
    }

    val parent =
      File(
        activity.getExternalFilesDir(Environment.DIRECTORY_MOVIES) ?: activity.filesDir,
        "InkMARC",
      ).apply { mkdirs() }
    return Output(displayName = displayName, file = File(parent, displayName))
  }

  private fun publishOutput(output: Output) {
    if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q && output.uri != null) {
      activity.contentResolver.update(
        output.uri,
        ContentValues().apply { put(MediaStore.Video.Media.IS_PENDING, 0) },
        null,
        null,
      )
    } else {
      output.file?.let { file ->
        MediaScannerConnection.scanFile(activity, arrayOf(file.absolutePath), arrayOf("video/mp4"), null)
      }
    }
  }

  private fun discardOutput(output: Output) {
    runCatching { output.descriptor?.close() }
    runCatching { output.uri?.let { activity.contentResolver.delete(it, null, null) } }
    runCatching { output.file?.delete() }
  }
}
