package com.example.inkmarc_capture

import android.content.ContentValues
import android.content.Context
import android.media.MediaCodec
import android.media.MediaCodecInfo
import android.media.MediaCodecList
import android.media.MediaFormat
import android.media.MediaMuxer
import android.net.Uri
import android.opengl.EGL14
import android.opengl.EGLConfig
import android.opengl.EGLContext
import android.opengl.EGLDisplay
import android.opengl.EGLExt
import android.opengl.EGLSurface
import android.opengl.GLES20
import android.os.Build
import android.os.Environment
import android.os.ParcelFileDescriptor
import android.provider.MediaStore
import android.util.Log
import android.view.Surface
import java.io.File
import java.io.IOException
import java.util.Locale

private const val MIME_TYPE_AVC = "video/avc"
private const val TAG_CAPS = "AvcCaps"

/**
 * Records the current GL scene into an MP4 file using MediaCodec + MediaMuxer.
 *
 * Usage (from a GL thread with a current EGL context, e.g. inside onDrawFrame):
 *
 *   val recorder = GlVideoRecorder(context, width, height)
 *   recorder.start("exercise_name")
 *
 *   // each frame:
 *   recorder.drawFrame { drawScene() }
 *
 *   // when done:
 *   recorder.stop()
 *
 * All methods MUST be called on the GL thread that owns the current EGL context.
 *
 * @param context The application context.
 * @param width The width of the video frame.
 * @param height The height of the video frame.
 * @param bitRate The desired video bit rate in bits per second.
 * @param frameRate The desired video frame rate.
 * @param iFrameIntervalSec The interval between I-frames (keyframes) in seconds.
 */
class GlVideoRecorder(
    private val context: Context,
    private val width: Int,
    private val height: Int,
    private val bitRate: Int = 8_000_000,
    private val frameRate: Int = 30,
    private val iFrameIntervalSec: Int = 1
) {

    companion object {
        private const val TAG = "GlVideoRecorder"
        private const val MIME_TYPE = "video/avc"
        // Timeout for draining the encoder, in microseconds.
        private const val DRAIN_TIMEOUT_US = 10_000L
        // Using 720p as a widely supported fallback resolution.
        private const val FALLBACK_WIDTH = 1280
        private const val FALLBACK_HEIGHT = 720
    }

    private var encoder: MediaCodec? = null
    private var muxer: MediaMuxer? = null
    private var trackIndex: Int = -1
    private var muxerStarted: Boolean = false

    private var inputSurface: Surface? = null

    // Actual encoder resolution, which may differ from the requested width/height.
    private var encWidth: Int = width
    private var encHeight: Int = height

    // EGL objects for the encoder surface.
    private var eglDisplay: EGLDisplay? = null
    private var eglSurface: EGLSurface? = null
    private var eglContext: EGLContext? = null

    private var outputUri: Uri? = null
    private var outputFileDescriptor: ParcelFileDescriptor? = null

    @Volatile
    private var encoderFailed = false

    /**
     * Returns true if the encoder has entered a failed state.
     * If this is true, all calls to [drawFrame] will be no-ops.
     */
    fun hasFailed(): Boolean = encoderFailed

    /**
     * Prepares the recorder for capturing frames.
     *
     * This sets up the [MediaCodec] encoder, [MediaMuxer], and the EGL surface for recording.
     * It must be called on a GL thread with a current EGL context and surface.
     *
     * @param exerciseName The name of the exercise to include in the filename.
     * @throws IOException if the output file cannot be created.
     * @throws RuntimeException if the EGL setup fails or the encoder cannot be configured.
     */
    fun start(exerciseName: String) {
        if (encoder != null) {
            Log.w(TAG, "start() called but encoder is already set up")
            return
        }

        encoderFailed = false

        val resolver = context.contentResolver
        val ts = System.currentTimeMillis()
        val participantId = ParticipantIdManager.getParticipantId(context)
        val fileName = "participant_${participantId}_${exerciseName}_$ts.mp4"
        val videoCollection = getVideoCollectionUri()
        val videoDetails = buildVideoContentValues(fileName)

        val uri = resolver.insert(videoCollection, videoDetails)
        outputUri = uri

        if (uri == null) {
            throw IOException("Failed to create new MediaStore record.")
        }

        try {
            val pfd = resolver.openFileDescriptor(uri, "w")
                ?: throw IOException("Failed to get file descriptor.")
            outputFileDescriptor = pfd

            // Prefer a software AVC encoder to avoid known hardware driver issues on some devices.
            val codecInfo = selectSoftwareAvcEncoderOrNull()
            val codec = if (codecInfo != null) {
                Log.d(TAG, "Using software AVC encoder: ${codecInfo.name}")
                MediaCodec.createByCodecName(codecInfo.name)
            } else {
                Log.d(TAG, "Falling back to default HW encoder for $MIME_TYPE")
                MediaCodec.createEncoderByType(MIME_TYPE)
            }

            val isPortrait = height > width

            // Choose an encoder size that roughly matches the on-screen aspect ratio.
            // For portrait, swap the fallback so we get 720x1280 instead of 1280x720.
            val (reqW, reqH) = if (isPortrait) {
                FALLBACK_HEIGHT to FALLBACK_WIDTH // Portrait: 720 x 1280
            } else {
                FALLBACK_WIDTH to FALLBACK_HEIGHT // Landscape: 1280 x 720
            }

            val (chosenW, chosenH) = pickEncoderSize(codec, reqW, reqH)
            encWidth = chosenW
            encHeight = chosenH

            Log.d(
                TAG,
                "Starting recording to $uri " +
                        "(requested=${width}x$height, isPortrait=$isPortrait, encoder=${encWidth}x$encHeight)"
            )

            // Configure MediaCodec with the chosen size.
            val format = MediaFormat.createVideoFormat(MIME_TYPE, encWidth, encHeight).apply {
                setInteger(
                    MediaFormat.KEY_COLOR_FORMAT,
                    MediaCodecInfo.CodecCapabilities.COLOR_FormatSurface
                )
                setInteger(MediaFormat.KEY_BIT_RATE, bitRate)
                setInteger(MediaFormat.KEY_FRAME_RATE, frameRate)
                setInteger(MediaFormat.KEY_I_FRAME_INTERVAL, iFrameIntervalSec)
            }

            codec.configure(format, null, null, MediaCodec.CONFIGURE_FLAG_ENCODE)
            val surface = codec.createInputSurface()
            codec.start()

            encoder = codec
            inputSurface = surface

            val mux = MediaMuxer(pfd.fileDescriptor, MediaMuxer.OutputFormat.MUXER_OUTPUT_MPEG_4)
            muxer = mux
            trackIndex = -1
            muxerStarted = false

            setupEglForEncoderSurface(surface)
        } catch (e: Exception) {
            Log.e(TAG, "Failed to start recorder, releasing resources", e)
            encoderFailed = true
            releaseResources()
            resolver.delete(uri, null, null)
            outputUri = null
            throw e
        }
    }

    /**
     * Draws a single frame to the video.
     *
     * This method will make the encoder's EGL context current, draw the scene,
     * swap buffers, and then restore the previous EGL context.
     *
     * @param drawScene A lambda that renders the GL scene. This will be executed on the
     *   encoder's EGL context.
     */
    fun drawFrame(drawScene: () -> Unit) {
        if (encoder == null || encoderFailed) {
            return
        }

        val display = eglDisplay ?: return
        val encSurface = eglSurface ?: return
        val ctx = eglContext ?: return
        val codec = encoder ?: return

        // Save the current EGL context so we can restore it later.
        val prevDisplay = EGL14.eglGetCurrentDisplay()
        val prevDraw = EGL14.eglGetCurrentSurface(EGL14.EGL_DRAW)
        val prevRead = EGL14.eglGetCurrentSurface(EGL14.EGL_READ)
        val prevCtx = EGL14.eglGetCurrentContext()

        try {
            // Switch to the encoder's EGL context.
            if (!EGL14.eglMakeCurrent(display, encSurface, encSurface, ctx)) {
                throw RuntimeException(
                    "eglMakeCurrent(encoderSurface) failed: 0x${Integer.toHexString(EGL14.eglGetError())}"
                )
            }

            GLES20.glViewport(0, 0, encWidth, encHeight)
            GLES20.glClear(GLES20.GL_COLOR_BUFFER_BIT)

            drawScene()

            EGLExt.eglPresentationTimeANDROID(display, encSurface, System.nanoTime())
            if (!EGL14.eglSwapBuffers(display, encSurface)) {
                throw RuntimeException(
                    "eglSwapBuffers(encoderSurface) failed: 0x${Integer.toHexString(EGL14.eglGetError())}"
                )
            }

            drainEncoder(codec, endOfStream = false)

        } finally {
            // Restore the previous EGL context.
            if (EGL14.eglGetCurrentContext() != prevCtx) {
                 if (!EGL14.eglMakeCurrent(prevDisplay, prevDraw, prevRead, prevCtx)) {
                    Log.e(TAG, "eglMakeCurrent(prev) failed after drawing: 0x${Integer.toHexString(EGL14.eglGetError())}")
                }
            }
        }
    }

    /**
     * Stops the recording, finalizes the video file, and releases all resources.
     *
     * After this method is called, the recorder can be started again to record a new file.
     */
    fun stop() {
        Log.d(TAG, "Stopping recording (failed=$encoderFailed)")
        if (!encoderFailed) {
             try {
                encoder?.let { drainEncoder(it, endOfStream = true) }
            } catch (e: Exception) {
                Log.e(TAG, "Error during final drain of encoder.", e)
            }
        }
        releaseResources()

        // After all resources are released, update the MediaStore to mark the video as complete.
        outputUri?.let { uri ->
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                val values = ContentValues().apply {
                    put(MediaStore.Video.Media.IS_PENDING, 0)
                }
                context.contentResolver.update(uri, values, null, null)
            }
        }
    }

    /**
     * Extracts all pending encoded data from the encoder and forwards it to the muxer.
     *
     * @param codec The [MediaCodec] encoder instance.
     * @param endOfStream If true, signals the end of the stream to the encoder.
     */
    private fun drainEncoder(codec: MediaCodec, endOfStream: Boolean) {
        val localMuxer = muxer
        if (encoderFailed || localMuxer == null) return

        try {
            if (endOfStream) {
                codec.signalEndOfInputStream()
            }

            val bufferInfo = MediaCodec.BufferInfo()
            var eosRetries = 10

            while (true) {
                val status = codec.dequeueOutputBuffer(bufferInfo, DRAIN_TIMEOUT_US)

                when {
                    status == MediaCodec.INFO_TRY_AGAIN_LATER -> {
                        if (!endOfStream) {
                            break
                        }
                        // For EOS, we retry a few times to make sure we get all data.
                        if (eosRetries > 0) {
                            eosRetries--
                            continue
                        }
                        Log.w(TAG, "drainEncoder: INFO_TRY_AGAIN_LATER timed out at EOS.")
                        break
                    }

                    status == MediaCodec.INFO_OUTPUT_FORMAT_CHANGED -> {
                        if (muxerStarted) {
                            throw RuntimeException("Format changed twice")
                        }
                        val newFormat = codec.outputFormat
                        trackIndex = localMuxer.addTrack(newFormat)
                        localMuxer.start()
                        muxerStarted = true
                        Log.d(TAG, "Muxer started. trackIndex=$trackIndex, format=$newFormat")
                    }

                    status >= 0 -> {
                        val encodedData = codec.getOutputBuffer(status)
                            ?: throw RuntimeException("encoderOutputBuffer $status was null")

                        if ((bufferInfo.flags and MediaCodec.BUFFER_FLAG_CODEC_CONFIG) != 0) {
                            bufferInfo.size = 0
                        }

                        if (bufferInfo.size != 0) {
                            if (!muxerStarted) {
                                throw RuntimeException("Muxer hasn't started")
                            }

                            encodedData.position(bufferInfo.offset)
                            encodedData.limit(bufferInfo.offset + bufferInfo.size)

                            localMuxer.writeSampleData(trackIndex, encodedData, bufferInfo)
                        }

                        codec.releaseOutputBuffer(status, false)

                        if ((bufferInfo.flags and MediaCodec.BUFFER_FLAG_END_OF_STREAM) != 0) {
                            Log.d(TAG, "End of stream reached")
                            break
                        }
                    }
                }
            }
        } catch (e: Exception) {
            Log.e(TAG, "drainEncoder failed, marking as failed and re-throwing", e)
            encoderFailed = true
            throw e
        }
    }

    /**
     * Releases all resources associated with the recorder (encoder, muxer, EGL objects).
     *
     * This makes the recorder instance ready to be started again. It is called internally
     * by [stop] and when an error occurs during [start].
     */
    private fun releaseResources() {
        if (
            encoder == null &&
            muxer == null &&
            inputSurface == null &&
            outputFileDescriptor == null &&
            eglDisplay == null &&
            eglSurface == null &&
            eglContext == null
        ) {
            return
        }
        Log.d(TAG, "Releasing all recording resources.")

        // --- Tear down MediaCodec / MediaMuxer / input surface ---

        try {
            encoder?.stop()
        } catch (e: Exception) {
            Log.w(TAG, "Error stopping encoder", e)
        }
        try {
            encoder?.release()
        } catch (e: Exception) {
            Log.w(TAG, "Error releasing encoder", e)
        }
        encoder = null

        try {
            if (muxerStarted) {
                muxer?.stop()
            }
        } catch (e: Exception) {
            Log.w(TAG, "Error stopping muxer", e)
        }
        try {
            muxer?.release()
        } catch (e: Exception) {
            Log.w(TAG, "Error releasing muxer", e)
        }
        muxer = null
        muxerStarted = false
        trackIndex = -1

        try {
            inputSurface?.release()
        } catch (e: Exception) {
            Log.w(TAG, "Error releasing input surface", e)
        }
        inputSurface = null

        try {
            outputFileDescriptor?.close()
        } catch (e: Exception) {
            Log.w(TAG, "Error closing output file descriptor", e)
        }
        outputFileDescriptor = null

        // --- Tear down encoder EGL objects only, without touching preview context ---

        val display = eglDisplay
        val surface = eglSurface
        val ctx = eglContext

        if (display != null && surface != null && surface != EGL14.EGL_NO_SURFACE) {
            EGL14.eglDestroySurface(display, surface)
        }

        if (display != null && ctx != null && ctx != EGL14.EGL_NO_CONTEXT) {
            // Only unbind if THIS encoder context is current; otherwise leave
            // the preview (GLSurfaceView) context alone.
            if (EGL14.eglGetCurrentContext() == ctx) {
                EGL14.eglMakeCurrent(
                    display,
                    EGL14.EGL_NO_SURFACE,
                    EGL14.EGL_NO_SURFACE,
                    EGL14.EGL_NO_CONTEXT
                )
            }
            EGL14.eglDestroyContext(display, ctx)
        }

        eglDisplay = null
        eglSurface = null
        eglContext = null

        Log.d(TAG, "Recording resources released, file=$outputUri")
    }

    /**
     * Scans the list of available codecs and returns the first software-based AVC encoder,
     * or null if none are found. Prefers software encoders to avoid hardware-specific issues.
     */
    private fun selectSoftwareAvcEncoderOrNull(): MediaCodecInfo? {
        val list = MediaCodecList(MediaCodecList.ALL_CODECS)
        for (info in list.codecInfos) {
            if (!info.isEncoder) continue
            if (!info.supportedTypes.contains(MIME_TYPE_AVC)) continue
            if (isSoftwareEncoder(info)) {
                Log.d(TAG_CAPS, "Selected SW encoder: ${info.name}")
                return info
            }
        }
        Log.w(TAG_CAPS, "No software AVC encoder found; falling back to default HW encoder.")
        return null
    }

    private fun isSoftwareEncoder(info: MediaCodecInfo): Boolean {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            return !info.isHardwareAccelerated
        }

        val codecName = info.name.lowercase(Locale.US)
        return codecName.startsWith("omx.google.") ||
            codecName.startsWith("c2.android.") ||
            codecName.contains(".sw.") ||
            codecName.contains("software")
    }

    /**
     * Selects a video size that is supported by the encoder and close to the requested dimensions.
     *
     * This method queries the codec's capabilities and attempts to find a supported width and
     * height that respects the codec's alignment requirements.
     *
     * @param codec The [MediaCodec] instance.
     * @param reqWidth The desired width.
     * @param reqHeight The desired height.
     * @return A [Pair] containing the chosen width and height.
     */
    private fun pickEncoderSize(codec: MediaCodec, reqWidth: Int, reqHeight: Int): Pair<Int, Int> {
        return try {
            val caps = codec.codecInfo
                .getCapabilitiesForType(MIME_TYPE)
                .videoCapabilities

            val widthRange = caps?.supportedWidths
            val heightRange = caps?.supportedHeights
            val widthAlign = caps?.widthAlignment
            val heightAlign = caps?.heightAlignment

            if (widthRange == null || heightRange == null || widthAlign == null || heightAlign == null) {
                throw RuntimeException("Encoder capabilities missing")
            }

            var w = reqWidth.coerceIn(widthRange.lower, widthRange.upper)
            var h = reqHeight.coerceIn(heightRange.lower, heightRange.upper)

            // Adjust to the required alignment.
            w -= (w % widthAlign)
            h -= (h % heightAlign)

            if (w <= 0 || h <= 0) {
                Log.w(TAG, "pickEncoderSize produced non-positive size ($w x $h), falling back.")
                Pair(FALLBACK_WIDTH, FALLBACK_HEIGHT) // A final, hardcoded safe value.
            } else {
                Log.d(
                    TAG,
                    "Encoder size chosen: $w x $h (align $widthAlign x $heightAlign, " +
                            "supportedW=$widthRange, supportedH=$heightRange)"
                )
                Pair(w, h)
            }
        } catch (e: Exception) {
            Log.w(TAG, "pickEncoderSize failed, falling back to ${FALLBACK_WIDTH}x${FALLBACK_HEIGHT}", e)
            Pair(FALLBACK_WIDTH, FALLBACK_HEIGHT)
        }
    }

    /**
     * Sets up an EGL window surface for the encoder, using the current EGL context.
     */
    private fun setupEglForEncoderSurface(surface: Surface) {
        val curDisplay = EGL14.eglGetCurrentDisplay()
        val curContext = EGL14.eglGetCurrentContext()
        val curDrawSurface = EGL14.eglGetCurrentSurface(EGL14.EGL_DRAW)

        if (curDisplay == EGL14.EGL_NO_DISPLAY ||
            curContext == EGL14.EGL_NO_CONTEXT ||
            curDrawSurface == EGL14.EGL_NO_SURFACE
        ) {
            throw RuntimeException(
                "GlVideoRecorder.start() must be called on a GL thread with a current EGL context"
            )
        }

        // Use the same EGLConfig as the preview surface.
        val config = getConfigFromSurface(curDisplay, curDrawSurface)

        // Create a window surface for the encoder input surface.
        val surfaceAttribs = intArrayOf(EGL14.EGL_NONE)
        val encEglSurface = EGL14.eglCreateWindowSurface(
            curDisplay,
            config,
            surface,
            surfaceAttribs,
            0
        )

        if (encEglSurface == null || encEglSurface == EGL14.EGL_NO_SURFACE) {
            throw RuntimeException("Unable to create EGLSurface for encoder")
        }

        // Create a NEW EGL context that shares with the current one.
        // This allows sharing textures, but avoids issues with making a context current
        // on two different surfaces at once.
        val version = IntArray(1)
        var clientVersion = 2
        if (EGL14.eglQueryContext(
                curDisplay,
                curContext,
                EGL14.EGL_CONTEXT_CLIENT_VERSION,
                version,
                0
            )
        ) {
            if (version[0] != 0) {
                clientVersion = version[0]
            }
        }

        val ctxAttribs = intArrayOf(
            EGL14.EGL_CONTEXT_CLIENT_VERSION,
            clientVersion,
            EGL14.EGL_NONE
        )

        val encEglContext = EGL14.eglCreateContext(
            curDisplay,
            config,
            curContext,   // Share with the preview context.
            ctxAttribs,
            0
        )

        if (encEglContext == null || encEglContext == EGL14.EGL_NO_CONTEXT) {
            // Clean up the surface we just created before bailing.
            EGL14.eglDestroySurface(curDisplay, encEglSurface)
            throw RuntimeException("Unable to create shared EGLContext for encoder")
        }

        eglDisplay = curDisplay
        eglSurface = encEglSurface
        eglContext = encEglContext

        Log.d(
            TAG,
            "EGL encoder surface created with shared context " +
                    "(clientVersion=$clientVersion)."
        )
    }

    /**
     * Returns the EGLConfig of the given surface.
     */
    private fun getConfigFromSurface(display: EGLDisplay, surface: EGLSurface): EGLConfig {
        val value = IntArray(1)
        val ok = EGL14.eglQuerySurface(
            display,
            surface,
            EGL14.EGL_CONFIG_ID,
            value,
            0
        )
        if (!ok) {
            throw RuntimeException("eglQuerySurface(EGL_CONFIG_ID) failed")
        }
        val configId = value[0]
        val attrs = intArrayOf(
            EGL14.EGL_CONFIG_ID, configId,
            EGL14.EGL_NONE
        )
        val configs = arrayOfNulls<EGLConfig>(1)
        val numConfigs = IntArray(1)
        val chose = EGL14.eglChooseConfig(
            display,
            attrs,
            0,
            configs,
            0,
            configs.size,
            numConfigs,
            0
        )
        if (!chose || numConfigs[0] <= 0 || configs[0] == null) {
            throw RuntimeException("Unable to find EGLConfig for encoder surface")
        }
        return configs[0]!!
    }

    private fun getVideoCollectionUri(): Uri {
        return if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            MediaStore.Video.Media.getContentUri(MediaStore.VOLUME_EXTERNAL_PRIMARY)
        } else {
            MediaStore.Video.Media.EXTERNAL_CONTENT_URI
        }
    }

    private fun buildVideoContentValues(fileName: String): ContentValues {
        return ContentValues().apply {
            put(MediaStore.Video.Media.DISPLAY_NAME, fileName)
            put(MediaStore.Video.Media.MIME_TYPE, MIME_TYPE)

            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                put(MediaStore.Video.Media.RELATIVE_PATH, "Movies/InkMARC")
                put(MediaStore.Video.Media.IS_PENDING, 1)
            } else {
                @Suppress("DEPRECATION")
                put(
                    MediaStore.Video.Media.DATA,
                    createLegacyMediaFile(Environment.DIRECTORY_MOVIES, fileName).absolutePath
                )
            }
        }
    }

    private fun createLegacyMediaFile(directoryType: String, fileName: String): File {
        @Suppress("DEPRECATION")
        val baseDirectory = Environment.getExternalStoragePublicDirectory(directoryType)
        val outputDirectory = File(baseDirectory, "InkMARC")
        if (!outputDirectory.exists()) {
            outputDirectory.mkdirs()
        }
        return File(outputDirectory, fileName)
    }
}
