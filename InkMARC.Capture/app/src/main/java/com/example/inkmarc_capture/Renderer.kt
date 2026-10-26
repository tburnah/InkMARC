package com.example.inkmarc_capture

import android.content.Context
import android.content.res.AssetManager
import android.graphics.Bitmap
import android.graphics.SurfaceTexture
import android.opengl.GLES11Ext
import android.opengl.GLES20
import android.opengl.GLSurfaceView
import android.os.Build
import android.util.Log
import androidx.annotation.RequiresApi
import androidx.core.graphics.createBitmap
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import javax.microedition.khronos.egl.EGLConfig
import javax.microedition.khronos.opengles.GL10
import androidx.core.graphics.set
import android.graphics.Canvas
import android.graphics.Color

private const val BYTES_PER_FLOAT = 4

/**
 * Custom GLSurfaceView.Renderer for rendering the camera preview, AprilTag detections,
 * and stylus-based drawing.
 *
 * @param context The application context.
 */
class Renderer(private val context: Context) : GLSurfaceView.Renderer, SurfaceTexture.OnFrameAvailableListener {

    companion object {
        private const val TAG = "Renderer"

        // Physical dimensions (metres)
        private const val SURFACE_WIDTH_M  = 0.224   // 22.4 cm
        private const val SURFACE_HEIGHT_M = 0.148   // 14.8 cm

        private const val TAG_SIZE_M   = 0.03        // 30 mm tag

        private const val TAG_COORD_SCALE = 2.0 / TAG_SIZE_M  // metres -> tag-local [-1,1] units

        // Tag centres in the usable area, as you gave (metres, origin at surface centre)
        private val TAG_CENTER_LOCATIONS: Map<Int, DoubleArray> = mapOf(
            // Bottom row (tags 8–11): y = +0.054
            8  to doubleArrayOf(-0.0920,  0.0540, 0.0),
            9  to doubleArrayOf(-0.0307,  0.0540, 0.0),
            10 to doubleArrayOf( 0.0307,  0.0540, 0.0),
            11 to doubleArrayOf( 0.0920,  0.0540, 0.0),

            // Middle row (tags 4–7): y = 0
            4  to doubleArrayOf(-0.0920,  0.0000, 0.0),
            5  to doubleArrayOf(-0.0307,  0.0000, 0.0),
            6  to doubleArrayOf( 0.0307,  0.0000, 0.0),
            7  to doubleArrayOf( 0.0920,  0.0000, 0.0),

            // Top row (tags 0–3): y = -0.054
            0  to doubleArrayOf(-0.0920, -0.0540, 0.0),
            1  to doubleArrayOf(-0.0307, -0.0540, 0.0),
            2  to doubleArrayOf( 0.0307, -0.0540, 0.0),
            3  to doubleArrayOf( 0.0920, -0.0540, 0.0)
        )

        // Full surface half-extents (these match SURFACE_* / 2, but written explicitly)
        private const val HALF_SURFACE_WIDTH_M  = SURFACE_WIDTH_M / 2.0   // 0.112
        private const val HALF_SURFACE_HEIGHT_M = SURFACE_HEIGHT_M / 2.0  // 0.074

        // Stylus ink properties
        private const val STYLUS_MAX_X = 44800f
        private const val STYLUS_MAX_Y = 29600f
        private const val PRESSURE_MAX = 4096f
        private const val BRUSH_MIN_RADIUS = 2
        private const val BRUSH_MAX_RADIUS = 8
        private const val BRUSH_COLOR = 0xFF000000.toInt()

        // Ink resources
        private const val INK_TEX_W = 2048
        private const val INK_TEX_H = 2048
    }

    // One color per tag (12 total)
    private val TAG_DEBUG_COLORS = arrayOf(
        floatArrayOf(1f, 0f, 0f, 1f),       // red
        floatArrayOf(1f, 1f, 0f, 1f),       // yellow
        floatArrayOf(1f, 0.4f, 0.7f, 1f),   // pink
        floatArrayOf(0f, 1f, 0f, 1f),       // green
        floatArrayOf(0.5f, 0f, 0.5f, 1f),   // purple
        floatArrayOf(1f, 0.65f, 0f, 1f),    // orange
        floatArrayOf(0f, 0f, 1f, 1f),       // blue
        floatArrayOf(0.6f, 0.3f, 0f, 1f),   // brown
        floatArrayOf(0f, 0.5f, 0.5f, 1f),   // teal
        floatArrayOf(1f, 0f, 1f, 1f),       // magenta
        floatArrayOf(0.9f, 0.9f, 0.98f, 1f),// lavender
        floatArrayOf(0.75f, 1f, 0f, 1f)     // lime
    )

    data class TagFrameFlags(
        val visible: Boolean,
        val centered: Boolean
    )

    private fun tagRow(id: Int): Int = id / 4   // 0=top,1=mid,2=bottom
    private fun tagCol(id: Int): Int = id % 4   // 0..3 left→right

    // Geometric weight: middle row & inner columns get more influence
    private fun geometricWeight(id: Int): Double {
        val row = tagRow(id)
        val col = tagCol(id)

        val rowWeight = if (row == 1) 1.0 else 0.8        // middle row best
        val colWeight = if (col == 1 || col == 2) 1.0 else 0.8 // inner columns best

        return rowWeight * colWeight
    }

    // Per-frame visibility / centering
    private fun visibilityWeight(flags: TagFrameFlags): Double {
        if (!flags.visible) return 0.0
        return if (flags.centered) 1.0 else 0.4
    }

    // Static base reliabilities; you can tune per tag, or just use 1.0 for all
    private val BASE_RELIABILITY = DoubleArray(12) { 1.0 }.apply {
        this[1] = 0.3   // tag 1 less trusted
    }

    private fun tagWeightForFrame(
        tagId: Int,
        flags: TagFrameFlags
    ): Double {
        if (tagId !in 0..11) return 0.0
        val vis = visibilityWeight(flags)
        if (vis == 0.0) return 0.0

        val geom = geometricWeight(tagId)
        val base = BASE_RELIABILITY[tagId]

        return base * geom * vis
    }

    // GL Resources
    private lateinit var surfaceTexture: SurfaceTexture
    private var textureId: Int = 0
    private var program: Int = 0
    private var plainProgram: Int = 0
    private var positionHandle: Int = 0
    private var texCoordHandle: Int = 0
    private var textureHandle: Int = 0
    private var texMatrixHandle: Int = 0

    private val vertices: FloatBuffer
    private val texCoords: FloatBuffer

    // Stream and view dimensions
    private var streamW = 0
    private var streamH = 0
    private var viewW = 0
    private var viewH = 0
    private var effectiveW = 0
    private var effectiveH = 0

    // Frame state
    private var frameAvailable = false
    private var lastQuarterTurn: Boolean? = null

    // Texture coordinate crop region
    private var u0f = 0f; private var u1f = 1f
    private var v0f = 0f; private var v1f = 1f

    // AprilTag Detection
    @Volatile private var latestDetections: Array<AprilTagDetection>? = null
    private val surfaceCornerPx = FloatArray(8) { Float.NaN }

    // Overlay Drawing
    private var overlayProgram = 0
    private var overlayPos = 0
    private var overlayColor = 0
    private val cornerTextColor = floatArrayOf(0f, 0f, 0f, 1f) // black text
    private lateinit var lineBuf: FloatBuffer

    // Tracing Overlay
    private var overlayBitmap: Bitmap? = null
    private var overlayUploadBuffer: ByteBuffer? = null
    private var overlayNeedsUpload = false
    private var overlayTextureId: Int = 0
    private var currentTraceName: String? = null

    // Transformation matrices
    private val stMatrix = FloatArray(16) { if (it % 5 == 0) 1f else 0f }
    private val drawTexMatrix = FloatArray(16)
    private val yFlipMatrix = FloatArray(16)
    private val tmpMatrix = FloatArray(16)
    private val tmpCompose = FloatArray(16) // scratch for matrix multiply

    private var flipX = false
    private var flipY = false

    /** The last reported X coordinate of the stylus, or null if not available. */
    @Volatile var stylusX: Int? = null
        private set
    /** The last reported Y coordinate of the stylus, or null if not available. */
    @Volatile var stylusY: Int? = null
        private set
    /** The last reported pressure of the stylus, or null if not available. */
    @Volatile var stylusPressure: Int? = null
        private set
    /** The last reported tilt of the stylus on the X-axis, or null if not available. */
    @Volatile var stylusTiltX: Int? = null
        private set
    /** The last reported tilt of the stylus on the Y-axis, or null if not available. */
    @Volatile var stylusTiltY: Int? = null
        private set

    /** If true, draw AprilTag quads for debugging. */
    @Volatile
    var showTagQuads: Boolean = false

    @Volatile
    var showTraceOverlay: Boolean = true

    // Ink resources
    private var inkBitmap: Bitmap? = null
    private var inkUploadBuffer: ByteBuffer? = null
    private var inkNeedsUpload = false
    private var inkTextureId: Int = 0
    private var inkProgram: Int = 0
    private var inkPosHandle: Int = 0
    private var inkTexCoordHandle: Int = 0
    private var inkSamplerHandle: Int = 0
    private var inkTexMatrixHandle: Int = 0
    private lateinit var inkVertexBuf: FloatBuffer
    private lateinit var inkTexCoordBuf: FloatBuffer
    private val inkIdentityMatrix: FloatArray = FloatArray(16).apply {
        for (i in indices) this[i] = if (i % 5 == 0) 1f else 0f
    }

    // Previous ink point (in ink-bitmap pixel coordinates) for stroke continuity
    private var prevInkX: Int? = null
    private var prevInkY: Int? = null

    // Homography from screen pixel coords -> ink UV (0..1 x 0..1)
    // Stored row-major: [h00,h01,h02, h10,h11,h12, h20,h21,h22]
    private val screenToInkH = FloatArray(9)
    private var inkScreenToInkHandle: Int = 0
    private var inkViewportSizeHandle: Int = 0
    private var inkFlipYHandle = -1
    private var inkFlipXHandle = -1

    private var texelSizeHandle = -1
    private var keyColorHandle = -1

    // Chroma-key uniforms
    private var chromaViewportSizeHandle: Int = -1
    private var chromaNdcToSurfaceHandle: Int = -1
    private var chromaHasSurfaceHandle: Int = -1

    // --- Cursor tuning ---
    private val cursorColor = floatArrayOf(1f, 0f, 0f, 1f) // red
    private val cursorHalfLenPx = 10f                      // half length of each arm
    private val cursorLineWidth = 3f

    // Scratch for inverse(H)
    private val inkToScreenH = FloatArray(9)

    private var exerciseName = ""

    // 7-segment display properties
    private val segs = arrayOf(
        floatArrayOf(-0.5f,  1.0f,  0.5f,  1.0f), // A (top)
        floatArrayOf( 0.5f,  1.0f,  0.5f,  0.0f), // B (upper-right)
        floatArrayOf( 0.5f,  0.0f,  0.5f, -1.0f), // C (lower-right)
        floatArrayOf(-0.5f,  0.0f,  0.5f,  0.0f), // D (middle)
        floatArrayOf(-0.5f, -1.0f,  0.5f, -1.0f), // E (bottom)
        floatArrayOf(-0.5f,  1.0f, -0.5f,  0.0f), // F (upper-left)
        floatArrayOf(-0.5f,  0.0f, -0.5f, -1.0f)  // G (lower-left)
    )
    private val digitMask = intArrayOf(
        0b1110111, // 0
        0b0000110, // 1
        0b1011011, // 2
        0b0011111, // 3
        0b0101110, // 4
        0b0111101, // 5
        0b1111101, // 6
        0b0000111, // 7
        0b1111111, // 8
        0b0101111, // 9
        0b0001000, // -
    )


    // Video Recording
    private var isRecording = false
    private var frameCount = 0
    private var videoRecorder: GlVideoRecorder? = null
    /** Controls whether the chroma key shader is active. */
    var showChromaKey = true

    init {
        val vertexData = floatArrayOf(-1f, -1f, 1f, -1f, -1f, 1f, 1f, 1f)
        vertices = ByteBuffer.allocateDirect(vertexData.size * BYTES_PER_FLOAT)
            .order(ByteOrder.nativeOrder()).asFloatBuffer()
        vertices.put(vertexData).position(0)

        val texCoordData = floatArrayOf(0f, 1f, 1f, 1f, 0f, 0f, 1f, 0f)
        texCoords = ByteBuffer.allocateDirect(texCoordData.size * BYTES_PER_FLOAT)
            .order(ByteOrder.nativeOrder()).asFloatBuffer()
        texCoords.put(texCoordData).position(0)
    }

    private fun hasValidSurfaceQuad(): Boolean =
        !surfaceCornerPx.any { it.isNaN() || !it.isFinite() }

    /**
     * Called once to set up the view's OpenGL ES environment.
     */
    override fun onSurfaceCreated(gl: GL10?, config: EGLConfig?) {
        textureId = createOesTexture()
        surfaceTexture = SurfaceTexture(textureId)
        surfaceTexture.setOnFrameAvailableListener(this)

        val vsrc = context.assets.readText("shaders/vertex_shader.glsl")
        val fsrc = context.assets.readText("shaders/chroma_key.glsl")

        val fsrcPlain = """
#version 300 es
#extension GL_OES_EGL_image_external_essl3 : require
precision mediump float;

uniform samplerExternalOES u_texture;
in vec2 v_TexCoord;
out vec4 fragColor;

void main() {
  fragColor = texture(u_texture, v_TexCoord);
}
""".trimIndent()

        program = GLES20.glCreateProgram()
        GLES20.glAttachShader(program, compileShader(GLES20.GL_VERTEX_SHADER, vsrc, "v"))
        GLES20.glAttachShader(program, compileShader(GLES20.GL_FRAGMENT_SHADER, fsrc, "f"))
        GLES20.glLinkProgram(program)
        checkLink(program)

        plainProgram = GLES20.glCreateProgram()
        GLES20.glAttachShader(plainProgram, compileShader(GLES20.GL_VERTEX_SHADER, vsrc, "v"))
        GLES20.glAttachShader(plainProgram, compileShader(GLES20.GL_FRAGMENT_SHADER, fsrcPlain, "plain"))
        GLES20.glLinkProgram(plainProgram)
        checkLink(plainProgram)

        positionHandle = GLES20.glGetAttribLocation(program, "a_Position")
        texCoordHandle = GLES20.glGetAttribLocation(program, "a_TexCoord")
        textureHandle = GLES20.glGetUniformLocation(program, "u_texture")
        texMatrixHandle = GLES20.glGetUniformLocation(program, "uTexMatrix")
        texelSizeHandle = GLES20.glGetUniformLocation(program, "u_texelSize")
        keyColorHandle = GLES20.glGetUniformLocation(program, "u_keyColor")

        // surface limiting uniforms on the chroma program
        chromaViewportSizeHandle = GLES20.glGetUniformLocation(program, "uViewportSize")
        chromaNdcToSurfaceHandle = GLES20.glGetUniformLocation(program, "uNdcToSurface")
        chromaHasSurfaceHandle   = GLES20.glGetUniformLocation(program, "uHasSurface")

        val fsrcInk = """
#version 300 es
precision mediump float;

uniform sampler2D u_inkTexture;
uniform mat3 uScreenToInk;     // homography: NDC -> UV
uniform vec2 uViewportSize;    // (viewW, viewH)
uniform float uFlipY;          // 0.0 = no flip, 1.0 = flip V
uniform float uFlipX;          // 0.0 = no flip, 1.0 = flip U

in vec2 v_TexCoord;

out vec4 fragColor;

void main() {
    // fragment pixel -> NDC
    vec2 ndc = (gl_FragCoord.xy / uViewportSize) * 2.0 - 1.0;

    // NDC -> UV
    vec3 q = uScreenToInk * vec3(ndc, 1.0);
    float w = q.z;
    if (abs(w) < 1e-6) {
        fragColor = vec4(0.0);
        return;
    }

    vec2 uv = q.xy / w;

    // Conditionally flip only when requested
    uv.y = mix(uv.y, 1.0 - uv.y, uFlipY);
    uv.x = mix(uv.x, 1.0 - uv.x, uFlipX);

    if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) {
        fragColor = vec4(0.0);
        return;
    }

    fragColor = texture(u_inkTexture, uv);
}
""".trimIndent()

        inkProgram = GLES20.glCreateProgram()
        GLES20.glAttachShader(inkProgram, compileShader(GLES20.GL_VERTEX_SHADER, vsrc, "ink_v"))
        GLES20.glAttachShader(inkProgram, compileShader(GLES20.GL_FRAGMENT_SHADER, fsrcInk, "ink_f"))
        GLES20.glLinkProgram(inkProgram)
        checkLink(inkProgram)
        inkViewportSizeHandle = GLES20.glGetUniformLocation(inkProgram, "uViewportSize")

        inkPosHandle = GLES20.glGetAttribLocation(inkProgram, "a_Position")
        inkTexCoordHandle = GLES20.glGetAttribLocation(inkProgram, "a_TexCoord")
        inkSamplerHandle = GLES20.glGetUniformLocation(inkProgram, "u_inkTexture")
        inkTexMatrixHandle = GLES20.glGetUniformLocation(inkProgram, "uTexMatrix") // unused, OK
        inkScreenToInkHandle = GLES20.glGetUniformLocation(inkProgram, "uScreenToInk")
        inkViewportSizeHandle = GLES20.glGetUniformLocation(inkProgram, "uViewportSize")
        inkFlipXHandle = GLES20.glGetUniformLocation(inkProgram, "uFlipX")
        inkFlipYHandle = GLES20.glGetUniformLocation(inkProgram, "uFlipY")

        android.opengl.Matrix.setIdentityM(yFlipMatrix, 0)
        yFlipMatrix[5] = -1f
        yFlipMatrix[13] = 1f

        ensureOverlayProgram()

        //ensureTraceOverlay("Q0.png")

        GLES20.glDisable(GLES20.GL_DEPTH_TEST)
        GLES20.glClearColor(0f, 0f, 0f, 1f)
    }

    /**
     * Called if the geometry of the view changes, such as when the device's screen orientation changes.
     */
    override fun onSurfaceChanged(gl: GL10?, width: Int, height: Int) {
        GLES20.glViewport(0, 0, width, height)
        viewW = width
        viewH = height
        recomputeGeometry()
    }

    /**
     * For each detected tag, determine whether it is visible in the rendered view
     * and whether it is fully inside (not cut by the screen edges).
     *
     * Visibility is based on the *rendered* output, i.e. after rotation, flipping,
     * and aspect-ratio cropping, by using cameraPxToNdc().
     */
    private fun computeFrameFlagsForDetections(
        dets: Array<AprilTagDetection>
    ): Map<Int, TagFrameFlags> {

        val result = mutableMapOf<Int, TagFrameFlags>()

        // Small margin so "centered" tags aren't considered cut off if they are
        // numerically very close to the edge.
        val edgeMargin = 0.02f

        for (det in dets) {
            // Only care about tags we know the layout for
            if (!TAG_CENTER_LOCATIONS.containsKey(det.id)) continue

            val c = det.corners
            if (c.size < 8) continue  // safety

            var minX = Float.POSITIVE_INFINITY
            var maxX = Float.NEGATIVE_INFINITY
            var minY = Float.POSITIVE_INFINITY
            var maxY = Float.NEGATIVE_INFINITY

            // Convert each tag corner from camera pixels -> NDC
            for (i in 0 until 4) {
                val ix = c[2 * i]
                val iy = c[2 * i + 1]
                val (nx, ny) = cameraPxToNdc(ix, iy)

                if (nx < minX) minX = nx
                if (nx > maxX) maxX = nx
                if (ny < minY) minY = ny
                if (ny > maxY) maxY = ny
            }

            // 1) Visible if the quad's bounding box overlaps the view [-1,1]x[-1,1].
            //    If it's entirely to one side, it's not visible.
            val overlapsHoriz = maxX >= -1f && minX <= 1f
            val overlapsVert  = maxY >= -1f && minY <= 1f
            val visible = overlapsHoriz && overlapsVert

            if (!visible) {
                result[det.id] = TagFrameFlags(visible = false, centered = false)
                continue
            }

            // 2) Centered if the *entire* quad is inside the view,
            //    with a small margin from the edges to avoid "almost cut".
            val insideHoriz = (minX >= -1f + edgeMargin) && (maxX <= 1f - edgeMargin)
            val insideVert  = (minY >= -1f + edgeMargin) && (maxY <= 1f - edgeMargin)
            val centered = insideHoriz && insideVert

            result[det.id] = TagFrameFlags(visible = true, centered = centered)
        }

        return result
    }

    /**
     * Called for each redraw of the view.
     */
    @RequiresApi(Build.VERSION_CODES.Q)
    override fun onDrawFrame(gl: GL10?) {
        synchronized(this) {
            if (frameAvailable) {
                surfaceTexture.updateTexImage()
                surfaceTexture.getTransformMatrix(stMatrix)

                val quarterTurn =
                    (kotlin.math.abs(stMatrix[0]) < 0.5f && kotlin.math.abs(stMatrix[5]) < 0.5f)
                if (lastQuarterTurn != quarterTurn) {
                    lastQuarterTurn = quarterTurn
                    if (streamW > 0) {
                        effectiveW = if (quarterTurn) streamH else streamW
                        effectiveH = if (quarterTurn) streamW else streamH

                        updateTexelSizeUniform(effectiveW, effectiveH)

                        recomputeGeometry()
                    }
                }
                frameAvailable = false
            }
        }

        android.opengl.Matrix.multiplyMM(tmpMatrix, 0, stMatrix, 0, yFlipMatrix, 0)
        System.arraycopy(tmpMatrix, 0, drawTexMatrix, 0, 16)

        // Use the composed matrix (stMatrix * yFlipMatrix) that the preview uses
        System.arraycopy(drawTexMatrix, 0, tmpCompose, 0, 16)
        updateFlipFlags(tmpCompose)

        // manage video recorder lifecycle based on isRecording flag ---
        if (isRecording && videoRecorder == null && viewW > 0 && viewH > 0) {
            // Ensure encoder gets even dimensions (many HW codecs dislike odd sizes)
            val recW = if (viewW % 2 == 0) viewW else viewW - 1
            val recH = if (viewH % 2 == 0) viewH else viewH - 1

            videoRecorder = GlVideoRecorder(context, recW, recH).also {
                it.start(exerciseName)
                Log.d(TAG, "Video recorder started at ${recW}x$recH (view was ${viewW}x$viewH)")
            }
        } else if (!isRecording && videoRecorder != null) {
            videoRecorder?.stop()
            //Log.d(TAG, "Video recorder stopped")
            videoRecorder = null
        }

        // Snapshot detections for this frame so both draws (screen + encoder)
        // use the same set.
        val detsSnapshot = latestDetections

        // Define the scene once, and reuse it for both screen and encoder.
        val baseScenePreview: () -> Unit = {
            // Draw camera background
            drawCameraQuad(false)

            if (detsSnapshot != null && detsSnapshot.isNotEmpty()) {
                val frameFlags = computeFrameFlagsForDetections(detsSnapshot)

                GLES20.glUseProgram(overlayProgram)
                GLES20.glEnable(GLES20.GL_BLEND)
                GLES20.glBlendFunc(GLES20.GL_SRC_ALPHA, GLES20.GL_ONE_MINUS_SRC_ALPHA)
                GLES20.glLineWidth(3f)
                GLES20.glEnableVertexAttribArray(overlayPos)

                if (showTagQuads) {
                    drawTagDetectionQuads(detsSnapshot)
                }

                drawSurfaceQuadUsingSingleTag(detsSnapshot, frameFlags)
                //(detsSnapshot)
                //drawSurfaceQuadPerTag(detsSnapshot)
                drawCornerDebugStrip()

                GLES20.glDisableVertexAttribArray(overlayPos)
                GLES20.glDisable(GLES20.GL_BLEND)
            }
        }

        val baseSceneRecording: () -> Unit = {
            // Draw camera background
            drawCameraQuad(true)

            if (detsSnapshot != null && detsSnapshot.isNotEmpty()) {
                val frameFlags = computeFrameFlagsForDetections(detsSnapshot)

                GLES20.glUseProgram(overlayProgram)
                GLES20.glEnable(GLES20.GL_BLEND)
                GLES20.glBlendFunc(GLES20.GL_SRC_ALPHA, GLES20.GL_ONE_MINUS_SRC_ALPHA)
                GLES20.glLineWidth(3f)
                GLES20.glEnableVertexAttribArray(overlayPos)

                if (showTagQuads) {
                    drawTagDetectionQuads(detsSnapshot)
                }

                //drawSurfaceQuadUsingSingleTag(detsSnapshot, frameFlags)
                //(detsSnapshot)
                //drawSurfaceQuadPerTag(detsSnapshot)
                drawCornerDebugStrip()

                GLES20.glDisableVertexAttribArray(overlayPos)
                GLES20.glDisable(GLES20.GL_BLEND)
            }
        }

        // Preview scene: base + ink overlay
        val previewScene: () -> Unit = {
            // 1) Add any new ink for this frame
            stampInkFromStylus()
            uploadInkIfNeeded()

            // 2) Draw camera + tags
            baseScenePreview()

            // 3) Draw tracing images under ink
            if (showTraceOverlay) {
                drawTracingOverlayQuad()
            }

            // 4) Draw ink on top of the surface quad
            drawInkQuad()

            // 5) Draw Cursor
            drawStylusCursor()
        }

        // 1) Draw to GLSurfaceView (screen) WITH ink
        previewScene()

        // 2) If recording, draw into encoder surface WITHOUT ink
        videoRecorder?.drawFrame(baseSceneRecording)

        videoRecorder?.let { recorder ->
            if (recorder.hasFailed()) {
                Log.e(TAG, "Video recorder failed; stopping recording.")
                recorder.stop()
                videoRecorder = null
                isRecording = false  // internal flag so we don't try to start it again next frame
            }
        }
    }

    /** Draw a small '+' at the stylus location (even when not touching). */
    private fun drawStylusCursor() {
        val sx = stylusX ?: return
        val sy = stylusY ?: return
        if (viewW <= 0 || viewH <= 0) return
        if (surfaceCornerPx.any { it.isNaN() }) return

        // Need the same homography used for ink projection
        if (!computeScreenToInkHomography()) return
        if (!invertMat3ColMajor(screenToInkH, inkToScreenH)) return

        // Same UV computation you use for stamping ink into the bitmap :contentReference[oaicite:2]{index=2}
        var u = (sx.toFloat() / STYLUS_MAX_X).coerceIn(0f, 1f)
        var v = 1f - (sy.toFloat() / STYLUS_MAX_Y).coerceIn(0f, 1f)

        // Must match drawInkQuad() which calls drawProjectedTexture(..., flipY=true, flipX=false) :contentReference[oaicite:3]{index=3}
        // So apply the same flip to the cursor UV before mapping back to NDC.
        // (flipX=false => no change)
        v = 1f - v  // flipY=true

        // UV -> NDC via inverse homography
        val (ndcX, ndcY) = inkUvToNdc(u, v, inkToScreenH) ?: return

        // Cursor size in NDC (pixel-constant on screen)
        val dx = (cursorHalfLenPx / viewW.toFloat()) * 2f
        val dy = (cursorHalfLenPx / viewH.toFloat()) * 2f

        GLES20.glUseProgram(overlayProgram)
        GLES20.glEnable(GLES20.GL_BLEND)
        GLES20.glBlendFunc(GLES20.GL_SRC_ALPHA, GLES20.GL_ONE_MINUS_SRC_ALPHA)
        GLES20.glLineWidth(cursorLineWidth)

        GLES20.glEnableVertexAttribArray(overlayPos)
        GLES20.glUniform4fv(overlayColor, 1, cursorColor, 0)

        // Horizontal line
        lineBuf.clear()
        lineBuf.put(ndcX - dx).put(ndcY)
            .put(ndcX + dx).put(ndcY)
            .position(0)
        GLES20.glVertexAttribPointer(overlayPos, 2, GLES20.GL_FLOAT, false, 0, lineBuf)
        GLES20.glDrawArrays(GLES20.GL_LINES, 0, 2)

        // Vertical line
        lineBuf.clear()
        lineBuf.put(ndcX).put(ndcY - dy)
            .put(ndcX).put(ndcY + dy)
            .position(0)
        GLES20.glVertexAttribPointer(overlayPos, 2, GLES20.GL_FLOAT, false, 0, lineBuf)
        GLES20.glDrawArrays(GLES20.GL_LINES, 0, 2)

        GLES20.glDisableVertexAttribArray(overlayPos)
        GLES20.glDisable(GLES20.GL_BLEND)
    }

    /** UV -> NDC using a 3x3 homography (column-major), returning null if invalid. */
    private fun inkUvToNdc(u: Float, v: Float, h: FloatArray): Pair<Float, Float>? {
        // Column-major mat3 multiply: [x;y;w] = H * [u;v;1]
        val x = h[0] * u + h[3] * v + h[6]
        val y = h[1] * u + h[4] * v + h[7]
        val w = h[2] * u + h[5] * v + h[8]
        if (!w.isFinite() || kotlin.math.abs(w) < 1e-6f) return null
        val nx = x / w
        val ny = y / w
        if (!nx.isFinite() || !ny.isFinite()) return null
        return nx to ny
    }

    /**
     * Invert a 3x3 matrix in column-major order.
     * m and out are length-9 arrays (OpenGL-style column-major).
     */
    private fun invertMat3ColMajor(m: FloatArray, out: FloatArray): Boolean {
        // Unpack (col-major)
        val m00 = m[0]; val m10 = m[1]; val m20 = m[2]
        val m01 = m[3]; val m11 = m[4]; val m21 = m[5]
        val m02 = m[6]; val m12 = m[7]; val m22 = m[8]

        val c00 =  (m11 * m22 - m21 * m12)
        val c01 = -(m10 * m22 - m20 * m12)
        val c02 =  (m10 * m21 - m20 * m11)

        val c10 = -(m01 * m22 - m21 * m02)
        val c11 =  (m00 * m22 - m20 * m02)
        val c12 = -(m00 * m21 - m20 * m01)

        val c20 =  (m01 * m12 - m11 * m02)
        val c21 = -(m00 * m12 - m10 * m02)
        val c22 =  (m00 * m11 - m10 * m01)

        val det = m00 * c00 + m01 * c01 + m02 * c02
        if (!det.isFinite() || kotlin.math.abs(det) < 1e-9f) return false
        val invDet = 1f / det

        // Adjugate transpose of cofactors; write back as column-major
        out[0] = c00 * invDet
        out[1] = c01 * invDet
        out[2] = c02 * invDet

        out[3] = c10 * invDet
        out[4] = c11 * invDet
        out[5] = c12 * invDet

        out[6] = c20 * invDet
        out[7] = c21 * invDet
        out[8] = c22 * invDet

        return true
    }

    private fun drawTagDetectionQuads(dets: Array<AprilTagDetection>) {
        // Green lines for per-tag quads
        //GLES20.glUniform4f(overlayColor, 0f, 1f, 0f, 1f)

        for (det in dets) {
            val c = det.corners

            // Convert to NDC using your existing mapping
            val ndc = Array(4) { 0f to 0f }
            for (i in 0 until 4) {
                val ix = c[2 * i]
                val iy = c[2 * i + 1]
                ndc[i] = cameraPxToNdc(ix, iy)
            }

            // Draw the 4 edges
            for (i in 0 until 4) {
                val j = (i + 1) % 4
                val (x0, y0) = ndc[i]
                val (x1, y1) = ndc[j]

                if (!x0.isFinite() || !y0.isFinite() || !x1.isFinite() || !y1.isFinite()) continue

                val col = TAG_DEBUG_COLORS[det.id]
                GLES20.glUniform4f(overlayColor, col[0], col[1], col[2], col[3])

                lineBuf.clear()
                lineBuf
                    .put(x0).put(y0)
                    .put(x1).put(y1)
                    .position(0)

                GLES20.glVertexAttribPointer(overlayPos, 2, GLES20.GL_FLOAT, false, 0, lineBuf)
                GLES20.glDrawArrays(GLES20.GL_LINES, 0, 2)
            }
        }
    }

    /**
     * Called when a new frame is available from the camera stream.
     */
    override fun onFrameAvailable(surfaceTexture: SurfaceTexture?) {
        synchronized(this) { frameAvailable = true }
    }

    /**
     * Returns the [SurfaceTexture] used for rendering the camera preview.
     * This should be passed to the CameraX Preview use case.
     */
    fun getSurfaceTexture(): SurfaceTexture = surfaceTexture

    /**
     * Sets the resolution of the camera input stream. Must be called on the GL thread.
     * @param w The width of the stream.
     * @param h The height of the stream.
     */
    fun setStreamResolutionOnGlThread(w: Int, h: Int) {
        streamW = w
        streamH = h
        effectiveW = w
        effectiveH = h

        updateTexelSizeUniform(effectiveW, effectiveH)

        recomputeGeometry()
    }

    private fun updateTexelSizeUniform(w: Int, h: Int) {
        if (texelSizeHandle < 0 || w <= 0 || h <= 0) return
        GLES20.glUseProgram(program)
        GLES20.glUniform2f(
            texelSizeHandle,
            1f / w.toFloat(),
            1f / h.toFloat()
        )
    }

    /**
     * Updates the renderer with the latest set of AprilTag detections.
     * This method is thread-safe.
     * @param dets The array of detected AprilTags, or null if none.
     */
    fun updateDetections(dets: Array<AprilTagDetection>?) {
        latestDetections = dets
    }

    /**
     * Updates the current stylus data from a MotionEvent.
     * This method is thread-safe.
     * @param x The X coordinate of the stylus.
     * @param y The Y coordinate of the stylus.
     * @param pressure The pressure of the stylus.
     * @param tiltX The tilt on the X-axis.
     * @param tiltY The tilt on the Y-axis.
     */
    fun updateStylusData(
        x: Int?,
        y: Int?,
        pressure: Int?,
        tiltX: Int?,
        tiltY: Int?
    ) {
        stylusX = x
        stylusY = y
        stylusPressure = pressure
        stylusTiltX = tiltX
        stylusTiltY = tiltY
    }

    /**
     * Starts video recording of the GLSurfaceView.
     */
    fun startRecording(exercise: String) {
        if (isRecording) return
        isRecording = true
        frameCount = 0
        exerciseName = exercise
    }

    /**
     * Stops the video recording.
     */
    fun stopRecording() {
        if (!isRecording) return
        isRecording = false
    }

    /**
     * Releases all GL resources. Should be called when the renderer is no longer needed.
     */
    fun release() {
        // Stop any ongoing recording to avoid leaks
        videoRecorder?.stop()
        videoRecorder = null

        surfaceTexture.release()
        if (program != 0) GLES20.glDeleteProgram(program)
        if (overlayProgram != 0) GLES20.glDeleteProgram(overlayProgram)
        if (plainProgram != 0) GLES20.glDeleteProgram(plainProgram)
        if (inkProgram != 0) GLES20.glDeleteProgram(inkProgram)
        if (textureId != 0) GLES20.glDeleteTextures(1, intArrayOf(textureId), 0)
        if (inkTextureId != 0) GLES20.glDeleteTextures(1, intArrayOf(inkTextureId), 0)
    }

    private fun recomputeGeometry() {
        if (effectiveW == 0 || effectiveH == 0 || viewW == 0 || viewH == 0) return

        val viewAR = viewW.toFloat() / viewH
        val texAR  = effectiveW.toFloat() / effectiveH

        // Full-screen vertices for the camera quad
        val v = floatArrayOf(-1f, -1f, 1f, -1f, -1f, 1f, 1f, 1f)
        vertices.clear(); vertices.put(v).position(0)

        var u0 = 0f; var u1 = 1f
        var v0 = 0f; var v1 = 1f
        if (viewAR > texAR) { // screen wider: crop texture vertically
            val vCrop = (1f - texAR / viewAR) / 2f
            v0 = vCrop
            v1 = 1f - vCrop
        } else { // screen taller: crop texture horizontally
            val uCrop = (1f - viewAR / texAR) / 2f
            u0 = uCrop
            u1 = 1f - uCrop
        }

        val t = floatArrayOf(u0, v1, u1, v1, u0, v0, u1, v0)
        texCoords.clear(); texCoords.put(t).position(0)

        // Save for coordinate mapping
        u0f = u0; u1f = u1
        v0f = v0; v1f = v1
    }

    private fun createOesTexture(): Int {
        val ids = IntArray(1)
        GLES20.glGenTextures(1, ids, 0)
        GLES20.glBindTexture(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, ids[0])
        GLES20.glTexParameteri(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, GLES20.GL_TEXTURE_MIN_FILTER, GLES20.GL_LINEAR)
        GLES20.glTexParameteri(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, GLES20.GL_TEXTURE_MAG_FILTER, GLES20.GL_LINEAR)
        GLES20.glTexParameteri(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, GLES20.GL_TEXTURE_WRAP_S, GLES20.GL_CLAMP_TO_EDGE)
        GLES20.glTexParameteri(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, GLES20.GL_TEXTURE_WRAP_T, GLES20.GL_CLAMP_TO_EDGE)
        return ids[0]
    }

    private fun compileShader(type: Int, src: String, tag: String): Int {
        val sh = GLES20.glCreateShader(type)
        GLES20.glShaderSource(sh, src)
        GLES20.glCompileShader(sh)
        val ok = IntArray(1)
        GLES20.glGetShaderiv(sh, GLES20.GL_COMPILE_STATUS, ok, 0)
        if (ok[0] == 0) throw RuntimeException("$tag shader compile failed: ${GLES20.glGetShaderInfoLog(sh)}")
        return sh
    }

    private fun checkLink(program: Int) {
        val ok = IntArray(1)
        GLES20.glGetProgramiv(program, GLES20.GL_LINK_STATUS, ok, 0)
        if (ok[0] == 0) throw RuntimeException("program link failed: ${GLES20.glGetProgramInfoLog(program)}")
    }

    private fun ensureOverlayProgram() {
        if (overlayProgram != 0) return

        val vsrc = "attribute vec2 aPos; void main() { gl_Position = vec4(aPos, 0.0, 1.0); }"
        val fsrc = "precision mediump float; uniform vec4 uColor; void main() { gl_FragColor = uColor; }"

        val vsh = compileShader(GLES20.GL_VERTEX_SHADER, vsrc, "overlay-vertex")
        val fsh = compileShader(GLES20.GL_FRAGMENT_SHADER, fsrc, "overlay-fragment")

        overlayProgram = GLES20.glCreateProgram()
        GLES20.glAttachShader(overlayProgram, vsh)
        GLES20.glAttachShader(overlayProgram, fsh)
        GLES20.glLinkProgram(overlayProgram)
        checkLink(overlayProgram)

        overlayPos = GLES20.glGetAttribLocation(overlayProgram, "aPos")
        overlayColor = GLES20.glGetUniformLocation(overlayProgram, "uColor")

        lineBuf = ByteBuffer.allocateDirect(4 * 2 * BYTES_PER_FLOAT).order(ByteOrder.nativeOrder()).asFloatBuffer()
    }

    private fun ensureTraceOverlay(name: String) {
        if (overlayTextureId != 0) return
        val bmp = loadTraceBitmapFromAssets(name)
        overlayTextureId = createTextureFromBitmap(bmp)
        bmp.recycle()
    }

    private fun ensureProjectionBuffers() {
        if (!::inkVertexBuf.isInitialized) {
            inkVertexBuf = ByteBuffer
                .allocateDirect(4 * 2 * BYTES_PER_FLOAT)
                .order(ByteOrder.nativeOrder())
                .asFloatBuffer()
        }
        if (!::inkTexCoordBuf.isInitialized) {
            inkTexCoordBuf = ByteBuffer
                .allocateDirect(4 * 2 * BYTES_PER_FLOAT)
                .order(ByteOrder.nativeOrder())
                .asFloatBuffer().apply {
                    put(
                        floatArrayOf(
                            0f, 0f,
                            1f, 0f,
                            1f, 1f,
                            0f, 1f
                        )
                    )
                    position(0)
                }
        }
    }

    private fun surfaceToTagLocal(
        surfaceX: Double,
        surfaceY: Double,
        tagCx: Double,
        tagCy: Double
    ): Pair<Double, Double> {
        val dx = surfaceX - tagCx
        val dy = surfaceY - tagCy
        val u = dx * TAG_COORD_SCALE
        val v = dy * TAG_COORD_SCALE
        return u to v
    }

    private fun applyDetectionHomography(
        det: AprilTagDetection,
        u: Double,
        v: Double
    ): Pair<Double, Double> {
        val detHomography = det.homography

        val x = detHomography[0] * u + detHomography[1] * v + detHomography[2]
        val y = detHomography[3] * u + detHomography[4] * v + detHomography[5]
        val w = detHomography[6] * u + detHomography[7] * v + detHomography[8]

        return (x / w) to (y / w)
    }

    private fun reorderToScreenTLTRBRBL(cornersPx: FloatArray): FloatArray {
        // cornersPx: [x0,y0,x1,y1,x2,y2,x3,y3] in some order
        val pts = Array(4) { i ->
            Pair(cornersPx[i*2], cornersPx[i*2+1])
        }

        val cx = pts.map { it.first }.average().toFloat()
        val cy = pts.map { it.second }.average().toFloat()

        fun quad(p: Pair<Float,Float>): Int {
            val (x,y) = p
            return when {
                x < cx && y < cy -> 0 // TL
                x >= cx && y < cy -> 1 // TR
                x >= cx && y >= cy -> 2 // BR
                else -> 3 // BL
            }
        }

        val out = FloatArray(8)
        for (p in pts) {
            val q = quad(p)
            out[q*2] = p.first
            out[q*2+1] = p.second
        }
        return out
    }

    private fun updateFlipFlags(texM: FloatArray) {
        val x0 = texM[12]
        val x1 = texM[0] + texM[12]
        flipX = x1 < x0

        val y0 = texM[13]
        val y1 = texM[5] + texM[13]
        flipY = y1 < y0
    }

    private fun cameraPxToNdc(px: Double, py: Double): Pair<Float, Float> {
        var rotatedPx = px
        var rotatedPy = py

        if (lastQuarterTurn == true) {
            rotatedPx = py
            rotatedPy = streamW - px
        }

        val u = rotatedPx / effectiveW
        val v = rotatedPy / effectiveH

        val uAdj = if (!flipX) 1.0 - u else u
        val vAdj = if (!flipY) 1.0 - v else v

        val uCrop = (uAdj - u0f) / (u1f - u0f)
        val vCrop = (vAdj - v0f) / (v1f - v0f)

        val ndcX = (uCrop * 2f - 1f).toFloat()
        val ndcY = (1f - vCrop * 2f).toFloat()

        return Pair(ndcX, ndcY)
    }

    /**
     * Solve for a 3x3 homography H that maps 4 source points (x,y) -> 4 dest points (u,v),
     * using the standard 8-equation system with h8 fixed to 1.
     *
     * srcPts, dstPts are length 8: [x0, y0, x1, y1, x2, y2, x3, y3]
     * outH is length 9, row-major.
     */
    private fun solveHomography4Points(
        srcPts: FloatArray,
        dstPts: FloatArray,
        outH: FloatArray
    ): Boolean {
        if (srcPts.size != 8 || dstPts.size != 8 || outH.size != 9) return false

        // We solve for h0..h7 with h8 = 1.
        val a = Array(8) { DoubleArray(9) } // 8x9 augmented matrix

        for (i in 0 until 4) {
            val x = srcPts[2 * i].toDouble()
            val y = srcPts[2 * i + 1].toDouble()
            val u = dstPts[2 * i].toDouble()
            val v = dstPts[2 * i + 1].toDouble()

            // Row for u equation
            val r0 = 2 * i
            a[r0][0] = x
            a[r0][1] = y
            a[r0][2] = 1.0
            a[r0][3] = 0.0
            a[r0][4] = 0.0
            a[r0][5] = 0.0
            a[r0][6] = -u * x
            a[r0][7] = -u * y
            a[r0][8] = u

            // Row for v equation
            val r1 = r0 + 1
            a[r1][0] = 0.0
            a[r1][1] = 0.0
            a[r1][2] = 0.0
            a[r1][3] = x
            a[r1][4] = y
            a[r1][5] = 1.0
            a[r1][6] = -v * x
            a[r1][7] = -v * y
            a[r1][8] = v
        }

        // Gaussian elimination on 8x9
        val n = 8
        var row = 0
        for (col in 0 until 8) {
            // Find pivot
            var pivot = row
            var maxAbs = kotlin.math.abs(a[pivot][col])
            for (r in (row + 1) until n) {
                val v = kotlin.math.abs(a[r][col])
                if (v > maxAbs) {
                    maxAbs = v
                    pivot = r
                }
            }
            if (maxAbs < 1e-9) continue // singular-ish; skip

            // Swap
            if (pivot != row) {
                val tmp = a[row]
                a[row] = a[pivot]
                a[pivot] = tmp
            }

            // Normalize row
            val div = a[row][col]
            for (c in col until 9) {
                a[row][c] /= div
            }

            // Eliminate other rows
            for (r in 0 until n) {
                if (r == row) continue
                val factor = a[r][col]
                if (kotlin.math.abs(factor) < 1e-12) continue
                for (c in col until 9) {
                    a[r][c] -= factor * a[row][c]
                }
            }

            row++
            if (row == n) break
        }

        // Extract h0..h7 from the reduced matrix (last column is the solution)
        val h = DoubleArray(9)
        for (i in 0 until 8) {
            h[i] = a[i][8]
        }
        h[8] = 1.0

        // h[] is row-major:
        // [ h00, h01, h02,
        //   h10, h11, h12,
        //   h20, h21, h22 ]
        //
        // OpenGL expects column-major layout when transpose=false:
        // [ h00, h10, h20,
        //   h01, h11, h21,
        //   h02, h12, h22 ]
        outH[0] = h[0].toFloat() // h00
        outH[1] = h[3].toFloat() // h10
        outH[2] = h[6].toFloat() // h20

        outH[3] = h[1].toFloat() // h01
        outH[4] = h[4].toFloat() // h11
        outH[5] = h[7].toFloat() // h21

        outH[6] = h[2].toFloat() // h02
        outH[7] = h[5].toFloat() // h12
        outH[8] = h[8].toFloat() // h22

        return true
    }

    /**
     * Compute H_screen->ink: screen pixel coords (surfaceCornerPx) -> ink UV [0,1]^2
     * Mapping:
     *   TL (surfaceCornerPx[0], surfaceCornerPx[1]) -> (0,0)
     *   TR (surfaceCornerPx[2], surfaceCornerPx[3]) -> (1,0)
     *   BR (surfaceCornerPx[4], surfaceCornerPx[5]) -> (1,1)
     *   BL (surfaceCornerPx[6], surfaceCornerPx[7]) -> (0,1)
     */
//    private fun computeScreenToInkHomography(): Boolean {
//        if (surfaceCornerPx.any { it.isNaN() || !it.isFinite() }) return false
////        val ordered = reorderToScreenTLTRBRBL(surfaceCornerPx)
//
//        val src = floatArrayOf(
//            surfaceCornerPx[0], surfaceCornerPx[1],
//            surfaceCornerPx[2], surfaceCornerPx[3],
//            surfaceCornerPx[4], surfaceCornerPx[5],
//            surfaceCornerPx[6], surfaceCornerPx[7]
//        )
//
//        val dst = floatArrayOf(
//            0f, 1f,  // TL in screen maps to top of image -> V=1
//            1f, 1f,  // TR
//            1f, 0f,  // BR
//            0f, 0f   // BL
//        )
//
//        return solveHomography4Points(src, dst, screenToInkH)
//    }
    private fun computeScreenToInkHomography(): Boolean {
        if (surfaceCornerPx.any { it.isNaN() || !it.isFinite() }) return false

        fun toNdcX(x: Float) = (x / viewW) * 2f - 1f
        fun toNdcY(y: Float) = (y / viewH) * 2f - 1f

        val srcNdc = floatArrayOf(
            toNdcX(surfaceCornerPx[0]), toNdcY(surfaceCornerPx[1]), // TL
            toNdcX(surfaceCornerPx[2]), toNdcY(surfaceCornerPx[3]), // TR
            toNdcX(surfaceCornerPx[4]), toNdcY(surfaceCornerPx[5]), // BR
            toNdcX(surfaceCornerPx[6]), toNdcY(surfaceCornerPx[7])  // BL
        )

        // Your V flip fix already established:
        val dstUv = floatArrayOf(
            0f, 1f,   // TL
            1f, 1f,   // TR
            1f, 0f,   // BR
            0f, 0f    // BL
        )

        return solveHomography4Points(srcNdc, dstUv, screenToInkH)
    }


    private fun drawCameraQuad(forRecording: Boolean = false) {
        val useChroma = showChromaKey && !forRecording
        GLES20.glUseProgram(if (useChroma) program else plainProgram)

        if (useChroma) {
            // 1) Viewport size for gl_FragCoord → NDC conversion
            GLES20.glUniform2f(
                chromaViewportSizeHandle,
                viewW.toFloat(),
                viewH.toFloat()
            )

            // 2) Try to compute NDC -> surface [0,1]^2 homography.
            //    If it fails or we don't have a quad yet, disable surface limiting.
            var hasSurface = false
            if (hasValidSurfaceQuad() && computeScreenToInkHomography()) {
                hasSurface = true
                GLES20.glUniformMatrix3fv(
                    chromaNdcToSurfaceHandle,
                    1,
                    false,
                    screenToInkH,
                    0
                )
            }
            GLES20.glUniform1i(chromaHasSurfaceHandle, if (hasSurface) 1 else 0)

            // set key color depending on pass
            if (forRecording) {
                // green in recording
                GLES20.glUniform4f(keyColorHandle, 0f, 1f, 0f, 1f)
            } else {
                // white in preview
                GLES20.glUniform4f(keyColorHandle, 1f, 1f, 1f, 1f)
            }
        }

        GLES20.glActiveTexture(GLES20.GL_TEXTURE0)
        GLES20.glBindTexture(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, textureId)

        GLES20.glUniform1i(textureHandle, 0)
        GLES20.glUniformMatrix4fv(texMatrixHandle, 1, false, drawTexMatrix, 0)

        GLES20.glEnableVertexAttribArray(positionHandle)
        GLES20.glVertexAttribPointer(positionHandle, 2, GLES20.GL_FLOAT, false, 0, vertices)
        GLES20.glEnableVertexAttribArray(texCoordHandle)
        GLES20.glVertexAttribPointer(texCoordHandle, 2, GLES20.GL_FLOAT, false, 0, texCoords)

        GLES20.glDrawArrays(GLES20.GL_TRIANGLE_STRIP, 0, 4)

        GLES20.glDisableVertexAttribArray(positionHandle)
        GLES20.glDisableVertexAttribArray(texCoordHandle)
    }

    // Helper struct for weighted samples
    private data class WeightedPt(val x: Double, val y: Double, val w: Double)

    private data class TagQuad(
        val id: Int,
        val weight: Double,
        val cornersImg: Array<Pair<Double, Double>>
    ) {
        override fun equals(other: Any?): Boolean {
            if (this === other) return true
            if (javaClass != other?.javaClass) return false

            other as TagQuad

            if (id != other.id) return false
            if (weight != other.weight) return false
            if (!cornersImg.contentEquals(other.cornersImg)) return false

            return true
        }

        override fun hashCode(): Int {
            var result = id
            result = 31 * result + weight.hashCode()
            result = 31 * result + cornersImg.contentHashCode()
            return result
        }
    }

    /**
     * Filter out per-tag outliers based on their 4 image-space corners.
     * Uses a robust modified z-score on the mean pairwise distance between tag quads.
     */
    private fun filterOutlierTagQuads(
        quads: List<TagQuad>,
        zThreshold: Double = 2.5
    ): List<TagQuad> {
        if (quads.size < 3) {
            // Not enough tags to do a meaningful outlier test
            return quads
        }

        val n = quads.size
        // Flatten each quad into an 8D vector: [x0,y0,x1,y1,x2,y2,x3,y3]
        val vectors = Array(n) { DoubleArray(8) }
        for (i in 0 until n) {
            val c = quads[i].cornersImg
            vectors[i][0] = c[0].first
            vectors[i][1] = c[0].second
            vectors[i][2] = c[1].first
            vectors[i][3] = c[1].second
            vectors[i][4] = c[2].first
            vectors[i][5] = c[2].second
            vectors[i][6] = c[3].first
            vectors[i][7] = c[3].second
        }

        // Pairwise distances and mean distance per tag
        val meanDist = DoubleArray(n)
        for (i in 0 until n) {
            var sum = 0.0
            for (j in 0 until n) {
                if (i == j) continue
                val vi = vectors[i]
                val vj = vectors[j]
                var d2 = 0.0
                for (k in 0 until 8) {
                    val diff = vi[k] - vj[k]
                    d2 += diff * diff
                }
                sum += kotlin.math.sqrt(d2)
            }
            meanDist[i] = sum / (n - 1)
        }

        // Robust scale: median and MAD
        val sorted = meanDist.sorted()
        val median = if (n % 2 == 1) {
            sorted[n / 2]
        } else {
            (sorted[n / 2 - 1] + sorted[n / 2]) * 0.5
        }

        val absDev = meanDist.map { kotlin.math.abs(it - median) }.sorted()
        val mad = if (n % 2 == 1) {
            absDev[n / 2]
        } else {
            (absDev[n / 2 - 1] + absDev[n / 2]) * 0.5
        }

        if (mad <= 1e-6) {
            // Everything is basically the same; nothing stands out
            return quads
        }

        val kept = mutableListOf<TagQuad>()
        for (i in 0 until n) {
            val z = 0.6745 * (meanDist[i] - median) / mad
            if (kotlin.math.abs(z) <= zThreshold) {
                kept += quads[i]
            } else {
                //Log.d("Renderer", "Excluding tag ${quads[i].id} as quad outlier (z=$z, meanDist=${meanDist[i]})")
            }
        }

        // Safety: if we somehow dropped everyone, keep the original list
        return if (kept.isEmpty()) quads else kept
    }

    private fun  drawSurfaceQuadUsingSingleTag(
        dets: Array<AprilTagDetection>,
        frameFlags: Map<Int, TagFrameFlags>
    ) {
        val usableDets = dets.filter { TAG_CENTER_LOCATIONS.containsKey(it.id) }
        if (usableDets.isEmpty()) {
            return
        }

        val halfW = HALF_SURFACE_WIDTH_M
        val halfH = HALF_SURFACE_HEIGHT_M

        val surfaceCorners = arrayOf(
            Pair(-halfW,  halfH), // top-left
            Pair( halfW,  halfH), // top-right
            Pair( halfW, -halfH), // bottom-right
            Pair(-halfW, -halfH)  // bottom-left
        )

        // --- NEW: compute per-tag image-space quads first ---
        val tagQuads = mutableListOf<TagQuad>()

        for (det in usableDets) {
            val layout = TAG_CENTER_LOCATIONS[det.id]!!
            val cx = layout[0]
            val cy = layout[1]

            val flags = frameFlags[det.id] ?: TagFrameFlags(visible = true, centered = true)
            val w = tagWeightForFrame(det.id, flags)
            if (w <= 0.0) {
                continue
            }

            val imgCornersForTag = Array(4) { 0.0 to 0.0 }

            surfaceCorners.forEachIndexed { idx, (sx, sy) ->
                val (u, v) = surfaceToTagLocal(sx, sy, cx, cy)
                val (ix, iy) = applyDetectionHomography(det, u, v)
                imgCornersForTag[idx] = ix to iy
            }

            tagQuads += TagQuad(det.id, w, imgCornersForTag)
        }

        if (tagQuads.isEmpty()) {
            return
        }

        // --- NEW: filter outlier tags at the quad level ---
        val inlierQuads = filterOutlierTagQuads(tagQuads)
        if (inlierQuads.isEmpty()) {
            Log.w(TAG, "All tag quads rejected as outliers – skipping surface draw")
            return
        }

        // --- 1) Collect weighted samples per surface corner from INLIER tags only ---
        val perCornerImagePoints = Array(4) { mutableListOf<WeightedPt>() }

        for (quad in inlierQuads) {
            val w = quad.weight
            val corners = quad.cornersImg
            for (idx in 0 until 4) {
                val (ix, iy) = corners[idx]
                perCornerImagePoints[idx].add(WeightedPt(ix, iy, w))
            }
        }

        val imgCorners = Array(4) { 0.0 to 0.0 }

        // --- 2) Robust weighted average per surface corner (your existing logic) ---
        for (idx in 0 until 4) {
            val samples = perCornerImagePoints[idx]
            if (samples.isEmpty()) {
                Log.w(TAG, "  corner $idx has no samples – aborting surface quad draw")
                return
            }

            if (samples.size == 1) {
                imgCorners[idx] = samples[0].x to samples[0].y
                continue
            }

            // First pass: weighted mean as initial center
            val sumW = samples.sumOf { it.w }
            val meanX = samples.sumOf { it.x * it.w } / sumW
            val meanY = samples.sumOf { it.y * it.w } / sumW

            data class PtDist(val x: Double, val y: Double, val w: Double, val d: Double)

            val withDist = samples.map { s ->
                val dx = s.x - meanX
                val dy = s.y - meanY
                val d = kotlin.math.sqrt(dx * dx + dy * dy)
                PtDist(s.x, s.y, s.w, d)
            }

            val sortedDistances = withDist.map { it.d }.sorted()
            val medianDist = sortedDistances[sortedDistances.size / 2]

            val threshold = kotlin.math.max(15.0, medianDist * 2.0)

            val filtered = if (withDist.size >= 3) {
                val kept = withDist.filter { it.d <= threshold }
                kept.ifEmpty { withDist }
            } else {
                withDist
            }

            val sumWf = filtered.sumOf { it.w }
            val finalX = filtered.sumOf { it.x * it.w } / sumWf
            val finalY = filtered.sumOf { it.y * it.w } / sumWf

            imgCorners[idx] = finalX to finalY
        }

        // --- 3) Convert to NDC and draw as before ---
        val ndcCorners = Array(4) { 0f to 0f }
        for (i in 0 until 4) {
            val (ix, iy) = imgCorners[i]
            val (ndcX, ndcY) = cameraPxToNdc(ix, iy)
            ndcCorners[i] = ndcX to ndcY
        }

//        run {
//            val ndcCorners = Array(4) { 0f to 0f }
//            val pxCorners  = Array(4) { 0f to 0f }
//
//            for (i in 0 until 4) {
//                val (ix, iy) = imgCorners[i]
//                val (ndcX, ndcY) = cameraPxToNdc(ix, iy)
//                ndcCorners[i] = ndcX to ndcY
//
//                val sxScr = ((ndcX + 1f) * 0.5f * viewW)
//                val syScr = ((ndcY + 1f) * 0.5f * viewH)
//                pxCorners[i] = sxScr to syScr
//            }
//
////            val sb = StringBuilder()
////            sb.append("SingleTag FINAL fused surface quad:\n")
////            for (i in 0 until 4) {
////                val (ix, iy) = imgCorners[i]
////                val (sxScr, syScr) = pxCorners[i]
////                sb.append(
////                    "  corner $i: img=(${ix.format2()}, ${iy.format2()}), " +
////                            "screen=(${sxScr.format1()}, ${syScr.format1()})\n"
////                )
////            }
//            //Log.d(TAG, sb.toString())
//        }

        GLES20.glUniform4f(overlayColor, 1f, 1f, 1f, 1f)

        for (i in 0 until 4) {
            val j = (i + 1) % 4
            val (x0, y0) = ndcCorners[i]
            val (x1, y1) = ndcCorners[j]

            if (!x0.isFinite() || !y0.isFinite() || !x1.isFinite() || !y1.isFinite()) {
                Log.w(TAG, "  edge $i has non-finite NDC values – skipping")
                continue
            }

            lineBuf.clear()
            lineBuf.put(x0).put(y0).put(x1).put(y1).position(0)
            GLES20.glVertexAttribPointer(overlayPos, 2, GLES20.GL_FLOAT, false, 0, lineBuf)
            GLES20.glDrawArrays(GLES20.GL_LINES, 0, 2)
        }

        // Save pixel-space corners for later use
        val newCornersPx = FloatArray(8)
        for (i in 0 until 4) {
            val (ndcX, ndcY) = ndcCorners[i]
            val sx = ((ndcX + 1f) * 0.5f * viewW)
            val sy = ((ndcY + 1f) * 0.5f * viewH)

            newCornersPx[i * 2]     = sx
            newCornersPx[i * 2 + 1] = sy
        }

        // instead of updateSurfaceCornersSmooth(newCornersPx)
        System.arraycopy(newCornersPx, 0, surfaceCornerPx, 0, 8)
    }

    private fun drawSurfaceQuadPerTag(dets: Array<AprilTagDetection>) {
        // Only use tags we know about
        Log.d(TAG, "Start Quads ----------")

        val usableDets = dets.filter { TAG_CENTER_LOCATIONS.containsKey(it.id) }
        if (usableDets.isEmpty()) return

        val halfW = HALF_SURFACE_WIDTH_M
        val halfH = HALF_SURFACE_HEIGHT_M

        // Surface corners in surface-local coordinates
        val surfaceCorners = arrayOf(
            Pair(-halfW,  halfH), // top-left
            Pair( halfW,  halfH), // top-right
            Pair( halfW, -halfH), // bottom-right
            Pair(-halfW, -halfH)  // bottom-left
        )

        for (det in usableDets) {
            val layout = TAG_CENTER_LOCATIONS[det.id] ?: continue
            val cx = layout[0]
            val cy = layout[1]

            // --- 1) Project surface corners using only this tag (image pixels) ---
            val imgCorners = Array(4) { 0.0 to 0.0 }

            for (idx in 0 until 4) {
                val (sx, sy) = surfaceCorners[idx]
                val (u, v) = surfaceToTagLocal(sx, sy, cx, cy)
                val (ix, iy) = applyDetectionHomography(det, u, v)
                imgCorners[idx] = ix to iy
            }

            // --- 2) Convert to NDC and also compute screen-pixel coords for logging ---
            val ndcCorners = Array(4) { 0f to 0f }
            val pxCorners  = Array(4) { 0f to 0f }

            for (i in 0 until 4) {
                val (ix, iy) = imgCorners[i]
                val (ndcX, ndcY) = cameraPxToNdc(ix, iy)
                ndcCorners[i] = ndcX to ndcY

                // NDC [-1,1] → screen pixels [0, viewW/viewH]
                val sx = ((ndcX + 1f) * 0.5f * viewW)
                val sy = ((ndcY + 1f) * 0.5f * viewH)
                pxCorners[i] = sx to sy
            }

            // --- 2.5) Log corners for this tag ---
            // imgCorners: original camera pixels (double)
            // pxCorners: final screen pixels after all transforms (float)
            val sb = StringBuilder()
            sb.append("Per-tag surface quad for tag ${det.id}:\n")
            for (i in 0 until 4) {
                val (ix, iy) = imgCorners[i]
                val (sx, sy) = pxCorners[i]
                sb.append("  corner $i: img=(${ix.format2()}, ${iy.format2()}), " +
                        "screen=(${sx.format1()}, ${sy.format1()})\n")
            }
            Log.d(TAG, sb.toString())

            // --- 3) Choose color based on tag id ---
            val c = TAG_DEBUG_COLORS[det.id]
            GLES20.glUniform4f(overlayColor, c[0], c[1], c[2], c[3])

            // --- 4) Draw the quad for this tag ---
            for (i in 0 until 4) {
                val j = (i + 1) % 4
                val (x0, y0) = ndcCorners[i]
                val (x1, y1) = ndcCorners[j]

                if (!x0.isFinite() || !y0.isFinite() || !x1.isFinite() || !y1.isFinite()) {
                    Log.w(TAG, "  per-tag edge $i for tag ${det.id} has non-finite NDC – skipping")
                    continue
                }

                lineBuf.clear()
                lineBuf
                    .put(x0).put(y0)
                    .put(x1).put(y1)
                    .position(0)

                GLES20.glVertexAttribPointer(
                    overlayPos,
                    2,
                    GLES20.GL_FLOAT,
                    false,
                    0,
                    lineBuf
                )
                GLES20.glDrawArrays(GLES20.GL_LINES, 0, 2)
            }
        }
    }

    private fun drawTracingOverlay() {
        if (overlayTextureId == 0) return
        if (surfaceCornerPx.any { it.isNaN() }) return

        if (!computeScreenToInkHomography()) return

        // Fill inkVertexBuf exactly as in drawInkQuad()
        // (you can refactor that buffer fill into a helper)
        //fillSurfaceQuadVertexBuf()

        GLES20.glUseProgram(inkProgram)
        GLES20.glEnable(GLES20.GL_BLEND)
        GLES20.glBlendFunc(GLES20.GL_SRC_ALPHA, GLES20.GL_ONE_MINUS_SRC_ALPHA)

        GLES20.glActiveTexture(GLES20.GL_TEXTURE2)
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, overlayTextureId)
        GLES20.glUniform1i(inkSamplerHandle, 2)

        GLES20.glUniformMatrix3fv(inkScreenToInkHandle, 1, false, screenToInkH, 0)
        GLES20.glUniform2f(inkViewportSizeHandle, viewW.toFloat(), viewH.toFloat())

        GLES20.glEnableVertexAttribArray(inkPosHandle)
        GLES20.glVertexAttribPointer(inkPosHandle, 2, GLES20.GL_FLOAT, false, 0, inkVertexBuf)

        GLES20.glEnableVertexAttribArray(inkTexCoordHandle)
        GLES20.glVertexAttribPointer(inkTexCoordHandle, 2, GLES20.GL_FLOAT, false, 0, inkTexCoordBuf)

        GLES20.glDrawArrays(GLES20.GL_TRIANGLE_FAN, 0, 4)

        GLES20.glDisableVertexAttribArray(inkPosHandle)
        GLES20.glDisableVertexAttribArray(inkTexCoordHandle)
        GLES20.glDisable(GLES20.GL_BLEND)
    }

    private fun Double.format2(): String = String.format("%.2f", this)
    private fun Float.format1(): String = String.format("%.1f", this)

    private fun logOverlayCornersWithNdc(tag: String = "Overlay") {
        val sb = StringBuilder()
        sb.append("$tag corners:\n")
        for (i in 0 until 4) {
            val sx = surfaceCornerPx[i * 2]
            val sy = surfaceCornerPx[i * 2 + 1]

            val ndcX = (sx / viewW) * 2f - 1f
            val ndcY = (sy / viewH) * 2f - 1f

            sb.append(
                "  corner $i: screen=(${sx.format1()}, ${sy.format1()}), " +
                        "ndc=(${ndcX.format1()}, ${ndcY.format1()})\n"
            )
        }
        Log.d(TAG, sb.toString())
    }

    private fun drawCornerDebugStrip() {
        if (surfaceCornerPx.any { it.isNaN() }) return

        val top = 1f
        val bottom = 0.75f
        val left = -1f
        val right = 1f

        lineBuf.clear()
        lineBuf
            .put(left).put(top)
            .put(left).put(bottom)
            .put(right).put(top)
            .put(right).put(bottom)
            .position(0)

        GLES20.glUniform4f(overlayColor, 1f, 1f, 1f, 1f)
        GLES20.glVertexAttribPointer(overlayPos, 2, GLES20.GL_FLOAT, false, 0, lineBuf)
        GLES20.glDrawArrays(GLES20.GL_TRIANGLE_STRIP, 0, 4)

        var baseY = 0.95f
        val lineStep = -0.07f
        val numberScale = 0.025f
        var baseX = -0.93f
        var count = 0
        for (corner in 0 until 4) {
            val xPx = surfaceCornerPx[corner * 2]
            val yPx = surfaceCornerPx[corner * 2 + 1]
            if (!xPx.isFinite() || !yPx.isFinite()) continue

            drawNumber(baseX, baseY, numberScale, xPx.toInt(), cornerTextColor)
            drawNumber(baseX + 0.3f, baseY, numberScale, yPx.toInt(), cornerTextColor)
            baseX += 0.6f
            count += 1
            if (count == 2) {
                baseY += lineStep
                baseX = -0.93f
            }
        }
        baseY = 0.95f
        drawNumber(baseX, baseY, numberScale, stylusX, cornerTextColor)
        drawNumber(baseX + 0.3f, baseY, numberScale, stylusY, cornerTextColor)
        drawNumber(baseX + 0.6f, baseY, numberScale, stylusPressure, cornerTextColor)
        drawNumber(baseX, baseY + lineStep, numberScale, stylusTiltX, cornerTextColor)
        drawNumber(baseX + 0.3f, baseY + lineStep, numberScale, stylusTiltY, cornerTextColor)
        val recW = if (viewW % 2 == 0) viewW else viewW - 1
        val recH = if (viewH % 2 == 0) viewH else viewH - 1
        baseX = -0.93f
        drawNumber(baseX, baseY + (lineStep * 2), numberScale, recW, cornerTextColor)
        drawNumber(baseX + 0.3f, baseY + (lineStep * 2), numberScale, recH, cornerTextColor)
    }

    @Suppress("SameParameterValue")
    private fun drawNumber(ndcX: Float, ndcY: Float, scale: Float, value: Int?, color: FloatArray) {
        GLES20.glUniform4fv(overlayColor, 1, color, 0)

        if (value == null) {
            var x = ndcX + (0 - (2 - 1) / 2f) * (1.4f * scale)
            drawDigit(x, ndcY, scale, 10)
            x = ndcX + (1 - (2 - 1) / 2f) * (1.4f * scale)
            drawDigit(x, ndcY, scale, 10)
            return
        }

        val isNegative = value < 0
        var v = kotlin.math.abs(value)

        val digitList = mutableListOf<Int>()
        if (v == 0) {
            digitList.add(0)
        } else {
            while (v > 0) { digitList.add(v % 10); v /= 10 }
            digitList.reverse()
        }

        if (isNegative) {
            digitList.add(0, 10)
        }

        val digits = digitList.toIntArray()
        val total = digits.size

        for ((i, d) in digits.withIndex()) {
            val x = ndcX + (i - (total - 1) / 2f) * (1.5f * scale)
            drawDigit(x, ndcY, scale, d)
        }
    }

    private fun drawDigit(ndcX: Float, ndcY: Float, scale: Float, digit: Int) {
        val mask = digitMask[digit.coerceIn(0,10)]
        for (s in 0..6) if ((mask and (1 shl s)) != 0) {
            val seg = segs[s]
            lineBuf.clear()
            lineBuf.put(ndcX + seg[0]*scale).put(ndcY + seg[1]*scale)
                .put(ndcX + seg[2]*scale).put(ndcY + seg[3]*scale)
                .position(0)
            GLES20.glVertexAttribPointer(overlayPos, 2, GLES20.GL_FLOAT, false, 0, lineBuf)
            GLES20.glDrawArrays(GLES20.GL_LINES, 0, 2)
        }
    }

    private fun drawInkQuad() {
        drawProjectedTexture(inkTextureId, GLES20.GL_TEXTURE1, true, false)
    }

    private fun drawTracingOverlayQuad() {
        //logOverlayCornersWithNdc("TracingOverlay")
        drawProjectedTexture(overlayTextureId, GLES20.GL_TEXTURE2, true, true)
    }

    private fun drawProjectedTexture(texId: Int, textureUnit: Int, flipY: Boolean, flipX: Boolean) {
        if (texId == 0) return
        if (viewW <= 0 || viewH <= 0) return
        if (surfaceCornerPx.any { it.isNaN() }) return

        ensureProjectionBuffers()

        if (!computeScreenToInkHomography()) return
        //Log.d(TAG, "screenToInkH = ${screenToInkH.joinToString(prefix = "[", postfix = "]") { "%.6f".format(it) }}")

        // Fill inkVertexBuf from surfaceCornerPx (same as drawInkQuad)
        inkVertexBuf.position(0)
        for (i in 0 until 4) {
            val sx = surfaceCornerPx[i * 2]
            val sy = surfaceCornerPx[i * 2 + 1]
            val ndcX = (sx / viewW.toFloat()) * 2f - 1f
            val ndcY = (sy / viewH.toFloat()) * 2f - 1f
            inkVertexBuf.put(ndcX).put(ndcY)
        }
        inkVertexBuf.position(0)

        GLES20.glUseProgram(inkProgram)
        GLES20.glEnable(GLES20.GL_BLEND)
        GLES20.glBlendFunc(GLES20.GL_SRC_ALPHA, GLES20.GL_ONE_MINUS_SRC_ALPHA)

        GLES20.glActiveTexture(textureUnit)
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, texId)
        GLES20.glUniform1i(inkSamplerHandle, textureUnit - GLES20.GL_TEXTURE0)

        GLES20.glUniformMatrix3fv(inkScreenToInkHandle, 1, false, screenToInkH, 0)
        GLES20.glUniform2f(inkViewportSizeHandle, viewW.toFloat(), viewH.toFloat())
        GLES20.glUniform1f(inkFlipXHandle, if (flipX) 1f else 0f)
        GLES20.glUniform1f(inkFlipYHandle, if (flipY) 1f else 0f)

        GLES20.glEnableVertexAttribArray(inkPosHandle)

        GLES20.glVertexAttribPointer(inkPosHandle, 2, GLES20.GL_FLOAT, false, 0, inkVertexBuf)

        GLES20.glEnableVertexAttribArray(inkTexCoordHandle)
        GLES20.glVertexAttribPointer(inkTexCoordHandle, 2, GLES20.GL_FLOAT, false, 0, inkTexCoordBuf)

        GLES20.glDrawArrays(GLES20.GL_TRIANGLE_FAN, 0, 4)

        GLES20.glDisableVertexAttribArray(inkPosHandle)
        GLES20.glDisableVertexAttribArray(inkTexCoordHandle)
        GLES20.glDisable(GLES20.GL_BLEND)
    }

    private fun stampInkFromStylus() {
        val sx = stylusX ?: return
        val sy = stylusY ?: return
        val pressure = stylusPressure ?: 0

        // Lifted: end the current stroke
        if (pressure <= 0) {
            prevInkX = null
            prevInkY = null
            return
        }

        ensureInkResources()

        val bmp = inkBitmap ?: return
        val w = bmp.width
        val h = bmp.height

        val u = (sx.toFloat() / STYLUS_MAX_X).coerceIn(0f, 1f)
        val v = 1f - (sy.toFloat() / STYLUS_MAX_Y).coerceIn(0f, 1f)

        val cx = (u * (w - 1)).toInt().coerceIn(0, w - 1)
        val cy = (v * (h - 1)).toInt().coerceIn(0, h - 1)

        val pNorm = (pressure.toFloat() / PRESSURE_MAX).coerceIn(0f, 1f)
        val radius = (BRUSH_MIN_RADIUS + pNorm * (BRUSH_MAX_RADIUS - BRUSH_MIN_RADIUS))
            .toInt()
            .coerceAtLeast(1)

        val px = prevInkX
        val py = prevInkY

        if (px == null || py == null) {
            // New stroke: just stamp the point
            stampCircle(bmp, cx, cy, radius, w, h)
        } else {
            // Continuing stroke: stamp circles along the segment px/py -> cx/cy
            stampSegment(bmp, px, py, cx, cy, radius, w, h)
        }

        // Remember for next frame
        prevInkX = cx
        prevInkY = cy

        inkNeedsUpload = true
    }

    private fun stampSegment(
        bmp: Bitmap,
        x0: Int, y0: Int,
        x1: Int, y1: Int,
        radius: Int,
        w: Int, h: Int
    ) {
        val dx = (x1 - x0).toFloat()
        val dy = (y1 - y0).toFloat()
        val dist = kotlin.math.sqrt(dx * dx + dy * dy)

        // Step length: smaller = denser stroke, fewer gaps
        val step = (radius.coerceAtLeast(1) * 0.5f).coerceAtLeast(1f)
        val steps = kotlin.math.ceil(dist / step).toInt().coerceAtLeast(1)

        for (i in 0..steps) {
            val t = i.toFloat() / steps.toFloat()
            val x = (x0 + dx * t).toInt()
            val y = (y0 + dy * t).toInt()
            stampCircle(bmp, x, y, radius, w, h)
        }
    }

    private fun stampCircle(bmp: Bitmap, cx: Int, cy: Int, radius: Int, w: Int, h: Int) {
        val r2 = radius * radius

        val xMin = (cx - radius).coerceAtLeast(0)
        val xMax = (cx + radius).coerceAtMost(w - 1)
        val yMin = (cy - radius).coerceAtLeast(0)
        val yMax = (cy + radius).coerceAtMost(h - 1)

        for (py in yMin..yMax) {
            val dy = py - cy
            val dy2 = dy * dy
            for (px in xMin..xMax) {
                val dx = px - cx
                if (dx * dx + dy2 <= r2) {
                    bmp[px, py] = BRUSH_COLOR
                }
            }
        }
    }

    private fun ensureInkResources() {
        if (inkBitmap == null) {
            inkBitmap = createBitmap(INK_TEX_W, INK_TEX_H)
            inkBitmap!!.eraseColor(0x00000000)

            inkUploadBuffer = ByteBuffer
                .allocateDirect(INK_TEX_W * INK_TEX_H * 4)
                .order(ByteOrder.nativeOrder())

            inkTexCoordBuf = ByteBuffer
                .allocateDirect(4 * 2 * BYTES_PER_FLOAT)
                .order(ByteOrder.nativeOrder())
                .asFloatBuffer().apply {
                    put(
                        floatArrayOf(
                            0f, 0f,
                            1f, 0f,
                            1f, 1f,
                            0f, 1f
                        )
                    )
                    position(0)
                }

            inkVertexBuf = ByteBuffer
                .allocateDirect(4 * 2 * BYTES_PER_FLOAT)
                .order(ByteOrder.nativeOrder())
                .asFloatBuffer()
        }

        if (inkTextureId == 0) {
            val ids = IntArray(1)
            GLES20.glGenTextures(1, ids, 0)
            inkTextureId = ids[0]

            GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, inkTextureId)
            GLES20.glTexParameteri(
                GLES20.GL_TEXTURE_2D,
                GLES20.GL_TEXTURE_MIN_FILTER,
                GLES20.GL_LINEAR
            )
            GLES20.glTexParameteri(
                GLES20.GL_TEXTURE_2D,
                GLES20.GL_TEXTURE_MAG_FILTER,
                GLES20.GL_LINEAR
            )
            GLES20.glTexParameteri(
                GLES20.GL_TEXTURE_2D,
                GLES20.GL_TEXTURE_WRAP_S,
                GLES20.GL_CLAMP_TO_EDGE
            )
            GLES20.glTexParameteri(
                GLES20.GL_TEXTURE_2D,
                GLES20.GL_TEXTURE_WRAP_T,
                GLES20.GL_CLAMP_TO_EDGE
            )

            uploadFullInkBitmap()
        }
    }

    private fun uploadFullInkBitmap() {
        val bmp = inkBitmap ?: return
        val buf = inkUploadBuffer ?: return

        buf.position(0)
        bmp.copyPixelsToBuffer(buf)
        buf.position(0)

        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, inkTextureId)
        GLES20.glTexImage2D(
            GLES20.GL_TEXTURE_2D,
            0,
            GLES20.GL_RGBA,
            bmp.width,
            bmp.height,
            0,
            GLES20.GL_RGBA,
            GLES20.GL_UNSIGNED_BYTE,
            buf
        )
    }

    private fun uploadInkIfNeeded() {
        if (!inkNeedsUpload || inkTextureId == 0) return
        uploadFullInkBitmap()
        inkNeedsUpload = false
    }

    private fun updateSurfaceCornersSmooth(
        newCorners: FloatArray,
        alpha: Float = 0.2f   // 0<alpha<=1; smaller = smoother
    ) {
        // First time: just take the new values
        if (surfaceCornerPx.any { it.isNaN() }) {
            System.arraycopy(newCorners, 0, surfaceCornerPx, 0, 8)
            return
        }

        for (i in 0 until 8) {
            val old = surfaceCornerPx[i]
            val new = newCorners[i]
            surfaceCornerPx[i] = old + alpha * (new - old)
        }
    }

    fun clearInk() {
        inkBitmap?.eraseColor(0x00000000)
        inkNeedsUpload = true
    }

    private fun loadTraceBitmapFromAssets(name: String): Bitmap {
        context.assets.open("traces/$name").use { input ->
            return android.graphics.BitmapFactory.decodeStream(input)
                .copy(Bitmap.Config.ARGB_8888, /*mutable=*/false)
        }
    }

    private fun createTextureFromBitmap(bmp: Bitmap): Int {
        val ids = IntArray(1)
        GLES20.glGenTextures(1, ids, 0)
        val texId = ids[0]

        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, texId)
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_MIN_FILTER, GLES20.GL_LINEAR)
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_MAG_FILTER, GLES20.GL_LINEAR)
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_WRAP_S, GLES20.GL_CLAMP_TO_EDGE)
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_WRAP_T, GLES20.GL_CLAMP_TO_EDGE)

        android.opengl.GLUtils.texImage2D(GLES20.GL_TEXTURE_2D, 0, bmp, 0)

        return texId
    }

    fun setTrace(name: String) {
        clearInk()
        currentTraceName = name.takeIf { it.isNotEmpty() }

        if (name.isEmpty()) {
            showTraceOverlay = false
        } else {
            showTraceOverlay = true
            if (overlayTextureId != 0) {
                GLES20.glDeleteTextures(1, intArrayOf(overlayTextureId), 0)
                overlayTextureId = 0
            }
            ensureTraceOverlay(name)
        }
    }

    /**
     * Creates a flat (non-perspective) bitmap of the current trace overlay
     * with the current ink drawn on top, both in texture space.
     *
     * Returns null if there is no active trace or no ink bitmap yet.
     *
     * NOTE: this does not write to disk; it just returns the Bitmap.
     * Callers can save it as PNG/JPEG as needed.
     */
    fun exportTraceWithInkFlat(): Bitmap? {
        val traceName = currentTraceName ?: return null

        // Make sure ink resources exist; this will create inkBitmap if needed
        ensureInkResources()
        val ink = inkBitmap ?: return null

        // Reload the trace bitmap from assets (ARGB_8888, immutable)
        val traceBmp = loadTraceBitmapFromAssets(traceName)

        val outW = traceBmp.width
        val outH = traceBmp.height

        // Copy ink bitmap so we are not racing with the drawing code
        val inkCopy: Bitmap = synchronized(this) {
            ink.copy(Bitmap.Config.ARGB_8888, /*mutable=*/false)
        }

        // If ink is not the same size as the trace, scale it
        val inkForComposite =
            if (inkCopy.width != outW || inkCopy.height != outH) {
                Bitmap.createScaledBitmap(inkCopy, outW, outH, true)
            } else {
                inkCopy
            }

        // Composite: white background -> trace -> ink
        val result = Bitmap.createBitmap(outW, outH, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(result)

        // 1) White base
        canvas.drawColor(Color.WHITE)

        // 2) Trace overlay
        canvas.drawBitmap(traceBmp, 0f, 0f, null)

        // 3) Ink on top
        canvas.drawBitmap(inkForComposite, 0f, 0f, null)

        // Clean up temporaries
        traceBmp.recycle()
        if (inkForComposite !== inkCopy) {
            inkCopy.recycle()
            inkForComposite.recycle()
        } else {
            inkCopy.recycle()
        }

        return result
    }

    private fun AssetManager.readText(path: String) = open(path).bufferedReader().use { it.readText() }
}
