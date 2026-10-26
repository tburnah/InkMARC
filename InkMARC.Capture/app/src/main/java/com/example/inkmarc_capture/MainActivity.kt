/**
 * The main activity for the application.
 *
 * This activity manages the camera preview, OpenGL rendering surface, and handles interactions
 * with a connected Wacom tablet via USB. It is responsible for:
 * - Requesting necessary permissions (Camera, Storage).
 * - Setting up the CameraX preview and image analysis streams.
 * - Initializing the `GLSurfaceView` and the custom `Renderer`.
 * - Handling the lifecycle of the camera, GLSurfaceView, and USB connection.
 * - Detecting and connecting to a Wacom tablet.
 * - Passing camera frames and stylus data to the native layer for processing and rendering.
 */
package com.example.inkmarc_capture

import android.Manifest
import android.app.PendingIntent
import android.content.BroadcastReceiver
import android.content.ContentValues
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.hardware.usb.UsbDevice
import android.hardware.usb.UsbManager
import android.net.Uri
import android.opengl.GLSurfaceView
import android.os.Build
import android.os.Bundle
import android.os.Environment
import android.os.Handler
import android.os.Looper
import android.provider.MediaStore
import android.util.Log
import android.util.Size
import android.view.Surface
import android.view.View
import androidx.activity.enableEdgeToEdge
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.core.resolutionselector.ResolutionSelector
import androidx.camera.core.resolutionselector.ResolutionStrategy
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.core.view.ViewCompat
import androidx.core.view.WindowInsetsCompat
import com.google.android.material.card.MaterialCardView
import com.google.android.material.button.MaterialButton
import com.google.android.material.textview.MaterialTextView
import java.io.File
import java.io.IOException
import java.nio.ByteBuffer
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

class MainActivity : AppCompatActivity() {

    enum class Exercise {
        T0, T1a, T1b, T2, T3a, T3b, T4a, T4b, T5a, T5b, T6a, T6b, T7, Q0, Done;

        fun next(): Exercise {
            return Exercise.entries[(ordinal + 1) % Exercise.entries.size]
        }
    }

    private var currentExercise = Exercise.T0
    private val traceFileNames = mutableSetOf<String>()
    private val instructionFileNames = mutableSetOf<String>()
    private val instructionTextByFileName = ConcurrentHashMap<String, String>()

    // --- Views ---
    /** The OpenGL surface used for rendering the camera feed and stylus input. */
    private lateinit var glSurfaceView: GLSurfaceView
    private lateinit var instructionOverlayCard: MaterialCardView
    private lateinit var instructionOverlayText: MaterialTextView

    // --- Camera ---
    /** A dedicated thread for running camera-related tasks, such as image analysis. */
    private lateinit var cameraExecutor: ExecutorService
    /** Background executor for MediaStore writes that should not block rendering. */
    private lateinit var mediaStoreExecutor: ExecutorService
    /** Background executor for loading exercise instruction text. */
    private lateinit var instructionExecutor: ExecutorService
    /** The CameraX provider instance, used to bind camera use cases to the activity's lifecycle. */
    private var cameraProvider: ProcessCameraProvider? = null

    // --- USB / Wacom ---
    /** Handles the low-level communication with the connected USB device. */
    private lateinit var usbHandler: UsbHandler
    /** Android system service for managing USB devices. */
    private lateinit var usbManager: UsbManager
    /** Action for the broadcast intent to request USB permission. */

    // --- State ---
    /** Flag to track whether video recording is currently active. */
    private var isRecording = false

    // --- Renderer ---
    /** The custom OpenGL renderer that draws the camera stream and stylus data. */
    private lateinit var renderer: Renderer

    private val mainHandler = Handler(Looper.getMainLooper())
    private val hideInstructionOverlayRunnable = Runnable {
        instructionOverlayCard.visibility = View.GONE
    }
    private var isRecordingStartScheduled = false
    private val startRecordingRunnable = Runnable {
        isRecordingStartScheduled = false
        startRecordingForCurrentExerciseIfEligible()
    }

    // --- Lifecycle Methods ---

    /**
     * Called when the activity is first created.
     *
     * Initializes views, camera, USB handling, and requests permissions.
     */
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContentView(R.layout.activity_main)
        ViewCompat.setOnApplyWindowInsetsListener(findViewById(R.id.main)) { v, insets ->
            val systemBars = insets.getInsets(WindowInsetsCompat.Type.systemBars())
            v.setPadding(systemBars.left, systemBars.top, systemBars.right, systemBars.bottom)
            insets
        }

        setupViews()
        setupExecutors()
        setupUsb()
        loadTraceFiles()
        beginExercise(currentExercise)

        // After setup, check for permissions and start the camera if granted.
        if (allPermissionsGranted()) {
            startCamera()
        } else {
            ActivityCompat.requestPermissions(
                this, REQUIRED_PERMISSIONS, REQUEST_CODE_PERMISSIONS
            )
        }
        // Initialize the native components.
        nativeInit()
    }

    /**
     * Called when the activity will start interacting with the user.
     *
     * Resumes the `GLSurfaceView`, starts the camera, and checks for a USB device.
     */
    override fun onResume() {
        super.onResume()
        glSurfaceView.onResume()
        if (allPermissionsGranted()) {
            startCamera()
        }
        scheduleRecordingStartForCurrentExerciseIfNeeded()
        checkForUsbDevice()
    }

    /**
     * Called when the activity is no longer in the foreground.
     *
     * This method ensures a graceful shutdown of resources in a specific order:
     * 1. Unbinds all camera use cases to stop receiving frames.
     * 2. Stops the USB handler to cease data reading.
     * 3. Queues a task on the GL thread to release OpenGL resources.
     * 4. Pauses the `GLSurfaceView`'s rendering thread.
     */
    override fun onPause() {
        // 1. Stop camera frames first.
        try {
            cameraProvider?.unbindAll()
        } catch (_: Throwable) {
            // Ignore any errors during unbinding.
        }

        cancelPendingRecordingStart()
        isRecording = false

        // 2. Stop the USB handler.
        usbHandler.stop()

        // 3. Release GL resources while the GL context is still active.
        glSurfaceView.queueEvent { renderer.release() }

        // 4. Pause the GL thread.
        glSurfaceView.onPause()

        super.onPause()
    }

    /**
     * Called before the activity is destroyed.
     *
     * Performs final cleanup of resources, including shutting down the camera executor,
     * stopping the USB handler, unregistering the broadcast receiver, and cleaning up
     * native resources.
     */
    override fun onDestroy() {
        // Ensure GL resources are released, in case onPause() was skipped.
        glSurfaceView.queueEvent { renderer.release() }
        cameraExecutor.shutdown()
        mediaStoreExecutor.shutdown()
        instructionExecutor.shutdown()
        mainHandler.removeCallbacks(hideInstructionOverlayRunnable)
        cancelPendingRecordingStart()

        // Final USB cleanup.
        usbHandler.stop()
        try {
            unregisterReceiver(usbPermissionReceiver)
        } catch (_: IllegalArgumentException) {
            // The receiver was not registered or was already unregistered, which is safe to ignore.
        }

        super.onDestroy()
        // Clean up native resources.
        nativeDestroy()
    }

    // --- Initial Setup ---

    /**
     * Initializes the UI components.
     *
     * Sets up the `GLSurfaceView`, its renderer, and the record button's click listener.
     */
    private fun setupViews() {
        glSurfaceView = findViewById(R.id.gl_surface_view)
        instructionOverlayCard = findViewById(R.id.instruction_overlay_card)
        instructionOverlayText = findViewById(R.id.instruction_overlay_text)
        glSurfaceView.setEGLContextClientVersion(3) // Use OpenGL ES 3.
        renderer = Renderer(this)
        glSurfaceView.setRenderer(renderer)
        glSurfaceView.renderMode = GLSurfaceView.RENDERMODE_CONTINUOUSLY // Continuously redraw the scene.

        val clearButton = findViewById<MaterialButton>(R.id.clear_button)
        clearButton.setOnClickListener {
            // Run on GL thread so we safely touch GL / inkBitmap state
            glSurfaceView.queueEvent {
                renderer.clearInk()
            }
            // Ask GL to redraw with cleared ink
            glSurfaceView.requestRender()
        }

        val exerciseButton = findViewById<MaterialButton>(R.id.overlay_button)
        exerciseButton.setOnClickListener {
            val exercise = currentExercise

            stopRecordingFlow()

            if (exercise.name.startsWith("Q")) {
                val resolver = contentResolver
                val participantId = ParticipantIdManager.getParticipantId(this)
                val ts = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date())
                val fileName = "participant_${participantId}_${exercise}_$ts.png"
                val imageUri = resolver.insert(
                    getImageCollectionUri(),
                    buildImageContentValues(fileName)
                )

                if (imageUri == null) {
                    Log.e(
                        "MainActivity",
                        "Failed to create MediaStore image record for $exercise"
                    )
                } else {
                    glSurfaceView.queueEvent {
                        val bmp = renderer.exportTraceWithInkFlat()
                        mediaStoreExecutor.execute {
                            saveTraceBitmap(imageUri, bmp, exercise)
                        }
                    }
                }
            }

            currentExercise = currentExercise.next()
            exerciseButton.text = currentExercise.toString()
            beginExercise(currentExercise)

            val traceFileName = "${currentExercise}.png"
            val assetPath = if (traceFileNames.contains(traceFileName)) traceFileName else ""

            glSurfaceView.queueEvent {
                renderer.setTrace(assetPath)
            }
            glSurfaceView.requestRender()
        }
    }

    private fun loadTraceFiles() {
        try {
            assets.list("traces")?.let { assetNames ->
                traceFileNames.addAll(assetNames.filter { it.endsWith(".png", ignoreCase = true) })
                instructionFileNames.addAll(assetNames.filter { it.endsWith(".txt", ignoreCase = true) })
                preloadInstructionText(instructionFileNames)
            }
        } catch (e: IOException) {
            Log.e("MainActivity", "Error loading trace files", e)
        }
    }


    /**
     * Initializes the camera executor service.
     */
    private fun setupExecutors() {
        cameraExecutor = Executors.newSingleThreadExecutor()
        mediaStoreExecutor = Executors.newSingleThreadExecutor()
        instructionExecutor = Executors.newSingleThreadExecutor()
    }

    /**
     * Initializes USB-related components.
     *
     * Gets the `UsbManager`, creates the `UsbHandler`, and registers a broadcast
     * receiver for USB permission requests.
     */
    private fun setupUsb() {
        usbManager = getSystemService(USB_SERVICE) as UsbManager
        usbHandler = UsbHandler(this) { x, y, pressure, tiltX, tiltY ->
            // Pass stylus data to the renderer on the GL thread.
            glSurfaceView.queueEvent {
                renderer.updateStylusData(x, y, pressure, tiltX, tiltY)
            }
        }

        val filter = IntentFilter(ACTION_USB_PERMISSION)
        val receiverFlags =
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) RECEIVER_NOT_EXPORTED else 0
        registerReceiver(usbPermissionReceiver, filter, receiverFlags)
    }

    // --- Permissions ---

    /**
     * Checks if all required permissions have been granted.
     *
     * @return `true` if all permissions are granted, `false` otherwise.
     */
    private fun allPermissionsGranted() = REQUIRED_PERMISSIONS.all {
        ContextCompat.checkSelfPermission(
            baseContext, it
        ) == PackageManager.PERMISSION_GRANTED
    }

    /**
     * Callback for the result from requesting permissions.
     *
     * If all permissions are granted, it starts the camera. Otherwise, a TODO is left
     * to handle the denial case.
     */
    override fun onRequestPermissionsResult(
        requestCode: Int, permissions: Array<String>, grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == REQUEST_CODE_PERMISSIONS) {
            if (allPermissionsGranted()) {
                startCamera()
            } else {
                // TODO: Handle the case where the user denies permission.
                //  For now, the camera will simply not start.
            }
        }
    }

    // --- Camera ---

    /**
     * Initializes the `ProcessCameraProvider` and binds the camera use cases.
     */
    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)
        cameraProviderFuture.addListener({
            // The future is guaranteed to be done here.
            cameraProvider = cameraProviderFuture.get()
            bindCameraUseCases()
        }, ContextCompat.getMainExecutor(this))
    }

    /**
     * Configures and binds the `Preview` and `ImageAnalysis` use cases to the activity's lifecycle.
     *
     * This function sets up the camera to provide a preview stream to the `GLSurfaceView`'s
     * underlying `SurfaceTexture` and an analysis stream for processing in the native layer.
     */
    private fun bindCameraUseCases() {
        val provider = cameraProvider ?: return // Return if the provider is not yet available.
        provider.unbindAll() // Unbind any existing use cases.

        val resolutionStrategy = ResolutionStrategy(
            Size(640, 480),
            ResolutionStrategy.FALLBACK_RULE_NONE
        )

        val resolutionSelector = ResolutionSelector.Builder()
            .setResolutionStrategy(resolutionStrategy)
            .build()

        // Configure the Preview use case.
        val preview = Preview.Builder()
            .setResolutionSelector(resolutionSelector)
            .setTargetRotation(Surface.ROTATION_0)
            .build()
            .apply {
                // Set the SurfaceProvider to route the camera preview to our renderer.
                surfaceProvider = Preview.SurfaceProvider { request ->
                    val res = request.resolution
                    // Inform the renderer of the stream resolution on the GL thread.
                    glSurfaceView.queueEvent {
                        renderer.setStreamResolutionOnGlThread(res.width, res.height)
                    }
                    // Get the SurfaceTexture from the renderer and provide it to the camera.
                    val st = renderer.getSurfaceTexture()
                    st.setDefaultBufferSize(res.width, res.height)
                    val surface = Surface(st)
                    request.provideSurface(surface, ContextCompat.getMainExecutor(this@MainActivity)) {
                        // This is called when the surface is no longer needed.
                        surface.release()
                    }
                }
            }

        // Configure the ImageAnalysis use case.
        val analysis = ImageAnalysis.Builder()
            .setResolutionSelector(resolutionSelector)
            .setTargetRotation(Surface.ROTATION_0)
            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
            .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_YUV_420_888)
            .build()
            .apply {
                // Set the analyzer to process frames.
                setAnalyzer(cameraExecutor, ::analyzeImage)
            }

        // Bind both use cases to the lifecycle with the default back camera.
        provider.bindToLifecycle(this, CameraSelector.DEFAULT_BACK_CAMERA, preview, analysis)
    }

    /**
     * Processes a single camera frame for AprilTag detection.
     *
     * This function receives an `ImageProxy` from CameraX, extracts the Y plane (luminance),
     * and passes its data to the native `nativeDetect` function. The results are then
     * passed to the `Renderer` to be drawn.
     *
     * @param img The image to analyze, provided by the CameraX `ImageAnalysis` use case.
     */
    private fun analyzeImage(img: ImageProxy) {
        try {
            val yPlane = img.planes[0]
            val yBuffer: ByteBuffer = yPlane.buffer

            // Create a duplicate of the buffer before passing to JNI to avoid
            // potential conflicts with other consumers of the buffer.
            val yForNative = yBuffer.duplicate().apply { position(0) }

            // Call the native function to detect AprilTags.
            val detections = nativeDetect(img.width, img.height, yPlane.rowStride, yForNative)

            // Pass the detection results to the renderer on the GL thread.
            glSurfaceView.queueEvent {
                renderer.updateDetections(detections ?: emptyArray())
            }
            glSurfaceView.requestRender() // Request a redraw to show new detections.

        } catch (t: Throwable) {
            Log.e("APRILTAG", "Analyzer error", t)
        } finally {
            img.close() // IMPORTANT: Close the image to release it for the next frame.
        }
    }

    private fun getImageCollectionUri(): Uri {
        return if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            MediaStore.Images.Media.getContentUri(MediaStore.VOLUME_EXTERNAL_PRIMARY)
        } else {
            MediaStore.Images.Media.EXTERNAL_CONTENT_URI
        }
    }

    private fun buildImageContentValues(fileName: String): ContentValues {
        return ContentValues().apply {
            put(MediaStore.Images.Media.DISPLAY_NAME, fileName)
            put(MediaStore.Images.Media.MIME_TYPE, "image/png")

            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                put(MediaStore.Images.Media.RELATIVE_PATH, "Pictures/InkMARC")
                put(MediaStore.Images.Media.IS_PENDING, 1)
            } else {
                @Suppress("DEPRECATION")
                put(
                    MediaStore.Images.Media.DATA,
                    createLegacyMediaFile(Environment.DIRECTORY_PICTURES, fileName).absolutePath
                )
            }
        }
    }

    private fun saveTraceBitmap(imageUri: Uri, bitmap: Bitmap?, exercise: Exercise) {
        if (bitmap == null) {
            Log.w("MainActivity", "No trace+ink bitmap exported for $exercise")
            contentResolver.delete(imageUri, null, null)
            return
        }

        try {
            contentResolver.openOutputStream(imageUri)?.use { out ->
                if (!bitmap.compress(Bitmap.CompressFormat.PNG, 100, out)) {
                    throw IOException("Bitmap compress() returned false")
                }
            } ?: throw IOException("Failed to open output stream for $imageUri")

            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                val doneValues = ContentValues().apply {
                    put(MediaStore.Images.Media.IS_PENDING, 0)
                }
                contentResolver.update(imageUri, doneValues, null, null)
            }

            Log.d("MainActivity", "Saved trace+ink PNG for $exercise to $imageUri")
        } catch (e: Exception) {
            Log.e("MainActivity", "Failed to save trace+ink for $exercise", e)
            contentResolver.delete(imageUri, null, null)
        } finally {
            bitmap.recycle()
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

    private fun preloadInstructionText(fileNames: Collection<String>) {
        instructionExecutor.execute {
            fileNames.forEach { fileName ->
                loadInstructionText(fileName)
            }
        }
    }

    private fun beginExercise(exercise: Exercise) {
        showInstructionForExercise(exercise)
        scheduleRecordingStart(exercise)
    }

    private fun showInstructionForExercise(exercise: Exercise) {
        val fileName = "${exercise}.txt"
        if (!instructionFileNames.contains(fileName)) {
            hideInstructionOverlay()
            return
        }

        val cachedText = instructionTextByFileName[fileName]
        if (cachedText != null) {
            displayInstructionOverlay(cachedText)
            return
        }

        instructionExecutor.execute {
            val text = loadInstructionText(fileName)
            mainHandler.post {
                if (currentExercise == exercise && text != null) {
                    displayInstructionOverlay(text)
                }
            }
        }
    }

    private fun loadInstructionText(fileName: String): String? {
        return instructionTextByFileName[fileName] ?: try {
            assets.open("traces/$fileName").bufferedReader().use { reader ->
                reader.readText().trim()
            }.takeIf { it.isNotEmpty() }?.also { text ->
                instructionTextByFileName[fileName] = text
            }
        } catch (e: IOException) {
            Log.e("MainActivity", "Error loading instruction text from $fileName", e)
            null
        }
    }

    private fun displayInstructionOverlay(text: String) {
        mainHandler.removeCallbacks(hideInstructionOverlayRunnable)
        instructionOverlayText.text = text
        instructionOverlayCard.alpha = 1f
        instructionOverlayCard.visibility = View.VISIBLE
        mainHandler.postDelayed(hideInstructionOverlayRunnable, INSTRUCTION_OVERLAY_DURATION_MS)
    }

    private fun hideInstructionOverlay() {
        mainHandler.removeCallbacks(hideInstructionOverlayRunnable)
        instructionOverlayCard.visibility = View.GONE
    }

    private fun scheduleRecordingStart(exercise: Exercise) {
        cancelPendingRecordingStart()
        if (exercise == Exercise.Done) {
            return
        }

        isRecordingStartScheduled = true
        mainHandler.postDelayed(startRecordingRunnable, RECORDING_START_DELAY_MS)
    }

    private fun scheduleRecordingStartForCurrentExerciseIfNeeded() {
        if (currentExercise == Exercise.Done || isRecording || isRecordingStartScheduled) {
            return
        }

        scheduleRecordingStart(currentExercise)
    }

    private fun cancelPendingRecordingStart() {
        isRecordingStartScheduled = false
        mainHandler.removeCallbacks(startRecordingRunnable)
    }

    private fun startRecordingForCurrentExerciseIfEligible() {
        if (isRecording || currentExercise == Exercise.Done) {
            return
        }

        isRecording = true
        val exerciseName = currentExercise.toString()
        glSurfaceView.queueEvent {
            renderer.startRecording(exerciseName)
        }
    }

    private fun stopRecordingFlow() {
        cancelPendingRecordingStart()
        if (!isRecording) {
            return
        }

        isRecording = false
        glSurfaceView.queueEvent { renderer.stopRecording() }
    }

    // --- USB / Wacom ---

    /**
     * Scans for a connected Wacom tablet and requests permission if needed.
     */
    private fun checkForUsbDevice() {
        Log.d("MainActivity", "Checking for USB devices...")
        val deviceList = usbManager.deviceList
        if (deviceList.isEmpty()) {
            Log.d("MainActivity", "No USB devices found.")
            return
        }

        Log.d("MainActivity", "Found ${deviceList.size} devices.")
        for (device in deviceList.values) {
            // Check for the Wacom vendor ID.
            if (device.vendorId == 1386) { // Wacom Vendor ID
                Log.d("MainActivity", "Wacom tablet found: ${device.deviceName}")
                if (usbManager.hasPermission(device)) {
                    Log.d("MainActivity", "Permission already granted. Starting handler.")
                    usbHandler.start(device)
                } else {
                    Log.d("MainActivity", "Requesting permission...")
                    val intent = Intent(ACTION_USB_PERMISSION)
                    // Use FLAG_IMMUTABLE for security on newer Android versions.
                    val pendingIntent = PendingIntent.getBroadcast(this, 0, intent, PendingIntent.FLAG_IMMUTABLE)
                    usbManager.requestPermission(device, pendingIntent)
                }
                return // Found and handled the first Wacom device.
            }
        }
        Log.d("MainActivity", "No Wacom tablet found.")
    }

    /**
     * A `BroadcastReceiver` that listens for the result of a USB permission request.
     *
     * If permission is granted, it starts the `UsbHandler`.
     */
    private val usbPermissionReceiver = object : BroadcastReceiver() {
        override fun onReceive(context: Context, intent: Intent) {
            // Check if the received intent is for our permission request.
            if (ACTION_USB_PERMISSION == intent.action) {
                synchronized(this) {
                    val device: UsbDevice? = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
                        intent.getParcelableExtra(UsbManager.EXTRA_DEVICE, UsbDevice::class.java)
                    } else {
                        @Suppress("DEPRECATION")
                        intent.getParcelableExtra(UsbManager.EXTRA_DEVICE)
                    }

                    if (intent.getBooleanExtra(UsbManager.EXTRA_PERMISSION_GRANTED, false)) {
                        device?.let {
                            Log.d("MainActivity", "USB permission granted for ${it.deviceName}")
                            usbHandler.start(it)
                        }
                    } else {
                        Log.d("MainActivity", "USB permission denied for ${device?.deviceName}")
                    }
                }
            }
        }
    }

    // --- Companion Object ---
    companion object {
        private const val INSTRUCTION_OVERLAY_DURATION_MS = 6_000L
        private const val RECORDING_START_DELAY_MS = 4_000L
        private const val REQUEST_CODE_PERMISSIONS = 10
        private val REQUIRED_PERMISSIONS =
            mutableListOf(
                Manifest.permission.CAMERA
            ).apply {
                // WRITE_EXTERNAL_STORAGE is only needed for Android 9 (Pie) and below.
                if (Build.VERSION.SDK_INT <= Build.VERSION_CODES.P) {
                    add(Manifest.permission.WRITE_EXTERNAL_STORAGE)
                }
            }.toTypedArray()

        /** Action for the broadcast intent to request USB permission. Must be unique. */
        private const val ACTION_USB_PERMISSION = "com.example.inkmarc_capture.USB_PERMISSION"

        // Load the native library containing the AprilTag detection logic.
        // This block is executed when the class is first loaded.
        init {
            System.loadLibrary("apriltag") // Replace "apriltag" with your actual .so library name.
        }

        /** Initializes native resources. Should be called once in `onCreate`. */
        @JvmStatic
        external fun nativeInit()

        /** Releases native resources. Should be called once in `onDestroy`. */
        @JvmStatic
        external fun nativeDestroy()

        /**
         * Detects AprilTags in a given YUV image buffer.
         *
         * @param width The width of the image in pixels.
         * @param height The height of the image in pixels.
         * @param stride The row stride of the image buffer in bytes.
         * @param yBuffer A `ByteBuffer` containing the Y (luminance) plane of the image.
         * @return An array of `AprilTagDetection` objects, or `null` if no tags are found.
         */
        @JvmStatic
        external fun nativeDetect(
            width: Int,
            height: Int,
            stride: Int,
            yBuffer: ByteBuffer
        ): Array<AprilTagDetection>?
    }
}
