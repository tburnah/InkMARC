#if ANDROID
using System;
using System.Collections.Generic;
using Android.Views;
using Android.Hardware.Camera2;
using Android.Content;
using Android.Graphics;
using Android.Runtime;
using Microsoft.Maui.Handlers;
using Microsoft.Maui;
using Android.Media;
using Android.Hardware.Camera2.Params;
using ASize = Android.Util.Size;

namespace InkMARC.Cue.Platforms.Android
{
    /// <summary>
    /// Handles the camera preview and recording functionality for Android.
    /// </summary>
    public class CameraPreviewHandler : ViewHandler<CameraPreview, TextureView>, IDisposable
    {
        private const int DefaultRecordingWidth = 1280;
        private const int DefaultRecordingHeight = 720;
        private const int MaxDimension = 448;
        private const int VideoBitRate = 1000000;
        private const int VideoFrameRate = 15;
        private const int OrientationHint = 90; // Portrait, change to 0 if landscape

        private CameraDevice? cameraDevice;
        private CameraCaptureSession? cameraSession;
        private TextureView? previewView;
        private MediaRecorder? mediaRecorder;
        private bool isRecording = false;
        private string? videoFilePath;
        private int recordingWidth = DefaultRecordingWidth;
        private int recordingHeight = DefaultRecordingHeight;

        /// <summary>
        /// Property mapper for CameraPreviewHandler.
        /// </summary>
        public static PropertyMapper<CameraPreview, CameraPreviewHandler> Mapper = new(ViewHandler.ViewMapper)
        {
            // You can add mappings here later
        };

        /// <summary>
        /// Initializes a new instance of the <see cref="CameraPreviewHandler"/> class.
        /// </summary>
        public CameraPreviewHandler() : base(Mapper)
        {
        }

        /// <summary>
        /// Creates the platform-specific view for the camera preview.
        /// </summary>
        /// <returns>The created <see cref="TextureView"/>.</returns>
        protected override TextureView CreatePlatformView()
        {
            System.Diagnostics.Debug.WriteLine("Creating TextureView");
            previewView = new TextureView(Context);
            InitializeCamera();
            return previewView;
        }

        /// <summary>
        /// Initializes the camera.
        /// </summary>
        void InitializeCamera()
        {
            if (previewView == null)
                return;

            if (previewView.IsAvailable)
            {
                OpenCamera();
            }
            else
            {
                previewView.SurfaceTextureListener = new CameraPreviewSurfaceTextureListener(this);
            }
        }

        /// <summary>
        /// Opens the camera.
        /// </summary>
        void OpenCamera()
        {
            System.Diagnostics.Debug.WriteLine("Opening camera...");
            CameraManager cameraManager = (CameraManager)Context.GetSystemService(Context.CameraService);
            try
            {
                string[] cameraIds = cameraManager.GetCameraIdList();
                string cameraId = cameraIds[0];
                cameraManager.OpenCamera(cameraId, new CameraStateCallback(this), null);
            }
            catch (Exception ex)
            {
                System.Diagnostics.Debug.WriteLine($"OpenCamera Exception: {ex.Message}");
            }
        }

        /// <summary>
        /// Starts the camera preview session.
        /// </summary>
        void StartPreviewSession()
        {
            if (cameraDevice == null || previewView == null || !previewView.IsAvailable)
                return;

            SurfaceTexture surfaceTexture = previewView.SurfaceTexture;
            surfaceTexture.SetDefaultBufferSize(previewView.Width, previewView.Height);
            Surface surface = new Surface(surfaceTexture);

            try
            {
                CaptureRequest.Builder builder = cameraDevice.CreateCaptureRequest(CameraTemplate.Preview);
                builder.AddTarget(surface);

                List<Surface> surfaces = new List<Surface> { surface };

                cameraDevice.CreateCaptureSession(surfaces, new CameraCaptureStateCallback(this, builder, surface), null);
            }
            catch (Exception ex)
            {
                System.Diagnostics.Debug.WriteLine($"StartPreviewSession Exception: {ex.Message}");
            }
        }

        /// <summary>
        /// Prepares the media recorder for recording.
        /// </summary>
        /// <param name="filePath">The file path to save the recorded video.</param>
        void PrepareMediaRecorder(string filePath)
        {
            var (nativeSize, _, _) = GetOptimalVideoSize();
            var (scaledWidth, scaledHeight) = ScaleToMax(nativeSize, MaxDimension);
            var (size, w, h) = GetSafeVideoSize();

            mediaRecorder = new MediaRecorder();
            mediaRecorder.SetVideoSource(VideoSource.Surface);
            mediaRecorder.SetOutputFormat(OutputFormat.Mpeg4);
            mediaRecorder.SetOutputFile(filePath);
            mediaRecorder.SetVideoEncodingBitRate(VideoBitRate);
            mediaRecorder.SetVideoFrameRate(VideoFrameRate);
            mediaRecorder.SetVideoSize(w, h);
            mediaRecorder.SetVideoEncoder(VideoEncoder.H264);

            recordingWidth = w;
            recordingHeight = h;

            mediaRecorder.SetOrientationHint(OrientationHint);
            mediaRecorder.Prepare();
        }

        /// <summary>
        /// Starts recording video.
        /// </summary>
        /// <param name="filePath">The file path to save the recorded video.</param>
        public void StartRecording(string filePath)
        {
            if (cameraDevice == null || previewView == null)
                return;

            try
            {
                PrepareMediaRecorder(filePath);

                var surfaceTexture = previewView.SurfaceTexture;
                surfaceTexture.SetDefaultBufferSize(recordingWidth, recordingHeight);
                Surface previewSurface = new Surface(surfaceTexture);
                Surface recorderSurface = mediaRecorder.Surface;

                var surfaces = new List<Surface> { previewSurface, recorderSurface };

                var builder = cameraDevice.CreateCaptureRequest(CameraTemplate.Record);
                builder.AddTarget(previewSurface);
                builder.AddTarget(recorderSurface);

                cameraDevice.CreateCaptureSession(
                    surfaces,
                    new RecordingCaptureSessionCallback(this, builder),
                    null
                );

                isRecording = true;
                videoFilePath = filePath;
            }
            catch (Exception ex)
            {
                System.Diagnostics.Debug.WriteLine($"StartRecording Exception: {ex.Message}");
            }
        }

        /// <summary>
        /// Stops recording video.
        /// </summary>
        public void StopRecording()
        {
            if (!isRecording || mediaRecorder == null)
                return;

            try
            {
                mediaRecorder.Stop();
                mediaRecorder.Reset();
                mediaRecorder.Release();
                mediaRecorder = null;
                isRecording = false;

                System.Diagnostics.Debug.WriteLine($"Video saved to {videoFilePath}");
            }
            catch (Exception ex)
            {
                System.Diagnostics.Debug.WriteLine($"StopRecording Exception: {ex.Message}");
            }
        }

        /// <summary>
        /// Gets the optimal video size for recording.
        /// </summary>
        /// <returns>A tuple containing the optimal size and its width and height.</returns>
        (ASize size, int width, int height) GetOptimalVideoSize()
        {
            CameraManager cameraManager = (CameraManager)Context.GetSystemService(Context.CameraService);
            var cameraId = cameraManager.GetCameraIdList()[0];
            var characteristics = cameraManager.GetCameraCharacteristics(cameraId);
            var map = (StreamConfigurationMap)characteristics.Get(CameraCharacteristics.ScalerStreamConfigurationMap);
            var sizes = map.GetOutputSizes(Java.Lang.Class.FromType(typeof(MediaRecorder)));

            ASize best = sizes[0];
            foreach (var size in sizes)
            {
                float aspectRatio = (float)size.Width / size.Height;
                float bestAspectRatio = (float)best.Width / best.Height;

                if (System.Math.Abs(size.Height - MaxDimension) < System.Math.Abs(best.Height - MaxDimension) &&
                    System.Math.Abs(aspectRatio - bestAspectRatio) < 0.01f)
                {
                    best = size;
                }
            }

            return (best, best.Width, best.Height);
        }

        /// <summary>
        /// Gets a safe video size for recording.
        /// </summary>
        /// <returns>A tuple containing the safe size and its width and height.</returns>
        (ASize widthHeight, int w, int h) GetSafeVideoSize()
        {
            var manager = (CameraManager)Context.GetSystemService(Context.CameraService);
            var cameraId = manager.GetCameraIdList()[0];
            var characteristics = manager.GetCameraCharacteristics(cameraId);
            var map = (StreamConfigurationMap)characteristics.Get(CameraCharacteristics.ScalerStreamConfigurationMap);
            var sizes = map.GetOutputSizes((int)ImageFormatType.Private);

            var safeSize = sizes.FirstOrDefault(s => s.Width == 640 && s.Height == 480);
            if (safeSize == null) safeSize = sizes.Last();

            return (safeSize, safeSize.Width, safeSize.Height);
        }

        /// <summary>
        /// Scales the given size to the maximum dimension while preserving the aspect ratio.
        /// </summary>
        /// <param name="originalSize">The original size.</param>
        /// <param name="maxDimension">The maximum dimension.</param>
        /// <returns>A tuple containing the scaled width and height.</returns>
        (int scaledWidth, int scaledHeight) ScaleToMax(ASize originalSize, int maxDimension)
        {
            int originalWidth = originalSize.Width;
            int originalHeight = originalSize.Height;

            float aspect = (float)originalWidth / originalHeight;

            if (originalWidth >= originalHeight)
            {
                if (originalWidth <= maxDimension)
                    return (originalWidth, originalHeight);

                int newWidth = maxDimension;
                int newHeight = (int)(newWidth / aspect);
                return (newWidth, newHeight);
            }
            else
            {
                if (originalHeight <= maxDimension)
                    return (originalWidth, originalHeight);

                int newHeight = maxDimension;
                int newWidth = (int)(newHeight * aspect);
                return (newWidth, newHeight);
            }
        }

        /// <summary>
        /// Connects the handler to the native view.
        /// </summary>
        /// <param name="nativeView">The native view.</param>
        protected override void ConnectHandler(TextureView nativeView)
        {
            base.ConnectHandler(nativeView);
        }

        /// <summary>
        /// Disconnects the handler from the native view.
        /// </summary>
        /// <param name="nativeView">The native view.</param>
        protected override void DisconnectHandler(TextureView nativeView)
        {
            base.DisconnectHandler(nativeView);
            Dispose();
        }

        /// <summary>
        /// Disposes the resources used by the handler.
        /// </summary>
        public void Dispose()
        {
            try
            {
                cameraSession?.Close();
                cameraSession = null;
                cameraDevice?.Close();
                cameraDevice = null;
                previewView?.Dispose();
                previewView = null;
            }
            catch (Exception ex)
            {
                System.Diagnostics.Debug.WriteLine($"Dispose Exception: {ex.Message}");
            }
        }

        /// <summary>
        /// Callback for handling the recording capture session state.
        /// </summary>
        private class RecordingCaptureSessionCallback : CameraCaptureSession.StateCallback
        {
            private readonly CameraPreviewHandler handler;
            private readonly CaptureRequest.Builder builder;

            public RecordingCaptureSessionCallback(CameraPreviewHandler handler, CaptureRequest.Builder builder)
            {
                this.handler = handler;
                this.builder = builder;
            }

            public override void OnConfigured(CameraCaptureSession session)
            {
                handler.cameraSession = session;
                try
                {
                    session.SetRepeatingRequest(builder.Build(), null, null);
                    handler.mediaRecorder?.Start();
                }
                catch (Exception ex)
                {
                    System.Diagnostics.Debug.WriteLine($"Recording session error: {ex.Message}");
                }
            }

            public override void OnConfigureFailed(CameraCaptureSession session)
            {
                System.Diagnostics.Debug.WriteLine("Recording session configuration failed.");
            }
        }

        /// <summary>
        /// Callback for handling the camera state.
        /// </summary>
        private class CameraStateCallback : CameraDevice.StateCallback
        {
            private readonly CameraPreviewHandler handler;
            public CameraStateCallback(CameraPreviewHandler handler)
            {
                this.handler = handler;
            }

            public override void OnOpened(CameraDevice camera)
            {
                handler.cameraDevice = camera;
                handler.StartPreviewSession();
            }

            public override void OnDisconnected(CameraDevice camera)
            {
                camera.Close();
                handler.cameraDevice = null;
            }

            public override void OnError(CameraDevice camera, [GeneratedEnum] CameraError error)
            {
                camera.Close();
                handler.cameraDevice = null;
                System.Diagnostics.Debug.WriteLine($"CameraStateCallback OnError: {error}");
            }
        }

        /// <summary>
        /// Callback for handling the camera capture session state.
        /// </summary>
        private class CameraCaptureStateCallback : CameraCaptureSession.StateCallback
        {
            private readonly CameraPreviewHandler handler;
            private readonly CaptureRequest.Builder builder;
            private readonly Surface surface;

            public CameraCaptureStateCallback(CameraPreviewHandler handler, CaptureRequest.Builder builder, Surface surface)
            {
                this.handler = handler;
                this.builder = builder;
                this.surface = surface;
            }

            public override void OnConfigured(CameraCaptureSession session)
            {
                if (handler.cameraDevice == null)
                    return;

                handler.cameraSession = session;
                try
                {
                    session.SetRepeatingRequest(builder.Build(), null, null);
                }
                catch (Exception ex)
                {
                    System.Diagnostics.Debug.WriteLine($"OnConfigured Exception: {ex.Message}");
                }
            }

            public override void OnConfigureFailed(CameraCaptureSession session)
            {
                System.Diagnostics.Debug.WriteLine("Camera capture session configuration failed.");
            }
        }

        /// <summary>
        /// Listener for handling the surface texture events.
        /// </summary>
        private class CameraPreviewSurfaceTextureListener : Java.Lang.Object, TextureView.ISurfaceTextureListener
        {
            private readonly CameraPreviewHandler handler;
            public CameraPreviewSurfaceTextureListener(CameraPreviewHandler handler)
            {
                this.handler = handler;
            }

            public void OnSurfaceTextureAvailable(SurfaceTexture surface, int width, int height)
            {
                handler.OpenCamera();
            }

            public bool OnSurfaceTextureDestroyed(SurfaceTexture surface)
            {
                return true;
            }

            public void OnSurfaceTextureSizeChanged(SurfaceTexture surface, int width, int height)
            {
                // Handle size changes if needed.
            }

            public void OnSurfaceTextureUpdated(SurfaceTexture surface)
            {
                // Called each time there's a new frame.
            }
        }
    }
}
#endif
