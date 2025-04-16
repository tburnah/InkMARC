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
        private const int MaxDimension = 448;
        private const int VideoBitRate = 500000;
        private const int VideoFrameRate = 15;
        private const int OrientationHint = 90; // Portrait, change to 0 if landscape

        private CameraDevice? cameraDevice;
        private CameraCaptureSession? cameraSession;
        private TextureView? previewView;
        private MediaRecorder? mediaRecorder;
        private bool isRecording = false;
        private string? videoFilePath;
        private Surface? previewSurface;

        /// <summary>
        /// Property mapper for CameraPreviewHandler.
        /// </summary>
        public static PropertyMapper<CameraPreview, CameraPreviewHandler> Mapper = new(ViewHandler.ViewMapper);

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
        private void InitializeCamera()
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
        private void OpenCamera()
        {
            System.Diagnostics.Debug.WriteLine("Opening camera...");
            var cameraManager = (CameraManager)Context.GetSystemService(Context.CameraService);
            try
            {
                var cameraId = cameraManager.GetCameraIdList().FirstOrDefault();
                if (cameraId != null)
                {
                    cameraManager.OpenCamera(cameraId, new CameraStateCallback(this), null);
                }
            }
            catch (Exception ex)
            {
                System.Diagnostics.Debug.WriteLine($"OpenCamera Exception: {ex.Message}");
            }
        }

        private void ApplyStableSettings(CaptureRequest.Builder builder)
        {
            builder.Set(CaptureRequest.ControlAfMode, (int)ControlAFMode.Edof);
            builder.Set(CaptureRequest.ControlAeMode, (int)ControlAEMode.On);
            builder.Set(CaptureRequest.ControlAeLock, true);
            builder.Set(CaptureRequest.ControlAwbMode, (int)ControlAwbMode.Auto);
            builder.Set(CaptureRequest.ControlAwbLock, true);
        }

        /// <summary>
        /// Starts the camera preview session.
        /// </summary>
        private void StartPreviewSession()
        {
            if (cameraDevice == null || previewView == null || !previewView.IsAvailable)
                return;

            var surfaceTexture = previewView.SurfaceTexture;
            surfaceTexture.SetDefaultBufferSize(previewView.Width, previewView.Height);
            previewSurface ??= new Surface(surfaceTexture);

            try
            {
                var builder = cameraDevice.CreateCaptureRequest(CameraTemplate.Preview);
                ApplyStableSettings(builder);
                builder.AddTarget(previewSurface);

                var surfaces = new List<Surface> { previewSurface };

                cameraSession?.Close();
                cameraSession?.Dispose();
                cameraSession = null;

                cameraDevice.CreateCaptureSession(surfaces, new CameraCaptureStateCallback(this, builder, previewSurface), null);
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
        private void PrepareMediaRecorder(string filePath)
        {
            var (nativeSize, _, _) = GetOptimalVideoSize();
            var (scaledWidth, scaledHeight) = ScaleToMax(nativeSize, MaxDimension);
            var (size, w, h) = GetSafeVideoSize();

            mediaRecorder ??= new MediaRecorder(Context);
            mediaRecorder.Reset();

            mediaRecorder.SetVideoSource(VideoSource.Surface);
            mediaRecorder.SetOutputFormat(OutputFormat.Mpeg4);
            mediaRecorder.SetOutputFile(filePath);
            mediaRecorder.SetVideoEncodingBitRate(VideoBitRate);
            mediaRecorder.SetVideoFrameRate(VideoFrameRate);
            mediaRecorder.SetVideoSize(w, h);
            mediaRecorder.SetVideoEncoder(VideoEncoder.H264);

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
                previewSurface ??= new Surface(surfaceTexture);
                var recorderSurface = mediaRecorder.Surface;

                var surfaces = new List<Surface> { previewSurface, recorderSurface };

                var builder = cameraDevice.CreateCaptureRequest(CameraTemplate.Record);
                ApplyStableSettings(builder);
                builder.AddTarget(previewSurface);
                builder.AddTarget(recorderSurface);

                cameraSession?.Close();
                cameraSession?.Dispose();
                cameraSession = null;

                cameraDevice.CreateCaptureSession(surfaces, new RecordingCaptureSessionCallback(this, builder), null);

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
        public async Task StopRecording()
        {
            if (!isRecording || mediaRecorder == null)
                return;

            try
            {
                mediaRecorder.Stop();
                mediaRecorder.Reset();
                mediaRecorder = null;
                isRecording = false;

                System.Diagnostics.Debug.WriteLine($"Video saved to {videoFilePath}");

                // Ensure clean transition back to preview
                cameraSession?.Close();
                cameraSession?.Dispose();
                cameraSession = null;

                await Task.Delay(200); // small buffer time helps stability

                StartPreviewSession();
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
        private (ASize size, int width, int height) GetOptimalVideoSize()
        {
            var cameraManager = (CameraManager)Context.GetSystemService(Context.CameraService);
            var cameraId = cameraManager.GetCameraIdList()[0];
            var characteristics = cameraManager.GetCameraCharacteristics(cameraId);
            var map = (StreamConfigurationMap)characteristics.Get(CameraCharacteristics.ScalerStreamConfigurationMap);
            var sizes = map.GetOutputSizes(Java.Lang.Class.FromType(typeof(MediaRecorder)));

            var best = sizes[0];
            foreach (var size in sizes)
            {
                var aspectRatio = (float)size.Width / size.Height;
                var bestAspectRatio = (float)best.Width / best.Height;

                if (Math.Abs(size.Height - MaxDimension) < Math.Abs(best.Height - MaxDimension) &&
                    Math.Abs(aspectRatio - bestAspectRatio) < 0.01f)
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
        private (ASize widthHeight, int w, int h) GetSafeVideoSize()
        {
            var manager = (CameraManager)Context.GetSystemService(Context.CameraService);
            var cameraId = manager.GetCameraIdList()[0];
            var characteristics = manager.GetCameraCharacteristics(cameraId);
            var map = (StreamConfigurationMap)characteristics.Get(CameraCharacteristics.ScalerStreamConfigurationMap);
            var sizes = map.GetOutputSizes((int)ImageFormatType.Private);

            var safeSize = sizes.FirstOrDefault(s => s.Width == 640 && s.Height == 480) ?? sizes.Last();

            return (safeSize, safeSize.Width, safeSize.Height);
        }

        /// <summary>
        /// Scales the given size to the maximum dimension while preserving the aspect ratio.
        /// </summary>
        /// <param name="originalSize">The original size.</param>
        /// <param name="maxDimension">The maximum dimension.</param>
        /// <returns>A tuple containing the scaled width and height.</returns>
        private (int scaledWidth, int scaledHeight) ScaleToMax(ASize originalSize, int maxDimension)
        {
            var originalWidth = originalSize.Width;
            var originalHeight = originalSize.Height;

            var aspect = (float)originalWidth / originalHeight;

            if (originalWidth >= originalHeight)
            {
                if (originalWidth <= maxDimension)
                    return (originalWidth, originalHeight);

                var newWidth = maxDimension;
                var newHeight = (int)(newWidth / aspect);
                return (newWidth, newHeight);
            }
            else
            {
                if (originalHeight <= maxDimension)
                    return (originalWidth, originalHeight);

                var newHeight = maxDimension;
                var newWidth = (int)(newHeight * aspect);
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
                previewSurface?.Release();
                previewSurface = null;
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
