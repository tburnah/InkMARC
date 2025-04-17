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
using Microsoft.Maui.Graphics.Platform;

namespace InkMARC.Evaluate.Platforms.Android
{
    /// <summary>
    /// Handles the camera preview and recording functionality for Android.
    /// </summary>
    public class CameraPreviewHandler : ViewHandler<CameraPreview, TextureView>, IDisposable
    {
        private CameraDevice? cameraDevice;
        private CameraCaptureSession? cameraSession;
        private TextureView? previewView;
        private MediaRecorder? mediaRecorder;
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
            inferenceLoop?.Cancel();
            Dispose();            
        }

        private CancellationTokenSource? inferenceLoop;
        private void StartInferenceLoop()
        {
            inferenceLoop = new CancellationTokenSource();
            Task.Run(async () =>
            {
                var runner = new OnnxModelRunner(Context, previewView.Width, previewView.Height);

                while (!inferenceLoop.IsCancellationRequested)
                {
                    if (previewView is not null && previewView.IsAvailable)
                    {
                        var bitmap = previewView.Bitmap;
                        if (bitmap != null)
                        {
                            var image = new PlatformImage(bitmap);
                            var result = runner.Predict(image);
                            MainThread.BeginInvokeOnMainThread(() =>
                            {
                                (VirtualView as CameraPreview)?.OnInferenceResult?.Invoke(result);
                            });
                            bitmap.Recycle();
                        }
                    }
                    else
                    {
                        System.Diagnostics.Debug.WriteLine("Preview view is not available.");
                    }
                }
            }, inferenceLoop.Token);
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

                if (width > 0 && height > 0)
                    handler.StartInferenceLoop(); // Start inference loop
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
