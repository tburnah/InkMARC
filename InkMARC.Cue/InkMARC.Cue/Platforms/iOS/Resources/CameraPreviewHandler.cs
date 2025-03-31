#if IOS
using System;
using AVFoundation;
using CoreGraphics;
using Foundation;
using Microsoft.Maui.Handlers;
using Microsoft.Maui;
using UIKit;

namespace InkMARC.Cue.Platforms.iOS
{
    public class CameraPreviewHandler : ViewHandler<CameraPreview, UIView>
    {
        AVCaptureSession captureSession;
        AVCaptureVideoPreviewLayer previewLayer;

        public CameraPreviewHandler(IPropertyMapper mapper, CommandMapper? commandMapper = null)
            : base(mapper, commandMapper)
        {
        }

        protected override UIView CreatePlatformView()
        {
            var nativeView = new UIView
            {
                BackgroundColor = UIColor.Black
            };

            // Create the capture session
            captureSession = new AVCaptureSession
            {
                SessionPreset = AVCaptureSession.PresetHigh
            };

            // Get the default video device (back camera)
            var videoDevice = AVCaptureDevice.GetDefaultDevice(AVMediaTypes.Video);
            if (videoDevice == null)
            {
                System.Diagnostics.Debug.WriteLine("No video device found");
                return nativeView;
            }

            NSError error;
            var videoInput = new AVCaptureDeviceInput(videoDevice, out error);
            if (error != null)
            {
                System.Diagnostics.Debug.WriteLine($"Error creating video input: {error.LocalizedDescription}");
                return nativeView;
            }

            // Add input to the capture session if possible
            if (captureSession.CanAddInput(videoInput))
            {
                captureSession.AddInput(videoInput);
            }
            else
            {
                System.Diagnostics.Debug.WriteLine("Could not add video input to the session.");
                return nativeView;
            }

            // Create and configure the preview layer
            previewLayer = new AVCaptureVideoPreviewLayer(captureSession)
            {
                Frame = nativeView.Bounds,
                VideoGravity = AVLayerVideoGravity.ResizeAspectFill
            };

            nativeView.Layer.AddSublayer(previewLayer);

            // Optionally update the preview layer's frame when the view resizes
            nativeView.AutoresizingMask = UIViewAutoresizing.FlexibleWidth | UIViewAutoresizing.FlexibleHeight;

            // Start the capture session
            captureSession.StartRunning();

            return nativeView;
        }

        protected override void DisconnectHandler(UIView platformView)
        {
            base.DisconnectHandler(platformView);

            // Clean up
            if (captureSession?.Running == true)
            {
                captureSession.StopRunning();
            }
            captureSession?.Dispose();
            previewLayer?.RemoveFromSuperLayer();
            previewLayer?.Dispose();
        }
    }
}
#endif
