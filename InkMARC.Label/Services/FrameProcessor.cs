using OpenCvSharp;
using OpenCvSharp.WpfExtensions;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Media.Imaging;

namespace InkMARC.Label.Services
{
    /// <summary>
    /// Provides image processing utilities for video frames.
    /// </summary>
    public static class FrameProcessor
    {
        private static readonly Scalar BlackScalar = Scalar.Black;

        /// <summary>
        /// Crops the center of the input frame to a square, applies rotation, and returns a BitmapSource for WPF.
        /// </summary>
        /// <param name="input">Input frame from VideoService.</param>
        /// <param name="rotation">Rotation in degrees.</param>
        /// <returns>A BitmapSource suitable for display in WPF, or null if input is invalid.</returns>
        public static BitmapSource? Process(Mat input, double rotation)
        {
            var rotated = ProcessToMatInternal(input, rotation);
            if (rotated == null)
                return null;

            var result = BitmapSourceConverter.ToBitmapSource(rotated);
            result.Freeze();
            rotated.Dispose();
            return result;
        }

        /// <summary>
        /// Crops the center of the input frame to a square and applies rotation, returning a new Mat.
        /// </summary>
        /// <param name="input">Input frame from VideoService.</param>
        /// <param name="rotation">Rotation in degrees.</param>
        /// <returns>A new Mat with the processed image, or null if input is invalid.</returns>
        public static Mat? ProcessToMat(Mat input, double rotation)
        {
            return ProcessToMatInternal(input, rotation);
        }

        /// <summary>
        /// Shared internal logic for processing the frame.
        /// </summary>
        private static Mat? ProcessToMatInternal(Mat input, double rotation)
        {
            if (input == null || input.Empty())
                return null;

            int width = input.Width;
            int height = input.Height;
            int squareSize = Math.Max(width, height);

            // Allocate square frame only once
            var squareFrame = new Mat(new Size(squareSize, squareSize), input.Type(), BlackScalar);
            int xOffset = (squareSize - width) >> 1;
            int yOffset = (squareSize - height) >> 1;
            var roi = new Rect(xOffset, yOffset, width, height);
            using (var roiMat = new Mat(squareFrame, roi))
            {
                input.CopyTo(roiMat);
            }

            var center = new Point2f(squareSize * 0.5f, squareSize * 0.5f);
            using var rotationMatrix = Cv2.GetRotationMatrix2D(center, rotation, 1.0);
            var rotatedFrame = new Mat();
            Cv2.WarpAffine(squareFrame, rotatedFrame, rotationMatrix, new Size(squareSize, squareSize), InterpolationFlags.Linear, BorderTypes.Constant, BlackScalar);
            squareFrame.Dispose();
            return rotatedFrame;
        }
    }
}
