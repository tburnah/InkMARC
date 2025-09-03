using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OpenCvSharp;

namespace InkMARC.Label.Services
{
    public sealed class BoundsPredictor : IDisposable
    {
        private readonly InferenceSession _session;
        private readonly string _inputName;
        private readonly string _outputName;
        private readonly int _inH = 448, _inW = 448;

        // ImageNet normalization used in training
        private static readonly float[] Mean = { 0.485f, 0.456f, 0.406f };
        private static readonly float[] Std = { 0.229f, 0.224f, 0.225f };

        public BoundsPredictor(string onnxPath, bool useCuda = false)
        {
            var opts = new SessionOptions();
            if (useCuda)
            {
                try { opts.AppendExecutionProvider_CUDA(); } catch { /* CUDA not available */ }
            }
            _session = new InferenceSession(onnxPath, opts);
            _inputName = _session.InputMetadata.Keys.First();   // "input" if exported as above
            _outputName = _session.OutputMetadata.Keys.First();  // "output"
        }

        /// <summary>
        /// Predict TL, TR, BR, BL in pixel coordinates for the original image.
        /// </summary>
        public Point2f[] Predict(Mat bgr)
        {
            if (bgr.Empty()) throw new ArgumentException("Input Mat is empty.");

            int origH = bgr.Rows, origW = bgr.Cols;

            // 1) Convert BGR -> RGB
            using var rgb = new Mat();
            Cv2.CvtColor(bgr, rgb, ColorConversionCodes.BGR2RGB);

            // 2) Resize to model input (448x448). This matches your training size.
            using var resized = new Mat();
            Cv2.Resize(rgb, resized, new Size(_inW, _inH), 0, 0, InterpolationFlags.Area);

            // 3) Create [1,3,H,W] tensor with [0..1] and ImageNet normalization
            var input = new DenseTensor<float>(new[] { 1, 3, _inH, _inW });

            // Fill in CHW order
            for (int y = 0; y < _inH; y++)
            {
                for (int x = 0; x < _inW; x++)
                {
                    // After BGR2RGB, channels are [R,G,B]
                    var px = resized.At<Vec3b>(y, x);
                    float r = px.Item0 / 255f;
                    float g = px.Item1 / 255f;
                    float b = px.Item2 / 255f;

                    input[0, 0, y, x] = (r - Mean[0]) / Std[0]; // R
                    input[0, 1, y, x] = (g - Mean[1]) / Std[1]; // G
                    input[0, 2, y, x] = (b - Mean[2]) / Std[2]; // B
                }
            }

            // 4) Run ONNX
            using var results = _session.Run(new[]
            {
            NamedOnnxValue.CreateFromTensor(_inputName, input)
        });
            var output = results.First(v => v.Name == _outputName).AsEnumerable<float>().ToArray();
            // output length == 8, normalized [0,1]: (x1,y1,...,x4,y4) in TL,TR,BR,BL order

            // 5) Clamp and denormalize to original pixel coords
            for (int i = 0; i < 8; i++)
                output[i] = Math.Min(1f, Math.Max(0f, output[i]));

            var corners = new Point2f[4];
            corners[0] = new Point2f(output[0] * origW, output[1] * origH); // TL
            corners[1] = new Point2f(output[2] * origW, output[3] * origH); // TR
            corners[2] = new Point2f(output[4] * origW, output[5] * origH); // BR
            corners[3] = new Point2f(output[6] * origW, output[7] * origH); // BL
            return corners;
        }

        public void Dispose() => _session?.Dispose();
    }

}
