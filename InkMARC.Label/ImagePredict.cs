using System;
using System.Linq;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System.Drawing;
using System.Drawing.Imaging;
using System.IO;
using System.Drawing.Drawing2D;

namespace InkMARC.Label
{
    public class ImagePredict : IDisposable
    {
        private readonly InferenceSession _session;

        public ImagePredict()
        {
            string modelPath = "resnet18_pytorch_20250416_111521.onnx"; 
            _session = new InferenceSession(modelPath);
        }
        public float PredictPressure(Bitmap image)
        {
            var imageTensor = ConvertBitmapToTensor(image);

            var inputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor("input", imageTensor)
            };

            using IDisposableReadOnlyCollection<DisposableNamedOnnxValue> results = _session.Run(inputs);
            return results.First().AsTensor<float>().First();
        }

        public void Dispose()
        {
            _session.Dispose();
        }

        private static Tensor<float> ConvertBitmapToTensor(Bitmap bitmap)
        {
            int width = bitmap.Width;
            int height = bitmap.Height;

            int targetWidth = 448;
            int targetHeight = 448;

            using Bitmap resizedBitmap = new Bitmap(targetWidth, targetHeight);
            using (Graphics graphics = Graphics.FromImage(resizedBitmap))
            {
                graphics.InterpolationMode = InterpolationMode.HighQualityBicubic;
                graphics.DrawImage(bitmap, 0, 0, targetWidth, targetHeight);
            }

            var tensor = new DenseTensor<float>(new[] { 1, 3, targetHeight, targetWidth });

            for (int y = 0; y < targetHeight; y++)
            {
                for (int x = 0; x < targetWidth; x++)
                {
                    Color pixel = resizedBitmap.GetPixel(x, y);
                    tensor[0, 0, y, x] = pixel.R / 255f; // Red channel
                    tensor[0, 1, y, x] = pixel.G / 255f; // Green channel
                    tensor[0, 2, y, x] = pixel.B / 255f; // Blue channel
                }
            }

            return tensor;
        }
    }
}
