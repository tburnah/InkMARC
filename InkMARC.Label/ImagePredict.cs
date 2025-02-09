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
    public class ImagePredict
    {
        private readonly InferenceSession _session;

        public ImagePredict()
        {
            string modelPath = "C:\\Users\\tburnah\\OneDrive - Massey University\\Desktop\\model2.onnx"; 
            _session = new InferenceSession(modelPath);
        }
        public float PredictPressure(Bitmap image)
        {
            var imageTensor = ConvertBitmapToTensor(image);

            var inputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor("args_0", imageTensor) // Adjust input name based on your model
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

            int targetWidth = 224;
            int targetHeight = 224;

            // Resize the image to 224x224
            using Bitmap resizedBitmap = new Bitmap(targetWidth, targetHeight);
            using (Graphics graphics = Graphics.FromImage(resizedBitmap))
            {
                graphics.InterpolationMode = InterpolationMode.HighQualityBicubic;
                graphics.DrawImage(bitmap, 0, 0, targetWidth, targetHeight);
            }

            var tensorData = new float[1 * targetHeight * targetWidth * 3]; // Flattened array

            int index = 0;
            for (int y = 0; y < targetHeight; y++)
            {
                for (int x = 0; x < targetWidth; x++)
                {
                    Color pixel = resizedBitmap.GetPixel(x, y);
                    tensorData[index++] = pixel.R / 255f; // Red
                    tensorData[index++] = pixel.G / 255f; // Green
                    tensorData[index++] = pixel.B / 255f; // Blue
                }
            }

            return new DenseTensor<float>(tensorData, new[] { 1, targetHeight, targetWidth, 3 });
        }
    }
}
