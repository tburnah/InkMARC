using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Android.Graphics;
using Android.Content;
using Microsoft.Maui.Graphics.Platform;
using Java.Nio;
using Rect = Android.Graphics.Rect;
using System.Diagnostics;
using Color = Android.Graphics.Color;

namespace InkMARC.Evaluate.Platforms.Android
{
    public class OnnxModelRunner : IModelRunner, IDisposable
    {
        private readonly InferenceSession session;

        public OnnxModelRunner(Context context)
        {
            Debug.WriteLine("Loading model...");
            try
            {
                using var assetStream = context.Assets.Open("resnet18_pytorch.onnx");
                using var ms = new MemoryStream();
                assetStream.CopyTo(ms);
                session = new InferenceSession(ms.ToArray());
            }
            catch (Exception ex)
            {
                Debug.WriteLine($"Error loading model: {ex.Message}");
                throw;
            }
        }

        public float Predict(Microsoft.Maui.Graphics.IImage image)
        {
            Debug.WriteLine("Predicting...");
            return DoPredict(image);
        }

        public static Bitmap PadToSquare(Bitmap original)
        {
            int width = original.Width;
            int height = original.Height;

            int size = Math.Max(width, height); // square size
            int padX = (size - width) / 2;
            int padY = (size - height) / 2;

            var paddedBitmap = Bitmap.CreateBitmap(size, size, Bitmap.Config.Argb8888);
            var canvas = new Canvas(paddedBitmap);
            canvas.DrawColor(Color.Black); // Black padding background
            canvas.DrawBitmap(original, padX, padY, null);

            return paddedBitmap;
        }

        public unsafe float DoPredict(Microsoft.Maui.Graphics.IImage image)
        {
            var stopwatch = Stopwatch.StartNew();

            const int targetSize = 448;

            var platformImage = image.ToPlatformImage() as PlatformImage;
            var bitmap = platformImage?.AsBitmap();

            var padded = PadToSquare(bitmap);
            var resized = Bitmap.CreateScaledBitmap(padded, 448, 448, true);

            if (resized == null)
                throw new InvalidOperationException("Failed to convert IImage to Bitmap.");

            var input = new DenseTensor<float>(new[] { 1, 3, targetSize, targetSize });

            // Lock the pixels
            var rect = new Rect(0, 0, targetSize, targetSize);
            var config = Bitmap.Config.Argb8888;

            using var lockedBitmap = resized.Copy(config, false);
            var buffer = ByteBuffer.AllocateDirect(targetSize * targetSize * 4);
            lockedBitmap.CopyPixelsToBuffer(buffer);
            buffer.Rewind();

            byte* pixelPtr = (byte*)buffer.GetDirectBufferAddress().ToPointer();

            for (int y = 0; y < targetSize; y++)
            {
                for (int x = 0; x < targetSize; x++)
                {
                    int offset = (y * targetSize + x) * 4;
                    byte a = pixelPtr[offset + 0]; // alpha, usually ignored
                    byte r = pixelPtr[offset + 1];
                    byte g = pixelPtr[offset + 2];
                    byte b = pixelPtr[offset + 3];

                    input[0, 0, y, x] = r / 255f;
                    input[0, 1, y, x] = g / 255f;
                    input[0, 2, y, x] = b / 255f;
                }
            }

            var inputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor("input", input)
            };

            stopwatch.Stop();
            Debug.WriteLine($"Time taken to prepare input: {stopwatch.ElapsedMilliseconds} ms");
            stopwatch.Restart();

            using var results = session.Run(inputs);
            stopwatch.Stop();
            Debug.WriteLine($"Time taken for inference: {stopwatch.ElapsedMilliseconds} ms");
            return results.First().AsEnumerable<float>().First();
        }

        public void Dispose() => session.Dispose();
    }
}
