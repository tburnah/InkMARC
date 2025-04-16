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

        // Reusable buffers
        private DenseTensor<float> inputTensor;
        private ByteBuffer? buffer;
        private Bitmap? reusableBitmap;

        private const int TargetSize = 448;

        public OnnxModelRunner(Context context)
        {
            Debug.WriteLine("Loading model...");
            try
            {
                using var assetStream = context.Assets.Open("resnet18_pytorch_20250416_111521.onnx");
                using var ms = new MemoryStream();
                assetStream.CopyTo(ms);
                session = new InferenceSession(ms.ToArray());

                // Initialize reusable tensor and buffer
                inputTensor = new DenseTensor<float>(new[] { 1, 3, TargetSize, TargetSize });
                buffer = ByteBuffer.AllocateDirect(TargetSize * TargetSize * 4);
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

            if (width == height)
                return original;

            int size = Math.Max(width, height);
            int padX = (size - width) / 2;
            int padY = (size - height) / 2;

            var paddedBitmap = Bitmap.CreateBitmap(size, size, Bitmap.Config.Argb8888);
            var canvas = new Canvas(paddedBitmap);
            canvas.DrawColor(Color.Black);
            canvas.DrawBitmap(original, padX, padY, null);

            return paddedBitmap;
        }

        public unsafe float DoPredict(Microsoft.Maui.Graphics.IImage image)
        {
            var stopwatch = Stopwatch.StartNew();

            var platformImage = image.ToPlatformImage() as PlatformImage;
            var bitmap = platformImage?.AsBitmap() ?? throw new InvalidOperationException("Failed to get bitmap.");

            Bitmap padded = PadToSquare(bitmap);

            // Only create the resized bitmap once
            if (reusableBitmap == null || reusableBitmap.Width != TargetSize || reusableBitmap.Height != TargetSize)
            {
                reusableBitmap?.Recycle();
                reusableBitmap = Bitmap.CreateBitmap(TargetSize, TargetSize, Bitmap.Config.Argb8888);
            }

            // Resize onto reusableBitmap
            var canvas = new Canvas(reusableBitmap);
            var srcRect = new Rect(0, 0, padded.Width, padded.Height);
            var dstRect = new Rect(0, 0, TargetSize, TargetSize);
            canvas.DrawBitmap(padded, srcRect, dstRect, null);

            // Use preallocated buffer
            buffer!.Rewind();
            reusableBitmap.CopyPixelsToBuffer(buffer);
            buffer.Rewind();

            byte* pixelPtr = (byte*)buffer.GetDirectBufferAddress().ToPointer();

            for (int y = 0; y < TargetSize; y++)
            {
                for (int x = 0; x < TargetSize; x++)
                {
                    int offset = (y * TargetSize + x) * 4;
                    byte a = pixelPtr[offset + 0];
                    byte r = pixelPtr[offset + 1];
                    byte g = pixelPtr[offset + 2];
                    byte b = pixelPtr[offset + 3];

                    inputTensor[0, 0, y, x] = r / 255f;
                    inputTensor[0, 1, y, x] = g / 255f;
                    inputTensor[0, 2, y, x] = b / 255f;
                }
            }

            stopwatch.Stop();
            Debug.WriteLine($"Time taken to prepare input: {stopwatch.ElapsedMilliseconds} ms");
            stopwatch.Restart();

            using var results = session.Run(new[] {
                NamedOnnxValue.CreateFromTensor("input", inputTensor)
            });

            stopwatch.Stop();
            Debug.WriteLine($"Time taken for inference: {stopwatch.ElapsedMilliseconds} ms");

            return results.First().AsEnumerable<float>().First();
        }

        public void Dispose()
        {
            session.Dispose();
            reusableBitmap?.Recycle();
        }
    }
}
