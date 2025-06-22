using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Android.Graphics;
using Android.Content;
using Microsoft.Maui.Graphics.Platform;
using Java.Nio;
using Rect = Android.Graphics.Rect;
using System.Diagnostics;
using Color = Android.Graphics.Color;
using Android.App;
using AndroidApp = Android.App.Application;
using Android.Media;
using Android.Hardware.Lights;
using static Android.Icu.Text.ListFormatter;

namespace InkMARC.Evaluate.Platforms.Android
{
    public class OnnxModelRunner : IModelRunner, IDisposable
    {
        private readonly InferenceSession session;

        // Reusable buffers
        private DenseTensor<float> inputTensor;
        private Bitmap? reusableBitmap;

        private const int TargetSize = 448;

        private readonly int[] pixelsBuffer = new int[TargetSize * TargetSize];
        private readonly float[] inputTensorFlat = new float[1 * 3 * TargetSize * TargetSize];
        private Bitmap? paddedBitmap = null;
        private Canvas? reusableCanvas;
        private int padX;
        private int padY;
        private bool square = false;
        private DenseTensor<float>? tensorFromFlat;
        private int planeSize;
        public OnnxModelRunner(Context context, int width, int height)
        {
            //Debug.WriteLine("Loading model...");
            try
            {
                //using var assetStream = context.Assets.Open("resnet18_pytorch_20250416_111521.onnx");
                using var assetStream = context.Assets.Open("resnet18_pytorch_20250416_111521_fp16_fixed.onnx");
                using var ms = new MemoryStream();
                assetStream.CopyTo(ms);

                Debug.WriteLine($"Model size in memory: {ms.Length} bytes");
                if (ms.Length == 0)
                {
                    Debug.WriteLine("WARNING: Model file is empty!");
                }

                var options = new SessionOptions();
                options.AppendExecutionProvider_Nnapi(
                    nnapiFlags: NnapiFlags.NNAPI_FLAG_USE_FP16 | NnapiFlags.NNAPI_FLAG_CPU_DISABLED
                );
                //options.AppendExecutionProvider_CPU();
                try
                {
                    var modelBytes = ms.ToArray();
                    Debug.WriteLine($"Attempting to create InferenceSession with {modelBytes.Length} bytes...");

                    session = new InferenceSession(modelBytes, options);

                    Debug.WriteLine("InferenceSession created successfully.");
                }
                catch (Exception ex)
                {
                    Debug.WriteLine($"Error setting NNAPI flags: {ex.Message}");
                    Debug.WriteLine($"StackTrace: {ex.StackTrace}");
                    throw;
                }


                
                var providers = Microsoft.ML.OnnxRuntime.OrtEnv.Instance().GetAvailableProviders();
                foreach (var provider in providers)
                {
                    Debug.WriteLine($"Execution provider: {provider}");
                }

                // Initialize reusable tensor and buffer
                inputTensor = new DenseTensor<float>(new[] { 1, 3, TargetSize, TargetSize });

                square = width == height;

                int size = Math.Max(width, height);
                padX = (size - width) / 2;
                padY = (size - height) / 2;
                paddedBitmap = Bitmap.CreateBitmap(size, size, Bitmap.Config.Argb8888);
                reusableCanvas = new Canvas(paddedBitmap);
                reusableCanvas.DrawColor(Color.Black);

                tensorFromFlat = new DenseTensor<float>(inputTensorFlat, new int[] { 1, 3, TargetSize, TargetSize });
                planeSize = TargetSize * TargetSize;

                //var activityManager = (ActivityManager)AndroidApp.Context.GetSystemService(Context.ActivityService);
                //var configInfo = activityManager.DeviceConfigurationInfo;
                //int glEsVersion = configInfo.ReqGlEsVersion;

                ////string version = $"OpenGL ES Version: {((glEsVersion >> 16) & 0xFFFF)}.{(glEsVersion & 0xFFFF)}";
                ////System.Diagnostics.Debug.WriteLine(version);
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

        // The image will be the same size always, so once padded and the black is drawn, we don't need to redo.

        public Bitmap PadToSquare(Bitmap original)
        {
            if (square)
                return original;

                reusableCanvas?.DrawBitmap(original, padX, padY, null);

                return paddedBitmap;
        }

        private static long GetCpuTimeMillis()
        {
            try
            {
                var stat = System.IO.File.ReadAllText("/proc/self/stat");
                var parts = stat.Split(' ');

                // According to procfs:
                // parts[13] = utime (user mode jiffies)
                // parts[14] = stime (kernel mode jiffies)
                long utime = long.Parse(parts[13]);
                long stime = long.Parse(parts[14]);

                // Jiffies to milliseconds
                long jiffies = utime + stime;

                // Most Android systems use 100Hz clock, so:
                long milliseconds = (jiffies * 1000) / 100;

                return milliseconds;
            }
            catch (Exception ex)
            {
                Debug.WriteLine($"Failed to get CPU time: {ex.Message}");
                return 0;
            }
        }

        public float DoPredict(Microsoft.Maui.Graphics.IImage image)
        {
            Debug.WriteLine("Predicting Start");
            var platformImage = image.ToPlatformImage() as PlatformImage;
            var bitmap = platformImage?.AsBitmap() ?? throw new InvalidOperationException("Failed to get bitmap.");

            Bitmap padded = PadToSquare(bitmap);

            // Create reusable resized bitmap once
            if (reusableBitmap == null || reusableBitmap.Width != TargetSize || reusableBitmap.Height != TargetSize)
            {
                reusableBitmap?.Recycle();
                reusableBitmap = Bitmap.CreateBitmap(TargetSize, TargetSize, Bitmap.Config.Argb8888);
            }

            // Resize padded image onto reusableBitmap
            var canvas = new Canvas(reusableBitmap);
            var srcRect = new Rect(0, 0, padded.Width, padded.Height);
            var dstRect = new Rect(0, 0, TargetSize, TargetSize);
            canvas.DrawBitmap(padded, srcRect, dstRect, null);

            // Use GetPixels instead of ByteBuffer and unsafe pointer logic            
            reusableBitmap.GetPixels(pixelsBuffer, 0, TargetSize, 0, 0, TargetSize, TargetSize);

            Parallel.For(0, TargetSize, y =>
            {
                int rowOffset = y * TargetSize;
                for (int x = 0; x < TargetSize; x++)
                {
                    int pixel = pixelsBuffer[rowOffset + x];
                    float r = ((pixel >> 16) & 0xFF) / 255f;
                    float g = ((pixel >> 8) & 0xFF) / 255f;
                    float b = (pixel & 0xFF) / 255f;
                    int baseOffset = rowOffset + x;
                    inputTensorFlat[0 * planeSize + baseOffset] = r;
                    inputTensorFlat[1 * planeSize + baseOffset] = g;
                    inputTensorFlat[2 * planeSize + baseOffset] = b;
                }
            });

            Debug.WriteLine("Image Prepared");

            var activityManager = (ActivityManager)AndroidApp.Context.GetSystemService(Context.ActivityService);
            var memoryInfoBefore = new ActivityManager.MemoryInfo();
            activityManager.GetMemoryInfo(memoryInfoBefore);
            long memoryBeforeMB = memoryInfoBefore.AvailMem / 1048576L;

            var stopwatch = Stopwatch.StartNew();
            long cpuTimeBefore = GetCpuTimeMillis();

            Debug.WriteLine("Session will Run");
            //tensorFromFlat = new DenseTensor<float>(inputTensorFlat, new int[] { 1, 3, TargetSize, TargetSize });
            //var tensorFromFlat = new DenseTensor<float>(inputTensorFlat, new int[] { 1, 3, TargetSize, TargetSize });
            using var results = session.Run(new[] {NamedOnnxValue.CreateFromTensor("input", tensorFromFlat)});

            Debug.WriteLine("Session Ran");

            stopwatch.Stop();
            long cpuTimeAfter = GetCpuTimeMillis();

            var memoryInfoAfter = new ActivityManager.MemoryInfo();
            activityManager.GetMemoryInfo(memoryInfoAfter);
            long memoryAfterMB = memoryInfoAfter.AvailMem / 1048576L;

            long memoryUsedMB = memoryBeforeMB - memoryAfterMB;
            long cpuTimeUsedMs = cpuTimeAfter - cpuTimeBefore;
            long wallClockTimeMs = stopwatch.ElapsedMilliseconds;

            Debug.WriteLine($"Time taken for inference (wall-clock): {wallClockTimeMs} ms");
            Debug.WriteLine($"CPU time used for inference: {cpuTimeUsedMs} ms");
            Debug.WriteLine($"Memory change during inference: {memoryUsedMB} MB");

            return results.First().AsEnumerable<float>().First();
        }

        public void Dispose()
        {
            session.Dispose();
            reusableBitmap?.Recycle();
        }
    }
}
