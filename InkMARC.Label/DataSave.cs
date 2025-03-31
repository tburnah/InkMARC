using HDF5CSharp;
using InkMARC.Models.Primatives;
using System;
using System.Drawing;
using System.Drawing.Imaging;

namespace InkMARC.Label
{
    public class DataSave
    {
        private static string fileName = "dataset.h5";
        private static long fileId = 0;
        private static bool isFileOpen = false;

        // Our chunked dataset objects.
        private static ChunkedDataset<float>? imageChunked = null;
        private static ChunkedDataset<int>? attributeChunked = null;
        private static bool datasetsInitialized = false;

        /// <summary>
        /// Creates (or re‐creates) the HDF5 file.
        /// </summary>
        public static void CreateFile(string name)
        {
            if (isFileOpen)
            {
                Hdf5.CloseFile(fileId);
                isFileOpen = false;
            }
            fileName = name;
            fileId = Hdf5.CreateFile(fileName);
            isFileOpen = true;
        }

        /// <summary>
        /// Converts a 3‑D image array (width, height, channels) into a 2‑D array (1, width*height*channels)
        /// so that it can be used with the ChunkedDataset constructor.
        /// </summary>
        public static float[,] FlattenImage(float[,,] imageData)
        {
            int width = imageData.GetLength(0);
            int height = imageData.GetLength(1);
            int channels = imageData.GetLength(2);
            float[,] flattened = new float[1, width * height * channels];
            int index = 0;
            for (int x = 0; x < width; x++)
            {
                for (int y = 0; y < height; y++)
                {
                    for (int c = 0; c < channels; c++)
                    {
                        flattened[0, index++] = imageData[x, y, c];
                    }
                }
            }
            return flattened;
        }

        /// <summary>
        /// Initializes the chunked datasets for images and the boolean “touch” attribute.
        /// Since the ChunkedDataset constructor only accepts a 2-D array, we must flatten our data.
        /// This method should be called with the very first frame.
        /// </summary>
        public static bool InitializeChunkedDatasets(Bitmap firstBitmap, bool touched, InkMARCPoint firstPoint)
        {
            if (!isFileOpen)
                return false;

            float[,,] firstImage3D = BitmapToFloatArray(firstBitmap);
            float[,] firstImage2D = FlattenImage(firstImage3D);

            int[,] firstAttributes = new int[1, 3]
            {
                { touched ? 1 : 0, (int)firstPoint.X, (int)firstPoint.Y }
            };

            imageChunked = new ChunkedDataset<float>("/images", fileId, firstImage2D);
            attributeChunked = new ChunkedDataset<int>("/attributes", fileId, firstAttributes);

            datasetsInitialized = true;
            return true;
        }

        public static bool WriteFrameEx(Bitmap bitmap, bool touched, InkMARCPoint point)
        {
            if (!isFileOpen || !datasetsInitialized)
                return false;

            float[,,] image3D = BitmapToFloatArray(bitmap);
            float[,] image2D = FlattenImage(image3D);

            int[,] attributeData = new int[1, 3]
            {
                { touched ? 1 : 0, (int)point.X, (int)point.Y }
            };

            imageChunked?.AppendDataset(image2D);
            attributeChunked?.AppendDataset(attributeData);

            return true;
        }

        /// <summary>
        /// Appends a new frame and its corresponding boolean “touch” attribute to the datasets.
        /// The image is flattened to a 2-D array of shape [1, width*height*channels],
        /// and the attribute is wrapped in a [1,1] array.
        /// </summary>
        public static bool WriteFrame(Bitmap bitmap, bool touched)
        {
            if (!isFileOpen || !datasetsInitialized)
                return false;

            float[,,] image3D = BitmapToFloatArray(bitmap);
            float[,] image2D = FlattenImage(image3D);

            bool[,] attributeData = new bool[1, 1] { { touched } };

            // Append the flattened image and attribute to the chunked datasets.
            imageChunked?.AppendDataset(image2D);
            attributeChunked?.AppendDataset(attributeData);

            return true;
        }

        /// <summary>
        /// Determines if the given point represents a touch.
        /// In this example, a touch is defined as any point with Pressure > 0.
        /// </summary>
        public static bool IsTouched(InkMARCPoint point)
        {
            return point.Pressure > 0f;
        }

        /// <summary>
        /// Converts a Bitmap to a 3‑D float array [width, height, 3] with normalized RGB values.
        /// Assumes the bitmap is in BGR (or BGRA) format.
        /// </summary>
        public static float[,,] BitmapToFloatArray(Bitmap bitmap)
        {
            int width = bitmap.Width;
            int height = bitmap.Height;
            float[,,] result = new float[width, height, 3];

            BitmapData bitmapData = bitmap.LockBits(
                new System.Drawing.Rectangle(0, 0, width, height),
                ImageLockMode.ReadOnly,
                bitmap.PixelFormat);

            int bytesPerPixel = Image.GetPixelFormatSize(bitmap.PixelFormat) / 8;
            int stride = bitmapData.Stride;
            IntPtr scan0 = bitmapData.Scan0;

            unsafe
            {
                byte* pixelData = (byte*)scan0;
                for (int y = 0; y < height; y++)
                {
                    for (int x = 0; x < width; x++)
                    {
                        byte* pixel = pixelData + y * stride + x * bytesPerPixel;
                        result[x, y, 0] = pixel[2] / 255f; // Red
                        result[x, y, 1] = pixel[1] / 255f; // Green
                        result[x, y, 2] = pixel[0] / 255f; // Blue
                    }
                }
            }

            bitmap.UnlockBits(bitmapData);
            return result;
        }

        /// <summary>
        /// Finalizes the datasets by disposing of the chunked dataset objects and closing the file.
        /// </summary>
        public static void FinalizeDatasets()
        {
            if (imageChunked != null)
            {
                imageChunked.Dispose();
                imageChunked = null;
            }
            if (attributeChunked != null)
            {
                attributeChunked.Dispose();
                attributeChunked = null;
            }
            if (isFileOpen)
            {
                Hdf5.CloseFile(fileId);
                isFileOpen = false;
            }
        }
    }
}
