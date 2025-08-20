using HDF5CSharp;
using InkMARC.Models.Primatives;
using OpenCvSharp;
using System;
using System.Drawing;
using System.Drawing.Imaging;

namespace InkMARC.Label
{
    public class LocationDataSave
    {
        private static string fileName = "location_dataset.h5";
        private static long fileId = 0;
        private static bool isFileOpen = false;

        // Our chunked dataset objects.
        private static ChunkedDataset<float>? imageChunked = null;
        private static ChunkedDataset<float>? attributeChunked = null;
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
        public static bool InitializeChunkedDatasets(Mat firstBitmap, bool touched, InkMARCPoint firstPoint, PerspectiveBounds bounds)
        {
            if (!isFileOpen)
                return false;

            float[,,] firstImage3D = MatToFloatArray(firstBitmap);
            float[,] firstImage2D = FlattenImage(firstImage3D);

            float[,] firstAttributes = new float[1, 13]
            {
                {
                    touched ? 1f : 0f,
                    firstPoint.X,
                    firstPoint.Y,
                    firstPoint.TiltX,
                    firstPoint.TiltY,
                    bounds.First.X,
                    bounds.First.Y,
                    bounds.Second.X,
                    bounds.Second.Y,
                    bounds.Third.X,
                    bounds.Third.Y,
                    bounds.Fourth.X,
                    bounds.Fourth.Y
                }
            };

            imageChunked = new ChunkedDataset<float>("/images", fileId, firstImage2D);
            attributeChunked = new ChunkedDataset<float>("/attributes", fileId, firstAttributes);

            datasetsInitialized = true;
            return true;
        }

        public static bool WriteFrameEx(Mat bitmap, bool touched, InkMARCPoint point, PerspectiveBounds bounds)
        {
            if (!isFileOpen || !datasetsInitialized)
                return false;

            float[,,] image3D = MatToFloatArray(bitmap);
            float[,] image2D = FlattenImage(image3D);

            float[,] attributeData = new float[1, 13]
            {
                { 
                    touched ? 1 : 0, 
                    point.X, 
                    point.Y,
                    point.TiltX,
                    point.TiltY,
                    bounds.First.X,
                    bounds.First.Y,
                    bounds.Second.X,
                    bounds.Second.Y,
                    bounds.Third.X,
                    bounds.Third.Y,
                    bounds.Fourth.X,
                    bounds.Fourth.Y
                }
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
        public static bool WriteFrame(Mat bitmap, bool touched)
        {
            if (!isFileOpen || !datasetsInitialized)
                return false;

            float[,,] image3D = MatToFloatArray(bitmap);
            float[,] image2D = FlattenImage(image3D);

            bool[,] attributeData = new bool[1, 1] { { touched } };

            // Append the flattened image and attribute to the chunked datasets.
            imageChunked?.AppendDataset(image2D);
            attributeChunked?.AppendDataset(attributeData);

            return true;
        }

        public static float[,,] MatToFloatArray(Mat mat)
        {
            if (mat.Type() != MatType.CV_8UC3)
                throw new ArgumentException("Only CV_8UC3 Mats are supported");

            int width = mat.Width;
            int height = mat.Height;

            float[,,] result = new float[width, height, 3];

            unsafe
            {
                byte* data = (byte*)mat.DataPointer;

                int step = (int)mat.Step();
                for (int y = 0; y < height; y++)
                {
                    byte* row = data + y * step;
                    for (int x = 0; x < width; x++)
                    {
                        int offset = x * 3;
                        result[x, y, 0] = row[offset + 2] / 255f; // R
                        result[x, y, 1] = row[offset + 1] / 255f; // G
                        result[x, y, 2] = row[offset + 0] / 255f; // B
                    }
                }
            }

            return result;
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
