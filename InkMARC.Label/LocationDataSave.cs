using HDF5CSharp;
using InkMARC.Models.Primatives;
using OpenCvSharp;
using System;
using System.Drawing;
using System.Drawing.Imaging;
using System.Runtime.InteropServices;


namespace InkMARC.Label
{
    public class LocationDataSave
    {
        private static readonly object Gate = new();
        private static long fileId = 0;
        private static bool isOpen = false;
        private static bool initialized = false;

        // Our chunked dataset objects
        private static ChunkedDataset<byte>? images;    // [N, 448*448*3]
        private static ChunkedDataset<float>? xy;       // [N, 2]
        private static ChunkedDataset<byte>? touch;     // [N, 1]
        private static ChunkedDataset<float>? bounds;   // [N, 8]
        private static ChunkedDataset<float>? tilt;     // [N, 2]

        // Reusable buffers to avoid allocations per frame
        private static byte[]? rgbFlat;     // 448*448*3
        private static byte[,]? imgRow2D;   // [1, K]
        private static float[,]? xyRow;     // [1,2]
        private static byte[,]? touchRow;   // [1,1]
        private static float[,]? boundsRow; // [1,8]
        private static float[,]? tiltRow;   // [1,2]

        // Constants
        private const int W = 448, H = 448, C = 3;
        private const int ImageFlatLen = W * H * C;

        /// <summary>
        /// Creates (or re‐creates) the HDF5 file.
        /// </summary>
        public static void CreateFile(string name)
        {
            lock (Gate)
            {
                CloseIfOpen();
                fileId = Hdf5.CreateFile(name);
                isOpen = true;

                // File-Level metadata                
                Hdf5.WriteAttribute(fileId, "image_width", W);
                Hdf5.WriteAttribute(fileId, "image_height", H);
                Hdf5.WriteAttribute(fileId, "channels", C);
                Hdf5.WriteAttribute(fileId, "color_order", "RGB");
                Hdf5.WriteAttribute(fileId, "coord_origin", "topleft");
                Hdf5.WriteAttribute(fileId, "bounds_order", "TL,TR,BR,BL");
            }
        }

        /// <summary>
        /// Call once with the first frame to define datasets, chunking & compression.
        /// </summary>
        /// <param name="firstBgrFrame"></param>
        /// <param name="firstTouched"></param>
        /// <param name="firstXY"></param>
        /// <param name="tl"></param>
        /// <param name="tr"></param>
        /// <param name="br"></param>
        /// <param name="bl"></param>
        /// <param name="firstTilt"></param>
        /// <returns></returns>
        public static bool Initialize(Mat firstBgrFrame, bool firstTouched, (float x, float y) firstXY,
                                      (float x, float y) tl, (float x, float y) tr,
                                      (float x, float y) br, (float x, float y) bl,
                                      (float tx, float ty) firstTilt)
        {
            lock (Gate)
            {
                if (!isOpen || initialized) return false;

                // Allocate reuse buffers
                rgbFlat = new byte[ImageFlatLen];
                imgRow2D = new byte[1, ImageFlatLen];
                xyRow = new float[1, 2];
                touchRow = new byte[1, 1];
                boundsRow = new float[1, 8];
                tiltRow = new float[1, 2];

                // Convert first frame BGR->RGB -> byte[]
                Mat rgb = new();
                Cv2.CvtColor(firstBgrFrame, rgb, ColorConversionCodes.BGR2RGB);
                Marshal.Copy(rgb.Data, rgbFlat, 0, ImageFlatLen);
                Buffer.BlockCopy(rgbFlat, 0, imgRow2D, 0, ImageFlatLen);

                // Fill attribute rows
                xyRow[0, 0] = firstXY.x; xyRow[0, 1] = firstXY.y;
                touchRow[0, 0] = (byte)(firstTouched ? 1 : 0);
                boundsRow[0, 0] = tl.x; boundsRow[0, 1] = tl.y;
                boundsRow[0, 2] = tr.x; boundsRow[0, 3] = tr.y;
                boundsRow[0, 4] = br.x; boundsRow[0, 5] = br.y;
                boundsRow[0, 6] = bl.x; boundsRow[0, 7] = bl.y;
                tiltRow![0, 0] = firstTilt.tx; tiltRow[0, 1] = firstTilt.ty;

                // Create chunked datasets (unlimited first dim)
                // Choose chunk sizes (rows per chunk)
                images = new ChunkedDataset<byte>("/images", fileId, imgRow2D);                
                xy = new ChunkedDataset<float>("/xy", fileId, xyRow);
                touch = new ChunkedDataset<byte>("/touch", fileId, touchRow);
                bounds = new ChunkedDataset<float>("/bounds", fileId, boundsRow);
                tilt = new ChunkedDataset<float>("/tilt", fileId, tiltRow);

                initialized = true;
                return true;
            }
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="bgrFrame"></param>
        /// <param name="touched"></param>
        /// <param name="xyVal"></param>
        /// <param name="tl"></param>
        /// <param name="tr"></param>
        /// <param name="br"></param>
        /// <param name="bl"></param>
        /// <param name="tiltVal"></param>
        /// <returns></returns>
        public static bool Append(Mat bgrFrame, bool touched, (float x, float y) xyVal,
                                          (float x, float y) tl, (float x, float y) tr,
                                          (float x, float y) br, (float x, float y) bl,
                                          (float tx, float ty) tiltVal)
        {
            lock (Gate)
            {
                if (!isOpen || !initialized) return false;

                // Reuse buffers
                if (rgbFlat == null || imgRow2D == null) return false;

                // Convert to RGB and flatten
                Mat rgb = new();
                Cv2.CvtColor(bgrFrame, rgb, ColorConversionCodes.BGR2RGB);
                Marshal.Copy(rgb.Data, rgbFlat, 0, ImageFlatLen);
                Buffer.BlockCopy(rgbFlat, 0, imgRow2D, 0, ImageFlatLen);

                xyRow![0, 0] = xyVal.x; xyRow[0, 1] = xyVal.y;
                touchRow![0, 0] = (byte)(touched ? 1 : 0);

                boundsRow![0, 0] = tl.x; boundsRow[0, 1] = tl.y;
                boundsRow[0, 2] = tr.x; boundsRow[0, 3] = tr.y;
                boundsRow[0, 4] = br.x; boundsRow[0, 5] = br.y;
                boundsRow[0, 6] = bl.x; boundsRow[0, 7] = bl.y;
                tiltRow![0, 0] = tiltVal.tx; tiltRow[0, 1] = tiltVal.ty;


                images!.AppendDataset(imgRow2D);
                xy!.AppendDataset(xyRow);
                touch!.AppendDataset(touchRow);
                bounds!.AppendDataset(boundsRow);
                tilt!.AppendDataset(tiltRow);

                return true;
            }
        }

        /// <summary>
        /// 
        /// </summary>
        public static void Flush()
        {
            lock (Gate)
            {
                images?.Flush();
                xy?.Flush();
                touch?.Flush();
                bounds?.Flush();
                tilt?.Flush();
            }
        }

        /// <summary>
        /// 
        /// </summary>
        public static void Close()
        {
            lock (Gate)
            {
                DisposeDatasets();
                CloseIfOpen();
            }
        }

        private static void DisposeDatasets()
        {
            images?.Dispose(); images = null;
            xy?.Dispose(); xy = null;
            touch?.Dispose(); touch = null;
            bounds?.Dispose(); bounds = null;
            tilt?.Dispose(); tilt = null;
            initialized = false;
        }

        private static void CloseIfOpen()
        {
            if (isOpen)
            {
                Hdf5.CloseFile(fileId);
                isOpen = false;
            }
        }
    }
}
