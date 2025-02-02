using HDF5CSharp;
using InkMARC.Models.Primatives;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Imaging;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace InkMARC.Prepare
{
    public class DataSave
    {
        private static string fileName = "dataset.h5";

        private static long fileId = 0;

        private static bool isFileOpen = false;

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

        public static void CloseFile()
        {
            if (isFileOpen)
            {
                Hdf5.CloseFile(fileId);
                isFileOpen = false;
            }
        }

        public static bool AddRecord(string recordName, Bitmap bitmap, InkMARCPoint point)
        {
            if (!isFileOpen)
                return false;

            string imageRecordName = recordName + "_image";
            string attributeRecordName = recordName + "_attribute";

            long groupId = Hdf5.CreateOrOpenGroup(fileId, "images");
            Hdf5.WriteDataset(groupId, imageRecordName, BitmapToFloatArray(bitmap));

            Hdf5.CloseGroup(groupId);
            groupId = Hdf5.CreateOrOpenGroup(fileId, "attributes");

            Hdf5.WriteDataset(groupId, attributeRecordName, InkMARCPointNormalized(point));
            Hdf5.CloseGroup(groupId);

            return true;
        }

        private static long imageGroupId; 
        private static long attributeGroupId;

        public static bool StartRecordBatch()
        {
            if (!isFileOpen)
                return false;
            imageGroupId = Hdf5.CreateOrOpenGroup(fileId, "images");
            attributeGroupId = Hdf5.CreateOrOpenGroup(fileId, "attributes");
            return true;
        }

        public static bool AddIndividualRecordBatch(string recordName, float[,,] bitmap, float[] point)
        {
            if (!isFileOpen)
                return false;
            string imageRecordName = recordName + "_image";
            string attributeRecordName = recordName + "_attribute";
            Hdf5.WriteDataset(imageGroupId, imageRecordName, bitmap);
            Hdf5.WriteDataset(attributeGroupId, attributeRecordName, point);
            return true;
        }

        public static bool EndRecordBatch()
        {
            if (!isFileOpen)
                return false;
            Hdf5.CloseGroup(imageGroupId);
            Hdf5.CloseGroup(attributeGroupId);
            return true;
        }

        public static bool AddRecordBatch(string recordName, List<(Bitmap, InkMARCPoint)> records)
        {
            if (!isFileOpen)
                return false;

            long imageGroupId = Hdf5.CreateOrOpenGroup(fileId, "images");
            long attributeGroupId = Hdf5.CreateOrOpenGroup(fileId, "attributes");

            foreach (var (bitmap, point) in records)
            {
                string imageRecordName = recordName + "_image";
                string attributeRecordName = recordName + "_attribute";

                Hdf5.WriteDataset(imageGroupId, imageRecordName, BitmapToFloatArray(bitmap));
                Hdf5.WriteDataset(attributeGroupId, attributeRecordName, InkMARCPointNormalized(point));
            }

            Hdf5.CloseGroup(imageGroupId);
            Hdf5.CloseGroup(attributeGroupId);

            return true;
        }

        public static float[] InkMARCPointNormalized(InkMARCPoint point)
        {
            return [point.X / 1089, point.Y / 750, point.Pressure, point.TiltX / 90, point.TiltY];
        }

        public static float[,,] BitmapToFloatArray(Bitmap bitmap)
        {
            int width = bitmap.Width;
            int height = bitmap.Height;
            float[,,] result = new float[width, height, 3];

            BitmapData bitmapData = bitmap.LockBits(new Rectangle(0, 0, width, height), ImageLockMode.ReadOnly, bitmap.PixelFormat);
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
                        result[x, y, 0] = pixel[2] / 255f; // R
                        result[x, y, 1] = pixel[1] / 255f; // G
                        result[x, y, 2] = pixel[0] / 255f; // B
                    }
                }
            }

            bitmap.UnlockBits(bitmapData);
            return result;
        }

        //private static float[,,] BitmapToFloatArray(Bitmap bitmap)
        //{
        //    float[,,] result = new float[bitmap.Width, bitmap.Height, 3];
        //    for (int x = 0; x < bitmap.Width; x++)
        //    {
        //        for (int y = 0; y < bitmap.Height; y++)
        //        {
        //            Color color = bitmap.GetPixel(x, y);
        //            result[x, y, 0] = color.R / 255f;
        //            result[x, y, 1] = color.G / 255f;
        //            result[x, y, 2] = color.B / 255f;
        //        }
        //    }
        //    return result;
        //}
    }
}
