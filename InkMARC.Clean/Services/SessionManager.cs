using HDF.PInvoke;
using OpenCvSharp;
using System;
using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Text;

namespace InkMARC.Clean.Services
{
    public sealed class SessionManager : IDisposable
    {
        // Dataset names (metadata only)
        private const string CornersName = "corners";         // [T,4,2] float32
        private const string LabelsName = "labels";           // [T,A] float32
        private const string LabelMaskName = "label_mask";    // [T,A] uint8
        private const string TimestampsName = "timestamps_ns";// [T] uint64

        // File attributes (recommended)
        private const string AttrWidth = "width";
        private const string AttrHeight = "height";
        private const string AttrChannels = "channels";
        private const string AttrFrameCount = "frame_count";
        private const string AttrAttrCount = "attr_count";
        private const string AttrFps = "fps";        

        // HDF5 handles
        private long _fileId = -1;
        private long _cornersId = -1;
        private long _labelsId = -1;
        private long _labelMaskId = -1;
        private long _timestampsId = -1;

        // Metadata
        public ulong FrameCount { get; private set; }
        public int Height { get; private set; }
        public int Width { get; private set; }
        public int Channels { get; private set; } = 3;
        public int AttrCount { get; private set; }
        public int Fps { get; private set; }

        // Cached dataspaces (keep open for session life)
        private long _cornFileSpace = -1;
        private long _cornMemSpace = -1;
        private readonly ulong[] _cornStart = new ulong[3];
        private static readonly ulong[] _cornCount = { 1, 4, 2 };

        private long _labFileSpace = -1;
        private long _labMemSpace = -1;
        private readonly ulong[] _labStart = new ulong[2];
        private ulong[] _labCount = Array.Empty<ulong>();

        private long _maskFileSpace = -1;
        private long _maskMemSpace = -1;
        private readonly ulong[] _maskStart = new ulong[2];
        private ulong[] _maskCount = Array.Empty<ulong>();

        private long _tsFileSpace = -1;
        private long _tsMemSpace = -1;
        private readonly ulong[] _tsStart = new ulong[1];
        private static readonly ulong[] _tsCount = { 1 };

        private long _dxpl = H5P.DEFAULT;

        private bool _disposed;

        private SessionManager() { }

        // -------------------------------
        // Factory: create new paired session (H5 + AVI)
        // Note: Video export/import responsibilities have been moved to VideoService.
        // SessionManager now only creates HDF5 metadata datasets and stores the video path as an attribute.
        // -------------------------------
        public static SessionManager CreateNew(
            string h5Path,            
            ulong frameCount,
            int height,
            int width,
            int attrCount,
            int fps,
            int chunkFrames = 256)
        {
            if (frameCount == 0) throw new ArgumentOutOfRangeException(nameof(frameCount));
            if (height <= 0) throw new ArgumentOutOfRangeException(nameof(height));
            if (width <= 0) throw new ArgumentOutOfRangeException(nameof(width));
            if (attrCount <= 0) throw new ArgumentOutOfRangeException(nameof(attrCount));
            if (fps <= 0) throw new ArgumentOutOfRangeException(nameof(fps));

            var mgr = new SessionManager
            {
                FrameCount = frameCount,
                Height = height,
                Width = width,
                Channels = 3,
                AttrCount = attrCount,
                Fps = fps                
            };

            mgr.CreateFileInternal(h5Path, chunkFrames);
            mgr.InitIoCaches();

            // Note: Video writer is no longer opened here. Use VideoService to create/write the AVI.

            return mgr;
        }

        // -------------------------------
        // Factory: open existing session
        // -------------------------------
        public static SessionManager OpenExisting(string h5Path, string? aviPathOverride = null, bool writeable = false)
        {
            var mgr = new SessionManager();
            mgr.OpenFileInternal(h5Path, writeable);

            mgr.InitIoCaches();

            return mgr;
        }

        // -------------------------------
        // Public API: write one frame (metadata only)
        // -------------------------------
        public void WriteFrame(
            ulong frameIndex,            
            ulong timestampNs,        // monotonic timestamp (or wallclock)
            float[] corners,          // length = 8
            float[] labels,           // length = AttrCount
            byte[] labelMask)         // length = AttrCount
        {
            EnsureNotDisposed();

            if (frameIndex >= FrameCount)
                throw new ArgumentOutOfRangeException(nameof(frameIndex));
            if (corners == null || corners.Length != 8)
                throw new ArgumentException("corners must be length 8.", nameof(corners));
            if (labels == null || labels.Length != AttrCount)
                throw new ArgumentException("labels length must equal AttrCount.", nameof(labels));
            if (labelMask == null || labelMask.Length != AttrCount)
                throw new ArgumentException("labelMask length must equal AttrCount.", nameof(labelMask));

            // 1) Write metadata to HDF5 at the given frame index
            WriteTimestamp(frameIndex, timestampNs);
            WriteCorners(frameIndex, corners);
            WriteLabels(frameIndex, labels);
            WriteLabelMask(frameIndex, labelMask);
        }

        // -------------------------------
        // Public API: read metadata row
        // -------------------------------
        public void ReadMetadata(
            ulong frameIndex,
            out ulong timestampNs,
            float[]? corners,
            float[]? labels,
            byte[]? labelMask)
        {
            EnsureNotDisposed();

            if (frameIndex >= FrameCount)
                throw new ArgumentOutOfRangeException(nameof(frameIndex));

            timestampNs = ReadTimestamp(frameIndex);

            if (corners != null)
            {
                if (corners.Length != 8) throw new ArgumentException("corners must be length 8.", nameof(corners));
                ReadCorners(frameIndex, corners);
            }

            if (labels != null)
            {
                if (labels.Length != AttrCount) throw new ArgumentException("labels length must equal AttrCount.", nameof(labels));
                ReadLabels(frameIndex, labels);
            }

            if (labelMask != null)
            {
                if (labelMask.Length != AttrCount) throw new ArgumentException("labelMask length must equal AttrCount.", nameof(labelMask));
                ReadLabelMask(frameIndex, labelMask);
            }
        }

        // -------------------------------
        // HDF5 write helpers
        // -------------------------------
        private unsafe void WriteTimestamp(ulong t, ulong timestampNs)
        {
            _tsStart[0] = t;
            var sel = H5S.select_hyperslab(_tsFileSpace, H5S.seloper_t.SET, _tsStart, null, _tsCount, null);
            if (sel < 0) throw new Exception("Failed to select hyperslab for timestamps.");

            ulong val = timestampNs;
            var status = H5D.write(_timestampsId, H5T.NATIVE_ULLONG, _tsMemSpace, _tsFileSpace, _dxpl, new IntPtr(&val));
            if (status < 0) throw new Exception("Failed to write timestamp.");
        }

        private unsafe void WriteCorners(ulong t, float[] corners)
        {
            _cornStart[0] = t; _cornStart[1] = 0; _cornStart[2] = 0;

            var sel = H5S.select_hyperslab(_cornFileSpace, H5S.seloper_t.SET, _cornStart, null, _cornCount, null);
            if (sel < 0) throw new Exception("Failed to select hyperslab for corners.");

            fixed (float* p = corners)
            {
                var status = H5D.write(_cornersId, H5T.NATIVE_FLOAT, _cornMemSpace, _cornFileSpace, _dxpl, (IntPtr)p);
                if (status < 0) throw new Exception("Failed to write corners.");
            }
        }

        private unsafe void WriteLabels(ulong t, float[] labels)
        {
            _labStart[0] = t; _labStart[1] = 0;

            var sel = H5S.select_hyperslab(_labFileSpace, H5S.seloper_t.SET, _labStart, null, _labCount, null);
            if (sel < 0) throw new Exception("Failed to select hyperslab for labels.");

            fixed (float* p = labels)
            {
                var status = H5D.write(_labelsId, H5T.NATIVE_FLOAT, _labMemSpace, _labFileSpace, _dxpl, (IntPtr)p);
                if (status < 0) throw new Exception("Failed to write labels.");
            }
        }

        private unsafe void WriteLabelMask(ulong t, byte[] mask)
        {
            _maskStart[0] = t; _maskStart[1] = 0;

            var sel = H5S.select_hyperslab(_maskFileSpace, H5S.seloper_t.SET, _maskStart, null, _maskCount, null);
            if (sel < 0) throw new Exception("Failed to select hyperslab for label_mask.");

            fixed (byte* p = mask)
            {
                var status = H5D.write(_labelMaskId, H5T.NATIVE_UCHAR, _maskMemSpace, _maskFileSpace, _dxpl, (IntPtr)p);
                if (status < 0) throw new Exception("Failed to write label mask.");
            }
        }

        // -------------------------------
        // HDF5 read helpers
        // -------------------------------
        private unsafe ulong ReadTimestamp(ulong t)
        {
            long fileSpace = H5D.get_space(_timestampsId);
            if (fileSpace < 0) throw new Exception("Failed to get /timestamps_ns dataspace.");

            try
            {
                ulong[] start = { t };
                var sel = H5S.select_hyperslab(fileSpace, H5S.seloper_t.SET, start, null, _tsCount, null);
                if (sel < 0) throw new Exception("Failed to select hyperslab for timestamps read.");

                ulong val = 0;
                var status = H5D.read(_timestampsId, H5T.NATIVE_ULLONG, _tsMemSpace, fileSpace, _dxpl, new IntPtr(&val));
                if (status < 0) throw new Exception("Failed to read timestamp.");
                return val;
            }
            finally
            {
                H5S.close(fileSpace);
            }
        }

        private unsafe void ReadCorners(ulong t, float[] buffer)
        {
            _cornStart[0] = t; _cornStart[1] = 0; _cornStart[2] = 0;

            var sel = H5S.select_hyperslab(_cornFileSpace, H5S.seloper_t.SET, _cornStart, null, _cornCount, null);
            if (sel < 0) throw new Exception("Failed to select hyperslab for corners read.");

            fixed (float* p = buffer)
            {
                var status = H5D.read(_cornersId, H5T.NATIVE_FLOAT, _cornMemSpace, _cornFileSpace, _dxpl, (IntPtr)p);
                if (status < 0) throw new Exception("Failed to read corners.");
            }
        }

        private unsafe void ReadLabels(ulong t, float[] buffer)
        {
            _labStart[0] = t; _labStart[1] = 0;

            var sel = H5S.select_hyperslab(_labFileSpace, H5S.seloper_t.SET, _labStart, null, _labCount, null);
            if (sel < 0) throw new Exception("Failed to select hyperslab for labels read.");

            fixed (float* p = buffer)
            {
                var status = H5D.read(_labelsId, H5T.NATIVE_FLOAT, _labMemSpace, _labFileSpace, _dxpl, (IntPtr)p);
                if (status < 0) throw new Exception("Failed to read labels.");
            }
        }

        private unsafe void ReadLabelMask(ulong t, byte[] buffer)
        {
            _maskStart[0] = t; _maskStart[1] = 0;

            var sel = H5S.select_hyperslab(_maskFileSpace, H5S.seloper_t.SET, _maskStart, null, _maskCount, null);
            if (sel < 0) throw new Exception("Failed to select hyperslab for label_mask read.");

            fixed (byte* p = buffer)
            {
                var status = H5D.read(_labelMaskId, H5T.NATIVE_UCHAR, _maskMemSpace, _maskFileSpace, _dxpl, (IntPtr)p);
                if (status < 0) throw new Exception("Failed to read label_mask.");
            }
        }

        // -------------------------------
        // Attributes helpers (scalar + string)
        // -------------------------------
        private static void WriteScalarAttribute(long locId, string name, int value)
        {
            long space = H5S.create(H5S.class_t.SCALAR);
            long attr = H5A.create(locId, name, H5T.NATIVE_INT, space);
            if (attr < 0) throw new Exception($"Failed to create attribute '{name}'.");

            try
            {
                unsafe
                {
                    int v = value;
                    var status = H5A.write(attr, H5T.NATIVE_INT, new IntPtr(&v));
                    if (status < 0) throw new Exception($"Failed to write attribute '{name}'.");
                }
            }
            finally
            {
                H5A.close(attr);
                H5S.close(space);
            }
        }

        private static void WriteScalarAttribute(long locId, string name, ulong value)
        {
            long space = H5S.create(H5S.class_t.SCALAR);
            long attr = H5A.create(locId, name, H5T.NATIVE_ULLONG, space);
            if (attr < 0) throw new Exception($"Failed to create attribute '{name}'.");

            try
            {
                unsafe
                {
                    ulong v = value;
                    var status = H5A.write(attr, H5T.NATIVE_ULLONG, new IntPtr(&v));
                    if (status < 0) throw new Exception($"Failed to write attribute '{name}'.");
                }
            }
            finally
            {
                H5A.close(attr);
                H5S.close(space);
            }
        }

        private static void WriteStringAttribute(long locId, string name, string value)
        {
            // Fixed-length string attribute (simple and portable)
            byte[] bytes = Encoding.UTF8.GetBytes(value);
            long type = H5T.copy(H5T.C_S1);
            H5T.set_size(type, new IntPtr(bytes.Length));
            H5T.set_strpad(type, H5T.str_t.NULLTERM);

            long space = H5S.create(H5S.class_t.SCALAR);
            long attr = H5A.create(locId, name, type, space);
            if (attr < 0) throw new Exception($"Failed to create string attribute '{name}'.");

            try
            {
                // Ensure null-terminated buffer
                byte[] nt = new byte[bytes.Length + 1];
                Buffer.BlockCopy(bytes, 0, nt, 0, bytes.Length);

                unsafe
                {
                    fixed (byte* p = nt)
                    {
                        var status = H5A.write(attr, type, (IntPtr)p);
                        if (status < 0) throw new Exception($"Failed to write string attribute '{name}'.");
                    }
                }
            }
            finally
            {
                H5A.close(attr);
                H5S.close(space);
                H5T.close(type);
            }
        }

        private int? TryReadIntAttribute(long locId, string name)
        {
            if (H5A.exists(locId, name) <= 0) return null;
            long attr = H5A.open(locId, name);
            if (attr < 0) return null;
            try
            {
                unsafe
                {
                    int v = 0;
                    var status = H5A.read(attr, H5T.NATIVE_INT, new IntPtr(&v));
                    if (status < 0) return null;
                    return v;
                }
            }
            finally
            {
                H5A.close(attr);
            }
        }

        private ulong? TryReadUlongAttribute(long locId, string name)
        {
            if (H5A.exists(locId, name) <= 0) return null;
            long attr = H5A.open(locId, name);
            if (attr < 0) return null;
            try
            {
                unsafe
                {
                    ulong v = 0;
                    var status = H5A.read(attr, H5T.NATIVE_ULLONG, new IntPtr(&v));
                    if (status < 0) return null;
                    return v;
                }
            }
            finally
            {
                H5A.close(attr);
            }
        }

        private string? TryReadStringAttribute(long locId, string name)
        {
            if (H5A.exists(locId, name) <= 0) return null;
            long attr = H5A.open(locId, name);
            if (attr < 0) return null;

            try
            {
                long atype = H5A.get_type(attr);
                long aspace = H5A.get_space(attr);

                try
                {
                    // Determine size
                    IntPtr size = H5T.get_size(atype);
                    int n = size.ToInt32();
                    if (n <= 0 || n > 64_000) return null;

                    byte[] buf = new byte[n + 1];
                    unsafe
                    {
                        fixed (byte* p = buf)
                        {
                            var status = H5A.read(attr, atype, (IntPtr)p);
                            if (status < 0) return null;
                        }
                    }

                    // Trim at null terminator
                    int end = Array.IndexOf(buf, (byte)0);
                    if (end < 0) end = buf.Length;
                    return Encoding.UTF8.GetString(buf, 0, end);
                }
                finally
                {
                    H5T.close(atype);
                    H5S.close(aspace);
                }
            }
            finally
            {
                H5A.close(attr);
            }
        }

        // -------------------------------
        // Internal: create file + datasets + attributes
        // -------------------------------
        private void CreateFileInternal(string h5Path, int chunkFrames)
        {
            _fileId = H5F.create(h5Path, H5F.ACC_TRUNC, H5P.DEFAULT, H5P.DEFAULT);
            if (_fileId < 0) throw new Exception("Failed to create HDF5 file.");

            // /corners : [T,4,2] float32
            {
                ulong[] dims = { FrameCount, 4, 2 };
                long spaceId = H5S.create_simple(3, dims, dims);
                long dcpl = H5P.create(H5P.DATASET_CREATE);

                ulong[] chunkDims = { Math.Min(FrameCount, (ulong)chunkFrames), 4, 2 };
                H5P.set_chunk(dcpl, 3, chunkDims);
                // No compression needed; you can add it later if desired

                _cornersId = H5D.create(_fileId, CornersName, H5T.NATIVE_FLOAT, spaceId, H5P.DEFAULT, dcpl, H5P.DEFAULT);
                if (_cornersId < 0) throw new Exception("Failed to create dataset 'corners'.");

                H5P.close(dcpl);
                H5S.close(spaceId);
            }

            // /labels : [T,A] float32
            {
                ulong[] dims = { FrameCount, (ulong)AttrCount };
                long spaceId = H5S.create_simple(2, dims, dims);
                long dcpl = H5P.create(H5P.DATASET_CREATE);

                ulong[] chunkDims = { Math.Min(FrameCount, (ulong)chunkFrames), (ulong)AttrCount };
                H5P.set_chunk(dcpl, 2, chunkDims);

                _labelsId = H5D.create(_fileId, LabelsName, H5T.NATIVE_FLOAT, spaceId, H5P.DEFAULT, dcpl, H5P.DEFAULT);
                if (_labelsId < 0) throw new Exception("Failed to create dataset 'labels'.");

                H5P.close(dcpl);
                H5S.close(spaceId);
            }

            // /label_mask : [T,A] uint8
            {
                ulong[] dims = { FrameCount, (ulong)AttrCount };
                long spaceId = H5S.create_simple(2, dims, dims);
                long dcpl = H5P.create(H5P.DATASET_CREATE);

                ulong[] chunkDims = { Math.Min(FrameCount, (ulong)chunkFrames), (ulong)AttrCount };
                H5P.set_chunk(dcpl, 2, chunkDims);

                _labelMaskId = H5D.create(_fileId, LabelMaskName, H5T.NATIVE_UCHAR, spaceId, H5P.DEFAULT, dcpl, H5P.DEFAULT);
                if (_labelMaskId < 0) throw new Exception("Failed to create dataset 'label_mask'.");

                H5P.close(dcpl);
                H5S.close(spaceId);
            }

            // /timestamps_ns : [T] uint64
            {
                ulong[] dims = { FrameCount };
                long spaceId = H5S.create_simple(1, dims, dims);
                long dcpl = H5P.create(H5P.DATASET_CREATE);

                ulong[] chunkDims = { Math.Min(FrameCount, (ulong)chunkFrames) };
                H5P.set_chunk(dcpl, 1, chunkDims);

                _timestampsId = H5D.create(_fileId, TimestampsName, H5T.NATIVE_ULLONG, spaceId, H5P.DEFAULT, dcpl, H5P.DEFAULT);
                if (_timestampsId < 0) throw new Exception("Failed to create dataset 'timestamps_ns'.");

                H5P.close(dcpl);
                H5S.close(spaceId);
            }

            // Write attributes for reproducibility
            WriteScalarAttribute(_fileId, AttrWidth, (int)Width);
            WriteScalarAttribute(_fileId, AttrHeight, (int)Height);
            WriteScalarAttribute(_fileId, AttrChannels, (int)Channels);
            WriteScalarAttribute(_fileId, AttrAttrCount, (int)AttrCount);
            WriteScalarAttribute(_fileId, AttrFps, (int)Fps);
            WriteScalarAttribute(_fileId, AttrFrameCount, (ulong)FrameCount);
        }

        private void OpenFileInternal(string path, bool writeable)
        {
            uint flags = writeable ? H5F.ACC_RDWR : H5F.ACC_RDONLY;
            _fileId = H5F.open(path, flags);
            if (_fileId < 0) throw new Exception("Failed to open HDF5 file.");

            _cornersId = H5D.open(_fileId, CornersName);
            _labelsId = H5D.open(_fileId, LabelsName);
            _labelMaskId = H5D.open(_fileId, LabelMaskName);
            _timestampsId = H5D.open(_fileId, TimestampsName);

            if (_cornersId < 0 || _labelsId < 0 || _labelMaskId < 0 || _timestampsId < 0)
                throw new Exception("Failed to open one or more datasets (corners/labels/label_mask/timestamps_ns).");

            // Read key attributes
            Width = TryReadIntAttribute(_fileId, AttrWidth) ?? throw new Exception("Missing HDF5 attr 'width'.");
            Height = TryReadIntAttribute(_fileId, AttrHeight) ?? throw new Exception("Missing HDF5 attr 'height'.");
            Channels = TryReadIntAttribute(_fileId, AttrChannels) ?? 3;
            AttrCount = TryReadIntAttribute(_fileId, AttrAttrCount) ?? throw new Exception("Missing HDF5 attr 'attr_count'.");
            Fps = TryReadIntAttribute(_fileId, AttrFps) ?? throw new Exception("Missing HDF5 attr 'fps'.");
            FrameCount = TryReadUlongAttribute(_fileId, AttrFrameCount) ?? ReadFrameCountFromDatasetsFallback();
        }

        private ulong ReadFrameCountFromDatasetsFallback()
        {
            // Fallback: get dims from /timestamps_ns (rank 1)
            long spaceId = H5D.get_space(_timestampsId);
            if (spaceId < 0) throw new Exception("Failed to read frame count.");
            try
            {
                ulong[] dims = new ulong[1];
                ulong[] maxdims = new ulong[1];
                H5S.get_simple_extent_dims(spaceId, dims, maxdims);
                return dims[0];
            }
            finally
            {
                H5S.close(spaceId);
            }
        }

        private void InitIoCaches()
        {
            // /corners
            _cornFileSpace = H5D.get_space(_cornersId);
            if (_cornFileSpace < 0) throw new Exception("InitIoCaches: failed to get /corners dataspace.");

            _cornMemSpace = H5S.create_simple(3, new ulong[] { 1, 4, 2 }, new ulong[] { 1, 4, 2 });
            if (_cornMemSpace < 0) throw new Exception("InitIoCaches: failed to create /corners memspace.");

            // /labels
            _labFileSpace = H5D.get_space(_labelsId);
            if (_labFileSpace < 0) throw new Exception("InitIoCaches: failed to get /labels dataspace.");

            _labCount = new ulong[] { 1, (ulong)AttrCount };
            _labMemSpace = H5S.create_simple(2, new ulong[] { 1, (ulong)AttrCount }, new ulong[] { 1, (ulong)AttrCount });
            if (_labMemSpace < 0) throw new Exception("InitIoCaches: failed to create /labels memspace.");

            // /label_mask
            _maskFileSpace = H5D.get_space(_labelMaskId);
            if (_maskFileSpace < 0) throw new Exception("InitIoCaches: failed to get /label_mask dataspace.");

            _maskCount = new ulong[] { 1, (ulong)AttrCount };
            _maskMemSpace = H5S.create_simple(2, new ulong[] { 1, (ulong)AttrCount }, new ulong[] { 1, (ulong)AttrCount });
            if (_maskMemSpace < 0) throw new Exception("InitIoCaches: failed to create /label_mask memspace.");

            // /timestamps_ns
            _tsFileSpace = H5D.get_space(_timestampsId);
            if (_tsFileSpace < 0) throw new Exception("InitIoCaches: failed to get /timestamps_ns dataspace.");

            _tsMemSpace = H5S.create_simple(1, new ulong[] { 1 }, new ulong[] { 1 });
            if (_tsMemSpace < 0) throw new Exception("InitIoCaches: failed to create /timestamps_ns memspace.");
        }

        // -------------------------------
        // Dispose
        // -------------------------------
        private void EnsureNotDisposed()
        {
            if (_disposed) throw new ObjectDisposedException(nameof(SessionManager));
        }

        public void Dispose()
        {
            if (_disposed) return;
            _disposed = true;

            try
            {
                // No video writer to dispose here anymore
            }
            catch { /* ignore */ }

            if (_tsMemSpace >= 0) H5S.close(_tsMemSpace);
            if (_tsFileSpace >= 0) H5S.close(_tsFileSpace);

            if (_cornMemSpace >= 0) H5S.close(_cornMemSpace);
            if (_cornFileSpace >= 0) H5S.close(_cornFileSpace);

            if (_labMemSpace >= 0) H5S.close(_labMemSpace);
            if (_labFileSpace >= 0) H5S.close(_labFileSpace);

            if (_maskMemSpace >= 0) H5S.close(_maskMemSpace);
            if (_maskFileSpace >= 0) H5S.close(_maskFileSpace);

            if (_cornersId >= 0) H5D.close(_cornersId);
            if (_labelsId >= 0) H5D.close(_labelsId);
            if (_labelMaskId >= 0) H5D.close(_labelMaskId);
            if (_timestampsId >= 0) H5D.close(_timestampsId);
            if (_fileId >= 0) H5F.close(_fileId);
        }
    }
}