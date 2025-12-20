using HDF.PInvoke;
using System;
using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Security.Cryptography;

namespace InkMARC.Clean.Services
{
    public sealed class SessionManager : IDisposable
    {
        // Dataset names
        private const string ImagesName = "images";
        private const string CornersName = "corners";
        private const string LabelsName = "labels";
        private const string LabelMaskName = "label_mask";

        // HDF5 handles
        private long _fileId = -1;
        private long _imagesId = -1;
        private long _cornersId = -1;
        private long _labelsId = -1;
        private long _labelMaskId = -1;

        // Metadata
        public ulong FrameCount { get; private set; }
        public int Height { get; private set; }
        public int Width { get; private set; }
        public int Channels { get; private set; } = 3;
        public int AttrCount { get; private set; }

        private bool _disposed;

        // Cached dataspaces (keep open for the life of the session)
        private long _imgFileSpace = -1;
        private ulong[] _imgStart = new ulong[4];
        private ulong[] _imgCountSingle;
        private long _imgMemSpaceSingle = -1;        // [1,H,W,C]
        private long _imgMemSpaceBatch = -1;    // [batch,H,W,C]


        private long _cornFileSpace = -1;
        private long _cornMemSpace = -1;
        private readonly ulong[] _cornStart = new ulong[3];
        private static readonly ulong[] _cornCount = { 1, 4, 2 };

        private long _labFileSpace = -1;
        private long _labMemSpace = -1;
        private readonly ulong[] _labStart = new ulong[2];
        private ulong[] _labCount;

        private long _maskFileSpace = -1;
        private long _maskMemSpace = -1;
        private readonly ulong[] _maskStart = new ulong[2];
        private ulong[] _maskCount;

        private ulong[] _imgMemDims;
        private static readonly ulong[] _cornMemDims = { 1, 4, 2 };
        private ulong[] _labMemDims;
        private ulong[] _maskMemDims;

        // --- Buffered image write state ---
        private int _imgBatchFrames = 16;          // write N frames at a time (set to your chunkFrames)
        private int _imgFrameBytes;               // H*W*C
        private byte[]? _imgBatchBuffer;          // N * frameBytes
        private int _imgBatchFill;                // how many frames currently buffered
        private ulong _imgBatchStartFrame;            // first frame index in the current batch
        private bool _imgBatchHasStart;


        private long _dxpl = H5P.DEFAULT; // optional: DXPL handle if you want a custom transfer plist

        private SessionManager() { }

        // -------------------------------
        // Factory: create new file
        // -------------------------------
        public static SessionManager CreateNew(
            string path,
            ulong frameCount,
            int height,
            int width,
            int attrCount,
            int chunkFrames = 16)
        {
            var mgr = new SessionManager();
            mgr.FrameCount = frameCount;
            mgr.Height = height;
            mgr.Width = width;
            mgr.AttrCount = attrCount;

            mgr._imgBatchFrames = Math.Max(1, chunkFrames);
            mgr.CreateFileInternal(path, chunkFrames);
            mgr.InitIoCaches();

            return mgr;
        }

        private void InitIoCaches()
        {
            // /images
            // ---------- /images ----------
            _imgFileSpace = H5D.get_space(_imagesId);
            if (_imgFileSpace < 0)
                throw new Exception("InitIoCaches: H5D.get_space(/images) failed.");

            _imgFrameBytes = Height * Width * Channels;

            // ---- SINGLE FRAME (READ + fallback WRITE) ----
            _imgCountSingle = new ulong[] { 1, (ulong)Height, (ulong)Width, (ulong)Channels };
            var memDimsSingle = new ulong[] { 1, (ulong)Height, (ulong)Width, (ulong)Channels };

            _imgMemSpaceSingle = H5S.create_simple(4, memDimsSingle, memDimsSingle);
            if (_imgMemSpaceSingle < 0)
                throw new Exception("InitIoCaches: create single-frame memspace failed.");

            // ---- BATCH (WRITE ONLY) ----
            if (_imgBatchFrames <= 0)
                throw new Exception("_imgBatchFrames must be > 0.");

            var memDimsBatch = new ulong[]
            {
                (ulong)_imgBatchFrames,
                (ulong)Height,
                (ulong)Width,
                (ulong)Channels
            };

            _imgMemSpaceBatch = H5S.create_simple(4, memDimsBatch, memDimsBatch);
            if (_imgMemSpaceBatch < 0)
                throw new Exception("InitIoCaches: create batch memspace failed.");

            _imgBatchBuffer = new byte[_imgBatchFrames * _imgFrameBytes];
            _imgBatchFill = 0;

            // /corners
            _cornFileSpace = H5D.get_space(_cornersId);
            if (_cornFileSpace < 0) throw new Exception("InitIoCaches: failed to get /corners dataspace.");

            _cornMemSpace = H5S.create_simple(3, _cornMemDims, _cornMemDims);
            if (_cornMemSpace < 0) throw new Exception("InitIoCaches: failed to create /corners memspace.");

            // /labels
            _labFileSpace = H5D.get_space(_labelsId);
            if (_labFileSpace < 0) throw new Exception("InitIoCaches: failed to get /labels dataspace.");

            _labCount = new ulong[] { 1, (ulong)AttrCount };
            _labMemDims = new ulong[] { 1, (ulong)AttrCount };
            _labMemSpace = H5S.create_simple(2, _labMemDims, _labMemDims);
            if (_labMemSpace < 0) throw new Exception("InitIoCaches: failed to create /labels memspace.");

            // /label_mask
            _maskFileSpace = H5D.get_space(_labelMaskId);
            if (_maskFileSpace < 0) throw new Exception("InitIoCaches: failed to get /label_mask dataspace.");

            _maskCount = new ulong[] { 1, (ulong)AttrCount };
            _maskMemDims = new ulong[] { 1, (ulong)AttrCount };
            _maskMemSpace = H5S.create_simple(2, _maskMemDims, _maskMemDims);
            if (_maskMemSpace < 0) throw new Exception("InitIoCaches: failed to create /label_mask memspace.");
        }

        // -------------------------------
        // Factory: open existing file
        // -------------------------------
        public static SessionManager OpenExisting(
            string path,
            bool writeable = false)
        {
            var mgr = new SessionManager();
            mgr.OpenFileInternal(path, writeable);
            mgr.InitIoCaches();
            return mgr;
        }

        // -------------------------------
        // Public API: Write one frame
        // -------------------------------
        public void WriteFrame(
            ulong frameIndex,
            byte[] imageRgb,        // length = H * W * 3
            float[] corners,        // length = 4 * 2
            float[] labels,         // length = AttrCount
            byte[] labelMask)       // length = AttrCount            
        {
            EnsureNotDisposed();
            if (frameIndex < 0 || frameIndex >= FrameCount)
                throw new ArgumentOutOfRangeException(nameof(frameIndex));

            if (imageRgb?.Length != Height * Width * Channels)
                throw new ArgumentException("Unexpected image buffer length.", nameof(imageRgb));
            if (corners?.Length != 4 * 2)
                throw new ArgumentException("corners must be length 8 (4 points x 2 coords).", nameof(corners));
            if (labels?.Length != AttrCount)
                throw new ArgumentException("labels length must equal AttrCount.", nameof(labels));
            if (labelMask?.Length != AttrCount)
                throw new ArgumentException("labelMask length must equal AttrCount.", nameof(labelMask));

            //WriteImageFrame(frameIndex, imageRgb);
            WriteImageFrameBuffered(frameIndex, imageRgb);
            WriteCorners(frameIndex, corners);
            WriteLabels(frameIndex, labels);
            WriteLabelMask(frameIndex, labelMask);
        }

        // -------------------------------
        // Public API: Read one frame
        // -------------------------------
        public void ReadFrame(
            ulong frameIndex,
            byte[]? imageRgb,        // may be null if you don't need it
            float[]? corners,
            float[]? labels,
            byte[]? labelMask)
        {
            EnsureNotDisposed();
            if (frameIndex < 0 || frameIndex >= FrameCount)
                throw new ArgumentOutOfRangeException(nameof(frameIndex));

            if (_imgBatchFill > 0)
                FlushImageBatch();

            if (imageRgb != null && imageRgb.Length != Height * Width * Channels)
                throw new ArgumentException("Unexpected image buffer length.", nameof(imageRgb));
            if (corners != null && corners.Length != 4 * 2)
                throw new ArgumentException("corners must be length 8 (4 points x 2 coords).", nameof(corners));
            if (labels != null && labels.Length != AttrCount)
                throw new ArgumentException("labels length must equal AttrCount.", nameof(labels));
            if (labelMask != null && labelMask.Length != AttrCount)
                throw new ArgumentException("labelMask length must equal AttrCount.", nameof(labelMask));

            if (imageRgb != null)
                ReadImageFrame(frameIndex, imageRgb);
            if (corners != null)
                ReadCorners(frameIndex, corners);
            if (labels != null)
                ReadLabels(frameIndex, labels);
            if (labelMask != null)
                ReadLabelMask(frameIndex, labelMask);
        }

        // -------------------------------
        // Internal: create file + datasets
        // -------------------------------
        private void CreateFileInternal(string path, int chunkFrames)
        {
            // Create file
            _fileId = H5F.create(path, H5F.ACC_TRUNC, H5P.DEFAULT, H5P.DEFAULT);
            if (_fileId < 0) throw new Exception("Failed to create HDF5 file.");

            // Create /images : [T, H, W, 3] (uint8)
            {
                ulong[] dims = { FrameCount, (ulong)Height, (ulong)Width, (ulong)Channels };
                long spaceId = H5S.create_simple(4, dims, dims);
                if (spaceId < 0) throw new Exception("Failed to create images dataspace.");

                long dcpl = H5P.create(H5P.DATASET_CREATE);
                if (dcpl < 0) throw new Exception("Failed to create DCPL for images.");

                ulong[] chunkDims = { Math.Min(FrameCount, (ulong)chunkFrames), (ulong)Height, (ulong)Width, (ulong)Channels };
                var status = H5P.set_chunk(dcpl, 4, chunkDims);
                if (status < 0) throw new Exception("Failed to set chunking for images.");

                status = H5P.set_deflate(dcpl, 4); // gzip level 4
                if (status < 0) throw new Exception("Failed to set deflate for images.");

                _imagesId = H5D.create(_fileId, ImagesName, H5T.NATIVE_UCHAR, spaceId, H5P.DEFAULT, dcpl, H5P.DEFAULT);
                if (_imagesId < 0) throw new Exception("Failed to create dataset 'images'.");

                H5P.close(dcpl);
                H5S.close(spaceId);
            }

            // /corners : [T, 4, 2] float32
            {
                ulong[] dims = { (ulong)FrameCount, 4, 2 };
                long spaceId = H5S.create_simple(3, dims, dims);
                long dcpl = H5P.create(H5P.DATASET_CREATE);
                ulong[] chunkDims = { Math.Min(FrameCount, (ulong)chunkFrames), 4, 2 };
                H5P.set_chunk(dcpl, 3, chunkDims);
                H5P.set_deflate(dcpl, 4);

                _cornersId = H5D.create(_fileId, CornersName, H5T.NATIVE_FLOAT, spaceId, H5P.DEFAULT, dcpl, H5P.DEFAULT);
                if (_cornersId < 0) throw new Exception("Failed to create dataset 'corners'.");

                H5P.close(dcpl);
                H5S.close(spaceId);
            }

            // /labels : [T, A] float32
            {
                ulong[] dims = { (ulong)FrameCount, (ulong)AttrCount };
                long spaceId = H5S.create_simple(2, dims, dims);
                long dcpl = H5P.create(H5P.DATASET_CREATE);
                ulong[] chunkDims = { Math.Min(FrameCount, (ulong)chunkFrames), (ulong)AttrCount };
                H5P.set_chunk(dcpl, 2, chunkDims);
                H5P.set_deflate(dcpl, 4);

                _labelsId = H5D.create(_fileId, LabelsName, H5T.NATIVE_FLOAT, spaceId, H5P.DEFAULT, dcpl, H5P.DEFAULT);
                if (_labelsId < 0) throw new Exception("Failed to create dataset 'labels'.");

                H5P.close(dcpl);
                H5S.close(spaceId);
            }

            // /label_mask : [T, A] uint8 (0/1)
            {
                ulong[] dims = { (ulong)FrameCount, (ulong)AttrCount };
                long spaceId = H5S.create_simple(2, dims, dims);
                long dcpl = H5P.create(H5P.DATASET_CREATE);
                ulong[] chunkDims = { Math.Min(FrameCount, (ulong)chunkFrames), (ulong)AttrCount };
                H5P.set_chunk(dcpl, 2, chunkDims);
                H5P.set_deflate(dcpl, 4);

                _labelMaskId = H5D.create(_fileId, LabelMaskName, H5T.NATIVE_UCHAR, spaceId, H5P.DEFAULT, dcpl, H5P.DEFAULT);
                if (_labelMaskId < 0) throw new Exception("Failed to create dataset 'label_mask'.");

                H5P.close(dcpl);
                H5S.close(spaceId);
            }

            // You can also write file attributes (height/width/attr_count) here with H5A.*
        }

        // -------------------------------
        // Internal: open existing file
        // -------------------------------
        private void OpenFileInternal(string path, bool writeable)
        {
            uint flags = writeable ? H5F.ACC_RDWR : H5F.ACC_RDONLY;
            _fileId = H5F.open(path, flags);
            if (_fileId < 0) throw new Exception("Failed to open HDF5 file.");

            _imagesId = H5D.open(_fileId, ImagesName);
            _cornersId = H5D.open(_fileId, CornersName);
            _labelsId = H5D.open(_fileId, LabelsName);
            _labelMaskId = H5D.open(_fileId, LabelMaskName);

            // Read dimensions from /images
            long spaceId = H5D.get_space(_imagesId);
            int rank = H5S.get_simple_extent_ndims(spaceId);
            if (rank != 4) throw new Exception("Expected images rank 4.");
            ulong[] dims = new ulong[4];
            ulong[] maxdims = new ulong[4];
            H5S.get_simple_extent_dims(spaceId, dims, maxdims);
            H5S.close(spaceId);

            FrameCount = dims[0];
            Height = (int)dims[1];
            Width = (int)dims[2];
            Channels = (int)dims[3];

            // Read AttrCount from /labels
            long labelSpace = H5D.get_space(_labelsId);
            ulong[] labelDims = new ulong[2];
            ulong[] labelMax = new ulong[2];
            H5S.get_simple_extent_dims(labelSpace, labelDims, labelMax);
            H5S.close(labelSpace);
            AttrCount = (int)labelDims[1];
        }

        // -------------------------------
        // Internal: Write helpers
        // -------------------------------
        private unsafe void WriteCorners(ulong t, float[] corners)
        {
            _cornStart[0] = t;
            _cornStart[1] = 0;
            _cornStart[2] = 0;

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
            _labStart[0] = t;
            _labStart[1] = 0;

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
            _maskStart[0] = t;
            _maskStart[1] = 0;

            var sel = H5S.select_hyperslab(_maskFileSpace, H5S.seloper_t.SET, _maskStart, null, _maskCount, null);
            if (sel < 0) throw new Exception("Failed to select hyperslab for label_mask.");

            fixed (byte* p = mask)
            {
                var status = H5D.write(_labelMaskId, H5T.NATIVE_UCHAR, _maskMemSpace, _maskFileSpace, _dxpl, (IntPtr)p);
                if (status < 0) throw new Exception("Failed to write label mask.");
            }
        }

        private unsafe void WriteImageFrameBuffered(ulong t, byte[] data)
        {
            // If buffering isn't initialized (shouldn't happen if InitIoCaches ran), fall back to direct write.
            if (_imgBatchBuffer == null || _imgBatchFrames <= 1)
            {
                WriteImageFrameDirect(t, data);
                return;
            }

            // If this is the first frame in the batch, record the start index.
            if (!_imgBatchHasStart)
            {
                _imgBatchStartFrame = t;
                _imgBatchHasStart = true;
                _imgBatchFill = 0;
            }

            // We assume sequential writes. If caller jumps around, flush current batch and start anew.
            ulong expectedT = _imgBatchStartFrame + (ulong)_imgBatchFill;
            if (t != expectedT)
            {
                FlushImageBatch(); // writes any pending frames
                _imgBatchStartFrame = t;
                _imgBatchHasStart = true;
                _imgBatchFill = 0;
            }

            // Copy this frame into the batch buffer.
            Buffer.BlockCopy(data, 0, _imgBatchBuffer, _imgBatchFill * _imgFrameBytes, _imgFrameBytes);
            _imgBatchFill++;

            // If batch is full, write it in one H5D.write.
            if (_imgBatchFill >= _imgBatchFrames)
            {
                FlushImageBatch();
            }
        }

        private unsafe void FlushImageBatch()
        {
            if (_imgBatchBuffer == null || !_imgBatchHasStart || _imgBatchFill <= 0)
                return;

            _imgStart[0] = _imgBatchStartFrame;
            _imgStart[1] = 0;
            _imgStart[2] = 0;
            _imgStart[3] = 0;

            ulong[] writeCount =
            {
                (ulong)_imgBatchFill,
                (ulong)Height,
                (ulong)Width,
                (ulong)Channels
            };

            var sel = H5S.select_hyperslab(_imgFileSpace, H5S.seloper_t.SET, _imgStart, null, writeCount, null);
            if (sel < 0) throw new Exception("Failed to select hyperslab for buffered image write.");

            long memSpace = _imgMemSpaceBatch;
            bool tempMem = false;

            // Tail batch (not full): create a matching memspace
            if (_imgBatchFill != _imgBatchFrames)
            {
                ulong[] memDims =
                {
                    (ulong)_imgBatchFill,
                    (ulong)Height,
                    (ulong)Width,
                    (ulong)Channels
                };
                memSpace = H5S.create_simple(4, memDims, memDims);
                if (memSpace < 0) throw new Exception("Failed to create memspace for partial buffered image write.");
                tempMem = true;
            }

            fixed (byte* p = _imgBatchBuffer)
            {
                var status = H5D.write(_imagesId, H5T.NATIVE_UCHAR, memSpace, _imgFileSpace, _dxpl, (IntPtr)p);
                if (status < 0)
                {
                    // Optional: print native HDF5 error stack to debug output
                    // H5E.print2(H5E.DEFAULT, IntPtr.Zero);
                    throw new Exception("Failed to write buffered image frames.");
                }
            }

            if (tempMem)
                H5S.close(memSpace);

            _imgBatchFill = 0;
            _imgBatchHasStart = false;
        }


        // Keep a direct-write version for fallback / non-buffer mode.
        private unsafe void WriteImageFrameDirect(ulong t, byte[] data)
        {
            _imgStart[0] = t; _imgStart[1] = 0; _imgStart[2] = 0; _imgStart[3] = 0;

            var sel = H5S.select_hyperslab(_imgFileSpace, H5S.seloper_t.SET, _imgStart, null, _imgCountSingle, null);
            if (sel < 0) throw new Exception("Failed to select hyperslab for image frame.");

            fixed (byte* p = data)
            {
                var status = H5D.write(_imagesId, H5T.NATIVE_UCHAR, _imgMemSpaceSingle, _imgFileSpace, _dxpl, (IntPtr)p);
                if (status < 0) throw new Exception("Failed to write image frame.");
            }
        }


        // -------------------------------
        // Internal: Read helpers
        // -------------------------------
        private unsafe void ReadImageFrame(ulong t, byte[] buffer)
        {

            if (buffer == null) throw new ArgumentNullException(nameof(buffer));
            if ((ulong)buffer.Length != (ulong)Height * (ulong)Width * (ulong)Channels)
                throw new ArgumentException("Unexpected image buffer length.", nameof(buffer));

            // Get a *fresh* file dataspace for this call
            long fileSpace = H5D.get_space(_imagesId);
            if (fileSpace < 0) throw new Exception("Failed to get images dataspace.");

            try
            {
                ulong[] start = { t, 0, 0, 0 };
                // Use the same _imgCountSingle you already computed: {1,H,W,C}
                var sel = H5S.select_hyperslab(fileSpace, H5S.seloper_t.SET, start, null, _imgCountSingle, null);
                if (sel < 0) throw new Exception("Failed to select hyperslab for image read.");

                fixed (byte* p = buffer)
                {
                    var status = H5D.read(_imagesId, H5T.NATIVE_UCHAR, _imgMemSpaceSingle, fileSpace, _dxpl, (IntPtr)p);
                    if (status < 0)
                    {
                        DumpHdf5ErrorsToDebug("H5D.read(/images)");
                        throw new Exception("Failed to read image frame.");
                    }
                }
            }
            finally
            {
                H5S.close(fileSpace);
            }
        }


        private unsafe void ReadCorners(ulong t, float[] buffer)
        {
            _cornStart[0] = t;
            _cornStart[1] = 0;
            _cornStart[2] = 0;

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
            _labStart[0] = t;
            _labStart[1] = 0;

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
            _maskStart[0] = t;
            _maskStart[1] = 0;

            var sel = H5S.select_hyperslab(_maskFileSpace, H5S.seloper_t.SET, _maskStart, null, _maskCount, null);
            if (sel < 0) throw new Exception("Failed to select hyperslab for label_mask read.");

            fixed (byte* p = buffer)
            {
                var status = H5D.read(_labelMaskId, H5T.NATIVE_UCHAR, _maskMemSpace, _maskFileSpace, _dxpl, (IntPtr)p);
                if (status < 0) throw new Exception("Failed to read label_mask.");
            }
        }

        private static void DumpHdf5ErrorsToDebug(string context)
        {
            try
            {
                long estack = H5E.get_current_stack();
                if (estack < 0)
                {
                    Debug.WriteLine($"[HDF5] {context}: Failed to get current error stack.");
                    return;
                }

                Debug.WriteLine($"[HDF5] Error stack ({context}):");

                H5E.walk(
                    estack,
                    H5E.direction_t.H5E_WALK_DOWNWARD,
                    (uint n, ref H5E.error_t err, IntPtr client_data) =>
                    {
                        string file = string.IsNullOrWhiteSpace(err.file_name) ? "<unknown file>" : err.file_name;
                        string func = string.IsNullOrWhiteSpace(err.func_name) ? "<unknown func>" : err.func_name;
                        string desc = string.IsNullOrWhiteSpace(err.desc) ? "<no desc>" : err.desc;

                        Debug.WriteLine($"  #{n}: {file}:{err.line} {func} - {desc}");
                        return 0; // continue
                    },
                    IntPtr.Zero);

                H5E.close_stack(estack);
            }
            catch (Exception ex)
            {
                Debug.WriteLine($"[HDF5] {context}: Exception while dumping error stack: {ex}");
            }
        }

        // -------------------------------
        // Dispose pattern
        // -------------------------------
        private void EnsureNotDisposed()
        {
            if (_disposed) throw new ObjectDisposedException(nameof(SessionManager));
        }

        public void Dispose()
        {
            if (_disposed) return;
            _disposed = true;

            if (_imgBatchFill > 0) FlushImageBatch();

            if (_imgMemSpaceBatch >= 0) H5S.close(_imgMemSpaceBatch);
            if (_imgMemSpaceSingle >= 0) H5S.close(_imgMemSpaceSingle);            
            if (_imgFileSpace >= 0) H5S.close(_imgFileSpace);

            if (_cornMemSpace >= 0) H5S.close(_cornMemSpace);
            if (_cornFileSpace >= 0) H5S.close(_cornFileSpace);

            if (_labMemSpace >= 0) H5S.close(_labMemSpace);
            if (_labFileSpace >= 0) H5S.close(_labFileSpace);

            if (_maskMemSpace >= 0) H5S.close(_maskMemSpace);
            if (_maskFileSpace >= 0) H5S.close(_maskFileSpace);

            if (_imagesId >= 0) H5D.close(_imagesId);
            if (_cornersId >= 0) H5D.close(_cornersId);
            if (_labelsId >= 0) H5D.close(_labelsId);
            if (_labelMaskId >= 0) H5D.close(_labelMaskId);
            if (_fileId >= 0) H5F.close(_fileId);
        }
    }
}
