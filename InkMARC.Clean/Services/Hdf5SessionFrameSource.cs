using InkMARC.Clean.Model;
using InkMARC.Clean.Services.Interfaces;
using OpenCvSharp;
using System;
using System.Runtime.InteropServices;

namespace InkMARC.Clean.Services
{
    /// <summary>
    /// Frame source backed by an InkMARC HDF5 session file using SessionManager.
    /// Reads frame images and metadata (corners, stylus) from the file.
    /// </summary>
    public sealed class Hdf5SessionFrameSource : IFrameSource
    {
        private SessionManager? _session;
        private bool _disposed;
        private readonly double _defaultFps;

        public event EventHandler<int>? FrameCountChanged;

        public int FrameCount { get; private set; }

        /// <summary>
        /// Nominal frames per second for this session.
        /// Currently a fixed default; you can later store/read this as an HDF5 attribute.
        /// </summary>
        public double FramesPerSecond { get; private set; }
        public int ViewW { get => throw new NotImplementedException(); set => throw new NotImplementedException(); }
        public int ViewH { get => throw new NotImplementedException(); set => throw new NotImplementedException(); }

        public string FileFilter => "HDF5 Files (*.h5;*.hdf5)|*.h5;*.hdf5|All Files|*.*";

        /// <summary>
        /// Create a new HDF5-backed frame source.
        /// </summary>
        /// <param name="defaultFps">
        /// Fallback FPS to use for timestamps when the session file does not store it explicitly.
        /// </param>
        public Hdf5SessionFrameSource(double defaultFps = 30.0)
        {
            _defaultFps = defaultFps;
            FramesPerSecond = defaultFps;
        }

        public bool SupportsPlay => true;

        public bool FileSeek => false;

        public void Open(string path)
        {
            ThrowIfDisposed();

            _session?.Dispose();
            _session = SessionManager.OpenExisting(path, writeable: false);

            // FrameCount is ulong in SessionManager; we clamp/check for int here.
            if (_session.FrameCount > int.MaxValue)
                throw new InvalidOperationException("HDF5 session has more frames than supported by IFrameSource.");

            FrameCount = (int)_session.FrameCount;

            // At the moment SessionManager does not expose FPS, so we use the default.
            // If you later store FPS as an attribute, you can add a property on SessionManager
            // and read it here instead of using _defaultFps.
            FramesPerSecond = _defaultFps;

            FrameCountChanged?.Invoke(this, FrameCount);
        }

        public FrameData? GetFrame(int index)
        {
            ThrowIfDisposed();

            if (_session == null || FrameCount == 0)
                return null;

            if (index < 0 || index >= FrameCount)
                return null;

            // Prepare buffers for a single frame
            int h = _session.Height;
            int w = _session.Width;
            int c = _session.Channels;

            byte[] imageRgb = new byte[h * w * c];
            var corners = new float[4 * 2];       // TL,TR,BR,BL → (x,y) each
            var labels = new float[_session.AttrCount];
            var labelMask = new byte[_session.AttrCount];

            // Read frame t = index
            _session.ReadFrame(
                frameIndex: (ulong)index,
                imageRgb: imageRgb,
                corners: corners,
                labels: labels,
                labelMask: labelMask);

            if (imageRgb.Length != h * w * 3)
            {
                throw new InvalidOperationException(
                    $"Unexpected image buffer length: got {imageRgb.Length}, expected {h * w * 3}.");
            }

            // Wrap image data into an OpenCV Mat (assumes 8UC3 packed RGB/BGR).
            var mat = new Mat(h, w, MatType.CV_8UC3);
            Marshal.Copy(imageRgb, 0, mat.Data, imageRgb.Length);

            // Convert to BGR for the rest of your OpenCV pipeline
            var matBgr = new Mat();
            Cv2.CvtColor(mat, matBgr, ColorConversionCodes.RGB2BGR);
            mat.Dispose(); // optional if you don't need it anymore

            // Map corners:
            var tl = new Point((int)Math.Round(corners[0]), (int)Math.Round(corners[1]));
            var tr = new Point((int)Math.Round(corners[2]), (int)Math.Round(corners[3]));
            var br = new Point((int)Math.Round(corners[4]), (int)Math.Round(corners[5]));
            var bl = new Point((int)Math.Round(corners[6]), (int)Math.Round(corners[7]));

            // Stylus mapping:
            //
            // ASSUMPTION (change to match your label layout):
            //   labels[0] → StylusX
            //   labels[1] → StylusY
            //   labels[2] → StylusPressure
            //   labels[3] → StylusTiltX
            //   labels[4] → StylusTiltY
            //
            // and labelMask[i] != 0 means "this attribute is present/valid".
            int? stylusX = null;
            int? stylusY = null;
            int? stylusPressure = null;
            int? stylusTiltX = null;
            int? stylusTiltY = null;
            bool hasStylusData = false;

            if (_session.AttrCount >= 5)
            {
                stylusX = labelMask[0] != 0 ? (int)Math.Round(labels[0]) : null;
                stylusY = labelMask[1] != 0 ? (int)Math.Round(labels[1]) : null;
                stylusPressure = labelMask[2] != 0 ? (int)Math.Round(labels[2]) : null;
                stylusTiltX = labelMask[3] != 0 ? (int)Math.Round(labels[3]) : null;
                stylusTiltY = labelMask[4] != 0 ? (int)Math.Round(labels[4]) : null;

                hasStylusData =
                    stylusX.HasValue || stylusY.HasValue ||
                    stylusPressure.HasValue || stylusTiltX.HasValue || stylusTiltY.HasValue;
            }

            // Construct FrameData for the ViewModel / pipeline.
            var frame = new FrameData
            {
                FrameIndex = index,
                Image = matBgr,

                TopLeft = tl,
                TopRight = tr,
                BottomRight = br,
                BottomLeft = bl,

                StylusX = stylusX,
                StylusY = stylusY,
                StylusPressure = stylusPressure,
                StylusTiltX = stylusTiltX,
                StylusTiltY = stylusTiltY,

                HasStylusData = hasStylusData,
                AdditionalText = null // fill if you later store per-frame text in HDF5
            };

            return frame;
        }

        public void Dispose()
        {
            if (_disposed) return;
            _disposed = true;

            _session?.Dispose();
            _session = null;

            GC.SuppressFinalize(this);
        }

        private void ThrowIfDisposed()
        {
            if (_disposed)
                throw new ObjectDisposedException(nameof(Hdf5SessionFrameSource));
        }

        public FrameData? GetFrameForExport(int index)
        {
            throw new NotImplementedException();
        }
    }
}
