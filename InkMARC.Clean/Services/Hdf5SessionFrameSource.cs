using InkMARC.Clean.Model;
using InkMARC.Clean.Services.Interfaces;
using OpenCvSharp;
using System;
using System.IO;
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
        private string _videoPath = string.Empty;

        private int _lastFrameIndex = -1;
        private readonly Mat _frameBuffer = new();        // reused decode buffer
        private float[] _corners;
        private float[] _labels;
        private byte[] _labelMask;

        public event EventHandler<int>? FrameCountChanged;

        private OpenCvSharp.VideoCapture? _cap;
        private readonly object _capLock = new object();

        public int FrameCount { get; private set; }

        /// <summary>
        /// Nominal frames per second for this session.
        /// Currently a fixed default; you can later store/read this as an HDF5 attribute.
        /// </summary>
        public double FramesPerSecond { get; private set; }
        public int ViewW { get => _frameBuffer.Width; set => throw new NotImplementedException(); }
        public int ViewH { get => _frameBuffer.Height; set => throw new NotImplementedException(); }

        public string FileFilter => "AVI File (*.avi)|*.avi|All Files|*.*";

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

        public void Open(string videoPath)
        {
            ThrowIfDisposed();

            if (string.IsNullOrWhiteSpace(videoPath))
                throw new ArgumentException("Path is null/empty.", nameof(videoPath));

            _videoPath = videoPath;

            // Derive H5 path from the chosen video path.
            // Example:
            //   participant_17_Q1a_20251202_140719_White.avi
            // ->participant_17_Q1a_20251202_140719.h5
            string h5Path = DeriveH5PathFromVideoPath(videoPath);

            _session?.Dispose();

            // IMPORTANT: OpenExisting expects the H5 path, not the AVI path.
            // Pass the selected video as an override so SessionManager knows which video to use.
            _session = SessionManager.OpenExisting(h5Path, aviPathOverride: videoPath, writeable: false);

            if (_session.FrameCount > int.MaxValue)
                throw new InvalidOperationException("HDF5 session has more frames than supported by IFrameSource.");

            FrameCount = (int)_session.FrameCount;

            FramesPerSecond = _defaultFps; // until you store/read FPS attribute

            // reset playback state
            lock (_capLock)
            {
                _cap?.Release();
                _cap?.Dispose();
                _cap = null;
            }
            _lastFrameIndex = -1;

            // allocate metadata buffers once
            _corners = new float[8];
            _labels = new float[_session.AttrCount];
            _labelMask = new byte[_session.AttrCount];

            FrameCountChanged?.Invoke(this, FrameCount);
        }

        private static string DeriveH5PathFromVideoPath(string videoPath)
        {
            string dir = Path.GetDirectoryName(videoPath) ?? string.Empty;
            string stem = Path.GetFileNameWithoutExtension(videoPath) ?? string.Empty;

            // Define the exact set of colour suffixes you export.
            // Keep this in sync with your export list.
            // Case-insensitive match.
            var knownColours = ColorPalette.BackgroundNames;

            // Strip a trailing "_Colour" if present.
            int lastUnderscore = stem.LastIndexOf('_');
            if (lastUnderscore > 0 && lastUnderscore < stem.Length - 1)
            {
                string suffix = stem[(lastUnderscore + 1)..];
                if (knownColours.Contains(suffix))
                {
                    stem = stem[..lastUnderscore];
                }
            }

            // Choose one convention. If you use baseName.h5:
            string candidate = Path.Combine(dir, stem + ".h5");
            if (File.Exists(candidate))
                return candidate;

            // If you use baseName_meta.h5:
            string candidateMeta = Path.Combine(dir, stem + "_meta.h5");
            if (File.Exists(candidateMeta))
                return candidateMeta;

            // Otherwise, fail loudly: the caller chose a video that doesn't map to metadata.
            throw new FileNotFoundException(
                "Could not find a corresponding H5 file for the selected video.",
                candidate);
        }

        public FrameData? GetFrame(int index)
        {
            ThrowIfDisposed();

            if (_session == null || FrameCount == 0)
                return null;

            if (index < 0 || index >= FrameCount)
                return null;

            int h = _session.Height;
            int w = _session.Width;

            // Ensure VideoCapture is open
            VideoCapture cap;
            lock (_capLock)
            {
                if (_cap == null)
                {
                    if (string.IsNullOrWhiteSpace(_videoPath))
                        return null;

                    _cap = new VideoCapture(_videoPath);

                    if (!_cap.IsOpened())
                        return null;
                }

                cap = _cap;
            }

            // Read metadata into reused buffers (no per-frame allocations)
            _session.ReadMetadata(
                frameIndex: (ulong)index,
                timestampNs: out ulong timestampNs,
                corners: _corners,
                labels: _labels,
                labelMask: _labelMask);

            // Read video frame with sequential optimization
            bool ok;
            lock (_capLock)
            {
                if (index == _lastFrameIndex + 1)
                {
                    ok = cap.Read(_frameBuffer);
                }
                else
                {
                    ok = cap.Set(VideoCaptureProperties.PosFrames, index);
                    if (ok) ok = cap.Read(_frameBuffer);
                }

                _lastFrameIndex = ok ? index : _lastFrameIndex;
            }

            if (!ok || _frameBuffer.Empty())
                return null;
            
            // (Optional) avoid resize unless truly necessary
            if (_frameBuffer.Rows != h || _frameBuffer.Cols != w)
            {
                Cv2.Resize(_frameBuffer, _frameBuffer, new Size(w, h));
            }

            // Map corners TL,TR,BR,BL
            var tl = new Point((int)Math.Round(_corners[0]), (int)Math.Round(_corners[1]));
            var tr = new Point((int)Math.Round(_corners[2]), (int)Math.Round(_corners[3]));
            var br = new Point((int)Math.Round(_corners[4]), (int)Math.Round(_corners[5]));
            var bl = new Point((int)Math.Round(_corners[6]), (int)Math.Round(_corners[7]));

            int? stylusX = null, stylusY = null, stylusPressure = null, stylusTiltX = null, stylusTiltY = null;
            bool hasStylusData = false;

            if (_session.AttrCount >= 5)
            {
                stylusX = _labelMask[0] != 0 ? (int)Math.Round(_labels[0]) : null;
                stylusY = _labelMask[1] != 0 ? (int)Math.Round(_labels[1]) : null;
                stylusPressure = _labelMask[2] != 0 ? (int)Math.Round(_labels[2]) : null;
                stylusTiltX = _labelMask[3] != 0 ? (int)Math.Round(_labels[3]) : null;
                stylusTiltY = _labelMask[4] != 0 ? (int)Math.Round(_labels[4]) : null;

                hasStylusData = stylusX.HasValue || stylusY.HasValue || stylusPressure.HasValue || stylusTiltX.HasValue || stylusTiltY.HasValue;
            }

            // IMPORTANT: FrameData must own its Mat; clone is safest (but expensive).
            // If you can change FrameData to allow a "borrowed Mat" for display-only, you can avoid this clone.
            var frame = new FrameData
            {
                FrameIndex = index,
                Image = _frameBuffer.Clone(),

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
                AdditionalText = null
            };

            return frame;
        }

        public void Dispose()
        {
            if (_disposed) return;
            _disposed = true;

            lock (_capLock)
            {
                _cap?.Release();
                _cap?.Dispose();
                _cap = null;
            }

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
