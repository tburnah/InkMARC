using OpenCvSharp;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Media.Imaging;

namespace InkMARC.Clean.Model
{
    /// <summary>
    /// A single frame plus all associated metadata needed by the UI / processing pipeline.
    /// NOTE: Caller is responsible for disposing <see cref="Image"/>.
    /// </summary>
    public sealed class FrameData
    {
        public int FrameIndex { get; init; }

        /// <summary>
        /// Raw image data for this frame (e.g. BGR 8UC3).
        /// Caller must Dispose() this when done.
        /// </summary>
        public Mat? Image { get; init; } = null!;

        public Mat? AuxImage { get; init; } = null!;

        public BitmapSource? AuxBitmapSource { get; init; }

        // Quad corners in pixel space (e.g., encoder or view space)
        public Point? TopLeft { get; init; }
        public Point? TopRight { get; init; }
        public Point? BottomRight { get; init; }
        public Point? BottomLeft { get; init; }

        // Stylus information (nullable when not available)
        public int? StylusX { get; init; }
        public int? StylusY { get; init; }
        public int? StylusPressure { get; init; }
        public int? StylusTiltX { get; init; }
        public int? StylusTiltY { get; init; }

        public String? AdditionalText { get; init; }
        public bool HasStylusData { get; init; }
    }
}
