using InkMARC.Clean.Model;
using InkMARC.Clean.Services.Interfaces;
using OpenCvSharp;
using OpenCvSharp.WpfExtensions;
using System.Windows.Media.Imaging;

namespace InkMARC.Clean.Services
{
    internal class VideoFileFrameSource : IFrameSource, IDisposable
    {
        private readonly IVideoService _videoService;
        private bool _disposed;

        // Config: expected fixed height of the data/text bar in pixels.
        // Tune this to match your video frames. Default 48.
        public int DataBarHeight { get; set; } = 48;

        // simple read-only text properties for corners and stylus
        private int? corner0X;        
        private int? corner0Y;        
        private int? corner1X;        
        private int? corner1Y;        
        private int? corner2X;        
        private int? corner2Y;        
        private int? corner3X;        
        private int? corner3Y;
        
        private int? stylusX;        
        private int? stylusY;        
        private int? stylusPressure;        
        private int? stylusTiltX;        
        private int? stylusTiltY;
        
        private int frameCount;

        private int? _detectedBorderHeight = null;

        public int ViewW { get; set; } = 1080;
        public int ViewH { get; set; } = 2161;

        internal VideoFileFrameSource(IVideoService videoService)
        {
            _videoService = videoService ?? throw new ArgumentNullException(nameof(videoService));
            _videoService.FrameCountChanged += VideoService_FrameCountChanged;
        }

        #region IFrameSource Implementation

        public bool SupportsPlay => true;

        public string FileFilter => "Video Files|*.mp4;*.avi;*.mov|All files|*.*";

        public bool FileSeek => false;

        public int FrameCount => frameCount;

        public double FramesPerSecond => throw new NotImplementedException();

        public event EventHandler<int>? FrameCountChanged;

        public void Dispose()
        {
            if (_disposed) return;
            _disposed = true;

            _videoService.FrameCountChanged -= VideoService_FrameCountChanged;
        }

        public FrameData? GetFrameForExport(int index)
        {
            try
            {
                using var mat = _videoService.GetFrameAt(index);
                if (mat == null || mat.Empty())
                    return null;

                var wid = mat.Width;
                var hei = mat.Height;
                // Detect/extract text strip (existing behaviour)
                var textMat = ExtractTextStripMat(mat, wid, hei);
                bool hasStylusInfo = (textMat != null);

                // If we can’t proceed with databar logic, preserve existing behaviour: return null
                // (Your original code returned null unless hasStylusInfo && borderHeight.HasValue)
                if (!hasStylusInfo)
                    return null;

                int borderH = _detectedBorderHeight ?? 0;              

                // OCR (existing behaviour)                
                ParseOcr(DataStripAnalyzer.ExtractText(textMat));

                // Cache surface points if invariant (same borderH/width/height)
                var p = GetSurfaceMask(_detectedBorderHeight, mat.Width, mat.Height);
                (Point tl, Point tr, Point br, Point bl)? pts =
                    p.HasValue ? (p.Value.Item1, p.Value.Item2, p.Value.Item3, p.Value.Item4) : null;

                // Decide content region
                if (borderH < mat.Height)
                {
                    int y = borderH;
                    int h = mat.Height - y;

                    if (h <= 0)
                    {
                        return new FrameData
                        {                            
                            BottomLeft = pts?.bl,
                            BottomRight = pts?.br,
                            FrameIndex = index,
                            HasStylusData = hasStylusInfo,
                            StylusPressure = stylusPressure,
                            StylusTiltX = stylusTiltX,
                            StylusTiltY = stylusTiltY,
                            StylusX = stylusX,
                            StylusY = stylusY,
                            TopLeft = pts?.tl,
                            TopRight = pts?.tr,
                        };
                    }

                    using var contentRoi = new Mat(mat, new Rect(0, y, mat.Width, h));
                    return new FrameData
                    {                        
                        BottomLeft = pts?.bl,
                        BottomRight = pts?.br,
                        FrameIndex = index,
                        HasStylusData = hasStylusInfo,
                        Image = contentRoi.Clone(),
                        StylusPressure = stylusPressure,
                        StylusTiltX = stylusTiltX,
                        StylusTiltY = stylusTiltY,
                        StylusX = stylusX,
                        StylusY = stylusY,
                        TopLeft = pts?.tl,
                        TopRight = pts?.tr,
                    }; ;
                }

                // borderH >= mat.Height, content is whole image
                return new FrameData
                {
                    AuxBitmapSource = null,
                    BottomLeft = pts?.bl,
                    BottomRight = pts?.br,
                    FrameIndex = index,
                    HasStylusData = hasStylusInfo,
                    Image = mat.Clone(),
                    StylusPressure = stylusPressure,
                    StylusTiltX = stylusTiltX,
                    StylusTiltY = stylusTiltY,
                    StylusX = stylusX,
                    StylusY = stylusY,
                    TopLeft = pts?.tl,
                    TopRight = pts?.tr,
                };
            }
            catch (Exception ex)
            {
                System.Diagnostics.Debug.WriteLine("LoadFrame error: " + ex);
                return null;
            }
        }

        public FrameData? GetFrame(int index)
        {
            try
            {
                using var mat = _videoService.GetFrameAt(index);
                if (mat == null || mat.Empty())
                    return null;

                int wid = mat.Width;
                int hei = mat.Height;

                // Detect/extract text strip (existing behaviour)
                var textBmp = ExtractTextStrip(mat, wid, hei);
                bool hasStylusInfo = (textBmp != null);

                // If we can’t proceed with databar logic, preserve existing behaviour: return null
                // (Your original code returned null unless hasStylusInfo && borderHeight.HasValue)
                if (!hasStylusInfo)
                    return null;

                int borderH = _detectedBorderHeight.Value;

                // ROI for strip
                using var stripRoi = new Mat(mat, new Rect(0, 0, mat.Width, borderH));

                // OCR (existing behaviour)
                string additionalText = string.Empty;
                try
                {
                    var raw = DataStripAnalyzer.ExtractText(stripRoi);
                    if (string.IsNullOrWhiteSpace(raw))
                    {
                        additionalText = "No text detected";
                    }
                    else
                    {
                        additionalText = raw;
                        ParseOcr(raw);
                    }
                }
                catch (Exception ex)
                {
                    additionalText = $"OCR error: {ex.Message}";
                }

                // Cache surface points if invariant (same borderH/width/height)
                var p = GetSurfaceMask(_detectedBorderHeight, mat.Width, mat.Height);
                (Point tl, Point tr, Point br, Point bl)? pts =
                    p.HasValue ? (p.Value.Item1, p.Value.Item2, p.Value.Item3, p.Value.Item4) : null;

                // Clone strip once (reused in all branches)
                Mat stripClone = stripRoi.Clone();

                // Decide content region
                if (borderH < mat.Height)
                {
                    int y = borderH;
                    int h = mat.Height - y;

                    if (h <= 0)
                    {
                        return new FrameData
                        {
                            AdditionalText = additionalText,
                            AuxImage = stripClone,
                            AuxBitmapSource = textBmp,
                            BottomLeft = pts?.bl,
                            BottomRight = pts?.br,
                            FrameIndex = index,
                            HasStylusData = hasStylusInfo,
                            Image = null,
                            StylusPressure = stylusPressure,
                            StylusTiltX = stylusTiltX,
                            StylusTiltY = stylusTiltY,
                            StylusX = stylusX,
                            StylusY = stylusY,
                            TopLeft = pts?.tl,
                            TopRight = pts?.tr,
                        };
                    }

                    using var contentRoi = new Mat(mat, new Rect(0, y, mat.Width, h));
                    return new FrameData
                    {
                        AdditionalText = additionalText,
                        AuxImage = stripClone,
                        AuxBitmapSource = textBmp,
                        BottomLeft = pts?.bl,
                        BottomRight = pts?.br,
                        FrameIndex = index,
                        HasStylusData = hasStylusInfo,
                        Image = contentRoi.Clone(),
                        StylusPressure = stylusPressure,
                        StylusTiltX = stylusTiltX,
                        StylusTiltY = stylusTiltY,
                        StylusX = stylusX,
                        StylusY = stylusY,
                        TopLeft = pts?.tl,
                        TopRight = pts?.tr,
                    };
                }

                // borderH >= mat.Height, content is whole image
                return new FrameData
                {
                    AdditionalText = additionalText,
                    AuxImage = stripClone,
                    AuxBitmapSource = textBmp,
                    BottomLeft = pts?.bl,
                    BottomRight = pts?.br,
                    FrameIndex = index,
                    HasStylusData = hasStylusInfo,
                    Image = mat.Clone(),
                    StylusPressure = stylusPressure,
                    StylusTiltX = stylusTiltX,
                    StylusTiltY = stylusTiltY,
                    StylusX = stylusX,
                    StylusY = stylusY,
                    TopLeft = pts?.tl,
                    TopRight = pts?.tr,
                };
            }
            catch (Exception ex)
            {
                System.Diagnostics.Debug.WriteLine("LoadFrame error: " + ex);
                return null;
            }
        }

        public void Open(string path)
        {
            try
            {
                _videoService.Open(path);
                frameCount = _videoService.FrameCount;                
            }
            catch (Exception ex)
            {
                System.Windows.MessageBox.Show($"Failed to open video: {ex.Message}");
            }
        }

        #endregion

        private void VideoService_FrameCountChanged(object? sender, int e)
        {
            frameCount = e;
            FrameCountChanged?.Invoke(this, e);
        }

        private Mat? ExtractTextStripMat(Mat frame, int w, int h)
        {
            if (frame == null || frame.Empty())
                return null;

            // If we already detected the border, just check a cheap existence test first
            if (_detectedBorderHeight.HasValue)
            {
                // quick check: require the top two rows' rightmost two pixels all white
                bool topRowWhite = IsRightPairWhite(frame, 0);
                bool secondRowWhite = (h > 1) && IsRightPairWhite(frame, 1);
                if (!(topRowWhite && secondRowWhite))
                    return null; // databar not present on this frame

                int stripH = _detectedBorderHeight.Value;
                if (stripH <= 0 || stripH > h) return null;

                var roi = new Rect(0, 0, w, stripH);
                return new Mat(frame, roi);
            }

            // Need to detect border height by scanning the rightmost pixel pairs from top down
            int maxScan = Math.Min(h, Math.Max(DataBarHeight * 8, 400));
            int y = 0;
            int whiteCount = 0;

            while (y < maxScan)
            {
                bool isWhite = IsRightPairWhite(frame, y);

                if (isWhite)
                {
                    whiteCount++;
                    y++;
                    continue;
                }

                // Found first non-white row after a white run.
                if (whiteCount > 0)
                {
                    // Confirm next two rows are also non-white (if available).
                    bool nextNonWhite1 = true;
                    bool nextNonWhite2 = true;

                    // y is already known non-white, so nextNonWhite1 is trivially true
                    // (keep the original structure intent: confirm subsequent rows)
                    if (y + 1 < maxScan)
                        nextNonWhite2 = !IsRightPairWhite(frame, y + 1);

                    if (nextNonWhite1 && nextNonWhite2)
                    {
                        _detectedBorderHeight = whiteCount;
                        break;
                    }
                }

                y++;
            }


            // if still not detected, give up for now
            if (!_detectedBorderHeight.HasValue)
                return null;

            // Now that we have a detected height, extract the strip
            int stripHeight = _detectedBorderHeight.Value;
            if (stripHeight <= 0 || stripHeight > h) return null;

            var roiFinal = new Rect(0, 0, w, stripHeight);
            return new Mat(frame, roiFinal);
        }

        private BitmapSource? ExtractTextStrip(Mat frame, int w, int h)
        {
            if (frame == null || frame.Empty())
                return null;

            using Mat? stripMat = ExtractTextStripMat(frame, w, h);
            if (stripMat == null)
                return null;

            var bmp = BitmapSourceConverter.ToBitmapSource(stripMat);
            bmp.Freeze();
            return bmp;
        }

        // fast helper to check if the two rightmost pixels at row y are (near) pure white
        private bool IsRightPairWhite(Mat frame, int row)
        {
            int w = frame.Width;
            if ((uint)row >= (uint)frame.Height || w <= 0) return false;

            int x1 = w - 1;
            int x2 = (w >= 2) ? (w - 2) : (w - 1);

            // Fast path for 8-bit 3-channel mats (BGR)
            if (frame.Type() == MatType.CV_8UC3 && frame.IsContinuous())
            {
                unsafe
                {
                    byte* basePtr = (byte*)frame.DataPointer;
                    int step = (int)frame.Step(); // bytes per row
                    byte* rowPtr = basePtr + row * step;

                    // Each pixel is 3 bytes: B,G,R
                    byte* p1 = rowPtr + x1 * 3;
                    byte* p2 = rowPtr + x2 * 3;

                    return (p1[0] >= 250 && p1[1] >= 250 && p1[2] >= 250) &&
                           (p2[0] >= 250 && p2[1] >= 250 && p2[2] >= 250);
                }
            }

            // Safe fallback (works for non-continuous / different types)
            Vec3b v1 = frame.Get<Vec3b>(row, x1);
            Vec3b v2 = frame.Get<Vec3b>(row, x2);

            return (v1.Item0 >= 250 && v1.Item1 >= 250 && v1.Item2 >= 250) &&
                   (v2.Item0 >= 250 && v2.Item1 >= 250 && v2.Item2 >= 250);
        }

        private void ParseOcr(string? raw)
        {
            if (string.IsNullOrWhiteSpace(raw))
                return;

            string? line0 = null;
            string? line1 = null;
            string? line2 = null;

            string? best = null;
            int bestLen = -1;

            int count = 0;

            // Single pass over lines
            var span = raw.AsSpan();
            int start = 0;

            for (int i = 0; i <= span.Length; i++)
            {
                if (i == span.Length || span[i] == '\n' || span[i] == '\r')
                {
                    var slice = span.Slice(start, i - start).Trim();
                    start = i + 1;

                    if (slice.Length == 0)
                        continue;

                    string line = slice.ToString();

                    // Track longest line (replaces OrderByDescending)
                    if (line.Length > bestLen)
                    {
                        bestLen = line.Length;
                        best = line;
                    }

                    if (count == 0) line0 = line;
                    else if (count == 1) line1 = line;
                    else if (count == 2) line2 = line;

                    count++;
                    if (count >= 3) break;
                }
            }

            if (count < 2)
                return;

            // ---- Line 1 ----
            if (!TryParseInts(line0!, 7, out var v0)) return;

            corner0X = v0[0];
            corner0Y = v0[1];
            corner1X = v0[2];
            corner1Y = v0[3];
            stylusX = v0[4];
            stylusY = v0[5];
            stylusPressure = v0[6];

            // ---- Line 2 ----
            if (!TryParseInts(line1!, 6, out var v1)) return;

            corner2X = v1[0];
            corner2Y = v1[1];
            corner3X = v1[2];
            corner3Y = v1[3];
            stylusTiltX = v1[4];
            stylusTiltY = v1[5];

            // ---- Line 3 (optional) ----
            if (count >= 3 && TryParseInts(line2!, 2, out var v2))
            {
                ViewH = v2[0] ?? 0;
                ViewW = v2[1] ?? 0;
            }

            return;

            // ---------- helpers ----------

            static bool TryParseInts(string line, int expected, out int?[] values)
            {
                values = new int?[expected];

                ReadOnlySpan<char> s = line.AsSpan();
                int idx = 0;

                int i = 0;
                while (i < s.Length && idx < expected)
                {
                    // Skip leading spaces
                    while (i < s.Length && s[i] == ' ') i++;
                    if (i >= s.Length) break;

                    // Find token end
                    int start = i;
                    while (i < s.Length && s[i] != ' ') i++;
                    ReadOnlySpan<char> tok = s.Slice(start, i - start);

                    // Accept placeholder for missing values
                    // (handles "--" exactly; if your OCR sometimes returns "- -" or similar, we can add that too)
                    if (tok.Length == 2 && tok[0] == '-' && tok[1] == '-')
                    {
                        values[idx++] = null;
                        continue;
                    }

                    // Normal int
                    if (!int.TryParse(tok, out int v))
                        return false;

                    values[idx++] = v;
                }

                // We must see exactly expected tokens (even if some are "--")
                return idx == expected;
            }
        }

        private (Point, Point, Point, Point)? GetSurfaceMask(int? yOffset, int encFrameW, int encFrameH)
        {
            if (!(corner0X.HasValue && corner0Y.HasValue &&
                  corner1X.HasValue && corner1Y.HasValue &&
                  corner2X.HasValue && corner2Y.HasValue &&
                  corner3X.HasValue && corner3Y.HasValue))
            {
                return null;
            }

            const double shrink = 1.0; // keep as-is for identical output

            int yOff = yOffset.GetValueOrDefault();

            // Precompute invariants
            double sx = encFrameW / (double)ViewW;
            double sy = encFrameH / (double)ViewH;

            double cx = encFrameW * 0.5;
            double cy = encFrameH * 0.5;

            // If shrink changes later, these keep the same math but avoid recomputing
            double invShrink = shrink; // just naming clarity

            static int RoundToInt(double v) => (int)Math.Round(v); // preserves current rounding behaviour

            Point Transform(int lx, int ly)
            {
                // 1) scale
                double x = lx * sx;
                double y = ly * sy;

                // 2) vertical flip
                y = encFrameH - y;

                // 3) shrink about centre (no-op for shrink == 1.0, but keep for identical results)
                x = cx + (x - cx) * invShrink;
                y = cy + (y - cy) * invShrink;

                // 4) subtract cropped bar offset
                y -= yOff;

                return new Point(RoundToInt(x), RoundToInt(y));
            }

            var t0 = Transform(corner0X.Value, corner0Y.Value);
            var t1 = Transform(corner1X.Value, corner1Y.Value);
            var t2 = Transform(corner2X.Value, corner2Y.Value);
            var t3 = Transform(corner3X.Value, corner3Y.Value);

            return (t3, t2, t1, t0);
        }
    }
}
