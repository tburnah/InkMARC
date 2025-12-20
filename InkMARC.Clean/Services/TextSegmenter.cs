using OpenCvSharp;
using System.Text;
using System;
using System.Collections.Generic;

namespace InkMARC.Clean.Services
{
    public static class TextSegmenter
    {
        // 7-segment display segment line definitions: (x1, y1, x2, y2)
        public readonly struct SegmentDef
        {
            public readonly float X1, Y1, X2, Y2;
            public SegmentDef(float x1, float y1, float x2, float y2)
            {
                X1 = x1; Y1 = y1; X2 = x2; Y2 = y2;
            }
        }

        public static class SevenSegmentData
        {
            public static readonly SegmentDef[] Segments =
            {
                new SegmentDef(-0.5f,  1.0f,  0.5f,  1.0f), // A (top)
                new SegmentDef( 0.5f,  1.0f,  0.5f,  0.0f), // B (upper-right)
                new SegmentDef( 0.5f,  0.0f,  0.5f, -1.0f), // C (lower-right)
                new SegmentDef(-0.5f,  0.0f,  0.5f,  0.0f), // D (middle)
                new SegmentDef(-0.5f, -1.0f,  0.5f, -1.0f), // E (bottom)
                new SegmentDef(-0.5f,  1.0f, -0.5f,  0.0f), // F (upper-left)
                new SegmentDef(-0.5f,  0.0f, -0.5f, -1.0f)  // G (lower-left)
            };

            // Precomputed centers (nx, ny) for segments so we do not recompute every time
            public static readonly (float Nx, float Ny)[] SegmentCenters;

            static SevenSegmentData()
            {
                SegmentCenters = new (float, float)[Segments.Length];
                for (int i = 0; i < Segments.Length; i++)
                {
                    var s = Segments[i];
                    float nx = (s.X1 + s.X2) * 0.5f;
                    float ny = (s.Y1 + s.Y2) * 0.5f;
                    SegmentCenters[i] = (nx, ny);
                }
            }
        }

        // Bit masks for which segments are lit for each digit.
        private static readonly int[] DigitMask =
        {
            0b111_0111, // 0
            0b000_0110, // 1
            0b101_1011, // 2
            0b001_1111, // 3
            0b010_1110, // 4
            0b011_1101, // 5
            0b111_1101, // 6
            0b000_0111, // 7
            0b111_1111, // 8
            0b010_1111, // 9
            0b000_1000, // - (just the middle segment)
        };

        private static char DecodeDigitChar(byte segMask)
        {
            // 0–9, then '-' as last entry
            for (int i = 0; i < DigitMask.Length; i++)
            {
                if (DigitMask[i] == segMask)
                {
                    if (i <= 9) return (char)('0' + i);
                    if (i == 10) return '-';
                }
            }

            // Fallback if pattern not recognized
            return '?';
        }

        /// <summary>
        /// Segments the top section into lines and then words and returns the decoded digits.
        /// </summary>
        public static string SegmentLinesAndWords(Mat src)
        {
            if (src.Empty()) throw new ArgumentException("Empty image", nameof(src));

            // Optional debug image for drawing rects; currently unused
            Mat result = src.Channels() == 3 ? src.Clone() : src.CvtColor(ColorConversionCodes.GRAY2BGR);

            // 1) Grayscale + binarize: text as white, background as black
            Mat gray = src.Channels() == 1 ? src.Clone() : src.CvtColor(ColorConversionCodes.BGR2GRAY);

            Mat bin = new Mat();
            Cv2.Threshold(gray, bin, 0, 255,
                ThresholdTypes.BinaryInv | ThresholdTypes.Otsu);
            // bin: text = 255, background = 0

            int rows = bin.Rows;
            int cols = bin.Cols;

            // ---------- STEP 1: segment into lines (horizontal histogram) ----------
            var rowSums = new int[rows];
            for (int y = 0; y < rows; y++)
            {
                rowSums[y] = Cv2.CountNonZero(bin.Row(y));
            }

            int maxRowSum = 0;
            for (int i = 0; i < rowSums.Length; i++)
            {
                if (rowSums[i] > maxRowSum) maxRowSum = rowSums[i];
            }
            int rowThreshold = Math.Max(1, maxRowSum / 20); // 5% of max by default

            var lineBands = new List<(int yStart, int yEnd)>();
            bool inBand = false;
            int bandStart = 0;

            for (int y = 0; y < rows; y++)
            {
                bool hasText = rowSums[y] >= rowThreshold;

                if (hasText && !inBand)
                {
                    inBand = true;
                    bandStart = y;
                }
                else if (!hasText && inBand)
                {
                    inBand = false;
                    lineBands.Add((bandStart, y - 1));
                }
            }
            if (inBand)
            {
                lineBands.Add((bandStart, rows - 1));
            }

            var allText = new StringBuilder();

            // ---------- STEP 2: for each line, segment into words (vertical histogram) ----------
            // Reuse buffers across lines where possible
            var colSums = new int[cols];
            var wordBands = new List<(int xStart, int xEnd)>(32);
            var finalWordBands = new List<(int xStart, int xEnd)>(32);
            var digitEntries = new List<(Rect Rect, byte Mask)>(32);

            foreach (var (yStart, yEnd) in lineBands)
            {
                int lineHeight = yEnd - yStart + 1;
                // ROI view, no copy
                Rect lineRect = new Rect(0, yStart, cols, lineHeight);
                Mat lineBin = new Mat(bin, lineRect);

                // Reset reusable collections
                Array.Clear(colSums, 0, cols);
                wordBands.Clear();
                finalWordBands.Clear();
                digitEntries.Clear();

                for (int x = 0; x < cols; x++)
                {
                    colSums[x] = Cv2.CountNonZero(lineBin.Col(x));
                }

                int maxColSum = 0;
                for (int i = 0; i < colSums.Length; i++)
                {
                    if (colSums[i] > maxColSum) maxColSum = colSums[i];
                }
                int colThreshold = Math.Max(1, maxColSum / 10); // 10% of max

                bool inWord = false;
                int wordStart = 0;

                for (int x = 0; x < cols; x++)
                {
                    bool hasInk = colSums[x] >= colThreshold;

                    if (hasInk && !inWord)
                    {
                        inWord = true;
                        wordStart = x;
                    }
                    else if (!hasInk && inWord)
                    {
                        inWord = false;
                        wordBands.Add((wordStart, x - 1));
                    }
                }
                if (inWord)
                {
                    wordBands.Add((wordStart, cols - 1));
                }

                // -------- split wide words from the RIGHT into 2 segments --------
                for (int idx = 0; idx < wordBands.Count; idx++)
                {
                    var (xStart0, xEnd0) = wordBands[idx];
                    int width = xEnd0 - xStart0 + 1;
                    int xStart = xStart0;
                    int xEnd = xEnd0;

                    if (width > 13)
                    {
                        // Right-most segment is exactly 13 pixels wide.
                        int rightStart = xEnd - 13 + 1;

                        // Left segment (if any pixels remain)
                        if (rightStart > xStart)
                        {
                            finalWordBands.Add((xStart, rightStart - 1));
                        }

                        // Right segment (always 13px wide)
                        finalWordBands.Add((rightStart, xEnd));
                    }
                    else
                    {
                        finalWordBands.Add((xStart, xEnd));
                    }
                }

                // Ensure minimum width (e.g. 12 px)
                for (int i = 0; i < finalWordBands.Count; ++i)
                {
                    var (xStart, xEnd) = finalWordBands[i];
                    if (xEnd - xStart < 12)
                    {
                        xStart = xEnd - 12;
                        finalWordBands[i] = (xStart, xEnd);
                    }
                }

                // ---------- Build digit entries for this line ----------
                for (int i = 0; i < finalWordBands.Count; i++)
                {
                    var (xStart, xEnd) = finalWordBands[i];

                    int wordWidth = xEnd - xStart + 1;
                    Rect wordRect = new Rect(xStart, yStart, wordWidth, lineHeight);

                    byte segMask = 0;

                    // Use precomputed centers
                    for (int s = 0; s < 7; s++)
                    {
                        var center = SevenSegmentData.SegmentCenters[s];
                        float nx = center.Nx;
                        float ny = center.Ny;

                        int localX = (int)Math.Round((nx + 0.5f) * (wordRect.Width - 1));
                        int localY = (int)Math.Round((1f - ny) * 0.5f * (wordRect.Height - 1));

                        // inset X by 1px from both sides so we don't sample exactly on the edges
                        if (wordRect.Width > 2)
                            localX = Math.Clamp(localX, 2, wordRect.Width - 2);

                        int imgX = wordRect.X + localX;
                        int imgY = wordRect.Y + localY;

                        imgX = Math.Clamp(imgX, 0, bin.Cols - 1);
                        imgY = Math.Clamp(imgY, 0, bin.Rows - 1);

                        byte val = bin.At<byte>(imgY, imgX);
                        if (val > 0)
                            segMask |= (byte)(1 << s);
                    }

                    digitEntries.Add((wordRect, segMask));
                }

                if (digitEntries.Count == 0)
                    continue; // nothing on this line

                // ---------- Group rects that are within 40px horizontally ----------
                // Sort in-place by X instead of LINQ OrderBy
                digitEntries.Sort((a, b) => a.Rect.X.CompareTo(b.Rect.X));

                var groups = new List<List<(Rect Rect, byte Mask)>>();
                var currentGroup = new List<(Rect Rect, byte Mask)> { digitEntries[0] };
                Rect last = digitEntries[0].Rect;

                for (int i = 1; i < digitEntries.Count; i++)
                {
                    var entry = digitEntries[i];
                    Rect nextRect = entry.Rect;
                    int gap = nextRect.X - (last.X + last.Width); // horizontal gap

                    if (gap <= 40)
                    {
                        currentGroup.Add(entry);

                        // keep 'last' as rightmost rect
                        if (nextRect.X + nextRect.Width > last.X + last.Width)
                            last = nextRect;
                    }
                    else
                    {
                        groups.Add(currentGroup);
                        currentGroup = new List<(Rect Rect, byte Mask)> { entry };
                        last = nextRect;
                    }
                }
                groups.Add(currentGroup);

                // Build text for this line
                var lineText = new StringBuilder();

                foreach (var group in groups)
                {
                    // group is already left-to-right due to sorted digitEntries
                    var wordText = new StringBuilder();
                    for (int i = 0; i < group.Count; i++)
                    {
                        var entry = group[i];
                        char ch = DecodeDigitChar(entry.Mask);
                        wordText.Append(ch);
                    }

                    if (wordText.Length > 0)
                    {
                        if (lineText.Length > 0)
                            lineText.Append(' '); // space between words

                        lineText.Append(wordText);
                    }

                    // Optional: blue box computation left as-is (drawing commented out)
                    int minX = int.MaxValue, minY = int.MaxValue;
                    int maxX = int.MinValue, maxY = int.MinValue;

                    for (int i = 0; i < group.Count; i++)
                    {
                        var r = group[i].Rect;
                        if (r.X < minX) minX = r.X;
                        if (r.Y < minY) minY = r.Y;
                        int rx2 = r.X + r.Width - 1;
                        int ry2 = r.Y + r.Height - 1;
                        if (rx2 > maxX) maxX = rx2;
                        if (ry2 > maxY) maxY = ry2;
                    }

                    Rect groupRect = new Rect(
                        minX,
                        minY,
                        maxX - minX + 1,
                        maxY - minY + 1
                    );

                    //Cv2.Rectangle(result, groupRect, new Scalar(255, 0, 0), 1);
                }

                if (lineText.Length > 0)
                {
                    allText.AppendLine(lineText.ToString());
                }
            }

            return allText.ToString();
        }
    }
}
