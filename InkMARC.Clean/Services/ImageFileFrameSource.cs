using InkMARC.Clean.Model;
using InkMARC.Clean.Services.Interfaces;
using System.IO;    
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using OpenCvSharp;

namespace InkMARC.Clean.Services
{
    internal class ImageFileFrameSource : IFrameSource
    {
        public int FrameCount => 1;

        public double FramesPerSecond => 1;

        private int _viewW;
        private int _viewH;
        public int ViewW { get => _viewW; set => _viewW = value; }
        public int ViewH { get => _viewH; set => _viewH = value; }

        public string FileFilter => "Image Files|*.png;*.jpg;*.jpeg;*.bmp|All files|*.*";

        public event EventHandler<int>? FrameCountChanged;

        FrameData? _frameData;

        public void Dispose()
        {
            if (_frameData != null)
            {
                _frameData.Image?.Dispose();
                _frameData = null;
            }            
        }

        public bool SupportsPlay => false;

        public bool FileSeek => true;

        public FrameData? GetFrame(int index)
        {
            if (index != 0 || _frameData == null)
            {
                return null;
            }
            return _frameData;
        }

        public void Open(string path)
        {
            if (!string.IsNullOrEmpty(path) && File.Exists(path))
            {
                if (_frameData != null)
                {
                    _frameData.Image?.Dispose();
                }

                var src = Cv2.ImRead(path, ImreadModes.Color);
                if (src == null || src.Empty())
                {
                    _frameData = null;
                    return;
                }

                // Determine which mask to use based on filename pattern like:
                // participant_17_Q1a_20251202_140719.png
                // we take the last character of the 3rd segment (index 2) to choose the mask
                string fileName = Path.GetFileName(path) ?? string.Empty;
                string? maskName = null;
                var parts = fileName.Split('_');
                if (parts.Length >= 3 && !string.IsNullOrEmpty(parts[2]))
                {
                    var seg = parts[2];
                    char last = seg[seg.Length - 1];
                    switch (char.ToUpperInvariant(last))
                    {
                        case 'A':
                            maskName = "Qa.png";
                            break;
                        case 'B':
                            maskName = "Qb.png";
                            break;
                        case 'F':
                            maskName = "QF.png";
                            break;
                        case '2':
                            maskName = "QF2.png";
                            break;
                        case '0':
                            maskName = "Q.png";
                            break;
                    }
                }

                Mat finalMat = src;

                if (maskName != null)
                {
                    // Try several candidate locations for the mask file
                    string? maskPath = null;
                    var candidates = new[]
                    {
                        Path.Combine(Path.GetDirectoryName(path) ?? string.Empty, "resources", "images", maskName),
                        Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "resources", "images", maskName),
                        Path.Combine(Directory.GetCurrentDirectory(), "resources", "images", maskName),
                    };

                    foreach (var c in candidates)
                    {
                        if (!string.IsNullOrEmpty(c) && File.Exists(c))
                        {
                            maskPath = c;
                            break;
                        }
                    }

                    if (maskPath != null)
                    {
                        try
                        {
                            // Load the blank form (same layout as the composite)
                            using var form = Cv2.ImRead(maskPath, ImreadModes.Color);
                            if (form != null && !form.Empty())
                            {
                                // 1) Compute absolute difference between composite and form
                                using var diff = new Mat();
                                Cv2.Absdiff(src, form, diff); // |src - form|

                                // 2) Convert to grayscale and optionally blur to reduce noise
                                using var diffGray = new Mat();
                                Cv2.CvtColor(diff, diffGray, ColorConversionCodes.BGR2GRAY);
                                Cv2.GaussianBlur(diffGray, diffGray, new Size(3, 3), 0);

                                // 3) Threshold: pixels with difference > T are considered "ink"
                                const double thresholdValue = 25;      // tune this if needed
                                using var inkMask = new Mat();
                                Cv2.Threshold(diffGray, inkMask, thresholdValue, 255,
                                              ThresholdTypes.Binary);

                                // clean small specks with morphology
                                using var kernel = Cv2.GetStructuringElement(MorphShapes.Rect, new Size(3, 3));
                                Cv2.MorphologyEx(inkMask, inkMask, MorphTypes.Open, kernel);

                                // 4) Apply mask to original composite to keep only ink pixels
                                using var inkMask3 = new Mat();
                                Cv2.CvtColor(inkMask, inkMask3, ColorConversionCodes.GRAY2BGR);

                                using var inkOnlyBgr = new Mat();
                                Cv2.BitwiseAnd(src, inkMask3, inkOnlyBgr);

                                using var inkFlipped = new Mat();
                                Cv2.Flip(inkOnlyBgr, inkFlipped, FlipMode.X);  // or 0

                                using var maskFlipped = new Mat();
                                Cv2.Flip(inkMask, maskFlipped, FlipMode.X);    // must flip mask the same way

                                // (a) Rotate the blank form 180 degrees (down)
                                using var rotatedForm = new Mat();
                                Cv2.Rotate(form, rotatedForm, RotateFlags.Rotate180);

                                // (b) Use the rotated form as the background
                                using var compositeBgr = rotatedForm.Clone();

                                // (c) Copy only ink pixels (where inkMask is white) onto the background.
                                //     White pixels in inkOnlyBgr are "transparent" because inkMask=0 there.
                                inkFlipped.CopyTo(compositeBgr, maskFlipped);                                

                                // (d) If you want a 4-channel BGRA output, convert:
                                using var final4 = new Mat();
                                Cv2.CvtColor(compositeBgr, final4, ColorConversionCodes.BGR2BGRA);

                                // clean up and assign final
                                src.Dispose();
                                finalMat = final4.Clone();

                            }
                        }
                        catch
                        {
                            // If anything goes wrong, fall back to original composite image
                        }
                    }
                }

                _frameData = new()
                {
                    Image = finalMat

                };
                ViewW = _frameData.Image.Width;
                ViewH = _frameData.Image.Height;
            }
        }

        public FrameData? GetFrameForExport(int index)
        {
            throw new NotImplementedException();
        }
    }
}
