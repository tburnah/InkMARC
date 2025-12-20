using OpenCvSharp;
using OpenCvSharp.Extensions;
using System;
using System.Diagnostics;
using System.Drawing;
using System.Drawing.Imaging;
using System.IO;

namespace InkMARC.Clean.Services
{
    /// <summary>
    /// Analyzes the top data/text strip and returns OCR text using Tesseract.
    /// Requires a "tessdata" folder with language data to be present in the application directory.
    /// </summary>
    public class DataStripAnalyzer 
    {
        /// <summary>
        /// Extract text from an OpenCv Mat representing the data strip.
        /// Returns null if OCR engine is not available or on error.
        /// </summary>
        public static string? ExtractText(Mat? stripMat)
        {
            if (stripMat == null || stripMat.Empty())
                return null;

            string result = TextSegmenter.SegmentLinesAndWords(stripMat);
          
            return result;
        }
    }
}
