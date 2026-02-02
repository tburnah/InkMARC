using System;
using System.Collections.Generic;
using System.Linq;
using OpenCvSharp;

namespace InkMARC.Clean.Services
{
    /// <summary>
    /// Encapsulates the list of background colours, their BGR scalar values and
    /// the corresponding foreground colour selection logic.
    /// </summary>
    public class ColorPalette
    {
        private static readonly string[] _backgroundNames = new[] { "White", "Black", "Gray", "SaddleBrown", "DarkGreen", "Tan" };

        public static IReadOnlyList<string> BackgroundNames => _backgroundNames;

        private readonly Dictionary<string, Scalar> _scalars;
        private readonly Dictionary<string, string> _foreground;

        public ColorPalette()
        {
            _scalars = new Dictionary<string, Scalar>(StringComparer.OrdinalIgnoreCase)
            {
                ["Black"] = new Scalar(0, 0, 0),
                ["White"] = new Scalar(255, 255, 255),
                ["Gray"] = new Scalar(128, 128, 128),
                ["SaddleBrown"] = new Scalar(139, 69, 19),
                ["DarkGreen"] = new Scalar(0, 100, 0),
                ["Tan"] = new Scalar(210, 180, 140),
            };

            _foreground = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase)
            {
                ["White"] = "Black",
                ["Black"] = "White",
                ["Gray"] = "White",
                ["SaddleBrown"] = "White",
                ["DarkGreen"] = "White",
                ["Tan"] = "Black",
            };
        }

        public Scalar GetScalar(string name)
        {
            if (string.IsNullOrEmpty(name)) return _scalars["White"];
            return _scalars.TryGetValue(name, out var s) ? s : _scalars["White"];
        }

        public string GetForeground(string name)
        {
            if (string.IsNullOrEmpty(name)) return _foreground["White"];
            return _foreground.TryGetValue(name, out var f) ? f : _foreground["White"];
        }

        public string Next(string current)
        {
            if (string.IsNullOrEmpty(current)) return _backgroundNames[0];
            int idx = Array.FindIndex(_backgroundNames, n => string.Equals(n, current, StringComparison.OrdinalIgnoreCase));
            if (idx < 0) return _backgroundNames[0];
            return _backgroundNames[(idx + 1) % _backgroundNames.Length];
        }
    }
}
