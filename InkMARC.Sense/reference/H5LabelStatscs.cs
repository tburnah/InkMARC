using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace InkMARC.Clean.Services
{
    public static class H5LabelStats
    {
        public sealed class FileStats
        {
            public string Path { get; init; } = string.Empty;
            public ulong FrameCount { get; init; }

            public ulong FramesWithAnyLabels { get; set; }
            public ulong FramesWithNoLabels { get; set; }

            // Only among frames with ANY labels present
            public ulong LabeledFramesPressureGt0 { get; set; }
            public ulong LabeledFramesPressureEq0 { get; set; }
            public ulong LabeledFramesPressureMissing { get; set; } // optional but recommended
        }

        public sealed class DirectoryStats
        {
            public List<FileStats> PerFile { get; } = new();

            public ulong TotalFrames => PerFile.Aggregate<FileStats, ulong>(0, (acc, f) => acc + f.FrameCount);
            public ulong TotalFramesWithAnyLabels => PerFile.Aggregate<FileStats, ulong>(0, (acc, f) => acc + f.FramesWithAnyLabels);
            public ulong TotalFramesWithNoLabels => PerFile.Aggregate<FileStats, ulong>(0, (acc, f) => acc + f.FramesWithNoLabels);

            public ulong TotalLabeledPressureGt0 => PerFile.Aggregate<FileStats, ulong>(0, (acc, f) => acc + f.LabeledFramesPressureGt0);
            public ulong TotalLabeledPressureEq0 => PerFile.Aggregate<FileStats, ulong>(0, (acc, f) => acc + f.LabeledFramesPressureEq0);
            public ulong TotalLabeledPressureMissing => PerFile.Aggregate<FileStats, ulong>(0, (acc, f) => acc + f.LabeledFramesPressureMissing);
        }

        /// <summary>
        /// Scan all .h5 files in a directory and compute label + pressure statistics.
        /// </summary>
        /// <param name="directory">Directory containing .h5 files</param>
        /// <param name="recursive">If true, scan subdirectories</param>
        /// <param name="pressureIndex">Index in labels/mask for pressure (default 2)</param>
        /// <param name="treatNegativePressureAsZero">
        /// If true, pressure &lt;= 0 counts as "== 0". If false, only exactly 0 counts as "== 0".
        /// </param>
        public static DirectoryStats ScanDirectory(
            string directory,
            bool recursive = false,
            int pressureIndex = 2,
            bool treatNegativePressureAsZero = true)
        {
            if (string.IsNullOrWhiteSpace(directory))
                throw new ArgumentException("Directory is null/empty.", nameof(directory));
            if (!Directory.Exists(directory))
                throw new DirectoryNotFoundException(directory);

            var stats = new DirectoryStats();

            var opt = recursive ? SearchOption.AllDirectories : SearchOption.TopDirectoryOnly;
            foreach (var h5Path in Directory.EnumerateFiles(directory, "*.h5", opt))
            {
                var fileStats = ScanSingleFile(
                    h5Path,
                    pressureIndex: pressureIndex,
                    treatNegativePressureAsZero: treatNegativePressureAsZero);

                stats.PerFile.Add(fileStats);
            }

            return stats;
        }

        /// <summary>
        /// Scan a single .h5 session file and compute label + pressure statistics.
        /// </summary>
        public static FileStats ScanSingleFile(
            string h5Path,
            int pressureIndex = 2,
            bool treatNegativePressureAsZero = true)
        {
            if (string.IsNullOrWhiteSpace(h5Path))
                throw new ArgumentException("Path is null/empty.", nameof(h5Path));
            if (!File.Exists(h5Path))
                throw new FileNotFoundException(h5Path);

            using var session = SessionManager.OpenExisting(h5Path, aviPathOverride: null, writeable: false);

            var fs = new FileStats
            {
                Path = h5Path,
                FrameCount = session.FrameCount
            };

            // Allocate once per file, reuse per frame
            var corners = new float[8];                  // required by your ReadMetadata signature, can pass null if you change it
            var labels = new float[session.AttrCount];
            var mask = new byte[session.AttrCount];

            // We don't actually need corners or timestamp for counting; but ReadMetadata currently reads timestamp anyway.
            for (ulong t = 0; t < session.FrameCount; t++)
            {
                session.ReadMetadata(
                    frameIndex: t,
                    timestampNs: out _,
                    corners: null,    // corners not needed for this task
                    labels: labels,
                    labelMask: mask);

                bool anyLabel = AnyNonZero(mask);
                if (!anyLabel)
                {
                    fs.FramesWithNoLabels++;
                    continue;
                }

                fs.FramesWithAnyLabels++;

                // Among labeled frames: evaluate pressure
                if (pressureIndex < 0 || pressureIndex >= mask.Length)
                {
                    // If the file doesn't even have that attribute, treat as "missing"
                    fs.LabeledFramesPressureMissing++;
                    continue;
                }

                if (mask[pressureIndex] == 0)
                {
                    // There are labels, but pressure specifically is not set for this frame
                    fs.LabeledFramesPressureMissing++;
                    continue;
                }

                float p = labels[pressureIndex];

                if (p > 0)
                {
                    fs.LabeledFramesPressureGt0++;
                }
                else
                {
                    // either p == 0, or (optionally) p < 0 treated as "0"
                    if (treatNegativePressureAsZero || p == 0)
                        fs.LabeledFramesPressureEq0++;
                    else
                        fs.LabeledFramesPressureMissing++; // or create a separate "negative pressure" bucket
                }
            }

            // Sanity: these should sum to FrameCount
            // (If you want, assert in Debug.)
            return fs;
        }

        private static bool AnyNonZero(byte[] buffer)
        {
            // Fast enough; if you care, you can unroll or use Span<byte>
            for (int i = 0; i < buffer.Length; i++)
                if (buffer[i] != 0)
                    return true;
            return false;
        }
    }
}
