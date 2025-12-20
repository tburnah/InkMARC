using System;
using System.IO;
using System.Linq;

namespace InkMARC.Clean.Services
{
    public static class FileHelpers
    {
        /// <summary>
        /// Returns the next file (by name, alphanumeric) in the same directory
        /// with the same extension as the given file, or null if there is none.
        /// </summary>
        public static string? GetNextFileInDirectory(string filePath)
            => GetAdjacentFile(filePath, +1);

        /// <summary>
        /// Returns the previous file (by name, alphanumeric) in the same directory
        /// with the same extension as the given file, or null if there is none.
        /// </summary>
        public static string? GetPreviousFileInDirectory(string filePath)
            => GetAdjacentFile(filePath, -1);

        /// <summary>
        /// Core helper: offset = +1 for next, -1 for previous.
        /// </summary>
        private static string? GetAdjacentFile(string filePath, int offset)
        {
            if (string.IsNullOrWhiteSpace(filePath))
                throw new ArgumentException("filePath is null or empty.", nameof(filePath));

            if (!File.Exists(filePath))
                throw new FileNotFoundException("The specified file does not exist.", filePath);

            var directory = Path.GetDirectoryName(filePath);
            if (string.IsNullOrEmpty(directory))
                return null;

            var fileName = Path.GetFileName(filePath);
            var ext = Path.GetExtension(filePath);      // e.g. ".png"

            // Use same file type: "*.png", "*.jpg", etc.
            var searchPattern = "*" + ext;

            // Get all files with the same extension in alphanumeric order
            var files = Directory
                .EnumerateFiles(directory, searchPattern)
                .OrderBy(f => Path.GetFileName(f), StringComparer.OrdinalIgnoreCase)
                .ToList();

            var index = files.FindIndex(f =>
                string.Equals(Path.GetFileName(f), fileName, StringComparison.OrdinalIgnoreCase));

            if (index == -1)
            {
                // File not found in list (different extension or race condition)
                return null;
            }

            var newIndex = index + offset;
            if (newIndex < 0 || newIndex >= files.Count)
            {
                // No previous/next file in range
                return null;
            }

            return files[newIndex];
        }
    }
}
