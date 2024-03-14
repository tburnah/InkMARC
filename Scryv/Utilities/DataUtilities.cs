using Scryv.Interfaces;
using Scryv.Primatives;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Text.Json;
using System.Threading.Tasks;
#if WINDOWS
using Windows.Storage;
#endif

namespace Scryv.Utilities
{
    /// <summary>
    /// Provides utility methods for working with data.
    /// </summary>
    public static class DataUtilities
    {
        public static string GetDataFolder() => FileSystem.AppDataDirectory;

        private static string GetVideosFolderPath()
        {
#if WINDOWS
            return KnownFolders.VideosLibrary.Path;
#else
            string folderPath = 
                Path.Combine(FileSystem.AppDataDirectory, "Videos");
            if (!Directory.Exists(folderPath))
            {
                Directory.CreateDirectory(folderPath);
            }
            return folderPath;
#endif
        }

        /// <summary>
        /// Gets the video file name based on the exercise and session ID.
        /// </summary>
        /// <param name="exercise">The exercise number.</param>
        /// <param name="camera">The camera index</param>
        /// <param name="timestamp">The filetime</param>
        /// <returns>The video file name.</returns>
        public static string GetVideoFileName(long timestamp, int exercise)
        {
#if IOS
            return $"video_{SessionContext.FilePathSessionID}_{timestamp}_{exercise}.mov";
#else
            return $"video_{SessionContext.FilePathSessionID}_{timestamp}_{exercise}.mp4";
#endif
        }

        /// <summary>
        /// Gets the data file name based on the current session ID.
        /// </summary>
        /// <returns>The data file name.</returns>
        public static string GetDataFileName() => Path.Combine(GetVideosFolderPath(), $"data_{SessionContext.FilePathSessionID}.json");

        /// <summary>
        /// Gets the data file name based on the current session ID.
        /// </summary>
        /// <returns>The data file name.</returns>
        public static string GetDataFileName(int exercise) => Path.Combine(GetVideosFolderPath(), $"data_{SessionContext.FilePathSessionID}_{exercise}.json");

        /// <summary>
        /// Saves the advanced drawing lines to a file.
        /// </summary>
        /// <param name="lines">The advanced drawing lines to save.</param>
        /// <param name="filePath">The file path to save the lines to.</param>
        public static void SaveAdvancedDrawingLines(List<IAdvancedDrawingLine> lines, string filePath)
        {
            List<ScryvDrawingLine> existingLines = new List<ScryvDrawingLine>();

            // Check if the file exists and has content
            if (File.Exists(filePath) && new FileInfo(filePath).Length > 0)
            {
                // Read the existing content and deserialize it
                string existingJson = File.ReadAllText(filePath);
                foreach (var line in JsonSerializer.Deserialize<List<ScryvDrawingLine>>(existingJson))
                {
                    existingLines.Add(line);
                }
            }

            // Add the new lines to the existing lines
            foreach (var  advancedLine in lines)
            {
                existingLines.Add(advancedLine as ScryvDrawingLine);
            }            

            // Serialize the combined list and write/overwrite the file
            string json = JsonSerializer.Serialize(existingLines, new JsonSerializerOptions { WriteIndented = true });
            File.WriteAllText(filePath, json);
        }
    }
}
