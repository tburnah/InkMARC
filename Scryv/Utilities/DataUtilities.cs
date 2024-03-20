using Camera.MAUI;
using OcuInkTrain.Interfaces;
using OcuInkTrain.Primatives;
using OcuInkTrain.ViewModel;
using System.Text.Json;
#if WINDOWS
using Windows.Storage;
#endif

namespace OcuInkTrain.Utilities
{
    /// <summary>
    /// Provides utility methods for working with data.
    /// </summary>
    public static class DataUtilities
    {
        /// <summary>
        /// Gets the data folder path.
        /// </summary>
        /// <returns></returns>
        public static string GetDataFolder() => FileSystem.AppDataDirectory;

        private static string? videoFolderPath = null;
        /// <summary>
        /// Gets the folder path for storing videos.
        /// </summary>
        /// <returns>The folder path for videos.</returns>
        public static string GetVideosFolderPath()
        {
            if (videoFolderPath is null)
            {
#if WINDOWS
                // Get the videos library
                var videosLibrary = Task.Run(async () => await StorageLibrary.GetLibraryAsync(KnownLibraryId.Videos)).GetAwaiter().GetResult();
                // Get the save folder path
                videoFolderPath = videosLibrary.SaveFolder.Path;
#else
                // Create the folder path
                string folderPath = Path.Combine(FileSystem.AppDataDirectory, "Videos");
                // Create the folder if it doesn't exist
                if (!Directory.Exists(folderPath))
                {
                    Directory.CreateDirectory(folderPath);
                }
                videoFolderPath = folderPath;
#endif
            }
            return videoFolderPath;
        }

        /// <summary>
        /// Gets the video file name based on the exercise and session ID.
        /// </summary>
        /// <param name="exercise">The exercise number.</param>
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

        private static JsonSerializerOptions jssOptions = new() { WriteIndented = true };

        /// <summary>
        /// Saves the advanced drawing lines to a file.
        /// </summary>
        /// <param name="lines">The advanced drawing lines to save.</param>
        /// <param name="filePath">The file path to save the lines to.</param>
        public static void SaveAdvancedDrawingLines(List<IAdvancedDrawingLine> lines, string filePath)
        {
            ExerciseData data = new()
            {
                CameraInfo = CameraWindowViewModel.Current?.Camera ?? null,
                DrawingLines = lines.OfType<ScryvDrawingLine>().ToList()
            };

            // Check if the file exists and has content
            if (File.Exists(filePath) && new FileInfo(filePath).Length > 0)
            {
                // Read the existing content and deserialize it
                string existingJson = File.ReadAllText(filePath);
                data = JsonSerializer.Deserialize<ExerciseData>(existingJson, jssOptions) ?? data;
            }

            // Add the new lines to the existing lines
            foreach (var  advancedLine in lines)
            {
                if (advancedLine is ScryvDrawingLine scryvLine && data is not null && data.DrawingLines is not null)
                    data.DrawingLines.Add(scryvLine);
            }            

            // Serialize the combined list and write/overwrite the file
            string json = JsonSerializer.Serialize(data, jssOptions);
            File.WriteAllText(filePath, json);
        }

        private class ExerciseData
        {
            public CameraInfo? CameraInfo { get; set; }
            public List<ScryvDrawingLine>? DrawingLines { get; set; }
        }
    }
}
