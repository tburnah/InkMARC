using Camera.MAUI;
using OcuInk.Models.Interfaces;
using OcuInk.Models.Primatives;
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
        public static string GetDataFileName() => Path.Combine(GetVideosFolderPath(), $"data_{DateTime.Now.ToFileTime()}_{SessionContext.FilePathSessionID}.json");

        /// <summary>
        /// Gets the data file name based on the current session ID.
        /// </summary>
        /// <returns>The data file name.</returns>
        public static string GetDataFileName(int exercise) => Path.Combine(GetVideosFolderPath(), $"data_{DateTime.Now.ToFileTime()}_{SessionContext.FilePathSessionID}_{exercise}.json");

        /// <summary>
        /// Gets the image file name based on the current session ID.`
        /// </summary>
        /// <param name="exercise"></param>
        /// <returns></returns>
        public static string GetImageFileName(int exercise) => Path.Combine(GetVideosFolderPath(), $"image_{DateTime.Now.ToFileTime()}_{SessionContext.FilePathSessionID}_{exercise}.png");

        private static JsonSerializerOptions jssOptions = new() { WriteIndented = true };

        /// <summary>
        /// Saves the advanced drawing lines to a file.
        /// </summary>
        /// <param name="lines">The advanced drawing lines to save.</param>
        /// <param name="filePath">The file path to save the lines to.</param>
        public static void SaveAdvancedDrawingLines(List<IAdvancedDrawingLine> lines, string filePath)
        {
            var otherCameraInfo = CameraWindowViewModel.Current?.Camera;
            
            ExerciseData data = new()
            {
                CameraInfo = new OcuInk.Models.Primatives.CameraInfo()
                {
                    Name = otherCameraInfo?.Name ?? "Unknown",
                    DeviceId = otherCameraInfo?.DeviceId ?? "Unknown",
                    Position = otherCameraInfo is null ? OcuInk.Models.Primatives.CameraPosition.Unknown 
                                                       : otherCameraInfo.Position == Camera.MAUI.CameraPosition.Front ? OcuInk.Models.Primatives.CameraPosition.Front 
                                                       : otherCameraInfo.Position == Camera.MAUI.CameraPosition.Back ? OcuInk.Models.Primatives.CameraPosition.Back 
                                                       : OcuInk.Models.Primatives.CameraPosition.Unknown,
                    SelectedResolution = otherCameraInfo is null ? new OcuInkSize(640, 480) 
                                                                 : new OcuInkSize((float)otherCameraInfo.SelectedResolution.Width, (float)otherCameraInfo.SelectedResolution.Height),
                    EncodingQuality = otherCameraInfo?.EncodingQuality ?? "VGA"                     
                },
                DrawingLines = lines.OfType<OcuInkDrawingLine>().ToList()
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
                if (advancedLine is OcuInkDrawingLine ocuInkLine && data is not null && data.DrawingLines is not null)
                    data.DrawingLines.Add(ocuInkLine);
            }            

            // Serialize the combined list and write/overwrite the file
            string json = JsonSerializer.Serialize(data, jssOptions);
            File.WriteAllText(filePath, json);
        }

        private class ExerciseData
        {
            public OcuInk.Models.Primatives.CameraInfo? CameraInfo { get; set; }
            public List<OcuInkDrawingLine>? DrawingLines { get; set; }
        }
    }
}
