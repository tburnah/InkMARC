using CommunityToolkit.Mvvm.ComponentModel;
using InkMARC.Label.Services;
using InkMARC.Models.Primatives;
using Microsoft.VisualBasic.FileIO;
using OpenCvSharp;
using System.IO;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace InkMARC.Label
{
    /// <summary>
    /// Represents session information for a video exercise.
    /// </summary>
    public partial class ProjectInfo : ObservableObject
    {
        /// <summary>
        /// Gets or sets the path to the video file.
        /// </summary>
        public string VideoPath { get; set; }

        /// <summary>
        /// Gets or sets the path to the data file.
        /// </summary>
        public string? DataPath { get; set; }

        /// <summary>
        /// Gets or sets the path to the bounds file, if applicable.
        /// </summary>
        public string? BoundsPath { get; set; }

        /// <summary>
        /// Gets or sets the path to the H5 file.
        /// </summary>
        public string? H5Path { get; set; }

        /// <summary>
        /// Gets or sets the session ID.
        /// </summary>
        public string SessionID { get; set; }

        /// <summary>
        /// Gets or sets the exercise number.
        /// </summary>
        public int Exercise { get; set; }

        /// <summary>
        /// Gets or sets the offset of the first point.
        /// </summary>
        public long FirstPointOffset { get; set; }

        /// <summary>
        /// Gets or sets the date and time of the video.
        /// </summary>
        public DateTime? VideoDateTime { get; set; }

        /// <summary>
        /// Gets or sets the date and time of the data.
        /// </summary>
        public DateTime? DataDataTime { get; set; }

        /// <summary>
        /// Gets a value indicating whether the session has data.
        /// </summary>
        public bool HasData => !string.IsNullOrEmpty(DataPath);

        /// <summary>
        /// Gets a value indicating whether the session has an H5 file.
        /// </summary>
        public bool HasH5 => !string.IsNullOrEmpty(H5Path);

        /// <summary>
        /// Gets a value indicating whether the session has bounds data.
        /// </summary>
        public bool HasBounds => !string.IsNullOrEmpty(BoundsPath);

        /// <summary>
        /// Gets or sets the start frame of the session.
        /// </summary>
        public int StartFrame { get; set; }

        /// <summary>
        /// Gets or sets the stop frame of the session.
        /// </summary>
        public int StopFrame { get; set; }

        /// <summary>
        /// List of frames to ignore during processing.
        /// </summary>
        /// <summary>
        /// Stores state changes for the session.
        /// </summary>
        [JsonInclude]
        public SortedList<int, bool> IgnoredFrames { get; set; } = [];

        [JsonInclude]
        public int Rotation { get; set; }

        [JsonInclude]
        public SortedList<int, (int x, int y)> BoundOffsets { get; private set; } = new();

        [JsonInclude]
        public SortedList<int, (int x, int y)> CornerOffsetTL { get; set; } = new();

        [JsonInclude]
        public SortedList<int, (int x, int y)> CornerOffsetTR { get; set; } = new();

        [JsonInclude]
        public SortedList<int, (int x, int y)> CornerOffsetBL { get; set; } = new();

        [JsonInclude]
        public SortedList<int, (int x, int y)> CornerOffsetBR { get; set; } = new();        

        [JsonInclude]
        public SortedList<int, float> BoundRotations { get; private set; } = new();

        [JsonInclude]
        public SortedList<int, float> BoundScales { get; private set; } = new();

        [JsonInclude]
        public List<float> TouchPredition { get; private set; } = new();

        [JsonConverter(typeof(Point2fDictionaryConverter))]
        [JsonInclude]
        public Dictionary<int, Point2f[]> CenterPoints { get; private set; } = new();

        [JsonConverter(typeof(Point2fDictionaryConverter))]
        [JsonInclude]
        public Dictionary<int, Point2f[]> InferredBounds { get; private set; } = new();

        [JsonIgnore]
        public List<InkMARCPoint>? DrawingLine;

        public float TouchThreshold { get; set; } = 0.5f;

        /// <summary>
        /// Updates the H5 file path and notifies property changes.
        /// </summary>
        /// <param name="h5Path">The new H5 file path.</param>
        public void UpdateH5Path(string h5Path)
        {
            H5Path = h5Path;
            OnPropertyChanged(nameof(H5Path));
            OnPropertyChanged(nameof(HasH5));
        }

        /// <summary>
        /// Stores state changes for the session.
        /// </summary>
        [JsonInclude]
        public SortedList<int, bool> StateChanges { get; set; } = [];

        /// <summary>
        /// Creates a blank instance of ProjectInfo
        /// </summary>
        public ProjectInfo()
        {
            VideoPath = string.Empty;
            SessionID = string.Empty;
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="ProjectInfo"/> class.
        /// </summary>
        /// <param name="sessionID">The session ID.</param>
        /// <param name="videoPath">The path to the video file.</param>
        /// <param name="exercise">The exercise number.</param>
        /// <param name="dataPath">The path to the data file.</param>
        /// <param name="h5Path">The path to the H5 file.</param>
        /// <param name="videoDateTime">The date and time of the video.</param>
        /// <param name="dataDataTime">The date and time of the data.</param>
        public ProjectInfo(string sessionID, string videoPath, int exercise, string? dataPath, string? h5Path, string? boundsPath, DateTime? videoDateTime, DateTime? dataDataTime)
        {
            SessionID = sessionID;
            VideoPath = videoPath;
            DataPath = dataPath;
            H5Path = h5Path;
            BoundsPath = boundsPath;
            Exercise = exercise;
            VideoDateTime = videoDateTime;
            DataDataTime = dataDataTime;
            FirstPointOffset = -1;
        }

        /// <summary>
        /// Saves the session information to a file.
        /// </summary>
        public void SaveToFile()
        {
            string directoryPath = Path.GetDirectoryName(VideoPath) ?? SpecialDirectories.MyDocuments;
            string filePath = Path.Combine(directoryPath, SessionID + "_" + Exercise + ".session");

            var options = new JsonSerializerOptions
            {
                WriteIndented = true,
                DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull
            };
            options.Converters.Add(new Point2fArrayConverter());
            options.Converters.Add(new Point2fDictionaryConverter());
            options.Converters.Add(new IntPairValueTupleConverter()); 

            string json = JsonSerializer.Serialize(this, options);
            File.WriteAllText(filePath, json);
        }

        /// <summary>
        /// Loads session information from a file.
        /// </summary>
        /// <param name="filePath">The path to the session file.</param>
        /// <returns>The loaded session information, or null if the file does not exist.</returns>
        public static ProjectInfo? LoadFromFile(string filePath)
        {
            if (!File.Exists(filePath))
                return null;

            string json = File.ReadAllText(filePath);

            var options = new JsonSerializerOptions();
            options.Converters.Add(new Point2fArrayConverter());
            options.Converters.Add(new Point2fDictionaryConverter());      
            options.Converters.Add(new IntPairValueTupleConverter());

            return JsonSerializer.Deserialize<ProjectInfo>(json, options);
        }
    }
}