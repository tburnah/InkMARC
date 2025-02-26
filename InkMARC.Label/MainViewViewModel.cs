using System.Drawing;
using System.IO;
using System.Text.Json;
using System.Text.RegularExpressions;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using InkMARC.Models.Primatives;
using OpenCvSharp;
using OpenCvSharp.Extensions;
using Microsoft.WindowsAPICodePack.Dialogs;
using System.Collections.ObjectModel;
using System.Globalization;
using System.Text.Json.Serialization;
using Microsoft.VisualBasic.FileIO;

namespace InkMARC.Label
{
    /// <summary>
    /// Represents session information for a video exercise.
    /// </summary>
    internal partial class SessionInfo : ObservableObject
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
        /// Gets or sets the start frame of the session.
        /// </summary>
        public int StartFrame { get; set; }

        /// <summary>
        /// Gets or sets the stop frame of the session.
        /// </summary>
        public int StopFrame { get; set; }

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
        public SortedDictionary<int, bool> StateChanges = new SortedDictionary<int, bool>();

        /// <summary>
        /// Creates a blank instance of SessionInfo
        /// </summary>
        public SessionInfo()
        {
            VideoPath = string.Empty;
            SessionID = string.Empty;
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="SessionInfo"/> class.
        /// </summary>
        /// <param name="sessionID">The session ID.</param>
        /// <param name="videoPath">The path to the video file.</param>
        /// <param name="exercise">The exercise number.</param>
        /// <param name="dataPath">The path to the data file.</param>
        /// <param name="h5Path">The path to the H5 file.</param>
        /// <param name="videoDateTime">The date and time of the video.</param>
        /// <param name="dataDataTime">The date and time of the data.</param>
        public SessionInfo(string sessionID, string videoPath, int exercise, string? dataPath, string? h5Path, DateTime? videoDateTime, DateTime? dataDataTime)
        {
            SessionID = sessionID;
            VideoPath = videoPath;
            DataPath = dataPath;
            H5Path = h5Path;
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
                WriteIndented = true, // Pretty print JSON
                DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull // Ignore null values
            };

            string json = JsonSerializer.Serialize(this, options);
            File.WriteAllText(filePath, json);
        }

        /// <summary>
        /// Loads session information from a file.
        /// </summary>
        /// <param name="filePath">The path to the session file.</param>
        /// <returns>The loaded session information, or null if the file does not exist.</returns>
        public static SessionInfo? LoadFromFile(string filePath)
        {
            if (!File.Exists(filePath))
                return null;

            string json = File.ReadAllText(filePath);
            return JsonSerializer.Deserialize<SessionInfo>(json);
        }
    }

    internal partial class MainViewViewModel : ObservableObject
    {
        #region Private Data

        private VideoCapture? videoCapture;
        private int frameCount = 0;
        private int frameIndex = 0;
        private string recordName = string.Empty;
        private List<InkMARCPoint>? _drawingLine;
        private double framesPerSecond = 0;
        private ulong firstDataTimeStamp = 0;
        private ObservableCollection<SessionInfo> sessions = new();
        private SessionInfo? currentExercise;
        private string? formattedJson;
        private int _frameWidth;
        private int _frameHeight;
        private int _squareSize;
        private Mat _squareFrame = null!;
        private Mat _rotatedFrame = null!;
        private Rect _roi;
        private Mat _rotationMatrix = null!;
        private bool isTouched = false;
        private int rotation;
        private Bitmap? currentBitmap = null;
        private bool currentState = false;
        private ImageSource? _currentImage;
        private bool showProgressBar = false;

        #endregion

        #region Public Properties

        public int FrameIndex
        {
            get => frameIndex;
            set => frameIndex = value;
        }

        public int FrameCount => frameCount;

        public int StartFrame
        {
            get => CurrentExercise?.StartFrame ?? 0;
            set => CurrentExercise.StartFrame = value;
        }

        public int StopFrame
        {
            get => CurrentExercise?.StopFrame ?? 0;
            set => CurrentExercise.StopFrame = value;
        }

        public bool CurrentState => currentState;

        public ImageSource? CurrentImage
        {
            get => _currentImage;
            private set
            {
                _currentImage = value;
                OnPropertyChanged(nameof(CurrentImage));
            }
        }
        public ObservableCollection<SessionInfo> Sessions => sessions;
        public System.Windows.Media.Brush IsTouched => isTouched ? System.Windows.Media.Brushes.Red : System.Windows.Media.Brushes.Black;

        public int MaxProgress => StopFrame - StartFrame;
        public bool ShowProgressBar
        {
            get => showProgressBar;
            private set
            {
                SetProperty(ref showProgressBar, value);
                OnPropertyChanged(nameof(ShowCurrentIndex));
            }
        }
        public int Rotation => rotation;
        public bool ShowCurrentIndex
        {
            get => !showProgressBar;
        }

        public bool HasData => CurrentExercise is not null && CurrentExercise.HasData;
        public long StartingPoint => CurrentExercise?.FirstPointOffset ?? -1;

        public SessionInfo CurrentExercise
        {
            get => currentExercise ?? new SessionInfo();
            set
            {
                if (SetProperty(ref currentExercise, value))
                {
                    CurrentExercise?.SaveToFile();
                    FrameIndex = 0;
                    OnPropertyChanged(nameof(FrameIndex));
                    LoadSessionVideo(value);
                    LoadSessionJson(value);
                    OnPropertyChanged(nameof(HasData));
                    OnPropertyChanged(nameof(StartFrame));
                    OnPropertyChanged(nameof(StopFrame));
                    OnPropertyChanged(nameof(StartingPoint));
                    
                    if (CurrentExercise is not null && CurrentExercise.StopFrame == 0)
                        CurrentExercise.StopFrame = FrameCount;
                }
            }
        }

        public string FormattedJson
        {
            get => formattedJson ?? string.Empty; 
            set => SetProperty(ref formattedJson, value);
        }

        #endregion

        #region Relay Commands

        [RelayCommand]
        private void ClearData()
        {
            CurrentExercise?.StateChanges.Clear();
            UpdateCurrentState();
        }

        [RelayCommand]
        private void RecordStart()
        {
            if (frameIndex < frameCount) StartFrame = frameIndex;
            if (StopFrame <= frameIndex) StopFrame = frameIndex + 1;
            OnPropertyChanged(nameof(StartFrame));
            OnPropertyChanged(nameof(StopFrame));
        }

        [RelayCommand]
        private void RecordStop()
        {
            StopFrame = frameIndex;
            if (frameIndex > 0) StopFrame = frameIndex - 1;
            if (StartFrame >= StopFrame) StartFrame = StopFrame - 1;
            OnPropertyChanged(nameof(StartFrame));
            OnPropertyChanged(nameof(StopFrame));
        }

        [RelayCommand]
        private async Task ExportData3Async()
        {
            OnPropertyChanged(nameof(MaxProgress));
            ShowProgressBar = true;

            if (string.IsNullOrEmpty(recordName))
                recordName = Path.GetFileNameWithoutExtension(CurrentExercise?.VideoPath) + ".h5";

            // Create the output file.
            DataSave.CreateFile(recordName);

            // Setup progress reporting.
            var progress = new Progress<int>(value =>
            {
                FrameIndex = value;
                OnPropertyChanged(nameof(FrameIndex));
            });
            int progCounter = 0;

            await Task.Run(() =>
            {
                if (videoCapture is null)
                    return;

                // Use a single Mat for reading frames.
                using Mat frame = new();

                // Set the capture to the first frame.
                if (!videoCapture.Set(VideoCaptureProperties.PosFrames, StartFrame))
                {
                    Console.WriteLine($"Failed to set frame index {StartFrame}");
                    return;
                }

                // Read the first frame.
                if (!videoCapture.Read(frame) || frame.Empty())
                {
                    Console.WriteLine("Failed to retrieve first frame.");
                    return;
                }

                // Initialize image processing resources.
                int frameWidth = frame.Width;
                int frameHeight = frame.Height;
                int squareSize = Math.Max(frameWidth, frameHeight);

                // Allocate reusable Mats.
                using Mat squareFrame = new(new OpenCvSharp.Size(squareSize, squareSize), frame.Type(), Scalar.Black);
                using Mat rotatedFrame = new(new OpenCvSharp.Size(448, 448), frame.Type());

                // Calculate ROI for centering the frame.
                int xOffset = (squareSize - frameWidth) / 2;
                int yOffset = (squareSize - frameHeight) / 2;
                Rect roi = new(xOffset, yOffset, frameWidth, frameHeight);

                // Compute the rotation matrix with an embedded scale factor to fit 448x448.
                float scaleFactor = 448.0f / squareSize;
                Point2f center = new Point2f(squareSize / 2f, squareSize / 2f);
                using Mat rotationMatrix = Cv2.GetRotationMatrix2D(center, rotation, scaleFactor);

                // Process the first frame.
                squareFrame.SetTo(Scalar.Black);
                using (Mat roiMat = new Mat(squareFrame, roi))
                {
                    frame.CopyTo(roiMat);
                }
                Cv2.WarpAffine(squareFrame, rotatedFrame, rotationMatrix, new OpenCvSharp.Size(448, 448));

                using (Bitmap firstImage = BitmapConverter.ToBitmap(rotatedFrame))
                {
                    DataSave.InitializeChunkedDatasets(firstImage, GetStateAtFrame(StartFrame));
                }
                progCounter++;
                ((IProgress<int>)progress).Report(progCounter);

                // Process remaining frames.
                for (frameIndex = StartFrame + 1; frameIndex < StopFrame; frameIndex++)
                {
                    if (!videoCapture.Read(frame) || frame.Empty())
                    {
                        Console.WriteLine($"Skipping frame {frameIndex} because image retrieval failed.");
                        continue;
                    }

                    squareFrame.SetTo(Scalar.Black);
                    using (Mat roiMat = new Mat(squareFrame, roi))
                    {
                        frame.CopyTo(roiMat);
                    }
                    Cv2.WarpAffine(squareFrame, rotatedFrame, rotationMatrix, new OpenCvSharp.Size(448, 448));

                    using (Bitmap image = BitmapConverter.ToBitmap(rotatedFrame))
                    {
                        DataSave.WriteFrame(image, GetStateAtFrame(frameIndex));
                    }

                    progCounter++;
                    if (progCounter % 10 == 0) // Update UI every 10 frames.
                    {
                        ((IProgress<int>)progress).Report(progCounter);
                    }
                }
            });

            // Finalize and update.
            DataSave.FinalizeDatasets();
            if (File.Exists(recordName))
            {
                CurrentExercise.UpdateH5Path(recordName);
            }
            ShowProgressBar = false;
        }


        [RelayCommand]
        private void IncreaseRotation()
        {
            rotation = (rotation + 90) % 360;
            OnPropertyChanged(nameof(Rotation));
            UpdateImage();
        }

        [RelayCommand]
        private void DecreaseRotation()
        {
            if (rotation > 0)
                rotation -= 90;
            else
                rotation = 270;

            OnPropertyChanged(nameof(Rotation));
            UpdateImage();
        }

        [RelayCommand]
        public void ToggleTouched()
        {
            isTouched = !isTouched;
            OnPropertyChanged(nameof(IsTouched));
            if (CurrentExercise?.StateChanges.ContainsKey(frameIndex) ?? false)
            {
                CurrentExercise?.StateChanges.Remove(frameIndex);
            }
            CurrentExercise?.StateChanges.Add(frameIndex, isTouched);
            UpdateCurrentState();
        }

        [RelayCommand]
        public async Task AnalyzeFramesForStateChangesAsync()
        {
            if (CurrentExercise == null)
                return;

            OnPropertyChanged(nameof(MaxProgress));
            ShowProgressBar = true;

            // Optionally clear any existing state changes
            CurrentExercise.StateChanges.Clear();

            bool sequenceActive = false; // Tracks if we are inside a sequence of frames with a datapoint
            int startFrame = CurrentExercise.StartFrame;
            int stopFrame = CurrentExercise.StopFrame;
            int originalFrameIndex = frameIndex;

            // Pre-calculate constants.
            double frameDurationMs = 1000.0 / framesPerSecond;
            double thresholdMs = frameDurationMs / 2.0;
            double thresholdUs = thresholdMs * 1000.0;

            const int progressUpdateFrequency = 10;

            var progress = new Progress<int>(value =>
            {
                FrameIndex = value;
                OnPropertyChanged(nameof(FrameIndex));
            });

            // Run the analysis on a background thread.
            await Task.Run(() =>
            {
                // Loop through each frame between start and stop
                for (int i = startFrame; i <= stopFrame; i++)
                {
                    // Check if there is a datapoint for this frame.
                    bool hasDataPoint = FindClosestDataPointOptimized(i, thresholdUs) != null;

                    if (!sequenceActive && hasDataPoint)
                    {
                        // We just entered a sequence where frames have a datapoint.
                        CurrentExercise.StateChanges[i] = true;
                        sequenceActive = true;
                    }
                    else if (sequenceActive && !hasDataPoint)
                    {
                        // We just left a sequence: record the first frame where no datapoint is available.
                        CurrentExercise.StateChanges[i] = false;
                        sequenceActive = false;
                    }

                    // Report progress as the number of frames processed.
                    // Report progress only every 'progressUpdateFrequency' frames.
                    if ((i - startFrame) % progressUpdateFrequency == 0)
                    {
                        ((IProgress<int>)progress).Report(i - startFrame + 1);
                    }
                }
            });

            if (sequenceActive && stopFrame >= startFrame)
            {
                CurrentExercise.StateChanges[stopFrame] = false;
            }

            ShowProgressBar = false;

            // Restore the original frame index, if desired.
            frameIndex = originalFrameIndex;
        }

        [RelayCommand]
        public void MoveOffset(string parameter)
        {
            if (int.TryParse(parameter, out int adjust))
            {
                if ((frameIndex + adjust) >= 0 && (frameIndex + adjust) < FrameCount)
                {
                    frameIndex += adjust;
                    OnPropertyChanged(nameof(FrameIndex));
                    UpdateImage();
                    UpdateCurrentState();
                    if (CurrentExercise.FirstPointOffset >= 0)
                    {
                        UpdateFormattedJson();
                    }
                }
            }
        }

        [RelayCommand]
        public void LoadFolder(object parameter)
        {
            var folderDialog = new CommonOpenFileDialog
            {
                IsFolderPicker = true,
                Title = "Select Folder Containing Video Files"
            };

            if (folderDialog.ShowDialog() != CommonFileDialogResult.Ok ||
                string.IsNullOrEmpty(folderDialog.FileName))
            {
                return; // User cancelled.
            }

            string directory = folderDialog.FileName;

            // Retrieve files using file extensions.
            var availableVideos = Directory.EnumerateFiles(directory)
                .Where(file => IsVideoFile(file))
                .ToList();

            var availableJson = Directory.GetFiles(directory, "*.json").ToList();
            var availableH5 = Directory.GetFiles(directory, "*.h5").ToList();
            var availableData = Directory.GetFiles(directory, "*.session").ToList();

            // Build dictionaries for quick lookup.
            var videoSessionIds = BuildSessionIdDictionary(availableVideos);
            var dataSessionIds = BuildSessionIdDictionary(availableJson);
            var h5SessionIds = BuildSessionIdDictionarySimple(availableH5);
            var sessionData = BuildSessionDataDictionary(availableData);

            // Combine the data from video, JSON, H5, and session files.
            foreach (var sessionEntry in videoSessionIds)
            {
                string sessionId = sessionEntry.Key;
                foreach (var exerciseEntry in sessionEntry.Value)
                {
                    int exercise = exerciseEntry.Key;
                    var videoInfo = exerciseEntry.Value;
                    string videoFile = videoInfo.Item1;
                    DateTime? videoDate = videoInfo.Item2;

                    // Initialize default values.
                    string dataFile = string.Empty;
                    string h5File = string.Empty;
                    DateTime? dataDate = null;

                    if (dataSessionIds.TryGetValue(sessionId, out var dataDict) &&
                        dataDict.TryGetValue(exercise, out var dataInfo))
                    {
                        dataFile = dataInfo.Item1;
                        dataDate = dataInfo.Item2;
                    }
                    if (h5SessionIds.TryGetValue(sessionId, out var h5Dict) &&
                        h5Dict.TryGetValue(exercise, out var h5FileName))
                    {
                        h5File = h5FileName;
                    }

                    if (sessionData.TryGetValue(sessionId, out var sessionDict) &&
                        sessionDict.TryGetValue(exercise, out var sessionFile))
                    {
                        var newSessionInfo = SessionInfo.LoadFromFile(sessionFile)
                                             ?? new SessionInfo(sessionId, videoFile, exercise, dataFile, h5File, videoDate, dataDate);
                        // Update paths if needed.
                        newSessionInfo.DataPath = dataFile;
                        newSessionInfo.H5Path = h5File;
                        Sessions.Add(newSessionInfo);
                    }
                    else
                    {
                        Sessions.Add(new SessionInfo(sessionId, videoFile, exercise, dataFile, h5File, videoDate, dataDate));
                    }
                }
            }
        }

        /// <summary>
        /// Checks if the file extension indicates a video file.
        /// </summary>
        private bool IsVideoFile(string file)
        {
            string ext = Path.GetExtension(file).ToLowerInvariant();
            return ext == ".mp4" || ext == ".avi" || ext == ".mov";
        }

        /// <summary>
        /// Builds a dictionary from session ID to a dictionary mapping exercise number to a tuple (file path, date).
        /// Used for video and JSON files.
        /// </summary>
        private Dictionary<string, Dictionary<int, Tuple<string, DateTime?>>> BuildSessionIdDictionary(IEnumerable<string> files)
        {
            var dict = new Dictionary<string, Dictionary<int, Tuple<string, DateTime?>>>();

            foreach (var file in files)
            {
                var parsed = ExtractSessionIDAndIndex(Path.GetFileName(file));
                if (parsed != null)
                {
                    if (!dict.ContainsKey(parsed.Item1))
                    {
                        dict[parsed.Item1] = new Dictionary<int, Tuple<string, DateTime?>>();
                    }
                    dict[parsed.Item1][parsed.Item2] = Tuple.Create(file, parsed.Item3);
                }
            }
            return dict;
        }

        /// <summary>
        /// Builds a dictionary for session data (.session files) mapping session ID and exercise number to the file path.
        /// </summary>
        private Dictionary<string, Dictionary<int, string>> BuildSessionDataDictionary(IEnumerable<string> files)
        {
            var dict = new Dictionary<string, Dictionary<int, string>>();
            foreach (var file in files)
            {
                var name = Path.GetFileNameWithoutExtension(file);
                var parts = name.Split('_');
                if (parts.Length == 2 && int.TryParse(parts[1], out int exercise))
                {
                    if (!dict.ContainsKey(parts[0]))
                    {
                        dict[parts[0]] = new Dictionary<int, string>();
                    }
                    dict[parts[0]][exercise] = file;
                }
            }
            return dict;
        }

        /// <summary>
        /// Builds a dictionary for H5 files mapping session ID and exercise number to the file path.
        /// </summary>
        private Dictionary<string, Dictionary<int, string>> BuildSessionIdDictionarySimple(IEnumerable<string> files)
        {
            var dict = new Dictionary<string, Dictionary<int, string>>();
            foreach (var file in files)
            {
                var parsed = ExtractSessionIDAndIndex(Path.GetFileName(file));
                if (parsed != null)
                {
                    if (!dict.ContainsKey(parsed.Item1))
                    {
                        dict[parsed.Item1] = new Dictionary<int, string>();
                    }
                    dict[parsed.Item1][parsed.Item2] = file;
                }
            }
            return dict;
        }

        [RelayCommand]
        private void MarkStartingPoint()
        {
            double currentFrameTime = FrameIndex * 1000.0 / framesPerSecond;
            CurrentExercise.FirstPointOffset = (long)currentFrameTime;
            OnPropertyChanged(nameof(StartingPoint));
        }

        [RelayCommand]
        public void LoadVideo(object parameter)
        {
            // Open a folder selection dialog
            CommonOpenFileDialog folderDialog = new CommonOpenFileDialog
            {
                IsFolderPicker = true,
                Title = "Select Folder Containing Video Files"
            };

            if (folderDialog.ShowDialog() != CommonFileDialogResult.Ok || string.IsNullOrEmpty(folderDialog.FileName))
            {
                return; // User canceled
            }

            string directory = folderDialog.FileName;

            // Get all .h5 files and extract their base names
            HashSet<string> h5BaseNames = new HashSet<string>(
                Directory.GetFiles(directory, "*.h5")
                .Select(file => Path.GetFileNameWithoutExtension(file))
            );

            // Get all video files and filter out those with an associated .h5 file
            List<string> availableVideos = Directory.GetFiles(directory, "*.mp4")
                .Concat(Directory.GetFiles(directory, "*.avi"))
                .Concat(Directory.GetFiles(directory, "*.mov"))
                .Where(file => !h5BaseNames.Contains(Path.GetFileNameWithoutExtension(file)))
                .ToList();

            // If no videos are available, notify the user and exit
            if (availableVideos.Count == 0)
            {
                System.Windows.MessageBox.Show("No available video files found without an associated .h5 file.",
                    "No Videos Found", System.Windows.MessageBoxButton.OK, System.Windows.MessageBoxImage.Information);
                return;
            }

            // Select the first available video
            string videoPath = availableVideos.First();
            recordName = Path.ChangeExtension(videoPath, ".h5");

            // Load the video
            if (!string.IsNullOrEmpty(videoPath))
            {
                if (videoCapture != null)
                {
                    videoCapture.Dispose();
                }
                videoCapture = new VideoCapture(videoPath);
                frameCount = videoCapture is null ? 0 : (int)videoCapture.Get(VideoCaptureProperties.FrameCount);
                OnPropertyChanged(nameof(FrameCount));
                OnPropertyChanged(nameof(FrameIndex));
                UpdateImage();
            }
        }

        #endregion

        private bool GetStateAtFrame(int frame)
        {
            // If there are no recorded state changes, decide on a default:
            if (CurrentExercise is null || CurrentExercise?.StateChanges.Count == 0)
                return false; // or however you wish to default

            if (CurrentExercise is not null)
            {
                // Iterate in reverse order of keys.
                foreach (var kvp in CurrentExercise.StateChanges.Reverse())
                {
                    if (kvp.Key <= frame)
                        return kvp.Value;
                }
            }

            // If no state change has occurred before this frame, return default.
            return false;
        }
        
        private void UpdateCurrentState()
        {
            if (SetProperty(ref currentState, GetStateAtFrame(frameIndex)))
            {
                OnPropertyChanged(nameof(CurrentState));
                OnPropertyChanged(nameof(IsTouched));
            }
        }

        private Bitmap? GetImage()
        {
            if (videoCapture is null)
                return null;

            // Set the desired frame.
            if (!videoCapture.Set(VideoCaptureProperties.PosFrames, frameIndex))
                return null;

            using Mat frame = new();
            if (!videoCapture.Read(frame) || frame.Empty())
            {
                Console.WriteLine($"Failed to read frame at index {frameIndex}");
                return null;
            }

            // Get frame dimensions.
            int width = frame.Width;
            int height = frame.Height;
            int squareSize = Math.Max(width, height);

            // Create a black square Mat to hold the centered image.
            using Mat squareFrame = new(new OpenCvSharp.Size(squareSize, squareSize), frame.Type(), Scalar.Black);

            // Calculate offsets to center the frame.
            int xOffset = (squareSize - width) / 2;
            int yOffset = (squareSize - height) / 2;
            Rect roi = new(xOffset, yOffset, width, height);

            // Copy the frame into the square's ROI.
            using (Mat roiMat = new Mat(squareFrame, roi))
            {
                frame.CopyTo(roiMat);
            }

            // Calculate the center of the square.
            Point2f center = new Point2f(squareFrame.Width / 2f, squareFrame.Height / 2f);

            // Get the rotation matrix for the given rotation (scale factor 1.0).
            using Mat rotationMatrix = Cv2.GetRotationMatrix2D(center, rotation, 1.0);

            // Apply the affine transformation (rotation).
            using Mat rotatedFrame = new();
            Cv2.WarpAffine(squareFrame, rotatedFrame, rotationMatrix, new OpenCvSharp.Size(squareFrame.Width, squareFrame.Height));

            // Convert the final rotated frame to a Bitmap.
            return BitmapConverter.ToBitmap(rotatedFrame);
        }

        /// <summary>
        /// Converts a Bitmap to a WPF ImageSource.
        /// </summary>
        private static BitmapImage ConvertMatToImageSource(Bitmap? bitmap)
        {
            if (bitmap is null)
                return new BitmapImage();

            using MemoryStream memoryStream = new();
            bitmap.Save(memoryStream, System.Drawing.Imaging.ImageFormat.Png);
            memoryStream.Seek(0, SeekOrigin.Begin);

            BitmapImage bitmapImage = new();
            bitmapImage.BeginInit();
            bitmapImage.CacheOption = BitmapCacheOption.OnLoad;
            bitmapImage.StreamSource = memoryStream;
            bitmapImage.EndInit();
            bitmapImage.Freeze(); // For thread safety if used across threads
            return bitmapImage;
        }

        private static Tuple<string, int, DateTime?>? ExtractSessionIDAndIndex(string fileName)
        {
            // Regex patterns for different filename variations
            string[] patterns =
            {
                // Pattern 1: type_sessionID_timestamp_index.extension
                @"^(?:data|video)_(?<sessionID>[a-zA-Z0-9]+)_(?<timestamp>[0-9T:\-.Z]+)_(?<index>\d+)\.\w+$",
    
                // Pattern 2: type_filetime_sessionID_index.extension (sessionID after timestamp)
                @"^(?:data|video)_(?<timestamp>\d+)_(?<sessionID>[a-zA-Z0-9]+)_(?<index>\d+)\.\w+$",

                // Pattern 3: type_sessionID_filetime_index.extension
                @"^(?:data|video)_(?<sessionID>[a-zA-Z0-9]+)_(?<timestamp>\d+)_(?<index>\d+)\.\w+$",
    
                // Pattern 4: type_sessionID_index.extension (no timestamp)
                @"^(?:data|video)_(?<sessionID>[a-zA-Z0-9]+)_(?<index>\d+)\.\w+$"
            };

            foreach (var pattern in patterns)
            {
                var match = Regex.Match(fileName, pattern, RegexOptions.IgnoreCase);
                if (match.Success)
                {
                    string sessionID = match.Groups["sessionID"].Value;
                    int index = int.Parse(match.Groups["index"].Value);

                    DateTime? extractedDateTime = null;
                    if (match.Groups["timestamp"].Success)
                    {
                        string timestampStr = match.Groups["timestamp"].Value;
                        extractedDateTime = ParseTimestamp(timestampStr);
                    }

                    return Tuple.Create(sessionID, index, extractedDateTime);
                }
            }

            // Return null if no pattern matches
            return null;
        }

        /// <summary>
        /// Parses a timestamp from the filename into a DateTime object.
        /// </summary>
        private static DateTime? ParseTimestamp(string timestampStr)
        {
            if (DateTimeOffset.TryParse(timestampStr, CultureInfo.InvariantCulture, DateTimeStyles.RoundtripKind, out DateTimeOffset dto))
            {
                return dto.UtcDateTime; // Convert to UTC DateTime
            }

            return null; // Invalid timestamp
        }

        private void LoadSessionVideo(object parameter)
        {
            if ((parameter is not null) && (parameter is SessionInfo sessionInfo))
            {
                // Select the first available video
                string videoPath = sessionInfo.VideoPath;

                if (string.IsNullOrEmpty(sessionInfo.H5Path))
                {
                    recordName = Path.ChangeExtension(videoPath, ".h5");
                }

                // Load the video
                if (!string.IsNullOrEmpty(videoPath))
                {
                    if (videoCapture != null)
                    {
                        videoCapture.Dispose();
                    }
                    videoCapture = new VideoCapture(videoPath);
                    frameCount = videoCapture is null ? 0 : (int)videoCapture.Get(VideoCaptureProperties.FrameCount);
                    framesPerSecond = videoCapture is null ? 0 : videoCapture.Get(VideoCaptureProperties.Fps);
                    if (framesPerSecond < 0)
                    {
                        throw new Exception("Invalid FPS Value");
                    }
                    OnPropertyChanged(nameof(FrameCount));
                    OnPropertyChanged(nameof(FrameIndex));
                    UpdateImage();
                }
            }
        }

        private void LoadSessionJson(object parameter)
        {
            if ((parameter is not null) && (parameter is SessionInfo sessionInfo))
            {
                string? jsonPath = sessionInfo.DataPath;

                if (!string.IsNullOrEmpty(jsonPath))
                {
                    try
                    {
                        if (File.Exists(jsonPath))
                        {
                            string jsonContent = File.ReadAllText(jsonPath);
                            ParseJson(jsonContent);
                        }
                    }
                    catch (Exception ex)
                    {
                        // Handle errors (e.g., file not found, permission issues)
                        FormattedJson = $"Error loading JSON: {ex.Message}";
                    }
                }
            }
        }

        private void ParseJson(string jsonContent)
        {
            try
            {
                var jsonDoc = JsonDocument.Parse(jsonContent);
                var root = jsonDoc.RootElement;

                if (root.TryGetProperty("DrawingLines", out JsonElement drawingLinesElement))
                {
                    var allDrawingLines = JsonSerializer.Deserialize<List<InkMARCDrawingLine>>(drawingLinesElement.GetRawText());
                    _drawingLine = new List<InkMARCPoint>();
                    if (allDrawingLines is not null)
                    {
                        foreach (var line in allDrawingLines)
                        {
                            if (line.Points is not null)
                            {
                                foreach (var point in line.Points)
                                {
                                    _drawingLine.Add(point);
                                }
                            }
                        }
                    }
                    firstDataTimeStamp = _drawingLine[0].Timestamp;
                }

                UpdateFormattedJson();
            }
            catch (JsonException)
            {
                FormattedJson = "Invalid JSON format.";
            }
        }

        /// <summary>
        /// Optimized version of FindClosestDataPoint. If _drawingLine is sorted by timestamp,
        /// you could further optimize this with a binary search.
        /// </summary>
        private InkMARCPoint? FindClosestDataPointOptimized(int currentFrameIndex, double thresholdUs)
        {
            // Compute the video time for the frame.
            double frameVideoTimeMs = currentFrameIndex * 1000.0 / framesPerSecond;
            double expectedDataTimestamp = firstDataTimeStamp + (frameVideoTimeMs - StartingPoint) * 1000.0;

            InkMARCPoint? closestPoint = null;
            double smallestDiff = double.MaxValue;

            if (_drawingLine is not null)
            {
                // Linear search: Consider binary search if _drawingLine is sorted.
                foreach (var point in _drawingLine)
                {
                    double diff = Math.Abs(point.Timestamp - expectedDataTimestamp);
                    if (diff < smallestDiff)
                    {
                        smallestDiff = diff;
                        closestPoint = point;
                    }
                }
            }

            return (smallestDiff <= thresholdUs) ? closestPoint : null;
        }

        /// <summary>
        /// Updates FormattedJson with the point that matches the current frame timestamp.
        /// </summary>
        private void UpdateFormattedJson()
        {
            if (_drawingLine is not null)
            {
                if (_drawingLine.Count == 0)
                {
                    FormattedJson = "No DrawingLines available.";
                    return;
                }

                InkMARCPoint? closestPoint = _drawingLine[0];
                if (CurrentExercise.FirstPointOffset >= 0)
                {
                    double frameDurationMs = 1000.0 / framesPerSecond;
                    double thresholdMs = frameDurationMs / 2.0;
                    double thresholdUs = thresholdMs * 1000.0;
                    closestPoint = FindClosestDataPointOptimized(frameIndex, thresholdUs);
                }

                if (closestPoint != null)
                {
                    FormattedJson = JsonSerializer.Serialize(closestPoint, new JsonSerializerOptions { WriteIndented = true });
                }
                else
                {
                    FormattedJson = "No matching point found.";

                }
            }
        }

        private void UpdateImage()
        {
            if (currentBitmap is not null)
            {
                currentBitmap.Dispose();
                currentBitmap = null;
            }
            currentBitmap = GetImage();
            CurrentImage = ConvertMatToImageSource(currentBitmap);
            OnPropertyChanged(nameof(CurrentImage));
        }
    }
}