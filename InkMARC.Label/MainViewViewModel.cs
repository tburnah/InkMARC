﻿using System.IO;
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
using System.Windows.Threading;
using System.Drawing;
using OpenCvSharp.WpfExtensions;
using System.Diagnostics;

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
        private bool isTouched = false;
        private int rotation;
        private bool currentState = false;
        private ImageSource? _currentImage;
        private bool showProgressBar = false;
        private int _sliderValue;
        private DispatcherTimer? _debounceTimer;
        private int lastFrameIndex = -1;
        private double thresholdUs = 0.0;

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

        public int SliderValue
        {
            get => _sliderValue;
            set
            {
                if (SetProperty(ref _sliderValue, value))
                {
                    StartDebounceTimer();
                }
            }
        }

        public int SliderTickFrequency => FrameCount > 100 ? FrameCount / 100 : 1;

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
                    var closestPoint = FindClosestDataPointOptimized(frameIndex, thresholdUs);
                    closestPoint ??= new InkMARCPoint(float.NaN, float.NaN, 0, 0, 0, 0);
                    DataSave.InitializeChunkedDatasets(firstImage, GetStateAtFrame(StartFrame), (InkMARCPoint)closestPoint);
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
                        var closestPoint = FindClosestDataPointOptimized(frameIndex, thresholdUs);
                        closestPoint ??= new InkMARCPoint(float.NaN, float.NaN, 0, 0, 0, 0);
                        DataSave.WriteFrameEx(image, GetStateAtFrame(frameIndex), (InkMARCPoint)closestPoint);
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
                StartFrame = 0;
                StopFrame = frameCount;
                double frameDurationMs = 1000.0 / framesPerSecond;
                double thresholdMs = frameDurationMs / 2.0;
                thresholdUs = thresholdMs * 1000.0;
                OnPropertyChanged(nameof(FrameCount));
                OnPropertyChanged(nameof(FrameIndex));
                OnPropertyChanged(nameof(StartFrame));
                OnPropertyChanged(nameof(StopFrame));
                OnPropertyChanged(nameof(SliderTickFrequency));
                UpdateImage();
            }
        }

        #endregion

        private void StartDebounceTimer()
        {
            // Initialize the timer if it's not already created.
            if (_debounceTimer == null)
            {
                _debounceTimer = new DispatcherTimer
                {
                    Interval = TimeSpan.FromSeconds(1) // Adjust the delay as needed.
                };
                _debounceTimer.Tick += DebounceTimer_Tick;
            }
            _debounceTimer.Stop(); // Restart the timer each time the value changes.
            _debounceTimer.Start();
        }

        private void DebounceTimer_Tick(object sender, EventArgs e)
        {
            _debounceTimer?.Stop();

            // Now update the video position using the debounced slider value.
            FrameIndex = SliderValue;
            UpdateImage();
        }

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

        private BitmapSource? GetImage()
        {
            if (videoCapture is null)
                return null;

            // Create a new Mat to hold the frame.
            Mat frame = new Mat();

            // If we're moving sequentially forward, avoid repositioning.
            if (frameIndex == lastFrameIndex + 1)
            {
                if (!videoCapture.Read(frame) || frame.Empty())
                {
                    Console.WriteLine($"Failed to read sequential frame at index {frameIndex}");
                    return null;
                }
            }
            else
            {
                // For non-sequential jumps, set the position and read.
                if (!videoCapture.Set(VideoCaptureProperties.PosFrames, frameIndex))
                {
                    Console.WriteLine($"Failed to set frame index {frameIndex}");
                    return null;
                }
                if (!videoCapture.Read(frame) || frame.Empty())
                {
                    Console.WriteLine($"Failed to read frame at index {frameIndex}");
                    return null;
                }
            }

            lastFrameIndex = frameIndex;
            BitmapSource processedImage = ProcessFrame(frame);
            return processedImage;
        }

        private BitmapSource ProcessFrame(Mat frame)
        {
            int width = frame.Width;
            int height = frame.Height;
            int squareSize = Math.Max(width, height);

            // Create a black square Mat to center the frame
            using Mat squareFrame = new(new OpenCvSharp.Size(squareSize, squareSize), frame.Type(), Scalar.Black);
            int xOffset = (squareSize - width) / 2;
            int yOffset = (squareSize - height) / 2;
            Rect roi = new(xOffset, yOffset, width, height);
            using (Mat roiMat = new Mat(squareFrame, roi))
            {
                frame.CopyTo(roiMat);
            }

            // Compute the rotation matrix for the given rotation angle
            Point2f center = new Point2f(squareSize / 2f, squareSize / 2f);
            using Mat rotationMatrix = Cv2.GetRotationMatrix2D(center, rotation, 1.0);

            // Apply the affine transformation (rotation)
            using Mat rotatedFrame = new();
            Cv2.WarpAffine(squareFrame, rotatedFrame, rotationMatrix, new OpenCvSharp.Size(squareSize, squareSize));

            // Convert the final rotated Mat directly to a BitmapSource
            BitmapSource bitmapSource = BitmapSourceConverter.ToBitmapSource(rotatedFrame);
            bitmapSource.Freeze(); // Freeze for thread safety
            return bitmapSource;
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
    
                // Pattern 4: type_sessionID_index.extension (index is 1–2 digits only)
                @"^(?:data|video)_(?<sessionID>[a-zA-Z0-9]+)_(?<index>\d{1,2})\.\w+$",

                // Patter 5: type_sessionID_Participant_index_AppView.extension
                @"^(?:data|video)_(?<sessionID>[a-zA-Z0-9]+)_Participant(?<index>\d+)_AppView\d+\.\w+$",

                // Pattern 6: type_sessionID_timestamp.extension (no index) 
                @"^(?:data|video)_(?<sessionID>[a-zA-Z0-9]+)_(?<timestamp>\d+)\.\w+$"
            };

            foreach (var pattern in patterns)
            {
                var match = Regex.Match(fileName, pattern, RegexOptions.IgnoreCase);
                if (match.Success)
                {
                    string sessionID = match.Groups["sessionID"].Value;

                    int index;
                    
                    if (!(match.Groups["timestamp"].Success && int.TryParse(match.Groups["index"].Value, out index)))
                    {
                        index = 0;
                    }

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
            else if (DateTime.FromFileTimeUtc(long.Parse(timestampStr)) is DateTime dt)
            {
                return dt;
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
                    StartFrame = 0;
                    StopFrame = frameCount;
                    framesPerSecond = videoCapture is null ? 0 : videoCapture.Get(VideoCaptureProperties.Fps);
                    double frameDurationMs = 1000.0 / framesPerSecond;
                    double thresholdMs = frameDurationMs / 2.0;
                    thresholdUs = thresholdMs * 1000.0;

                    if (framesPerSecond < 0)
                    {
                        throw new Exception("Invalid FPS Value");
                    }
                    OnPropertyChanged(nameof(FrameCount));
                    OnPropertyChanged(nameof(FrameIndex));
                    OnPropertyChanged(nameof(StartFrame));
                    OnPropertyChanged(nameof(StopFrame));
                    OnPropertyChanged(nameof(SliderTickFrequency));
                    UpdateImage();
                }
            }
        }

        [RelayCommand]
        private void ListSessionIdsFromFolder()
        {
            var folderDialog = new CommonOpenFileDialog
            {
                IsFolderPicker = true,
                Title = "Select Folder Containing Session JSON Files"
            };

            if (folderDialog.ShowDialog() != CommonFileDialogResult.Ok || string.IsNullOrEmpty(folderDialog.FileName))
                return;

            string folderPath = folderDialog.FileName;
            var jsonFiles = Directory.GetFiles(folderPath, "data_*.json");

            var sessionIds = new HashSet<string>();

            foreach (var file in jsonFiles)
            {
                var fileName = Path.GetFileName(file);
                var parsed = ExtractSessionIDAndIndex(fileName);
                if (parsed != null)
                {
                    sessionIds.Add(parsed.Item1);
                }
            }

            if (sessionIds.Count == 0)
            {
                throw new Exception("No session IDs found in the selected folder.");
            }
            else
            {
                foreach (var sessionId in sessionIds)
                {
                    var TimeSpan = GetFullSessionDrawingDuration(folderPath, sessionId);
                    Console.WriteLine($"Session ID: {sessionId}, Duration: {TimeSpan}");
                }
            }
        }

        private Dictionary<string, string> MatchSessionsToVideosWithinThreshold(Dictionary<string, TimeSpan> sessionDurations, Dictionary<string, TimeSpan> videoDurations, double maxAllowedDifferenceSeconds = 30.0)
        {
            var matched = new Dictionary<string, string>();
            var remainingVideos = new Dictionary<string, TimeSpan>(videoDurations); // copy so we can remove matched videos

            foreach (var session in sessionDurations.OrderBy(sd => sd.Value))
            {
                string sessionId = session.Key;
                TimeSpan sessionTime = session.Value;

                string? bestMatch = null;
                double bestDiff = double.MaxValue;

                foreach (var video in remainingVideos)
                {
                    var diff = (video.Value - sessionTime).TotalSeconds;

                    if (diff >= 0 && diff <= maxAllowedDifferenceSeconds && diff < bestDiff)
                    {
                        bestDiff = diff;
                        bestMatch = video.Key;
                    }
                }

                if (bestMatch != null)
                {
                    matched[sessionId] = bestMatch;
                    remainingVideos.Remove(bestMatch);
                }
                else
                {
                    Console.WriteLine($"No suitable video match found for session {sessionId} (duration: {sessionTime})");
                }
            }

            return matched;
        }

        [RelayCommand]
        private async Task OrganizeSessionsByFolder()
        {
            this.FormattedJson = string.Empty;
            var folderDialog = new CommonOpenFileDialog
            {
                IsFolderPicker = true,
                Title = "Select Folder Containing Session JSON & Video Files"
            };

            if (folderDialog.ShowDialog() != CommonFileDialogResult.Ok || string.IsNullOrEmpty(folderDialog.FileName))
                return;

            string folderPath = folderDialog.FileName;

            var jsonFiles = Directory.GetFiles(folderPath, "data_*.json");
            var videoFiles = Directory.GetFiles(folderPath)
                .Where(f => IsVideoFile(f))
                .ToList();

            var sessionJsonGroups = jsonFiles
                .Select(file => (file, parsed: ExtractSessionIDAndIndex(Path.GetFileName(file))))
                .Where(x => x.parsed != null)
                .GroupBy(x => x.parsed.Item1)
                .ToDictionary(
                    g => g.Key,
                    g => g
                        .OrderBy(x => x.parsed.Item2)
                        .Select(x => x.file)
                        .ToList()
                );

            // Video durations
            var videoDurations = new Dictionary<string, TimeSpan>();
            foreach (var videoPath in videoFiles)
            {
                try
                {
                    using var cap = new OpenCvSharp.VideoCapture(videoPath);
                    double fps = cap.Get(VideoCaptureProperties.Fps);
                    double frameCount = cap.Get(VideoCaptureProperties.FrameCount);
                    double durationSeconds = frameCount / fps;
                    videoDurations[videoPath] = TimeSpan.FromSeconds(durationSeconds);
                }
                catch
                {
                    Console.WriteLine($"Failed to read video duration for {videoPath}");
                }
            }

            // Session durations
            var sessionDurations = new Dictionary<string, TimeSpan>();
            foreach (var session in sessionJsonGroups)
            {
                try
                {
                    sessionDurations[session.Key] = GetFullSessionDrawingDuration(folderPath, session.Key);
                }
                catch
                {
                    Console.WriteLine($"Failed to calculate session duration for {session.Key}");
                }
            }

            var matches = MatchSessionsToVideosWithinThreshold(sessionDurations, videoDurations, 30.0);

            // Set up progress bar
            ShowProgressBar = true;
            int total = sessionJsonGroups.Count;
            int current = 0;

            var progress = new Progress<int>(value =>
            {
                SliderValue = value;
                OnPropertyChanged(nameof(SliderValue));
            });

            await Task.Run(() =>
            {
                foreach (var sessionId in sessionJsonGroups.Keys)
                {
                    if (!matches.TryGetValue(sessionId, out string? videoFile))
                    {
                        Console.WriteLine($"Skipping unmatched session: {sessionId}");
                        current++;
                        ((IProgress<int>)progress).Report(current * 100 / total);
                        continue;
                    }

                    var dataFiles = sessionJsonGroups[sessionId];
                    string sessionFolder = Path.Combine(folderPath, sessionId);
                    Directory.CreateDirectory(sessionFolder);

                    var creationTime = File.GetCreationTimeUtc(videoFile);
                    string timestamp = creationTime.ToFileTime().ToString();
                    string newVideoName = $"video_{sessionId}_{timestamp}{Path.GetExtension(videoFile)}";
                    string videoDest = Path.Combine(sessionFolder, newVideoName);

                    ResizeVideoWithFFmpeg(videoFile, videoDest, 448); // FFmpeg resizing

                    // Merge JSON
                    var allDrawingLines = new List<JsonElement>();
                    foreach (var file in dataFiles)
                    {
                        string json = File.ReadAllText(file);
                        using var doc = JsonDocument.Parse(json);
                        if (doc.RootElement.TryGetProperty("DrawingLines", out JsonElement lines))
                        {
                            foreach (var line in lines.EnumerateArray())
                            {
                                allDrawingLines.Add(line.Clone());
                            }
                        }
                    }

                    // Extract timestamp from first file name
                    var firstDataFile = Path.GetFileNameWithoutExtension(dataFiles.First());
                    var nameParts = firstDataFile.Split('_');

                    string dataTimestamp = nameParts.Length > 2 ? nameParts[1] : DateTime.UtcNow.ToFileTime().ToString();

                    using var stream = File.Create(Path.Combine(sessionFolder, $"data_{sessionId}_{dataTimestamp}.json"));

                    using var writer = new Utf8JsonWriter(stream, new JsonWriterOptions { Indented = true });
                    writer.WriteStartObject();
                    writer.WritePropertyName("DrawingLines");
                    writer.WriteStartArray();
                    foreach (var line in allDrawingLines)
                    {
                        line.WriteTo(writer);
                    }
                    writer.WriteEndArray();
                    writer.WriteEndObject();

                    Console.WriteLine($"Organized session {sessionId} → matched video: {Path.GetFileName(videoFile)}");

                    current++;
                    ((IProgress<int>)progress).Report(current * 100 / total);
                }
            });

            ShowProgressBar = false;
            System.Windows.MessageBox.Show("Sessions organized successfully.", "Done", System.Windows.MessageBoxButton.OK, System.Windows.MessageBoxImage.Information);
        }

        private void ResizeVideoWithFFmpeg(string inputPath, string outputPath, int maxDimension = 448)
        {
            string ffmpegPath = "ffmpeg";

            // Try running ffmpeg -version to check availability
            bool ffmpegAvailable = true;
            try
            {
                var checkProcess = new Process
                {
                    StartInfo = new ProcessStartInfo
                    {
                        FileName = ffmpegPath,
                        Arguments = "-version",
                        RedirectStandardOutput = true,
                        RedirectStandardError = true,
                        UseShellExecute = false,
                        CreateNoWindow = true
                    }
                };
                checkProcess.Start();
                checkProcess.WaitForExit();

                if (checkProcess.ExitCode != 0)
                    ffmpegAvailable = false;
            }
            catch
            {
                ffmpegAvailable = false;
            }

            // Prompt for ffmpeg.exe if not available
            if (!ffmpegAvailable)
            {
                System.Windows.MessageBox.Show("FFmpeg was not found in the system path. Please select ffmpeg.exe manually.",
                    "FFmpeg Not Found", System.Windows.MessageBoxButton.OK, System.Windows.MessageBoxImage.Warning);

                var openDialog = new Microsoft.WindowsAPICodePack.Dialogs.CommonOpenFileDialog
                {
                    Title = "Locate ffmpeg.exe",
                    Filters = { new CommonFileDialogFilter("FFmpeg Executable", "exe") }
                };

                if (openDialog.ShowDialog() == CommonFileDialogResult.Ok && File.Exists(openDialog.FileName))
                {
                    ffmpegPath = openDialog.FileName;
                }
                else
                {
                    System.Windows.MessageBox.Show("FFmpeg path not provided. Aborting operation.",
                        "Error", System.Windows.MessageBoxButton.OK, System.Windows.MessageBoxImage.Error);
                    return;
                }
            }

            string args = $"-hwaccel cuda -hwaccel_output_format cuda -i \"{inputPath}\" " +
                          $"-vf \"scale_cuda={maxDimension}:-2\" " +
                          "-c:v h264_nvenc -preset fast -crf 28 -an " +
                          $"\"{outputPath}\"";

            var startInfo = new ProcessStartInfo
            {
                FileName = ffmpegPath,
                Arguments = args,
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                UseShellExecute = false,
                CreateNoWindow = true
            };

            using var process = new Process { StartInfo = startInfo };

            process.OutputDataReceived += (s, e) => { /* rarely needed for ffmpeg */ };
            process.ErrorDataReceived += (s, e) =>
            {
                if (!string.IsNullOrWhiteSpace(e.Data))
                {
                    FormattedJson += e.Data + Environment.NewLine;
                }
            };

            process.Start();
            process.BeginOutputReadLine();
            process.BeginErrorReadLine();
            process.WaitForExit();

            if (process.ExitCode != 0)
            {
                string error = process.StandardError.ReadToEnd();
                System.Windows.MessageBox.Show($"FFmpeg failed:\n{error}", "FFmpeg Error",
                    System.Windows.MessageBoxButton.OK, System.Windows.MessageBoxImage.Error);
            }
            else
            {
                Console.WriteLine($"FFmpeg resized and saved to {outputPath}");
            }
        }

        public static TimeSpan GetFullSessionDrawingDuration(string folderPath, string sessionId)
        {
            var files = Directory.GetFiles(folderPath, $"data_*_{sessionId}_*.json");

            long? sessionStart = null;
            long? sessionEnd = null;

            foreach (var file in files)
            {
                var json = File.ReadAllText(file);
                using var doc = JsonDocument.Parse(json);

                if (!doc.RootElement.TryGetProperty("DrawingLines", out var linesArray)) continue;

                foreach (var line in linesArray.EnumerateArray())
                {
                    if (!line.TryGetProperty("Points", out var pointsArray)) continue;

                    foreach (var point in pointsArray.EnumerateArray())
                    {
                        if (point.TryGetProperty("Timestamp", out var tsProp) && tsProp.TryGetInt64(out var timestamp))
                        {
                            if (sessionStart == null || timestamp < sessionStart) sessionStart = timestamp;
                            if (sessionEnd == null || timestamp > sessionEnd) sessionEnd = timestamp;
                        }
                    }
                }
            }

            if (sessionStart == null || sessionEnd == null)
                throw new Exception("No timestamps found across session files.");

            long durationMicroseconds = sessionEnd.Value - sessionStart.Value;
            return TimeSpan.FromMilliseconds(durationMicroseconds / 1000.0);
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
            //if (currentBitmap is not null)
            //{
            //    currentBitmap.Dispose();
            //    currentBitmap = null;
            //}
            CurrentImage = GetImage();
            //currentBitmap = GetImage();
            //CurrentImage = ConvertMatToImageSource(currentBitmap);
            OnPropertyChanged(nameof(CurrentImage));
        }
    }
}