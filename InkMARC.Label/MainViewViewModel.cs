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

namespace InkMARC.Label
{
    internal partial class SessionInfo : ObservableObject
    {
        public string VideoPath { get; set; }
        public string? DataPath { get; set; }
        public string? H5Path { get; set; }
        public string SessionID { get; set; }
        public int Exercise { get; set; }
        public long FirstPointOffset { get; set; }
        public DateTime? VideoDateTime { get; set; }
        public DateTime? DataDataTime { get; set; }
        public bool HasData => !string.IsNullOrEmpty(DataPath);
        public bool HasH5 => !string.IsNullOrEmpty(H5Path);
        public int StartFrame { get; set; }
        public int StopFrame { get; set; }

        public void UpdateH5Path(string h5Path)
        {
            H5Path = h5Path;
            OnPropertyChanged(nameof(H5Path));
            OnPropertyChanged(nameof(HasH5));
        }

        public SortedDictionary<int, bool> StateChanges = new SortedDictionary<int, bool>();

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

        // Method to save SessionInfo to a file
        public void SaveToFile()
        {
            string directoryPath = Path.GetDirectoryName(VideoPath);
            string filePath = Path.Combine(directoryPath, SessionID + "_" + Exercise + ".session");

            var options = new JsonSerializerOptions
            {
                WriteIndented = true, // Pretty print JSON
                DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull // Ignore null values
            };

            string json = JsonSerializer.Serialize(this, options);
            File.WriteAllText(filePath, json);
        }

        // Static method to load SessionInfo from a file
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
        private VideoCapture? videoCapture;
        private int frameCount = 0;
        private int frameIndex = 0;
        private string recordName = string.Empty;
        private List<InkMARCPoint>? _drawingLine;
        private double framesPerSecond = 0;
        private ulong firstDataTimeStamp = 0;

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

            // Create the file.
            DataSave.CreateFile(recordName);

            // Setup a progress reporter.
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

                // Create a Mat to hold the raw frame
                Mat frame = new();

                // If this is the first frame, explicitly position the capture.
                if (!videoCapture.Set(VideoCaptureProperties.PosFrames, StartFrame))
                {
                    Console.WriteLine($"Failed to set frame index {StartFrame}");
                    return;
                }

                // Read the first frame
                if (!videoCapture.Read(frame) || frame.Empty())
                {
                    Console.WriteLine("Failed to retrieve first frame.");
                    return;
                }

                // Initialize reusable image processing resources
                _frameWidth = frame.Width;
                _frameHeight = frame.Height;
                _squareSize = Math.Max(_frameWidth, _frameHeight);

                // Allocate reusable Mats
                _squareFrame = new Mat(new OpenCvSharp.Size(_squareSize, _squareSize), frame.Type(), Scalar.Black);
                _rotatedFrame = new Mat(new OpenCvSharp.Size(448, 448), frame.Type()); // Now 448x448 directly!

                // Calculate ROI for centering the frame
                int xOffset = (_squareSize - _frameWidth) / 2;
                int yOffset = (_squareSize - _frameHeight) / 2;
                _roi = new Rect(xOffset, yOffset, _frameWidth, _frameHeight);

                // Compute rotation matrix with scaling factor (Resize + Rotate in one step)
                float scaleFactor = 448.0f / _squareSize; // Scale image to fit in 448x448
                Point2f center = new Point2f(_squareSize / 2f, _squareSize / 2f);
                _rotationMatrix = Cv2.GetRotationMatrix2D(center, rotation, scaleFactor); // Embed scaling!

                // Process the first frame
                _squareFrame.SetTo(Scalar.Black);
                using (Mat roiMat = new Mat(_squareFrame, _roi))
                {
                    frame.CopyTo(roiMat);
                }
                Cv2.WarpAffine(_squareFrame, _rotatedFrame, _rotationMatrix, new OpenCvSharp.Size(448, 448)); // Now rotates & scales!

                using (Bitmap firstImage = BitmapConverter.ToBitmap(_rotatedFrame))
                {
                    DataSave.InitializeChunkedDatasets(firstImage, GetStateAtFrame(StartFrame));
                }

                progCounter++;
                ((IProgress<int>)progress).Report(progCounter);

                // Process remaining frames
                for (frameIndex = StartFrame + 1; frameIndex < StopFrame; frameIndex++)
                {
                    // Read the next frame
                    if (!videoCapture.Read(frame) || frame.Empty())
                    {
                        Console.WriteLine($"Skipping frame {frameIndex} because image retrieval failed.");
                        continue;
                    }

                    // Process frame (reuse Mats)
                    _squareFrame.SetTo(Scalar.Black);
                    using (Mat roiMat = new Mat(_squareFrame, _roi))
                    {
                        frame.CopyTo(roiMat);
                    }
                    Cv2.WarpAffine(_squareFrame, _rotatedFrame, _rotationMatrix, new OpenCvSharp.Size(448, 448)); // Rotates & Scales

                    // Convert and save the frame
                    using (Bitmap image = BitmapConverter.ToBitmap(_rotatedFrame))
                    {
                        DataSave.WriteFrame(image, GetStateAtFrame(frameIndex));
                    }

                    progCounter++;
                    if (progCounter % 10 == 0) // Update UI every 10 frames
                    {
                        ((IProgress<int>)progress).Report(progCounter);
                    }
                }
            });

            // Finalize the datasets and clean up.
            DataSave.FinalizeDatasets();
            if (File.Exists(recordName))
            {
                CurrentExercise.UpdateH5Path(recordName);
            }
            ShowProgressBar = false;
        }

        private int _frameWidth;
        private int _frameHeight;
        private int _squareSize;
        private Mat _squareFrame = null!;
        private Mat _rotatedFrame = null!;
        private Rect _roi;
        private Mat _rotationMatrix = null!;

        public System.Windows.Media.Brush IsTouched => isTouched ? System.Windows.Media.Brushes.Red : System.Windows.Media.Brushes.Black;

        private bool isTouched = false;
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
        private int rotation;

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

        public Bitmap? currentBitmap = null;

        public bool ShowCurrentIndex
        {
            get => !showProgressBar;
        }

        [RelayCommand]
        public void ToggleTouched()
        {
            isTouched = !isTouched;
            OnPropertyChanged(nameof(IsTouched));
            CurrentExercise?.StateChanges.Add(frameIndex, isTouched);
            UpdateCurrentState();
        }

        public bool GetStateAtFrame(int frame)
        {
            // If there are no recorded state changes, decide on a default:
            if (CurrentExercise is null || CurrentExercise?.StateChanges.Count == 0)
                return false; // or however you wish to default

            // Iterate in reverse order of keys.
            foreach (var kvp in CurrentExercise.StateChanges.Reverse())
            {
                if (kvp.Key <= frame)
                    return kvp.Value;
            }
            // If no state change has occurred before this frame, return default.
            return false;
        }

        public bool HasData => CurrentExercise is not null && CurrentExercise.HasData;

        [RelayCommand]
        public async void AnalyzeFramesForStateChangesAsync()
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

            // Optionally save the current frameIndex so you can restore it afterward.
            int originalFrameIndex = frameIndex;

            var progress = new Progress<int>(value =>
            {
                FrameIndex = value;
                OnPropertyChanged(nameof(FrameIndex));
            }
            );

            // Run the analysis on a background thread.
            await Task.Run(() =>
            {
                // Loop through each frame between start and stop
                for (int i = startFrame; i <= stopFrame; i++)
                {
                    // Set the frameIndex so that FindClosestDataPoint() uses the correct value.
                    frameIndex = i;
                    // Check if there is a datapoint for this frame.
                    bool hasDataPoint = FindClosestDataPoint() != null;

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
                    ((IProgress<int>)progress).Report(i - startFrame + 1);
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


        public bool currentState = false;
        public bool CurrentState => currentState;

        private void UpdateCurrentState()
        {
            if (SetProperty(ref currentState, GetStateAtFrame(frameIndex)))
                OnPropertyChanged(nameof(CurrentState));
        }

        private ImageSource _currentImage;
        private bool showProgressBar = false;

        public ImageSource CurrentImage
        {
            get => _currentImage;
            private set
            {
                _currentImage = value;
                OnPropertyChanged(nameof(CurrentImage));
            }
        }

        private Mat squareFrame;

        private Mat rotationMatrix;

        private void UpdateMatrices(Mat frame)
        {
            // Calculate the center of the image
            Point2f center = new Point2f(frame.Width / 2f, frame.Height / 2f);

            // Determine the size for the square. We want the square to fit the rotated image.
            int squareSize = Math.Max(frame.Width, frame.Height);


        }

        private Bitmap? GetImage()
        {
            if (videoCapture is null)
                return null;

            // Retrieve the frame
            Mat frame = new();
            if (videoCapture.Set(VideoCaptureProperties.PosFrames, frameIndex))
            {
                bool success = videoCapture.Read(frame);
                if (!success || frame.Empty())
                {
                    Console.WriteLine($"Failed to read frame at index {frameIndex}");
                    return null;
                }

                // Calculate the center of the image
                Point2f center = new Point2f(frame.Width / 2f, frame.Height / 2f);

                // Determine the size for the square. We want the square to fit the rotated image.
                int squareSize = Math.Max(frame.Width, frame.Height);

                // Create a black square Mat of the computed size.
                // The type should match that of rotatedFrame.
                Mat squareFrame = new Mat(new OpenCvSharp.Size(squareSize, squareSize), frame.Type(), Scalar.Black);

                // Calculate offsets to center the rotated image in the square
                int xOffset = (squareSize - frame.Width) / 2;
                int yOffset = (squareSize - frame.Height) / 2;

                // Define the region of interest (ROI) on the square image where the rotated image will be placed.
                Rect roi = new Rect(xOffset, yOffset, frame.Width, frame.Height);

                // Copy the rotated frame into the ROI of the square frame.
                frame.CopyTo(new Mat(squareFrame, roi));

                center = new Point2f(squareFrame.Width / 2f, squareFrame.Height / 2f);

                // Get the rotation matrix for the specified angle (and scale factor 1)
                Mat rotationMatrix = Cv2.GetRotationMatrix2D(center, rotation, 1.0);

                // Optionally, if you want to ensure the entire rotated image fits in the frame,
                // you might need to compute the bounding rectangle. For a simple case, we'll use the original size.
                Mat rotatedFrame = new Mat();
                Cv2.WarpAffine(squareFrame, rotatedFrame, rotationMatrix, new OpenCvSharp.Size(squareFrame.Width, squareFrame.Height));

                // Convert the final square frame to a Bitmap and return it.
                return BitmapConverter.ToBitmap(rotatedFrame);
            }
            return null;
        }

        // Helper method to convert OpenCV Mat to WPF ImageSource
        private ImageSource ConvertMatToImageSource(Bitmap? bitmap)
        {
            if (bitmap is null) return new BitmapImage();
            using var memoryStream = new MemoryStream();
            bitmap.Save(memoryStream, System.Drawing.Imaging.ImageFormat.Png);
            memoryStream.Seek(0, SeekOrigin.Begin);

            BitmapImage bitmapImage = new BitmapImage();
            bitmapImage.BeginInit();
            bitmapImage.CacheOption = BitmapCacheOption.OnLoad;
            bitmapImage.StreamSource = memoryStream;
            bitmapImage.EndInit();

            return bitmapImage;
        }

        private ObservableCollection<SessionInfo> sessions = new();
        private SessionInfo currentExercise;
        private string formattedJson;

        public ObservableCollection<SessionInfo> Sessions => sessions;

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
            CommonOpenFileDialog folderDialog = new CommonOpenFileDialog
            {
                IsFolderPicker = true,
                Title = "Select Folder Containing Video Files"
            };

            if (folderDialog.ShowDialog() != CommonFileDialogResult.Ok)
            {
                return; // User cancelled
            }

            string directory = folderDialog.FileName;

            List<string> availableVideos = Directory.GetFiles(directory, "*.mp4")
                .Concat(Directory.GetFiles(directory, "*.avi"))
                .Concat(Directory.GetFiles(directory, "*.mov")).ToList();

            List<string> availableJson = Directory.GetFiles(directory, "*.json").ToList();

            List<string> availableH5 = Directory.GetFiles(directory, "*.h5").ToList();

            List<string> availableData = Directory.GetFiles(directory, "*.session").ToList();

            Dictionary<string, Dictionary<int, string>> sessionData = new();
            Dictionary<string, Dictionary<int, Tuple<string, DateTime?>>> videoSessionIds = new();
            Dictionary<string, Dictionary<int, Tuple<string, DateTime?>>> dataSessionIds = new();
            Dictionary<string, Dictionary<int, string>> H5SessionIds = new();

            foreach (string fileName in availableVideos)
            {
                var parsed = ExtractSessionIDAndIndex(Path.GetFileName(fileName));
                if (parsed is not null)
                {
                    if (!videoSessionIds.ContainsKey(parsed.Item1))
                        videoSessionIds.Add(parsed.Item1, new Dictionary<int, Tuple<string, DateTime?>>());
                    videoSessionIds[parsed.Item1].Add(parsed.Item2, new Tuple<string, DateTime?>(fileName, parsed.Item3));
                }
            }

            foreach (string fileName in availableData)
            {
                var parsed = Path.GetFileNameWithoutExtension(fileName);
                var separated = parsed.Split('_');
                if (separated.Length == 2)
                {
                    if (!sessionData.ContainsKey(separated[0]))
                        sessionData.Add(separated[0], new Dictionary<int, string>());
                    sessionData[separated[0]].Add(int.Parse(separated[1]), fileName);
                }
            }

            foreach (string fileName in availableJson)
            {
                var parsed = ExtractSessionIDAndIndex(Path.GetFileName(fileName));
                if (parsed is not null)
                {
                    if (!dataSessionIds.ContainsKey(parsed.Item1))
                        dataSessionIds.Add(parsed.Item1, new Dictionary<int, Tuple<string, DateTime?>>());
                    dataSessionIds[parsed.Item1].Add(parsed.Item2, new Tuple<string, DateTime?>(fileName, parsed.Item3));
                }
            }

            foreach (string fileName in availableH5)
            {
                var parsed = ExtractSessionIDAndIndex(Path.GetFileName(fileName));
                if (parsed is not null)
                {
                    if (!H5SessionIds.ContainsKey(parsed.Item1))
                        H5SessionIds.Add(parsed.Item1, new Dictionary<int, string>());
                    H5SessionIds[parsed.Item1].Add(parsed.Item2, fileName);
                }
            }

            foreach (string sessionId in videoSessionIds.Keys)
            {
                foreach (int exercise in videoSessionIds[sessionId].Keys)
                {
                    string videoFile = videoSessionIds[sessionId][exercise].Item1;
                    string dataFile = string.Empty;
                    string h5File = string.Empty;
                    DateTime? videoDate = videoSessionIds[sessionId][exercise].Item2;
                    DateTime? dataDate = null;

                    if (dataSessionIds.ContainsKey(sessionId) && dataSessionIds[sessionId].ContainsKey(exercise))
                    {
                        dataFile = dataSessionIds[sessionId][exercise].Item1;
                        dataDate = dataSessionIds[sessionId][exercise].Item2;
                    }
                    if (H5SessionIds.ContainsKey(sessionId) && H5SessionIds[sessionId].ContainsKey(exercise))
                    {
                        h5File = H5SessionIds[sessionId][exercise];
                    }
                    if (sessionData.ContainsKey(sessionId) && sessionData[sessionId].ContainsKey(exercise))
                    {
                        var newSessionInfo = SessionInfo.LoadFromFile(sessionData[sessionId][exercise]);
                        if (newSessionInfo is not null)
                        {
                            newSessionInfo.DataPath = dataFile;
                            newSessionInfo.H5Path = h5File;                            
                        }
                        else
                        {
                            newSessionInfo = new SessionInfo(sessionId, videoFile, exercise, dataFile, h5File, videoDate, dataDate);                            
                        }
                        Sessions.Add(newSessionInfo);
                    }
                    else
                    {
                        Sessions.Add(new SessionInfo(sessionId, videoFile, exercise, dataFile, h5File, videoDate, dataDate));
                    }
                }
            }
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
                string jsonPath = sessionInfo.DataPath;

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
                    foreach (var line in allDrawingLines)
                    {
                        foreach (var point in line.Points)
                        {
                            _drawingLine.Add(point);
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

        public long StartingPoint => CurrentExercise?.FirstPointOffset ?? -1;

        [RelayCommand]
        private void MarkStartingPoint()
        {
            double currentFrameTime = FrameIndex * 1000.0 / framesPerSecond;
            CurrentExercise.FirstPointOffset = (long)currentFrameTime;
            OnPropertyChanged(nameof(StartingPoint));
        }

        public InkMARCPoint? FindClosestDataPoint()
        {
            // 1. Compute the video time (in ms) for this frame
            double frameVideoTimeMs = frameIndex * 1000.0 / framesPerSecond;

            double frameDurationMs = 1000.0 / framesPerSecond;
            double thresholdMs = frameDurationMs / 2.0;  // or simply 500.0 / framesPerSecond

            // 2. Compute the expected data timestamp (in microseconds) for this video time
            double expectedDataTimestamp = firstDataTimeStamp + (frameVideoTimeMs - StartingPoint) * 1000.0;

            // 3. Convert the threshold from milliseconds to microseconds
            double thresholdUs = thresholdMs * 1000.0;

            // 4. Iterate over data points to find the closest one
            InkMARCPoint? closestPoint = null;
            double smallestDiff = double.MaxValue;

            foreach (var point in _drawingLine)
            {
                double diff = Math.Abs((double)point.Timestamp - expectedDataTimestamp);
                if (diff < smallestDiff)
                {
                    smallestDiff = diff;
                    closestPoint = point;
                }
            }

            // 5. If the difference is within our threshold, return the point; otherwise return null
            return (smallestDiff <= thresholdUs) ? closestPoint : null;
        }
        
        /// <summary>
        /// Updates FormattedJson with the point that matches the current frame timestamp.
        /// </summary>
        private void UpdateFormattedJson()
        {
            if (_drawingLine.Count == 0)
            {
                FormattedJson = "No DrawingLines available.";
                return;
            }

            InkMARCPoint? closestPoint = _drawingLine[0];
            if (CurrentExercise.FirstPointOffset >= 0)
            {
                closestPoint = FindClosestDataPoint();
            }


            //// Find the closest timestamp match
            //var closestPoint = _drawingLines.Points.OrderBy(p => Math.Abs(p.Timestamp - _frameTimestamp)).FirstOrDefault();

            if (closestPoint != null)
            {
                FormattedJson = JsonSerializer.Serialize(closestPoint, new JsonSerializerOptions { WriteIndented = true });
            }
            else
            {
                FormattedJson = "No matching point found.";

            }
        }

        public SessionInfo CurrentExercise
        {
            get => currentExercise;
            set
            {
                if (SetProperty(ref currentExercise, value))
                {
                    if (CurrentExercise is not null)
                    {
                        CurrentExercise.SaveToFile();
                    }
                    FrameIndex = 0;
                    OnPropertyChanged(nameof(FrameIndex));
                    LoadSessionVideo(value);
                    LoadSessionJson(value);
                    OnPropertyChanged(nameof(HasData));
                    OnPropertyChanged(nameof(StartFrame));
                    OnPropertyChanged(nameof(StopFrame));
                    OnPropertyChanged(nameof(StartingPoint));
                    if (CurrentExercise.StopFrame == 0)
                        CurrentExercise.StopFrame = FrameCount;
                }
            }
        }

        public string FormattedJson
        {
            get => formattedJson;
            set => SetProperty(ref formattedJson, value);
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

            if (folderDialog.ShowDialog() != CommonFileDialogResult.Ok)
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