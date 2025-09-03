using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using InkMARC.Label.Services;
using InkMARC.Label.Views;
using InkMARC.Models.Primatives;
using InkMARC.Services.Video;
using MaterialDesignThemes.Wpf;
using Microsoft.WindowsAPICodePack.Dialogs;
using OpenCvSharp;
using System.Collections.ObjectModel;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Text.Json;
using System.Text.RegularExpressions;
using System.Windows;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Threading;

namespace InkMARC.Label
{
    internal partial class LocationLabellingViewModel : ObservableObject
    {
        private static readonly Brush ActiveBrush = Brushes.SkyBlue.Clone();

        private static readonly Brush InactiveBrush = Brushes.DimGray.Clone();

        #region Private Data
        private readonly VideoService _videoService;

        [ObservableProperty]
        private bool useBoundsInference = false;

        [ObservableProperty]
        private int frameIndex = 0;

        [ObservableProperty]
        private ImageSource? currentImage;

        [ObservableProperty]
        private bool currentState = false;

        [ObservableProperty]
        private bool currentIgnored = false;

        [ObservableProperty]
        private int selectedCorner = 0;

        [ObservableProperty]
        private bool isGuideVisible = false;

        [ObservableProperty]
        private ObservableCollection<SessionInfo> sessions = [];

        private string recordName = string.Empty;
        private List<InkMARCPoint>? _drawingLine;
        private ulong firstDataTimeStamp = 0;

        private SessionInfo? currentExercise;
        [ObservableProperty]
        private string? formattedJson;
        private bool isTouched = false;
        private bool isIgnored = false;

        [ObservableProperty]
        private int _sliderValue;

        private DispatcherTimer? _debounceTimer;
        private DispatcherTimer? _autoCornerTimer;

        private int lastFrameIndex = -1;
        private readonly Dictionary<int, Point2f[]> frameData = [];
        private Mat[] templates = new Mat[4];
        private Point2f[] centerPoints = new Point2f[4];

        [ObservableProperty]
        Point2f[] inferredCorners = new Point2f[4];

        private bool _isBulkProcessing = false;

        [ObservableProperty]
        private int maxProgress = 0;

        [ObservableProperty]
        private bool isSelectingPoints = false;
        private List<Point2f> _framePoints = new();
        private List<Point2f> _rotatedPoints = new();
        private List<Point2f> _scaledPoints = new();

        [ObservableProperty]
        private bool _isTrackingInProgress;

        [ObservableProperty]
        private double trackingProgress;

        [ObservableProperty]
        private double smoothingProgress;

        public int FrameCount => _videoService.FrameCount;

        private readonly int[] _xOffsets = new int[5];
        private readonly int[] _yOffsets = new int[5];

        [ObservableProperty]
        private bool _isAutoModeInProgress = false;

        private BoundsPredictor predictor = new BoundsPredictor("Resources/bounds_resnet18_448.onnx", useCuda: false);

    #endregion

    static LocationLabellingViewModel()
        {
            if (ActiveBrush.CanFreeze) ActiveBrush.Freeze();
            if (InactiveBrush.CanFreeze) InactiveBrush.Freeze();
        }
        public LocationLabellingViewModel()
        {
            _videoService = new VideoService();
            _videoService.FrameCountChanged += (s, e) => OnPropertyChanged(nameof(FrameCount));
            // Initialize templates to empty Mat objects
            for (int i = 0; i < templates.Length; i++)
            {
                templates[i] = new Mat();
            }
        }

        #region Public Properties

        public string CurrentSessionName => currentExercise?.SessionID ?? "No Session";

        public bool IsBulkProcessing
        {
            get => _isBulkProcessing;
            set => SetProperty(ref _isBulkProcessing, value);
        }

        public Point2f[] CenterPoints
        {
            get => centerPoints;
        }

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

        public float BoundRotation
        {
            get
            {
                var list = CurrentExercise.BoundRotations;
                return list.TryGetPredecessorValue(FrameIndex, out var rot) ? rot : 0f;
            }
            set
            {
                var list = CurrentExercise.BoundRotations;
                int frameIndex = FrameIndex;

                // Optional remove-on-zero:
                if (value == 0 && FrameIndex > 0 && list.TryGetPredecessorValue(FrameIndex - 1, out var prev) && prev == 0)
                {
                    list.Remove(frameIndex);
                    OnPropertyChanged(nameof(BoundRotation));
                    return;
                }

                list[frameIndex] = value;
                
                OnPropertyChanged(nameof(BoundRotation));
            }
        }

        public System.Windows.Media.Brush ZeroSelectedBrush => SelectedCorner == 0 && FramePoints.Count > 0 ? ActiveBrush : InactiveBrush;

        public System.Windows.Media.Brush OneSelectedBrush => SelectedCorner == 1 && FramePoints.Count > 0 ? ActiveBrush : InactiveBrush;

        public System.Windows.Media.Brush TwoSelectedBrush => SelectedCorner == 2 && FramePoints.Count > 0 ? ActiveBrush : InactiveBrush;

        public System.Windows.Media.Brush ThreeSelectedBrush => SelectedCorner == 3 && FramePoints.Count > 0 ? ActiveBrush : InactiveBrush;

        public System.Windows.Media.Brush FourSelectedBrush => SelectedCorner == 4 && FramePoints.Count > 0 ? ActiveBrush : InactiveBrush;

        public string CurrentBoundsString
        {
            get
            {
                string result = string.Empty;
                Point2f[] point2Fs = new Point2f[4];
                for (int i = 0; i < ScaledPoints.Count; i++)
                {
                    var pt = ScaledPoints[i];
                    var x = pt.X + XOffset + XOffsets[i + 1];
                    var y = pt.Y + YOffset + YOffsets[i + 1];
                    point2Fs[i] = new Point2f(x, y);
                }
                EnsureTLTRBRBL(point2Fs);
                result = $"TL({point2Fs[0].X:F1}, {point2Fs[0].Y:F1}), TR({point2Fs[1].X:F1}, {point2Fs[1].Y:F1}), BR({point2Fs[2].X:F1}, {point2Fs[2].Y:F1}), BL({point2Fs[3].X:F1}, {point2Fs[3].Y:F1})";
                return result;
            }
        }
   
        public float BoundScale
        {
            get
            {
                var list = CurrentExercise.BoundScales;
                return list.TryGetPredecessorValue(FrameIndex, out var scale) ? scale : 1f;
            }
            set
            {
                var list = CurrentExercise.BoundScales;
                int frameIndex = FrameIndex;
                if (value == 1 && FrameIndex > 0 && list.TryGetPredecessorValue(FrameIndex - 1, out var prev) && prev == 1)
                {
                    list.Remove(frameIndex);
                    OnPropertyChanged(nameof(BoundScale));
                    return;
                }

                list[frameIndex] = value;
                OnPropertyChanged(nameof(BoundScale));
            }
        }

        public int[] XOffsets
        {
            get
            {
                var result = GetXOffsets(FrameIndex);

                _xOffsets[0] = result[0];
                _xOffsets[1] = result[1];
                _xOffsets[2] = result[2];
                _xOffsets[3] = result[3];
                _xOffsets[4] = result[4];

                return _xOffsets;
            }
        }

        public int[] YOffsets
        {
            get
            {
                var result = GetYOffsets(FrameIndex);
                _yOffsets[0] = result[0];
                _yOffsets[1] = result[1];
                _yOffsets[2] = result[2];
                _yOffsets[3] = result[3];
                _yOffsets[4] = result[4];
                return _yOffsets;
            }
        }

        public int XOffset
        {
            get
            {
                var list = CurrentExercise.BoundOffsets; // SortedList<int, (int x, int y)>
                return list.TryGetPredecessorValue(FrameIndex, out var tup) ? tup.x : 0;
            }
            set
            {
                UpdateList(CurrentExercise.BoundOffsets, value, true);
                OnPropertyChanged(nameof(XOffset));
            }
        }

        public int YOffset
        {
            get
            {
                var list = CurrentExercise.BoundOffsets; 
                return list.TryGetPredecessorValue(FrameIndex, out var tup) ? tup.y : 0;
            }
            set
            {
                UpdateList(CurrentExercise.BoundOffsets, value, false);
                OnPropertyChanged(nameof(YOffset));
            }
        }

        public List<float> TouchPredictions => CurrentExercise?.TouchPredition ?? new List<float>();

        public float TouchThreshold
        {
            get => CurrentExercise?.TouchThreshold ?? 0.5f;
            set
            {
                if (CurrentExercise != null)
                {
                    CurrentExercise.TouchThreshold = value;
                    OnPropertyChanged(nameof(TouchThreshold));
                }
            }
        }

        public System.Windows.Media.Brush IsTouched => isTouched ? ActiveBrush : InactiveBrush;

        public PackIconKind IsIgnored => isIgnored ? PackIconKind.EyeOff : PackIconKind.Eye;

        public int Rotation
        {
            get => CurrentExercise?.Rotation ?? 0;
            set
            {
                if (CurrentExercise != null)
                {
                    CurrentExercise.Rotation = value;
                    OnPropertyChanged(nameof(Rotation));
                }
            }
        }

        public ObservableCollection<System.Windows.Point> SelectedPoints { get; } = new();

        public bool HasExercise => CurrentExercise is not null && !string.IsNullOrEmpty(CurrentExercise.VideoPath);

        public bool HasData => CurrentExercise is not null && CurrentExercise.HasData;

        public bool HasH5 => CurrentExercise is not null && CurrentExercise.HasH5;

        public bool HasBounds => CurrentExercise is not null && CurrentExercise.HasBounds;

        public long StartingPoint => CurrentExercise?.FirstPointOffset ?? -1;

        public SessionInfo CurrentExercise
        {
            get => currentExercise ?? new SessionInfo();
            set
            {
                if (SetProperty(ref currentExercise, value))
                {
                    if (currentExercise.BoundOffsets.Count == 0)
                        CurrentExercise.BoundOffsets.Add(0, (0, 0));
                    FrameIndex = 0;

                    LoadSessionVideo(value);
                    if (CurrentExercise is not null && CurrentExercise.StopFrame == 0)
                        CurrentExercise.StopFrame = _videoService.FrameCount;
                    LoadSessionJson(value);
                    LoadSessionBounds(value);

                    RefreshBindings();
                }
            }
        }

        public int IgnoredVersion { get; private set; }

        public int StateChangeNotifier { get; private set; }

        public SortedList<int, bool> IgnoredFrames => CurrentExercise?.IgnoredFrames ?? [];


        /// <summary>
        /// Stores state changes for the session.
        /// </summary>
        public SortedList<int, bool> StateChanges => CurrentExercise?.StateChanges ?? [];        

        public SortedList<int, bool> DataStateValues { get; set; } = [];

        partial void OnSliderValueChanged(int value)
        {
            StartDebounceTimer();
        }

        public int SliderTickFrequency => _videoService.FrameCount > 100 ? _videoService.FrameCount / 100 : 1;

        public List<Point2f> FramePoints
        {
            get => _framePoints;
            set
            {
                _framePoints = value;
                if (!IsBulkProcessing)
                    OnPropertyChanged();
            }
        }

        public List<Point2f> RotatedPoints
        {
            get => _rotatedPoints;
            private set => SetProperty(ref _rotatedPoints, value);            
        }

        public List<Point2f> ScaledPoints
        {
            get => _scaledPoints;
            private set => SetProperty(ref _scaledPoints, value);
        }

        #endregion

        #region Relay Commands

        [RelayCommand]
        public void SelectCorner(string parameter)
        {
            if (int.TryParse(parameter, out var corner))
            {
                if (corner < 5 && corner >= 0)
                {
                    SelectedCorner = corner;                    
                    OnPropertyChanged(nameof(ZeroSelectedBrush));
                    OnPropertyChanged(nameof(OneSelectedBrush));
                    OnPropertyChanged(nameof(TwoSelectedBrush));
                    OnPropertyChanged(nameof(ThreeSelectedBrush));
                    OnPropertyChanged(nameof(FourSelectedBrush));
                }    
            }
        }

        [RelayCommand]
        public void StartPointSelection()
        {
            IsSelectingPoints = true;
            SelectedPoints.Clear();
            MessageBox.Show("Click 4 points on the image. Press Enter to confirm.", "Select Points", MessageBoxButton.OK, MessageBoxImage.Information);
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
            if (frameIndex < _videoService.FrameCount) StartFrame = frameIndex;
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
            var dialog = new ExportOptionsWindow
            {
                Owner = Application.Current.Windows.OfType<System.Windows.Window>().FirstOrDefault(w => w.IsActive)
            };

            bool? result = dialog.ShowDialog();
            if (result == true)
            {
                maxProgress = StopFrame - StartFrame;
                OnPropertyChanged(nameof(MaxProgress));
                IsTrackingInProgress = true;

                if (string.IsNullOrEmpty(recordName))
                    recordName = Path.GetFileNameWithoutExtension(CurrentExercise?.VideoPath) + ".h5";
                if (dialog.ExportSession)
                {
                    // Export session data
                    currentExercise?.SaveToFile();
                }
                if (dialog.ExportLocation)
                {
                    // Export location data
                    await ExportSessionToHdf5Async();
                }
                if (dialog.ExportImage)
                {
                    ExportImage(FrameIndex);
                }
            }
            else
            {
                // User cancelled the export
            }
            IsTrackingInProgress = false;
        }


        [RelayCommand]
        private void IncrementTouchThreshold()
        {
            if (currentExercise is not null)
            {
                if (TouchThreshold < 0.99f)
                    TouchThreshold += 0.01f;
            }
        }

        [RelayCommand]
        private void DecrementTouchThreshold()
        {
            if (currentExercise is not null)
            {
                if (TouchThreshold > 0.01f)
                    TouchThreshold -= 0.01f;
            }
        }

        [RelayCommand]
        private void IncrementXOffset()
        {
            SetXOffset(XOffsets[SelectedCorner] + 1, SelectedCorner);
        }

        [RelayCommand]
        private void IncrementYOffset()
        {
            SetYOffset(YOffsets[SelectedCorner] + 1, SelectedCorner);            
        }

        [RelayCommand]
        private void DecrementXOffset()
        {
            SetXOffset(XOffsets[SelectedCorner] - 1, SelectedCorner);
        }

        [RelayCommand]
        private void DecrementYOffset()
        {
            SetYOffset(YOffsets[SelectedCorner] - 1, SelectedCorner);
        }

        [RelayCommand]
        private void IncreaseRotation()
        {
            Rotation = (Rotation + 90) % 360;
            UpdateImage();
        }

        [RelayCommand]
        private void DecreaseRotation()
        {
            if (Rotation > 0)
                Rotation -= 90;
            else
                Rotation = 270;
            UpdateImage();
        }

        [RelayCommand]
        public void ToggleTouched()
        {
            isTouched = !isTouched;
            OnPropertyChanged(nameof(IsTouched));
            if (CurrentExercise?.StateChanges.ContainsKey(FrameIndex) ?? false)
            {
                CurrentExercise?.StateChanges.Remove(FrameIndex);
            }
            CurrentExercise?.StateChanges.Add(FrameIndex, isTouched);
            StateChangeNotifier++;
            UpdateCurrentState();
            OnPropertyChanged(nameof(StateChanges));
            OnPropertyChanged(nameof(StateChangeNotifier));
        }

        [RelayCommand]
        public void ToggleBoundsInference()
        {
            UseBoundsInference = !UseBoundsInference;            
        }

        [RelayCommand]
        public void ToggleIgnored()
        {
            isIgnored = !isIgnored;
            if (CurrentExercise.IgnoredFrames.ContainsKey(FrameIndex))
            {
                CurrentExercise.IgnoredFrames.Remove(FrameIndex);
            }
            CurrentExercise.IgnoredFrames.Add(FrameIndex, isIgnored);
            IgnoredVersion++;
            UpdateIgnoredState();
        }

        [RelayCommand]
        public void ToggleGuideVisible()
        {
            IsGuideVisible = !IsGuideVisible;            
        }

        [RelayCommand]
        public async Task AnalyzeFramesForStateChangesAsync()
        {
            if (CurrentExercise == null)
                return;

            maxProgress = StopFrame - StartFrame;
            OnPropertyChanged(nameof(MaxProgress));
            IsTrackingInProgress = true;

            // Optionally clear any existing state changes
            CurrentExercise.StateChanges.Clear();

            bool sequenceActive = false; // Tracks if we are inside a sequence of frames with a datapoint
            int startFrame = CurrentExercise.StartFrame;
            int stopFrame = CurrentExercise.StopFrame;
            int originalFrameIndex = frameIndex;

            const int progressUpdateFrequency = 10;

            var progress = new Progress<int>(value =>
            {
                TrackingProgress = value;
            });

            // Run the analysis on a background thread.
            await Task.Run(() =>
            {
                // Loop through each frame between start and stop
                for (int i = startFrame; i <= stopFrame; i++)
                {
                    // Check if there is a datapoint for this frame.
                    bool hasDataPoint = FindClosestDataPointOptimized(i) != null;

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

            IsTrackingInProgress = false;

            // Restore the original frame index, if desired.
            frameIndex = originalFrameIndex;
        }

        [RelayCommand]
        public async Task ExtractFramesForStateChangesAsync()
        {
            if (CurrentExercise == null)
                return;

            OnPropertyChanged(nameof(MaxProgress));
            IsTrackingInProgress = true;

            var map = new SortedList<int, bool>();

            int startFrame = CurrentExercise.StartFrame;
            int stopFrame = CurrentExercise.StopFrame;
            int originalFrameIndex = FrameIndex;

            const int progressUpdateFrequency = 10;

            var progress = new Progress<int>(value =>
            {
                TrackingProgress = value;
            });

            await Task.Run(() =>
            {
                bool? previousState = null;

                for (int i = startFrame; i <= stopFrame; i++)
                {
                    bool hasDataPoint = FindClosestDataPointOptimized(i) != null;

                    // Record a change only if the state differs from the previous state
                    if (previousState == null || previousState.Value != hasDataPoint)
                    {
                        map[i] = hasDataPoint;
                        previousState = hasDataPoint;
                    }

                    if ((i - startFrame) % progressUpdateFrequency == 0)
                    {
                        ((IProgress<int>)progress).Report(i - startFrame + 1);
                    }
                }
            });

            IsTrackingInProgress = false;
            FrameIndex = originalFrameIndex;
            DataStateValues = map;
            OnPropertyChanged(nameof(DataStateValues));
        }

        [RelayCommand]
        public void MoveOffset(string parameter)
        {
            if (!int.TryParse(parameter, out int adjust))
                return;

            int newIndex = FrameIndex + adjust;
            if (newIndex < 0 || newIndex >= _videoService.FrameCount)
                return;

            // Update frame index and slider
            FrameIndex = newIndex;
            SliderValue = newIndex;

            UpdateImage();
            UpdateCurrentState();
            UpdateIgnoredState();

            if (CurrentExercise.FirstPointOffset >= 0)
                UpdateFormattedJson();

            if (frameData.TryGetValue(FrameIndex, out var points))
            {
                FramePoints = points.Select(p => new Point2f(p.X, p.Y)).ToList();
                if (CurrentExercise.CenterPoints.TryGetValue(FrameIndex, out Point2f[]? value))
                {
                    centerPoints = value;
                    OnPropertyChanged(nameof(CenterPoints));
                }
            }
            else
            {
                FramePoints = new List<Point2f>();
            }
            UpdateBounds();
            UpdateInferredBounds();
        }

        [RelayCommand]
        private void SmoothPointsCommand()
        {
            if (CurrentExercise is not null && CurrentExercise.CenterPoints.Count > 0)
            {
                SmoothPointTriplets(CurrentExercise.CenterPoints, 5.0f);
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

            var availableJson = Directory
                .GetFiles(directory, "*.json")
                .Where(f => !f.EndsWith("_smoothed.json", StringComparison.OrdinalIgnoreCase))
                .ToList();
            var availableH5 = Directory.GetFiles(directory, "*.h5").ToList();
            var availableData = Directory.GetFiles(directory, "*.session").ToList();
            var availableBounds = Directory.GetFiles(directory, "*_smoothed.json").ToList();

            // Build dictionaries for quick lookup.
            var videoSessionIds = BuildSessionIdDictionary(availableVideos);
            var dataSessionIds = BuildSessionIdDictionary(availableJson);
            var h5SessionIds = BuildSessionIdDictionarySimple(availableH5);
            var sessionData = BuildSessionDataDictionary(availableData);
            var boundsData = BuildSessionIdDictionary(availableBounds);

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
                    string boundsFile = string.Empty;
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
                    if (boundsData.TryGetValue(sessionId, out var boundsDict) && boundsDict.TryGetValue(exercise, out var boundsFileName))
                    {
                        boundsFile = boundsFileName.Item1;
                    }

                    if (sessionData.TryGetValue(sessionId, out var sessionDict) &&
                        sessionDict.TryGetValue(exercise, out var sessionFile))
                    {
                        var newSessionInfo = SessionInfo.LoadFromFile(sessionFile)
                                             ?? new SessionInfo(sessionId, videoFile, exercise, dataFile, h5File, boundsFile, videoDate, dataDate);
                        newSessionInfo.VideoPath = videoFile;
                        newSessionInfo.DataPath = dataFile;
                        newSessionInfo.H5Path = h5File;
                        newSessionInfo.BoundsPath = boundsFile;
                        Sessions.Add(newSessionInfo);
                    }
                    else
                    {
                        Sessions.Add(new SessionInfo(sessionId, videoFile, exercise, dataFile, h5File, boundsFile, videoDate, dataDate));
                    }
                }
            }
            CurrentExercise = Sessions.First();
            MoveOffset("0");
            OnPropertyChanged(nameof(ZeroSelectedBrush));
            OnPropertyChanged(nameof(OneSelectedBrush));
            OnPropertyChanged(nameof(TwoSelectedBrush));
            OnPropertyChanged(nameof(ThreeSelectedBrush));
            OnPropertyChanged(nameof(FourSelectedBrush));
        }

        [RelayCommand] void ToggleAutoMode()
        {
            IsAutoModeInProgress = !IsAutoModeInProgress;
        }

        [RelayCommand]
        private async Task MarkStartingPoint()
        {
            double currentFrameTime = FrameIndex * 1000.0 / _videoService.FramesPerSecond;
            CurrentExercise.FirstPointOffset = (long)currentFrameTime;
            OnPropertyChanged(nameof(StartingPoint));
            await ExtractFramesForStateChangesAsync();
            OnPropertyChanged(nameof(DataStateValues));
        }

        [RelayCommand]
        private async Task IncrementStartingPoint()
        {
            double frameTime = 1000.0 / _videoService.FramesPerSecond;
            CurrentExercise.FirstPointOffset += (long)frameTime;
            OnPropertyChanged(nameof(StartingPoint));
            await ExtractFramesForStateChangesAsync();
            OnPropertyChanged(nameof(DataStateValues));
        }

        [RelayCommand]
        private async Task DecrementStartingPoint()
        {
            double frameTime = 1000.0 / _videoService.FramesPerSecond;
            CurrentExercise.FirstPointOffset -= (long)frameTime;
            OnPropertyChanged(nameof(StartingPoint));
            await ExtractFramesForStateChangesAsync();
            OnPropertyChanged(nameof(DataStateValues));
        }

        //[RelayCommand]
        //public void LoadVideo(object parameter)
        //{
        //    // Open a folder selection dialog
        //    CommonOpenFileDialog folderDialog = new CommonOpenFileDialog
        //    {
        //        IsFolderPicker = true,
        //        Title = "Select Folder Containing Video Files"
        //    };

        //    if (folderDialog.ShowDialog() != CommonFileDialogResult.Ok || string.IsNullOrEmpty(folderDialog.FileName))
        //    {
        //        return; // User canceled
        //    }

        //    string directory = folderDialog.FileName;

        //    // Get all .h5 files and extract their base names
        //    HashSet<string> h5BaseNames = new HashSet<string>(
        //        Directory.GetFiles(directory, "*.h5")
        //        .Select(file => Path.GetFileNameWithoutExtension(file))
        //    );

        //    // Get all video files and filter out those with an associated .h5 file
        //    List<string> availableVideos = Directory.GetFiles(directory, "*.mp4")
        //        .Concat(Directory.GetFiles(directory, "*.avi"))
        //        .Concat(Directory.GetFiles(directory, "*.mov"))
        //        .Where(file => !h5BaseNames.Contains(Path.GetFileNameWithoutExtension(file)))
        //        .ToList();

        //    // If no videos are available, notify the user and exit
        //    if (availableVideos.Count == 0)
        //    {
        //        System.Windows.MessageBox.Show("No available video files found without an associated .h5 file.",
        //            "No Videos Found", System.Windows.MessageBoxButton.OK, System.Windows.MessageBoxImage.Information);
        //        return;
        //    }

        //    // Select the first available video
        //    string videoPath = availableVideos.First();
        //    recordName = Path.ChangeExtension(videoPath, ".h5");

        //    // Load the video
        //    if (!string.IsNullOrEmpty(videoPath))
        //    {
        //        if (videoCapture != null)
        //        {
        //            videoCapture.Dispose();
        //        }
        //        videoCapture = new VideoCapture(videoPath);
        //        frameCount = videoCapture is null ? 0 : (int)videoCapture.Get(VideoCaptureProperties.FrameCount);
        //        StartFrame = 0;
        //        StopFrame = frameCount;
        //        double frameDurationMs = 1000.0 / framesPerSecond;
        //        double thresholdMs = frameDurationMs / 2.0;
        //        thresholdUs = thresholdMs * 1000.0;
        //        OnPropertyChanged(nameof(FrameCount));
        //        OnPropertyChanged(nameof(FrameIndex));
        //        OnPropertyChanged(nameof(StartFrame));
        //        OnPropertyChanged(nameof(StopFrame));
        //        OnPropertyChanged(nameof(SliderTickFrequency));
        //        UpdateImage();
        //    }
        //}

        [RelayCommand]
        public async Task RunTemplateMatchingOnAllFramesAsyncOld()
        {
            if (!_videoService.IsOpen || CurrentExercise is null) return;

            // Progress setup (headless: no message boxes or image updates)
            MaxProgress = StopFrame - StartFrame + 1;
            TrackingProgress = 0;              // reset bar
            IsTrackingInProgress = true;

            // ~1% progress updates to avoid UI thrash
            int reportEvery = Math.Max(1, MaxProgress / 100);
            var progress = new Progress<int>(v => TrackingProgress = v);

            await Task.Run(() =>
            {
                // Init templates using the corrected centers at StartFrame
                using var first = _videoService.GetFrameAt(StartFrame);
                using var processedFirst = FrameProcessor.ProcessToMat(first, Rotation);
                var corr0 = GetCorrectedBoundsForFrame(StartFrame);

                for (int k = 0; k < 4; ++k)
                {
                    bool ok = templates[k] is not null && !templates[k].Empty();
                    if (!ok && !float.IsNaN(corr0[k].X))
                    {
                        Mat tpl = null!;
                        // Template window smaller than search window is fine
                        CapturePointTemplates(processedFirst, 6, corr0[k], ref tpl);
                        ok = tpl is not null && !tpl.Empty();
                        if (ok) templates[k] = tpl;
                    }
                    if (!ok) return; // bail if we can’t initialize templates
                }

                // Main loop (headless; progress only)
                for (int i = StartFrame; i <= StopFrame; ++i)
                {
                    using var frame = _videoService.GetFrameAt(i);
                    using var processed = FrameProcessor.ProcessToMat(frame, Rotation);
                    if (processed is null || processed.Empty()) continue;

                    var corrected = GetCorrectedBoundsForFrame(i);

                    // Match only the three measured corners as before
                    for (int j = 0; j < 3; j++)
                    {
                        if (!float.IsNaN(corrected[j].X))
                            ExtractCornerAtCorrectedPoint(j, processed, corrected[j]);
                    }

                    // Commit without touching UI-bound image props
                    var copy = (Point2f[])centerPoints.Clone();
                    if (!CurrentExercise.CenterPoints.TryAdd(i, copy))
                        CurrentExercise.CenterPoints[i] = copy;

                    // Throttled progress
                    int done = i - StartFrame + 1;
                    if (done % reportEvery == 0 || done == MaxProgress)
                        ((IProgress<int>)progress).Report(done);
                }

                // Persist once at the end
                CurrentExercise.SaveToFile();
            });

            // Finalize progress; still no popups
            TrackingProgress = MaxProgress;
            IsTrackingInProgress = false;
        }

        [RelayCommand]
        public void RunPythonTrackingFromSelectedPoints()
        {
            string videoPath = CurrentExercise?.VideoPath ?? "";
            if (!File.Exists(videoPath))
            {
                MessageBox.Show("Video path is missing or invalid.");
                return;
            }

            var pointsArray = SelectedPoints.Select(p => new[] { (int)p.X, (int)p.Y }).ToList();
            string jsonPoints = JsonSerializer.Serialize(pointsArray);

            string pythonExe = "C:\\Users\\tburnah\\source\\repos\\InkMARC_Locate_Tools\\.venv\\Scripts\\python.exe";
            string scriptPath = "C:\\Users\\tburnah\\source\\repos\\InkMARC_Locate_Tools\\SuperGluePretrainedNetwork\\tablettrack6.py";

            MaxProgress = 1;

            Task.Run(() =>
            {
                var psi = new ProcessStartInfo
                {
                    FileName = pythonExe,
                    Arguments = $"\"{scriptPath}\" \"{videoPath}\" \"{jsonPoints}\"",
                    UseShellExecute = false,
                    RedirectStandardOutput = true,
                    RedirectStandardError = true,
                    CreateNoWindow = true
                };

                var process = new Process { StartInfo = psi };

                process.OutputDataReceived += (s, e) =>
                {
                    if (string.IsNullOrWhiteSpace(e.Data))
                        return;

                    string line = e.Data.Trim();

                    if (line.StartsWith("PROGRESS:TRACK:"))
                    {
                        var parts = line.Substring("PROGRESS:TRACK:".Length).Split('/');
                        if (parts.Length == 2 &&
                            int.TryParse(parts[0], out int current) &&
                            int.TryParse(parts[1], out int total))
                        {
                            double progress = current / (double)total;
                            Application.Current.Dispatcher.Invoke(() =>
                            {
                                TrackingProgress = progress;
                                IsTrackingInProgress = true;
                            });
                        }
                    }
                    else if (line.StartsWith("PROGRESS:SMOOTH:"))
                    {
                        var parts = line.Substring("PROGRESS:SMOOTH:".Length).Split('/');
                        if (parts.Length == 2 &&
                            int.TryParse(parts[0], out int current) &&
                            int.TryParse(parts[1], out int total))
                        {
                            double progress = current / (double)total;
                            Application.Current.Dispatcher.Invoke(() =>
                            {
                                SmoothingProgress = progress;
                            });
                        }
                    }
                    else
                    {
                        Console.WriteLine($"[Python] {line}");
                    }
                };

                process.ErrorDataReceived += (s, e) =>
                {
                    if (!string.IsNullOrWhiteSpace(e.Data))
                        Console.WriteLine($"[Error] {e.Data}");
                };

                try
                {
                    process.Start();
                    process.BeginOutputReadLine();
                    process.BeginErrorReadLine();
                    process.WaitForExit();

                    Application.Current.Dispatcher.Invoke(() =>
                    {
                        IsTrackingInProgress = false;
                        SmoothingProgress = 1.0;
                        TrackingProgress = 1.0;

                        MessageBox.Show("Tracking complete. Smoothed file saved.", "Success", MessageBoxButton.OK, MessageBoxImage.Information);
                    });
                }
                catch (Exception ex)
                {
                    Application.Current.Dispatcher.Invoke(() =>
                    {
                        MessageBox.Show($"Error running script: {ex.Message}", "Error", MessageBoxButton.OK, MessageBoxImage.Error);
                    });
                }
            });
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

        [RelayCommand]
        public async Task PredictTouchForAllFramesAsync()
        {
            if (!_videoService.IsOpen || CurrentExercise is null)
            {
                MessageBox.Show("Video or session not loaded.");
                return;
            }

            IsTrackingInProgress = true;

            int totalFrames = StopFrame - StartFrame + 1;
            int originalFrameIndex = FrameIndex;

            // Ensure the list is long enough
            if (CurrentExercise.TouchPredition.Count < _videoService.FrameCount)
            {
                for (int i = CurrentExercise.TouchPredition.Count; i < _videoService.FrameCount; i++)
                    CurrentExercise.TouchPredition.Add(0.0f);
            }

            var progress = new Progress<int>(value =>
            {
                TrackingProgress = value;
            });

            await Task.Run(() =>
            {
                using ImagePredict predictor = new();

                var mat = new OpenCvSharp.Mat();
                mat = _videoService.GetFrameAt(StartFrame);

                for (int i = StartFrame; i <= StopFrame; i++)
                {
                    if (mat is null || mat.Empty())
                    {
                        Console.WriteLine($"Skipped frame {i}");
                        continue;
                    }

                    using var bitmap = OpenCvSharp.Extensions.BitmapConverter.ToBitmap(mat);
                    float pressure = predictor.PredictPressure(bitmap);

                    CurrentExercise.TouchPredition[i] = pressure;

                    if ((i - StartFrame) % 10 == 0)
                        ((IProgress<int>)progress).Report(i);

                    mat = _videoService.GetNextFrame();
                }
            });

            CurrentExercise.SaveToFile();

            IsTrackingInProgress = false;

            FrameIndex = originalFrameIndex;
            OnPropertyChanged(nameof(CurrentExercise.TouchPredition));
            MessageBox.Show("Touch prediction complete.", "Done", MessageBoxButton.OK, MessageBoxImage.Information);
        }

        [RelayCommand]
        private void RotateClockwise()
        {
            BoundRotation -= 0.5f;
            Debug.WriteLine("Rotation = " + BoundRotation.ToString());
            UpdateBounds();            
        }

        [RelayCommand]
        private void RotateCounterclockwise()
        {
            BoundRotation += 0.5f;
            Debug.WriteLine("Rotation = " + BoundRotation.ToString());
            UpdateBounds();            
        }

        [RelayCommand]
        private void IncreaseScale()
        {
            BoundScale += 0.01f;
            Debug.WriteLine("Scale = " + BoundScale.ToString());
            UpdateBounds();
        }

        [RelayCommand]
        private void DecreaseScale()
        {
            if (BoundScale > 0.1f)
            {
                BoundScale -= 0.01f;
                Debug.WriteLine("Scale = " + BoundScale.ToString());
                UpdateBounds();
            }
        }

        [RelayCommand]
        private void PullToTemplateMatch()
        {
            if (SelectedCorner > 0 && SelectedCorner < 5)
            {
                if (CenterPoints is not null && CenterPoints.Length > 0)
                {
                    Point2f scaledPoint = new (ScaledPoints[SelectedCorner - 1].X + XOffset, ScaledPoints[SelectedCorner - 1].Y + YOffset);
                    double distance = double.MaxValue;
                    int closest = 0;
                    for (int i = 0; i < CenterPoints.Length; i++)
                    {
                        var currentDistance = Math.Sqrt(Math.Pow(CenterPoints[i].X - scaledPoint.X, 2) + Math.Pow(CenterPoints[i].Y - scaledPoint.Y, 2));
                        if (currentDistance < distance)
                        {
                            distance = currentDistance;
                            closest = i;
                        }
                    }
                    float xDif = CenterPoints[closest].X - scaledPoint.X;
                    float yDif = CenterPoints[closest].Y - scaledPoint.Y;
                    SetXOffset((int)xDif, SelectedCorner);
                    SetYOffset((int)yDif, SelectedCorner);
                }
            }
        }

        [RelayCommand]
        private async Task ExportSessionToHdf5Async()
        {
            if (CurrentExercise is null || string.IsNullOrWhiteSpace(CurrentExercise.VideoPath))
            {
                System.Windows.MessageBox.Show("No session/video loaded.");
                return;
            }

            // Decide output path
            var h5Path = !string.IsNullOrWhiteSpace(CurrentExercise.H5Path)
                ? CurrentExercise.H5Path!
                : System.IO.Path.ChangeExtension(CurrentExercise.VideoPath, ".h5");

            IsTrackingInProgress = true;
            TrackingProgress = 0;

            await Task.Run(() =>
            {
                try
                {
                    // 1) Create the file
                    LocationDataSave.CreateFile(h5Path);

                    // 2) Find first valid frame in range to seed Initialize()
                    bool initialized = false;

                    int start = StartFrame;
                    int stop = Math.Max(start, Math.Min(StopFrame, _videoService.FrameCount - 1)); ;
                    int j = 0;

                    Point2f[] bounds = new Point2f[4];

                    // Clamp & sanity
                    if (start < 0) start = 0;
                    if (stop < start) stop = start;

                    for (int i = start; i <= stop; i++)
                    {
                        // respect ignored frames if present
                        if (CurrentExercise.IgnoredFrames != null &&
                            CurrentExercise.IgnoredFrames.TryGetValue(i, out bool ig) && ig)
                        {
                            continue; // skip ignored frames
                        }

                        using var src = _videoService.GetFrameAt(i);
                        if (src is null || src.Empty()) continue;

                        // Make a 448x448, correctly-rotated RGB/BGR Mat for saving
                        using var frame448 = PrepareFrame448(src, Rotation);

                        // Per-frame attributes
                        // xy: best-effort from your data-points
                        var pen = FindClosestDataPointOptimized(i);    // returns null if none
                        var xy = pen is null ? (float.NaN, float.NaN) : ((float)pen.Value.X, (float)pen.Value.Y);
                        var tilt = pen is null ? (0f, 0f) : ((float)pen.Value.TiltX, (float)pen.Value.TiltY);

                        // touched: prefer prediction list if present, else inferred from pen presence
                        bool touched = GetStateAtFrame(i);

                        // bounds
                        if (frameData.TryGetValue(i, out var points))
                        {

                            if (points is null || points.Length != 4)
                            {
                                SetNaN(bounds);
                            }
                            else
                            {

                                var pts = new List<Point2f>(points);
                                var degrees = CurrentExercise.BoundRotations.TryGetPredecessorValue(i, out var rot) ? rot : 0f;
                                if (degrees != 0f)
                                {
                                    GeometryHelper.RotateAroundCentroidInPlace(pts, degrees);
                                }

                                for (j = 0; j < pts.Count; j++)
                                    bounds[j] = pts[j];

                                var resize = CurrentExercise.BoundScales.TryGetPredecessorValue(i, out var scale) ? scale : 1f;
                                if (resize != 1.0f)
                                {
                                    bounds = QuadScalerCv.ScaleQuadAboutTopLeft(bounds, resize);
                                }

                                var xOffsets = GetXOffsets(i);
                                var yOffsets = GetYOffsets(i);
                                for (j = 0; j < bounds.Length; j++)
                                {
                                    bounds[j].X = bounds[j].X + xOffsets[0] + xOffsets[j + 1];
                                    bounds[j].Y = bounds[j].Y + yOffsets[0] + yOffsets[j + 1];
                                }
                            }
                        }
                        else
                        {  
                            // no bounds for this frame
                            SetNaN(bounds);
                        }

                        EnsureTLTRBRBL(bounds);

                        if (!initialized)
                        {
                            // First write defines datasets/chunks
                            if (!LocationDataSave.Initialize(
                                    frame448,
                                    touched,
                                    xy,
                                    (bounds[0].X, bounds[0].Y),
                                    (bounds[1].X, bounds[1].Y),
                                    (bounds[2].X, bounds[2].Y),
                                    (bounds[3].X, bounds[3].Y),
                                    tilt))
                            {
                                throw new InvalidOperationException("Failed to initialize HDF5 datasets.");
                            }
                            initialized = true;
                        }
                        else
                        {
                            // Append subsequent frames
                            if (!LocationDataSave.Append(
                                    frame448,
                                    touched,
                                    xy,
                                    (bounds[0].X, bounds[0].Y),
                                    (bounds[1].X, bounds[1].Y),
                                    (bounds[2].X, bounds[2].Y),
                                    (bounds[3].X, bounds[3].Y),
                                    tilt))
                            {
                                throw new InvalidOperationException($"Failed to append frame {i}.");
                            }
                        }

                        if ((i - start) % 10 == 0)
                        {
                            // coarse progress
                            var total = Math.Max(1, stop - start + 1);
                            var done = (i - start + 1);
                            System.Windows.Application.Current.Dispatcher.Invoke(() =>
                                TrackingProgress = (double)done / total);
                        }
                    }

                    // 3) finalize
                    LocationDataSave.Flush();
                    LocationDataSave.Close();

                    // Update the session so UI reflects the new file
                    CurrentExercise.UpdateH5Path(h5Path);
                }
                catch (Exception ex)
                {
                    System.Windows.Application.Current.Dispatcher.Invoke(() =>
                        System.Windows.MessageBox.Show($"Export failed: {ex.Message}", "Error"));
                }
            });

            // local helpers
            static void SetNaN(Point2f[] b)
            {
                b[0] = new Point2f(float.NaN, float.NaN);
                b[1] = new Point2f(float.NaN, float.NaN);
                b[2] = new Point2f(float.NaN, float.NaN);
                b[3] = new Point2f(float.NaN, float.NaN);
            }

            IsTrackingInProgress = false;
            TrackingProgress = 1.0;
        }

        /// <summary>
        /// Convert src frame to a square 448x448 Mat with current Rotation.
        /// Strategy:
        /// - paste src into a square black canvas,
        /// - scale that square to 448,
        /// - rotate via warpAffine, keeping 448x448 output.
        /// </summary>
        private static Mat PrepareFrame448(Mat srcBgr, int rotationDegrees)
        {
            // Square-pad
            int w = srcBgr.Width, h = srcBgr.Height;
            int side = Math.Max(w, h);

            var square = new Mat(new OpenCvSharp.Size(side, side), srcBgr.Type(), Scalar.Black);
            var roi = new OpenCvSharp.Rect((side - w) / 2, (side - h) / 2, w, h);
            using (var target = new Mat(square, roi))
            {
                srcBgr.CopyTo(target);
            }

            // Scale to 448
            var scaled = new Mat();
            Cv2.Resize(square, scaled, new OpenCvSharp.Size(448, 448), 0, 0, InterpolationFlags.Area);
            square.Dispose();

            // Rotate about center (deg clockwise is negative for OpenCV)
            if ((rotationDegrees % 360) != 0)
            {
                var center = new OpenCvSharp.Point2f(224, 224);
                using var M = Cv2.GetRotationMatrix2D(center, rotationDegrees, 1.0);
                var rotated = new Mat(new OpenCvSharp.Size(448, 448), srcBgr.Type());
                Cv2.WarpAffine(scaled, rotated, M, rotated.Size(), InterpolationFlags.Linear, BorderTypes.Constant, Scalar.Black);
                scaled.Dispose();
                return rotated; // BGR
            }

            return scaled; // already 448x448 BGR
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
            IsTrackingInProgress = true;

            int total = sessionJsonGroups.Count;
            int current = 0;

            var progress = new Progress<int>(value =>
            {
                TrackingProgress = value;
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

            IsTrackingInProgress = false;

            System.Windows.MessageBox.Show("Sessions organized successfully.", "Done", System.Windows.MessageBoxButton.OK, System.Windows.MessageBoxImage.Information);
        }

        #endregion

        #region Helpers
        public void ToggleAutoModePlaying()
        {
            if (IsAutoModeInProgress)
            {
                if (_autoCornerTimer is null || !_autoCornerTimer.IsEnabled)
                {
                    StartAutoCornerTimer();
                }
                else
                {
                    _autoCornerTimer.Stop();
                }
            }
        }

        private void UpdateBounds()
        {
            RotateSelected(BoundRotation);
            ScaleSelected();
        }

        private void RotateSelected(float degrees)
        {
            if (FramePoints is null || FramePoints.Count != 4) return;
            
            var pts = FramePoints.ToList();
            if (degrees != 0f)
            {
                GeometryHelper.RotateAroundCentroidInPlace(pts, degrees);
            }
            RotatedPoints.Clear();

            for (int i = 0; i < pts.Count; i++)
                RotatedPoints.Add(pts[i]);
            OnPropertyChanged(nameof(RotatedPoints));
        }

        private void SmoothPointTriplets(Dictionary<int, Point2f[]> points, float threshold = 5.0f)
        {
            if (points.Count < 5)
                return; // Not enough data to smooth

            var sortedKeys = points.Keys.OrderBy(k => k).ToList();

            for (int i = 0; i < sortedKeys.Count; i++)
            {
                var currentKey = sortedKeys[i];
                var currentValue = points[currentKey];

                // Get previous 2 keys/values if available
                var prev1 = i > 0 ? points[sortedKeys[i - 1]] : default;
                var prev2 = i > 1 ? points[sortedKeys[i - 2]] : default;

                // Get next 2 keys/values if available
                var next1 = i < sortedKeys.Count - 1 ? points[sortedKeys[i + 1]] : default;
                var next2 = i < sortedKeys.Count - 2 ? points[sortedKeys[i + 2]] : default;

                if (prev1 is null || prev2 is null || next1 is null || next2 is null)
                    continue; // Not enough data to smooth

                // Use currentValue, prev1, prev2, next1, next2 as needed
                var prevAvg = AveragePoints(prev2, prev1);
                var nextAvg = AveragePoints(next1, next2);

                if (prevAvg.Length != 3 || nextAvg.Length != 3)
                    continue; // Skip if averages are not valid

                Point2f newA = SmoothIfNeeded(points[i][0], prevAvg[0], nextAvg[0], threshold);
                Point2f newB = SmoothIfNeeded(points[i][1], prevAvg[1], nextAvg[1], threshold);
                Point2f newC = SmoothIfNeeded(points[i][2], prevAvg[2], nextAvg[2], threshold);

                points[i] = [newA, newB, newC];
            }
        }

        private static Point2f[] AveragePoints(Point2f[] p1, Point2f[] p2)
        {
            if (p1.Length != 3 || p2.Length != 3)
                return [];
            return [
                Average(p1[0], p2[0]),
                Average(p1[1], p2[1]),
                Average(p1[2], p2[2])
            ];
        }

        private static Point2f Average(Point2f p1, Point2f p2)
        {
            return new Point2f(
                (p1.X + p2.X) / 2.0f,
                (p1.Y + p2.Y) / 2.0f
            );
        }

        private static Point2f SmoothIfNeeded(Point2f current, Point2f prevAvg, Point2f nextAvg, float threshold)
        {
            var avgX = (prevAvg.X + nextAvg.X) / 2.0f;
            var avgY = (prevAvg.Y + nextAvg.Y) / 2.0f;
            var distance = MathF.Sqrt((current.X - avgX) * (current.X - avgX) + (current.Y - avgY) * (current.Y - avgY));

            if (distance > threshold)
                return new Point2f(avgX, avgY);

            return current;
        }

        private void ExtractCorner(int index, Mat bitmapSource, Point2f[] framePoints, int offsetX, int offsetY)
        {
            // Cache to avoid repeated casting and property access
            if (framePoints.Length < 3)
                return;

            // Ensure FramePoints has the requested index
            if (index >= framePoints.Length)
                return;

            // Calculate current point with offset
            var point = framePoints[index];
            var centerPoint = new Point2f(point.X + offsetX, point.Y + offsetY);

            Mat currentPos = new();
            CapturePointTemplates(bitmapSource, 6, centerPoint, ref currentPos);

            var result = MatchWithChamfer(currentPos, templates[index]);
            if (result is null)
                return;

            // Calculate corners relative to top-left of the extracted region
            Point2f sP = new((float)point.X - 6, (float)point.Y - 6);
            var corner1 = new Point2f(sP.X + result.Item1.X, sP.Y + result.Item1.Y);
            var corner2 = new Point2f(sP.X + result.Item2.X, sP.Y + result.Item2.Y);
            var corner3 = new Point2f(sP.X + result.Item3.X, sP.Y + result.Item3.Y);
            var corner4 = new Point2f(sP.X + result.Item4.X, sP.Y + result.Item4.Y);

            // Compute average center
            Point2f newCenter = new(
                (corner1.X + corner2.X + corner3.X + corner4.X) / 4.0f,
                (corner1.Y + corner2.Y + corner3.Y + corner4.Y) / 4.0f
            );

            // Only update and notify if value changed
            if (!centerPoints[index].Equals(newCenter))
            {
                centerPoints[index] = newCenter;
                OnPropertyChanged(nameof(CenterPoints));
            }
            currentPos.Dispose();
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

        private PerspectiveBounds GetPerspective()
        {
            // Apply offset to each point before ordering
            var offsetInput = FramePoints
                .Select(p => new Point2f(p.X + XOffset, p.Y + YOffset))
                .ToArray();

            var output = new Point2f[4];    // reuse this for every frame

            OrderClockwise(offsetInput, output);
            return new PerspectiveBounds()
            {
                First = output[0],
                Second = output[1],
                Third = output[2],
                Fourth = output[3]
            };
        }

        public static void OrderClockwise(Point2f[] points, Point2f[] result)
        {
            if (points.Length != 4)
                throw new ArgumentException("Exactly 4 points required.");
            if (result.Length != 4)
                throw new ArgumentException("Result array must have length 4.");

            // Step 1: Compute the centroid of the 4 points
            double centerX = 0, centerY = 0;
            foreach (var pt in points)
            {
                centerX += pt.X;
                centerY += pt.Y;
            }
            centerX /= 4;
            centerY /= 4;

            // Step 2: Sort points by angle from centroid to get clockwise order
            var sorted = points
                .Select(p => new
                {
                    Point = p,
                    Angle = Math.Atan2(p.Y - centerY, p.X - centerX)
                })
                .OrderBy(p => p.Angle)
                .ToArray();

            // Find the point with smallest Y (topmost), break ties by X (leftmost)
            int startIndex = 0;
            double minY = sorted[0].Point.Y;
            double minX = sorted[0].Point.X;

            for (int i = 1; i < 4; i++)
            {
                var p = sorted[i].Point;
                if (p.Y < minY || (Math.Abs(p.Y - minY) < 1e-3 && p.X < minX))
                {
                    minY = p.Y;
                    minX = p.X;
                    startIndex = i;
                }
            }

            // Rotate the array so that top-left is first
            for (int i = 0; i < 4; i++)
            {
                result[i] = sorted[(startIndex + i) % 4].Point;
            }
        }

        private Point2f[] GetCorrectedBoundsForFrame(int i)
        {
            // Default: NaNs if we have no points for this frame
            var nan = new Point2f(float.NaN, float.NaN);
            if (!frameData.TryGetValue(i, out var raw) || raw is null || raw.Length < 4)
                return new[] { nan, nan, nan, nan };

            // Start from raw points
            var pts = raw.ToList();

            // 1) rotate about centroid (if any)
            float deg = CurrentExercise.BoundRotations.TryGetPredecessorValue(i, out var rot) ? rot : 0f;
            if (Math.Abs(deg) > float.Epsilon)
                GeometryHelper.RotateAroundCentroidInPlace(pts, deg);

            // 2) scale about TL (if any)
            var bounds = pts.ToArray(); // TL,TR,BR,BL order expected elsewhere
            float scl = CurrentExercise.BoundScales.TryGetPredecessorValue(i, out var s) ? s : 1f;
            if (Math.Abs(scl - 1f) > 1e-6f)
                bounds = QuadScalerCv.ScaleQuadAboutTopLeft(bounds, scl);

            // 3) add general + per-corner offsets
            var xOffs = GetXOffsets(i); // [general, TL, TR, BL, BR]
            var yOffs = GetYOffsets(i);
            for (int j = 0; j < 4; j++)
            {
                bounds[j].X += xOffs[0] + xOffs[j + 1];
                bounds[j].Y += yOffs[0] + yOffs[j + 1];
            }

            EnsureTLTRBRBL(bounds);
            return bounds;
        }

        private void ExtractCornerAtPoint(int index, Mat bitmapSource, Point2f centerPoint)
        {
            // Defensive check
            if (index < 0 || index > 3) return;

            Mat currentPos = new();
            // window size unchanged (12), centered at the final point
            CapturePointTemplates(bitmapSource, 6, centerPoint, ref currentPos);

            var result = MatchWithChamfer(currentPos, templates[index]);
            if (result is null) return;

            // Compute corners relative to the top-left of the 12x12 window
            // (same math as your existing version, just anchored on centerPoint)
            Point2f sP = new(centerPoint.X - 6, centerPoint.Y - 6);
            var corner1 = new Point2f(sP.X + result.Item1.X, sP.Y + result.Item1.Y);
            var corner2 = new Point2f(sP.X + result.Item2.X, sP.Y + result.Item2.Y);
            var corner3 = new Point2f(sP.X + result.Item3.X, sP.Y + result.Item3.Y);

            // Store into the working triplet array
            // Keep your original index mapping behavior
            int[] indexMapping = { 1, 2, 3 }; // matches your previous usage
            if (index < indexMapping.Length)
                centerPoints[index] = new Point2f(
                    (corner1.X + corner2.X + corner3.X) / 3.0f,
                    (corner1.Y + corner2.Y + corner3.Y) / 3.0f
                );
        }

        private void ExtractCornerAtCorrectedPoint(int cornerIndex, Mat image, Point2f finalCenter)
        {
            // 12×12 search window around the fully corrected center
            if (!TryCaptureWindow(image, 12, finalCenter, out Mat currentPos, out Point2f roiTopLeft))
                return;

            var result = MatchWithChamfer(currentPos, templates[cornerIndex]);
            if (result is null) { currentPos.Dispose(); return; }

            // The four matched points are in ROI-local coords; translate by actual ROI top-left:
            var c1 = new Point2f(roiTopLeft.X + result.Item1.X, roiTopLeft.Y + result.Item1.Y);
            var c2 = new Point2f(roiTopLeft.X + result.Item2.X, roiTopLeft.Y + result.Item2.Y);
            var c3 = new Point2f(roiTopLeft.X + result.Item3.X, roiTopLeft.Y + result.Item3.Y);
            var c4 = new Point2f(roiTopLeft.X + result.Item4.X, roiTopLeft.Y + result.Item4.Y);

            // Average to center; update silently (no UI invalidation here)
            centerPoints[cornerIndex] = new Point2f(
                (c1.X + c2.X + c3.X + c4.X) / 4f,
                (c1.Y + c2.Y + c3.Y + c4.Y) / 4f
            );

            currentPos.Dispose();
        }

        private void StartAutoCornerTimer()
        {
            if (SelectedCorner > 0)
            {
                if (_autoCornerTimer == null)
                {
                    _autoCornerTimer = new DispatcherTimer
                    {
                        Interval = TimeSpan.FromMilliseconds(200)
                    };
                    _autoCornerTimer.Tick += _autoCornerTimer_Tick;
                }
                _autoCornerTimer?.Stop();
                _autoCornerTimer?.Start();
            }
        }

        private void _autoCornerTimer_Tick(object? sender, EventArgs e)
        {
            MoveOffset("1");
        }

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
            if (CurrentExercise?.StateChanges is not SortedList<int, bool> list || list.Count == 0)
                return false;

            bool result = false;
            list.TryGetPredecessorValue(frame, out result);
            return result;            
        }

        private bool GetIgnoredStateAtFrame(int frame)
        {
            if (CurrentExercise?.IgnoredFrames is not SortedList<int, bool> list || list.Count == 0)
                return false;

            bool result = false;
            list.TryGetPredecessorValue(frame, out result);
            return result;
        }

        private void UpdateCurrentState()
        {
            CurrentState = GetStateAtFrame(FrameIndex);
            OnPropertyChanged(nameof(CurrentState));
            OnPropertyChanged(nameof(IsTouched));
        }

        private void UpdateIgnoredState()
        {            
            CurrentIgnored = GetIgnoredStateAtFrame(FrameIndex);           
            OnPropertyChanged(nameof(IsIgnored));
            OnPropertyChanged(nameof(IgnoredFrames));
            OnPropertyChanged(nameof(IgnoredVersion));
        }

        private BitmapSource? GetImage()
        {
            if (!_videoService.IsOpen)
                return null;

            // Create a new Mat to hold the frame.
            Mat? frame = new Mat();

            // If we're moving sequentially forward, avoid repositioning.
            if (FrameIndex == lastFrameIndex + 1)
            {
                frame = _videoService.GetNextFrame();
                if (frame is null || frame.Empty())
                {
                    Console.WriteLine($"Failed to read sequential frame at index {FrameIndex}");
                    return null;
                }
            }
            else
            {
                frame = _videoService.GetFrameAt(FrameIndex);
                if (frame is null || frame.Empty())
                {
                    Console.WriteLine($"Failed to read frame at index {FrameIndex}");
                    return null;
                }
            }

            lastFrameIndex = FrameIndex;
            BitmapSource? processedImage = FrameProcessor.Process(frame, Rotation);
            return processedImage;
        }

        //private BitmapSource ProcessFrame(Mat frame)
        //{
        //    int width = frame.Width;
        //    int height = frame.Height;
        //    int squareSize = Math.Max(width, height);

        //    // Create a black square Mat to center the frame
        //    using Mat squareFrame = new(new OpenCvSharp.Size(squareSize, squareSize), frame.Type(), Scalar.Black);
        //    int xOffset = (squareSize - width) / 2;
        //    int yOffset = (squareSize - height) / 2;
        //    OpenCvSharp.Rect roi = new(xOffset, yOffset, width, height);
        //    using (Mat roiMat = new Mat(squareFrame, roi))
        //    {
        //        frame.CopyTo(roiMat);
        //    }

        //    // Compute the rotation matrix for the given rotation angle
        //    Point2f center = new Point2f(squareSize / 2f, squareSize / 2f);
        //    using Mat rotationMatrix = Cv2.GetRotationMatrix2D(center, Rotation, 1.0);

        //    // Apply the affine transformation (rotation)
        //    using Mat rotatedFrame = new();
        //    Cv2.WarpAffine(squareFrame, rotatedFrame, rotationMatrix, new OpenCvSharp.Size(squareSize, squareSize));

        //    // Convert the final rotated Mat directly to a BitmapSource
        //    BitmapSource bitmapSource = BitmapSourceConverter.ToBitmapSource(rotatedFrame);
        //    bitmapSource.Freeze(); // Freeze for thread safety
        //    return bitmapSource;
        //}

        private static Tuple<string, int, DateTime?>? ExtractSessionIDAndIndex(string fileName)
        {
            // Regex patterns for different filename variations
            string[] patterns =
            {
                // Pattern 1: type_sessionID_timestamp_smoothed.json
                @"^(?:data|video)_(?<sessionID>[a-zA-Z0-9]+)_(?<timestamp>\d+)_smoothed\.json$",

                // Pattern 2: type_sessionID_timestamp_index.extension
                @"^(?:data|video)_(?<sessionID>[a-zA-Z0-9]+)_(?<timestamp>[0-9T:\-.Z]+)_(?<index>\d+)\.\w+$",
    
                // Pattern 3: type_filetime_sessionID_index.extension (sessionID after timestamp)
                @"^(?:data|video)_(?<timestamp>\d+)_(?<sessionID>[a-zA-Z0-9]+)_(?<index>\d+)\.\w+$",

                // Pattern 4: type_sessionID_filetime_index.extension
                @"^(?:data|video)_(?<sessionID>[a-zA-Z0-9]+)_(?<timestamp>\d+)_(?<index>\d+)\.\w+$",
    
                // Pattern 5: type_sessionID_index.extension (index is 1–2 digits only)
                @"^(?:data|video)_(?<sessionID>[a-zA-Z0-9]+)_(?<index>\d{1,2})\.\w+$",

                // Pattern 7: type_sessionID_Participant_index_AppView.extension
                @"^(?:data|video)_(?<sessionID>[a-zA-Z0-9]+)_Participant(?<index>\d+)_AppView\d+\.\w+$",

                // Pattern 7: type_sessionID_timestamp.extension (no index) 
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
                    _videoService.Open(videoPath);
                    if (!_videoService.IsOpen)
                    {
                        Console.WriteLine($"Failed again!");
                        return;
                    }

                    //OnPropertyChanged(nameof(FrameCount));
                    OnPropertyChanged(nameof(FrameIndex));
                    OnPropertyChanged(nameof(StartFrame));
                    OnPropertyChanged(nameof(StopFrame));
                    OnPropertyChanged(nameof(SliderTickFrequency));
                    UpdateImage();
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

        private void UpdateInferredBounds()
        {
            if (UseBoundsInference)
            {
                using var src = _videoService.GetFrameAt(FrameIndex);
                if (src is null || src.Empty()) return;
                var image = PrepareFrame448(src, (int)(CurrentExercise.Rotation));

                InferredCorners = predictor.Predict(image);
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
                        // Handle errors (e.pt., file not found, permission issues)
                        FormattedJson = $"Error loading JSON: {ex.Message}";
                    }
                }

                ExtractFramesForStateChangesAsync();
            }
        }

        private void LoadSessionBounds(object parameter)
        {
            if ((parameter is not null) && (parameter is SessionInfo sessionInfo))
            {
                string? jsonPath = sessionInfo.BoundsPath;

                if (!string.IsNullOrEmpty(jsonPath))
                {
                    try
                    {
                        if (File.Exists(jsonPath))
                        {
                            string jsonText = File.ReadAllText(jsonPath);

                            frameData.Clear();

                            var rawData = JsonSerializer.Deserialize<Dictionary<string, List<List<double>>>>(jsonText);

                            foreach (var kvp in rawData)
                            {
                                if (int.TryParse(kvp.Key, out int frameIndex))
                                {
                                    var pointList = new Point2f[4];
                                    if (kvp.Value.Count != 4)
                                    {
                                        FormattedJson = $"Invalid data for frame {frameIndex}: expected 4 points, got {kvp.Value.Count}";
                                        continue;
                                    }
                                    int index = 0;
                                    foreach (var coord in kvp.Value)
                                    {
                                        if (coord.Count == 2)
                                            pointList[index++] = new Point2f((float)coord[0], (float)coord[1]);
                                    }

                                    frameData[frameIndex] = pointList;
                                }
                            }
                        }
                    }
                    catch (Exception ex)
                    {
                        // Handle errors (e.pt., file not found, permission issues)
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

        private int[] GetXOffsets(int index)
        {
            var general = CurrentExercise.BoundOffsets;
            var TL = CurrentExercise.CornerOffsetTL;
            var TR = CurrentExercise.CornerOffsetTR;
            var BL = CurrentExercise.CornerOffsetBL;
            var BR = CurrentExercise.CornerOffsetBR;

            var result = new int[5];
            result[0] = general.TryGetPredecessorValue(index, out var g) ? g.x : 0;
            result[1] = TL.TryGetPredecessorValue(index, out var tl) ? tl.x : 0;
            result[2] = TR.TryGetPredecessorValue(index, out var tr) ? tr.x : 0;
            result[3] = BL.TryGetPredecessorValue(index, out var bl) ? bl.x : 0;
            result[4] = BR.TryGetPredecessorValue(index, out var br) ? br.x : 0;

            return result;
        }

        private int[] GetYOffsets(int index)
        {
            var general = CurrentExercise.BoundOffsets;
            var TL = CurrentExercise.CornerOffsetTL;
            var TR = CurrentExercise.CornerOffsetTR;
            var BL = CurrentExercise.CornerOffsetBL;
            var BR = CurrentExercise.CornerOffsetBR;

            var result = new int[5];
            result[0] = general.TryGetPredecessorValue(index, out var g) ? g.y : 0;
            result[1] = TL.TryGetPredecessorValue(index, out var tl) ? tl.y : 0;
            result[2] = TR.TryGetPredecessorValue(index, out var tr) ? tr.y : 0;
            result[3] = BL.TryGetPredecessorValue(index, out var bl) ? bl.y : 0;
            result[4] = BR.TryGetPredecessorValue(index, out var br) ? br.y : 0;
            return result;
        }

        private bool TryCaptureWindow(Mat image, int size, Point2f center, out Mat roi, out Point2f roiTopLeft)
        {
            roi = null!;
            roiTopLeft = default;

            int half = size / 2;
            int cx = (int)center.X;
            int cy = (int)center.Y;

            int x = Math.Max(0, cx - half);
            int y = Math.Max(0, cy - half);
            int w = Math.Min(image.Width - x, size);
            int h = Math.Min(image.Height - y, size);

            if (w <= 0 || h <= 0) return false;

            roiTopLeft = new Point2f(x, y);
            roi = new Mat(image, new OpenCvSharp.Rect(x, y, w, h)).Clone();
            return true;
        }

        /// <summary>
        /// Optimized version of FindClosestDataPoint. If _drawingLine is sorted by timestamp,
        /// you could further optimize this with a binary search.
        /// </summary>
        private InkMARCPoint? FindClosestDataPointOptimized(int currentFrameIndex)
        {
            // Compute the video time for the frame.
            double frameVideoTimeMs = currentFrameIndex * 1000.0 / _videoService.FramesPerSecond;
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

            return (smallestDiff <= _videoService.ThresholdMicroseconds) ? closestPoint : null;
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
                    closestPoint = FindClosestDataPointOptimized(frameIndex);
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

        private void ExportImage(int frame)
        {
            var fileName = Path.GetFileNameWithoutExtension(CurrentExercise?.VideoPath) + $"frame_{frame:D5}.png";
            using var src = _videoService.GetFrameAt(frame);
            if (src is null || src.Empty()) return;

            PrepareFrame448(src, (int)(CurrentExercise.Rotation))?.SaveImage(Path.Combine(Path.GetDirectoryName(CurrentExercise?.VideoPath) ?? "", fileName));
        }

        private static void CapturePointTemplates(Mat image, int size, Point2f point, ref Mat output)
        {
            int xCenter = (int)point.X;
            int yCenter = (int)point.Y;

            int halfSize = (int)(size / 2);
            int x = Math.Max(0, xCenter - halfSize);
            int y = Math.Max(0, yCenter - halfSize);
            int width = Math.Min(image.Width - x, size);
            int height = Math.Min(image.Height - y, size);

            if (width > 0 && height > 0)
            {
                OpenCvSharp.Rect roi = new OpenCvSharp.Rect(x, y, width, height);
                output = new Mat(image, roi).Clone();  // Clone to decouple from original
                Cv2.ImWrite("template.png", output); // Save for debugging
            }
        }

        private double CalculateChamferScore(Mat sceneDist, Mat templateEdges, int offsetX, int offsetY)
        {
            // Ensure the ROI is fully within the bounds of sceneDist
            if (offsetX < 0 || offsetY < 0 ||
                offsetX + templateEdges.Cols > sceneDist.Cols ||
                offsetY + templateEdges.Rows > sceneDist.Rows)
            {
                return double.MaxValue; // Invalid region
            }

            // Extract region of interest
            OpenCvSharp.Rect roi = new(offsetX, offsetY, templateEdges.Cols, templateEdges.Rows);
            Mat distRoi = new(sceneDist, roi);

            // Sum distances where templateEdges > 0
            double sum = 0;
            double count = 0;

            for (int j = 0; j < templateEdges.Rows; ++j)
            {
                for (int i = 0; i < templateEdges.Cols; ++i)
                {
                    if (templateEdges.At<byte>(j, i) > 0) // Edge pixel
                    {
                        sum += distRoi.At<float>(j, i);
                        count++;
                    }
                }
            }

            return count > 0 ? sum / count : double.MaxValue; // Avoid division by zero
        }
        private void ScaleSelected()
        {                        
            var pts = RotatedPoints.ToArray();
            if (BoundScale != 1.0f)
            {
                pts = QuadScalerCv.ScaleQuadAboutTopLeft(pts, BoundScale);
            }
            ScaledPoints.Clear();

            for (int i = 0; i < pts.Length; i++)
                ScaledPoints.Add(pts[i]);
            OnPropertyChanged(nameof(ScaledPoints));
            OnPropertyChanged(nameof(CurrentBoundsString));
        }

        static void EnsureTLTRBRBL(Point2f[] pts)
        {
            if (pts == null || pts.Length != 4) return;
            if (pts.Any(p => float.IsNaN(p.X) || float.IsNaN(p.Y))) return;

            // Top two: smallest Y; Bottom two: largest Y
            var byY = pts.OrderBy(p => p.Y).ToArray();
            var top = byY.Take(2).OrderBy(p => p.X).ToArray();   // left->right
            var bottom = byY.Skip(2).OrderBy(p => p.X).ToArray();   // left->right

            var TL = top[0];
            var TR = top[1];
            var BL = bottom[0];
            var BR = bottom[1];

            // Write back in TL,TR,BR,BL order
            pts[0] = TL;
            pts[1] = TR;
            pts[2] = BR;
            pts[3] = BL;
        }

        private Tuple<Point2f, Point2f, Point2f, Point2f>? MatchWithChamfer(Mat imgScene, Mat imgTemplate)
        {
            try
            {
                // Get Edges
                Mat sceneGray = imgScene.CvtColor(ColorConversionCodes.BGR2GRAY);
                Mat templateGray = imgTemplate.CvtColor(ColorConversionCodes.BGR2GRAY);

                //Cv2.ImWrite("template_gray.png", templateGray);

                Mat sceneEdges = new();
                Mat templateEdges = new();

                Cv2.Canny(sceneGray, sceneEdges, 12, 150);
                Cv2.Canny(templateGray, templateEdges, 12, 150);

                Mat invertedSceneEdges = new();
                Cv2.BitwiseNot(sceneEdges, invertedSceneEdges);

                //Cv2.ImWrite("template_edges.png", templateEdges);

                Mat sceneDist = new();
                Cv2.DistanceTransform(invertedSceneEdges, sceneDist, DistanceTypes.L2, DistanceTransformMasks.Mask3);

                Mat distVis = new Mat();
                Cv2.Normalize(sceneDist, distVis, 0, 255, NormTypes.MinMax);
                distVis.ConvertTo(distVis, MatType.CV_8U); // Convert to 8-bit for saving

                // Optional: Apply colormap to visualize depth more clearly
                Mat distColor = new Mat();
                Cv2.ApplyColorMap(distVis, distColor, ColormapTypes.Jet);

                // Save both grayscale and color visualizations
                //Cv2.ImWrite("scene_distance_gray.png", distVis);
                //Cv2.ImWrite("scene_distance_color.png", distColor);

                // Slide template over scene to find best match
                double bestScore = double.MaxValue;
                Point2f bestPoint = new();

                int heatmapRows = sceneDist.Rows - templateEdges.Rows + 1;
                int heatmapCols = sceneDist.Cols - templateEdges.Cols + 1;

                Mat chamferScoreMap = new Mat(heatmapRows, heatmapCols, MatType.CV_32F, Scalar.All(0));

                for (int y = 0; y <= sceneDist.Rows - templateEdges.Rows; ++y)
                {
                    for (int x = 0; x <= sceneDist.Cols - templateEdges.Cols; ++x)
                    {
                        // Calculate Chamfer score
                        double score = CalculateChamferScore(sceneDist, templateEdges, x, y);

                        chamferScoreMap.Set(y, x, (float)score); // Record the score

                        if (score < bestScore)
                        {
                            bestScore = score;
                            bestPoint = new Point2f(x, y);
                        }
                    }
                }

                // Normalize to 0–255 for display
                Mat scoreVis = new Mat();
                Cv2.Normalize(chamferScoreMap, scoreVis, 0, 255, NormTypes.MinMax);
                scoreVis.ConvertTo(scoreVis, MatType.CV_8U);

                // Optional: apply colormap for heatmap-style visualization
                Mat heatmapColor = new Mat();
                Cv2.ApplyColorMap(scoreVis, heatmapColor, ColormapTypes.Jet);

                // Save both
                //Cv2.ImWrite("chamfer_score_gray.png", scoreVis);
                //Cv2.ImWrite("chamfer_score_heatmap.png", heatmapColor);

                // Return matching rectangle corners
                if (bestScore < double.MaxValue)
                {
                    Point2f topLeft = new Point2f(bestPoint.X, bestPoint.Y);
                    Point2f topRight = new Point2f(bestPoint.X + imgTemplate.Cols, bestPoint.Y);
                    Point2f bottomRight = new Point2f(bestPoint.X + imgTemplate.Cols, bestPoint.Y + imgTemplate.Rows);
                    Point2f bottomLeft = new Point2f(bestPoint.X, bestPoint.Y + imgTemplate.Rows);
                    return Tuple.Create(topLeft, topRight, bottomRight, bottomLeft);
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error during Chamfer matching: {ex.Message}");
            }

            return null;
        }

        private void SetXOffset(int value, int corner)
        {
            switch (corner)
            {
                case 0:
                    XOffset = value;
                    break;
                case 1:
                    UpdateList(CurrentExercise.CornerOffsetTL, value, true);
                    OnPropertyChanged(nameof(XOffsets));
                    break;
                case 2:
                    UpdateList(CurrentExercise.CornerOffsetTR, value, true);
                    OnPropertyChanged(nameof(XOffsets));
                    break;
                case 3:
                    UpdateList(CurrentExercise.CornerOffsetBL, value, true);
                    OnPropertyChanged(nameof(XOffsets));
                    break;
                case 4:
                    UpdateList(CurrentExercise.CornerOffsetBR, value, true);
                    OnPropertyChanged(nameof(XOffsets));
                    break;
            }
        }

        private void SetYOffset(int value, int corner)
        {
            switch (corner)
            {
                case 0:
                    YOffset = value;
                    break;
                case 1:
                    UpdateList(CurrentExercise.CornerOffsetTL, value, false);
                    OnPropertyChanged(nameof(YOffsets));
                    break;
                case 2:
                    UpdateList(CurrentExercise.CornerOffsetTR, value, false);
                    OnPropertyChanged(nameof(YOffsets));
                    break;
                case 3:
                    UpdateList(CurrentExercise.CornerOffsetBL, value, false);
                    OnPropertyChanged(nameof(YOffsets));
                    break;
                case 4:
                    UpdateList(CurrentExercise.CornerOffsetBR, value, false);
                    OnPropertyChanged(nameof(YOffsets));
                    break;
            }
        }

        private void UpdateList(SortedList<int, (int x, int y)> list, int value, bool isX)
        {
            if (isX)
            {
                int tempYOffset = list.TryGetPredecessorValue(FrameIndex, out var pt) ? pt.y : 0;
                list[FrameIndex] = (value, tempYOffset);
            }
            else
            {
                int tempXOffset = list.TryGetPredecessorValue(FrameIndex, out var pt) ? pt.x : 0;
                list[FrameIndex] = (tempXOffset, value);
            }
        }

        //private Tuple<Point2f, Point2f, Point2f, Point2f>? MatchWithOrb(Mat imgScene, Mat imgTemplate)
        //{
        //    if (imgScene.Empty() || imgTemplate.Empty())
        //    {
        //        return null;
        //    }


        //    Cv2.ImWrite("scene_debug_c.png", imgScene);
        //    Cv2.ImWrite("template_debug_c.png", imgTemplate);

        //    // Convert to grayscale if not already
        //    if (imgScene.Channels() > 1) imgScene = imgScene.CvtColor(ColorConversionCodes.BGR2GRAY);
        //    if (imgTemplate.Channels() > 1) imgTemplate = imgTemplate.CvtColor(ColorConversionCodes.BGR2GRAY);

        //    Cv2.ImWrite("scene_debug.png", imgScene);
        //    Cv2.ImWrite("template_debug.png", imgTemplate);

        //    var orb = AKAZE.Create();

        //    Mat des1 = new Mat();
        //    Mat des2 = new Mat();
        //    // Detect keypoints and descriptors
        //    orb.DetectAndCompute(imgTemplate, null, out KeyPoint[] kp1, des1);
        //    orb.DetectAndCompute(imgScene, null, out KeyPoint[] kp2, des2);

        //    if (des1.Empty() || des2.Empty())
        //    {
        //        Console.WriteLine("No descriptors found.");
        //        return null;
        //    }

        //    // Use BFMatcher with Hamming distance for ORB
        //    var bf = new BFMatcher(NormTypes.Hamming, crossCheck: true);
        //    var matches = bf.Match(des1, des2);

        //    // Sort matches and keep the best
        //    var goodMatches = matches.OrderBy(m => m.Distance).Take(50).ToArray();

        //    // Optional debug draw
        //    using (var matchImg = new Mat())
        //    {
        //        Cv2.DrawMatches(imgTemplate, kp1, imgScene, kp2, goodMatches, matchImg);
        //        Cv2.ImWrite("match_debug.png", matchImg);
        //    }

        //    // Compute Homography
        //    if (goodMatches.Length >= 4)
        //    {
        //        var srcPoints = goodMatches.Select(m => kp1[m.QueryIdx].Pt).ToArray();
        //        var dstPoints = goodMatches.Select(m => kp2[m.TrainIdx].Pt).ToArray();

        //        var H = Cv2.FindHomography(InputArray.Create(srcPoints), InputArray.Create(dstPoints), HomographyMethods.Ransac, 5.0);

        //        if (!H.Empty())
        //        {
        //            // Map template corners to scene
        //            var corners = new[]
        //            {
        //                new Point2f(0, 0),
        //                new Point2f(imgTemplate.Cols, 0),
        //                new Point2f(imgTemplate.Cols, imgTemplate.Rows),
        //                new Point2f(0, imgTemplate.Rows)
        //            };
        //            var transformedCorners = Cv2.PerspectiveTransform(corners, H);

        //            return Tuple.Create(
        //                transformedCorners[0],
        //                transformedCorners[1],
        //                transformedCorners[2],
        //                transformedCorners[3]
        //            );
        //        }
        //    }
        //    else
        //    {
        //        Console.WriteLine("Not enough good matches found.");
        //    }
        //    return null;
        //}

        private void RefreshBindings()
        {
            OnPropertyChanged(nameof(CurrentSessionName));
            OnPropertyChanged(nameof(HasExercise));
            OnPropertyChanged(nameof(HasData));
            OnPropertyChanged(nameof(HasH5));
            OnPropertyChanged(nameof(HasBounds));
            OnPropertyChanged(nameof(CurrentExercise));
            OnPropertyChanged(nameof(StateChanges));
            OnPropertyChanged(nameof(DataStateValues));
            OnPropertyChanged(nameof(StartFrame));
            OnPropertyChanged(nameof(StopFrame));
            OnPropertyChanged(nameof(StartingPoint));
            OnPropertyChanged(nameof(FrameIndex));
            OnPropertyChanged(nameof(CurrentImage));
            OnPropertyChanged(nameof(TouchPredictions));
            OnPropertyChanged(nameof(TouchThreshold));
        }

        private void UpdateImage()
        {
            CurrentImage = GetImage();
            OnPropertyChanged(nameof(CurrentImage));
        }

        #endregion
    }
}