using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using InkMARC.Clean.Model;
using InkMARC.Clean.Services;
using InkMARC.Clean.Services.Interfaces;
using Microsoft.Win32;
using OpenCvSharp;
using OpenCvSharp.WpfExtensions;
using System.IO;
using System.Windows.Input;
using System.Windows.Media.Imaging;

namespace InkMARC.Clean.ViewModels
{
    /// <summary>
    /// ViewModel for the main application window. Manages video/frame loading,
    /// export operations and image mask processing used by the UI.
    /// </summary>
    public partial class MainViewModel : ObservableObject, IDisposable
    {
        private readonly IVideoService _videoService;
        private readonly ColorPalette _palette;
        private CancellationTokenSource? _playCts;

        private IFrameSource? _frameSource;
        private bool _disposed;

        // Reused mats to avoid allocations every frame
        private Mat? _surfaceMask;
        private Mat? _hsv;
        private Mat? _colorMask;
        private Mat? _tmpMask;
        private Point[]? _lastSurfacePoly;

        private Mat? _ycrcb;
        private Mat? _skinMask;
        private Mat? _notSkinMask;
        private Mat? _pureGreenMask;

        private Mat? _lab;
        private Mat? _labA;
        private Mat? _labB;
        private Mat? _penMaskLab;
        private Mat? _bGeAMask;

        private Mat? _bgBgr;
        private Mat? _bgNoise;
        private Mat? _bgGrad16;

        private Mat? _bg16;
        private Mat? _noise8s;
        private Mat? _noise16s;
        private Mat? _col16;
        private Mat? _grad1_16;

        private Mat? _hsvOverlay;
        private Mat? _hCh;
        private Mat? _sCh;
        private Mat? _vCh;

        // Cached base+gradient (16S)
        private Mat? _bgBaseGrad16;
        private Scalar _bgBaseCached = new(double.NaN, double.NaN, double.NaN);
        private int _bgRowsCached = -1, _bgColsCached = -1, _bgGradAmpCached = int.MinValue;

        // Noise bank (16S)
        private Mat[]? _noise16Bank;
        private int _noiseBankRows = -1, _noiseBankCols = -1, _noiseBankAmp = int.MinValue;
        private const int NoiseBankSize = 32;

        /// <summary>
        /// Creates a new instance of <see cref="MainViewModel"/>.
        /// </summary>
        /// <param name="videoService">The video service used for frame access.</param>
        public MainViewModel(IVideoService videoService)
        {
            _videoService = videoService ?? throw new ArgumentNullException(nameof(videoService));
            _videoService.FrameCountChanged += VideoService_FrameCountChanged;

            // Centralised colour palette
            _palette = new ColorPalette();

            OpenVideoCommand = new RelayCommand(OpenVideo);
            MoveFrameCommand = new RelayCommand<int>(MoveFrame);
            TogglePlayCommand = new RelayCommand(TogglePlay);
            ProcessAllCommand = new RelayCommand(ProcessAll);
            ExportDatasetCommand = new RelayCommand(ExportDataset);
            ExportMultipleDatasetsCommand = new RelayCommand(ExportMultipleDatasets);
            ExportImageCommand = new RelayCommand(ExportImage);
            CycleResolutionCommand = new RelayCommand(CycleSizes);
            CycleColorsCommand = new RelayCommand(CycleColors);
            OpenPictureCommand = new RelayCommand(OpenPicture);
            OpenDatasetCommand = new RelayCommand(OpenDataset);
            StatsCommand = new RelayCommand(GetH5Stats);
        }

        #region progress reporting

        [ObservableProperty] private bool _isExporting;
        [ObservableProperty] private double _exportProgress01;
        [ObservableProperty] private string _exportStatusText = string.Empty;

        [ObservableProperty] private int _exportProgressFrames;
        [ObservableProperty] private int _exportProgressTotalFrames;

        [ObservableProperty]
        private double exportProgress; // 0..100

        [ObservableProperty]
        private string exportStatus = string.Empty;

        private CancellationTokenSource? _exportCts;

        [RelayCommand(CanExecute = nameof(CanCancelExport))]
        private void CancelExport()
        {
            _playCts?.Cancel();
        }

        private bool CanCancelExport() => IsExporting;

        private double _hueMin = 39;
        public double HueMin
        {
            get => _hueMin;
            set
            {
                if (_hueMin == value) return;
                _hueMin = value;
                if (_hueMin > HueMax) HueMax = _hueMin;   // clamp
                OnPropertyChanged(nameof(HueMin));
                OnPropertyChanged(nameof(HueRangeText));
                TriggerRecomputeMasks();
            }
        }

        private double _hueMax = 130;
        public double HueMax
        {
            get => _hueMax;
            set
            {
                if (_hueMax == value) return;
                _hueMax = value;
                if (_hueMax < HueMin) HueMin = _hueMax;   // clamp
                OnPropertyChanged(nameof(HueMax));
                OnPropertyChanged(nameof(HueRangeText));
                TriggerRecomputeMasks();
            }
        }

        public string HueRangeText => $"{HueMin:0} – {HueMax:0}";

        private double _valueThreshold = 110;
        public double ValueThreshold
        {
            get => _valueThreshold;
            set
            {
                if (_valueThreshold == value) return;
                _valueThreshold = value;
                OnPropertyChanged(nameof(ValueThreshold));
                TriggerRecomputeMasks();
            }
        }

        private double _saturationThreshold = 27;
        public double SaturationThreshold
        {
            get => _saturationThreshold;
            set
            {
                if (_saturationThreshold == value) return;
                _saturationThreshold = value;
                OnPropertyChanged(nameof(SaturationThreshold));
                TriggerRecomputeMasks();
            }
        }

        private void TriggerRecomputeMasks()
        {
           LoadFrame(_frameCount > 0 ? _currentFrameIndex : 0);
        }


        #endregion

        private void VideoService_FrameCountChanged(object? sender, int e)
        {
            FrameCount = e;
        }

        private int _frameCount;

        /// <summary>
        /// Gets or sets the number of frames available in the currently opened source.
        /// </summary>
        public int FrameCount
        {
            get => _frameCount;
            set { if (_frameCount != value) { _frameCount = value; OnPropertyChanged(nameof(FrameCount)); } }
        }

        private int _currentFrameIndex;

        /// <summary>
        /// Gets or sets the index of the currently displayed frame. Setting this will
        /// clamp the value and trigger a frame load.
        /// </summary>
        public int CurrentFrameIndex
        {
            get => _currentFrameIndex;
            set
            {
                if (_currentFrameIndex != value)
                {
                    _currentFrameIndex = Math.Max(0, Math.Min(value, Math.Max(0, FrameCount - 1)));
                    OnPropertyChanged(nameof(CurrentFrameIndex));
                    LoadFrame(_currentFrameIndex);
                }
            }
        }

        [ObservableProperty]
        private string currentFilePath = string.Empty;

        [ObservableProperty]
        private BitmapSource? currentFrameImage;
        [ObservableProperty]
        private BitmapSource? currentTextStripImage;

        [ObservableProperty]
        private int? bottomLeftX;
        [ObservableProperty]
        private int? bottomLeftY;
        [ObservableProperty]
        private int? bottomRightX;
        [ObservableProperty]
        private int? bottomRightY;
        [ObservableProperty]
        private int? topRightX;
        [ObservableProperty]
        private int? topRightY;
        [ObservableProperty]
        private int? topLeftX;
        [ObservableProperty]
        private int? topLeftY;

        [ObservableProperty]
        private int? stylusX;
        [ObservableProperty]
        private int? stylusY;
        [ObservableProperty]
        private int? stylusPressure;
        [ObservableProperty]
        private int? stylusTiltX;
        [ObservableProperty]
        private int? stylusTiltY;

        [ObservableProperty]
        private double viewW = 1080.0;
        [ObservableProperty]
        private double viewH = 2161.0;

        private const int HorizontalPadding = 300;

        [ObservableProperty]
        private string backgroundColor = "White";
        [ObservableProperty]
        private string foregroundColor = "Black";

        [ObservableProperty]
        private bool hasYellowBackground = false;

        private Scalar _backgroundScalar = new(255, 255, 255);

        /// <summary>
        /// When true, the computed surface mask will be shown/used in rendering.
        /// </summary>
        public bool ShowSurfaceMask { get; set; }

        /// <summary>
        /// Raw text recognized from the frame's auxiliary strip (if any).
        /// </summary>
        public string? CurrentRawOcr { get; set; }

        private bool _hasTextBar;

        /// <summary>
        /// Indicates whether the current frame contains an auxiliary text/strip area.
        /// </summary>
        public bool HasTextBar
        {
            get => _hasTextBar;
            private set { if (_hasTextBar != value) { _hasTextBar = value; OnPropertyChanged(nameof(HasTextBar)); } }
        }

        // Commands
        /// <summary>Command that opens a video file.</summary>
        public ICommand OpenVideoCommand { get; }
        /// <summary>Command that processes all frames sequentially.</summary>
        public ICommand ProcessAllCommand { get; }
        /// <summary>Command that exports the current dataset.</summary>
        public ICommand ExportDatasetCommand { get; }
        /// <summary>Command that exports multiple datasets in batch.</summary>
        public ICommand ExportMultipleDatasetsCommand { get; }
        /// <summary>Exports a single frame image.</summary>
        public ICommand ExportImageCommand { get; }
        /// <summary>Command that moves the current frame index by a delta.</summary>
        public ICommand MoveFrameCommand { get; }
        /// <summary>Command that toggles play/pause for frame playback.</summary>
        public ICommand TogglePlayCommand { get; }
        /// <summary>Command that cycles the viewport resolution.</summary>
        public ICommand CycleResolutionCommand { get; }
        /// <summary>Command that cycles background/foreground colours.</summary>
        public ICommand CycleColorsCommand { get; }
        /// <summary>Command that opens a single image file.</summary>
        public ICommand OpenPictureCommand { get; }
        /// <summary>Command that opens a dataset file.</summary>
        public ICommand OpenDatasetCommand { get; }

        public ICommand StatsCommand { get; }

        private async void Open()
        {
            if (_frameSource is null) return;

            var dlg = new Microsoft.Win32.OpenFileDialog
            {
                Filter = _frameSource?.FileFilter ?? "All files|*.*"
            };

            if (dlg.ShowDialog() == true)
            {
                OpenInternal(dlg.FileName);
            }
        }

        private void OpenInternal(string path)
        {
            try
            {
                CurrentFilePath = path;

                _frameSource?.Open(path);

                FrameCount = _frameSource?.FrameCount ?? 0;

                CurrentFrameIndex = 0;
                LoadFrame(0);
            }
            catch (Exception ex)
            {
                System.Windows.MessageBox.Show($"Failed to open video: {ex.Message}");
            }

        }

        private async void OpenVideo()
        {
            if (_frameSource is not VideoFileFrameSource)
            {
                if (_frameSource is IDisposable oldFs)
                {
                    oldFs.Dispose();
                }
                _frameSource = new VideoFileFrameSource(_videoService);
            }
            Open();
        }

        private static string? PickOutputDirectory(string? initialDir = null)
        {
            using var dlg = new System.Windows.Forms.FolderBrowserDialog
            {
                Description = "Select export output folder",
                UseDescriptionForTitle = true,
                SelectedPath = !string.IsNullOrWhiteSpace(initialDir) ? initialDir : Environment.CurrentDirectory,
                ShowNewFolderButton = true
            };

            return dlg.ShowDialog() == System.Windows.Forms.DialogResult.OK
                ? dlg.SelectedPath
                : null;
        }

        private void ExportMultipleDatasets()
        {
            if (_playCts != null)
            {
                var cts = _playCts;
                _playCts = null;
                try { cts.Cancel(); }
                catch { }
                finally { cts.Dispose(); }
            }

            // Pick input files
            var openDlg = new OpenFileDialog
            {
                Title = "Select files to export",
                Multiselect = true,
                Filter = "Video files|*.mp4;*.mov;*.avi;*.mkv;*.wmv;*.m4v|" + "All files|*.*"
            };

            if (openDlg.ShowDialog() != true || openDlg.FileNames.Length == 0)
                return;

            var initialDir = Path.GetDirectoryName(openDlg.FileNames[0]);
            var outDir = PickOutputDirectory(initialDir);

            // Pick output directory once
            if (string.IsNullOrWhiteSpace(outDir))
                return;

            bool prevShowSurfaceMask = ShowSurfaceMask;
            ShowSurfaceMask = true;

            var prevBackgroundScalar = _backgroundScalar;
            var prevBackgroundColor = BackgroundColor;
            var prevForegroundColor = ForegroundColor;

            _exportCts?.Cancel();
            _exportCts?.Dispose();
            _exportCts = new CancellationTokenSource();
            var ct = _exportCts.Token;

            IsExporting = true;
            ExportProgress = 0;
            ExportStatus = "Starting batch export...";

            var files = openDlg.FileNames;

            Task.Run(() =>
            {
                try
                {
                    var dispatcher = System.Windows.Application.Current.Dispatcher;

                    for (int fileIdx = 0; fileIdx < files.Length; fileIdx++)
                    {
                        ct.ThrowIfCancellationRequested();

                        var inputPath = files[fileIdx];
                        var baseName = Path.GetFileNameWithoutExtension(inputPath);

                        // create a fresh frame source per file (so mixed types work cleanly)
                        using var fs = new VideoFileFrameSource(_videoService);
                        fs.ViewW = (int)ViewW;
                        fs.ViewH = (int)ViewH;
                        fs.Open(inputPath);

                        int frameCount = fs.FrameCount;

                        // progress for this file = 0..1, map to overall 0..1
                        ExportDatasetCore(
                            frameSource: fs,
                            frameCount: frameCount,
                            baseDir: outDir!,
                            baseName: baseName,
                            ct: ct,
                            progress: (pct01WithinFile, status) =>
                            {
                                double overall01 = (fileIdx + pct01WithinFile) / Math.Max(1.0, files.Length);

                                dispatcher.BeginInvoke(new Action(() =>
                                {
                                    ExportProgress = overall01 * 100.0;
                                    ExportStatus = $"[{fileIdx + 1}/{files.Length}] {status}";
                                }));
                            });
                    }

                    System.Windows.Application.Current.Dispatcher.BeginInvoke(new Action(() =>
                    {
                        ExportProgress = 100;
                        ExportStatus = "Batch export complete.";
                    }));
                }
                catch (OperationCanceledException)
                {
                    System.Windows.Application.Current.Dispatcher.BeginInvoke(new Action(() =>
                    {
                        ExportStatus = "Batch export cancelled.";
                    }));
                }
                catch (Exception ex)
                {
                    System.Windows.Application.Current.Dispatcher.BeginInvoke(new Action(() =>
                    {
                        ExportStatus = "Batch export failed: " + ex.Message;
                        System.Windows.MessageBox.Show("Batch export failed: " + ex);
                    }));
                }
                finally
                {
                    System.Windows.Application.Current.Dispatcher.BeginInvoke(new Action(() =>
                    {
                        ShowSurfaceMask = prevShowSurfaceMask;
                        _backgroundScalar = prevBackgroundScalar;
                        BackgroundColor = prevBackgroundColor;
                        ForegroundColor = prevForegroundColor;
                        IsExporting = false;
                    }));
                }
            }, ct);
        }

        /// <summary>
        /// Opens an image file frame source and shows the open dialog.
        /// </summary>
        public void OpenPicture()
        {
            if (_frameSource is not ImageFileFrameSource)
            {
                if (_frameSource is IDisposable oldFs)
                {
                    oldFs.Dispose();
                }
                _frameSource = new ImageFileFrameSource();
            }
            Open();
        }

        /// <summary>
        /// Opens a dataset (HDF5) frame source and shows the open dialog.
        /// </summary>
        public void OpenDataset()
        {
            if (_frameSource is not Hdf5SessionFrameSource)
            {
                if (_frameSource is IDisposable oldFs)
                {
                    oldFs.Dispose();
                }
                _frameSource = new Hdf5SessionFrameSource();
            }
            Open();
        }

        private void CycleSizes()
        {
            if (ViewW == 1080)
            {
                ViewW = 1220;
                ViewH = 2550;
            }
            else
            {
                ViewW = 1080;
                ViewH = 2161;
            }
            if (_frameSource is not null)
            {
                _frameSource.ViewW = (int)ViewW;
                _frameSource.ViewH = (int)ViewH;
            }
        }

        private void CycleColors()
        {
            // Cycle using palette to centralise colour definitions
            BackgroundColor = _palette.Next(BackgroundColor);
            ForegroundColor = _palette.GetForeground(BackgroundColor);
            _backgroundScalar = _palette.GetScalar(BackgroundColor);
        }

        private void MoveFrame(int delta)
        {
            if (_frameSource is not null && !_frameSource.FileSeek)
            {
                if (FrameCount == 0) return;
                CurrentFrameIndex = Math.Max(0, Math.Min(FrameCount - 1, CurrentFrameIndex + delta));
            }
            else
            {
                if (delta > 0)
                {
                    var next = FileHelpers.GetNextFileInDirectory(CurrentFilePath);
                    if (!string.IsNullOrEmpty(next))
                        OpenInternal(next);
                }
                else
                {
                    var prev = FileHelpers.GetPreviousFileInDirectory(CurrentFilePath);
                    if (!string.IsNullOrEmpty(prev))
                        OpenInternal(prev);
                }
            }
        }

        private void TogglePlay()
        {
            if (_playCts != null)
            {
                var cts = _playCts;
                _playCts = null;

                try { cts.Cancel(); }
                catch {  }
                finally { cts.Dispose(); }

                return;
            }

            if (FrameCount == 0) return;

            var newCts = new CancellationTokenSource();
            _playCts = newCts;
            var ct = newCts.Token;

            Task.Run(async () =>
            {
                try
                {
                    while (!ct.IsCancellationRequested)
                    {
                        var delayMs = Math.Max(1, 1000.0 / Math.Max(1.0, _frameSource?.FramesPerSecond ?? 0));
                        await Task.Delay(TimeSpan.FromMilliseconds(delayMs), ct).ConfigureAwait(false);

                        if (ct.IsCancellationRequested) break;

                        var next = CurrentFrameIndex + 1;
                        if (next >= FrameCount) break;

                        System.Windows.Application.Current.Dispatcher.Invoke(() => CurrentFrameIndex = next);
                    }
                }
                catch (TaskCanceledException)
                {
                    // expected on cancel
                }
                finally
                {
                    if (ReferenceEquals(_playCts, newCts))
                    {
                        _playCts = null;
                    }
                    newCts.Dispose();
                }
            }, ct);
        }

        private void ProcessAll()
        {
            if (_playCts != null)
            {
                var cts = _playCts;
                _playCts = null;

                try { cts.Cancel(); }
                catch {  }
                finally { cts.Dispose(); }
            }

            if (FrameCount == 0) return;

            CurrentFrameIndex = 0;

            var newCts = new CancellationTokenSource();
            _playCts = newCts;
            var ct = newCts.Token;

            Task.Run(async () =>
            {
                try
                {
                    while (!ct.IsCancellationRequested)
                    {
                        if (ct.IsCancellationRequested) break;

                        var next = CurrentFrameIndex + 1;
                        if (next >= FrameCount) break;

                        System.Windows.Application.Current.Dispatcher.Invoke(() => CurrentFrameIndex = next);
                    }
                }
                catch (TaskCanceledException)
                {
                    // expected on cancel
                }
                finally
                {
                    if (ReferenceEquals(_playCts, newCts))
                    {
                        _playCts = null;
                    }
                    newCts.Dispose();
                }
            }, ct);
        }

        private void ExportDatasetCore(IFrameSource frameSource,
                                                    int frameCount,
                                                    string baseDir,
                                                    string baseName,
                                                    CancellationToken ct,
                                                    Action<double, string>? progress = null)
        {
            const int attrCount = 5;

            var colours = ColorPalette.BackgroundNames.ToArray();
            var bgScalars = colours.Select(name => _palette.GetScalar(name)).ToArray();

            var dispatcher = System.Windows.Application.Current.Dispatcher;

            // 1) Find first usable frame to establish dimensions
            FrameData? firstFrame = null;
            for (int i = 0; i < frameCount; i++)
            {
                ct.ThrowIfCancellationRequested();
                firstFrame = frameSource.GetFrameForExport(i);
                if (firstFrame?.Image != null && !firstFrame.Image.Empty())
                    break;

                firstFrame?.Image?.Dispose();
                firstFrame?.AuxImage?.Dispose();
                firstFrame = null;
            }

            if (firstFrame?.Image == null)
                throw new InvalidOperationException("Could not read any frames to export.");

            int yCrop = firstFrame.AuxImage?.Height ?? 0;
            int exportH = firstFrame.Image.Height - yCrop;
            int exportW = firstFrame.Image.Width;

            firstFrame.Image.Dispose();
            firstFrame.AuxImage?.Dispose();

            // 2) Buffers
            var cornersBuffer = new float[8];
            var labelsBuffer = new float[attrCount];
            var labelMaskBuffer = new byte[attrCount];

            var zeroCorners = new float[8];
            var zeroLabels = new float[attrCount];
            var zeroMask = new byte[attrCount];

            // A real black frame to keep AVI aligned + to satisfy validation if needed
            using var zeroBgr = new Mat(exportH, exportW, MatType.CV_8UC3);
            zeroBgr.SetTo(new Scalar(0, 0, 0));

            // 3) Timing
            int fps = frameSource.FramesPerSecond > 0 ? (int)Math.Round(frameSource.FramesPerSecond) : 30;
            if (fps <= 0) fps = 30;
            ulong nsPerFrame = (ulong)(1_000_000_000L / fps);

            // 4) Output paths: ONE H5, MANY AVIs
            string outH5Path = Path.Combine(baseDir, $"{baseName}.h5");

            // Open single metadata session 
            using var session = SessionManager.CreateNew(
                h5Path: outH5Path,
                frameCount: (ulong)frameCount,
                height: exportH,
                width: exportW,
                attrCount: attrCount,
                fps: fps,
                chunkFrames: 256);

            // Open all AVI writers once
            var writers = new OpenCvSharp.VideoWriter[colours.Length];

            // Preallocate per-variant output frames to avoid Clone() allocations
            var outFrames = new Mat[colours.Length];
            for (int i = 0; i < outFrames.Length; i++)
                outFrames[i] = new Mat(exportH, exportW, MatType.CV_8UC3);

            try
            {
                for (int pass = 0; pass < colours.Length; pass++)
                {
                    string outAviPath = Path.Combine(baseDir, $"{baseName}_{colours[pass]}.avi");

                    writers[pass] = new OpenCvSharp.VideoWriter(
                        outAviPath,
                        OpenCvSharp.FourCC.MJPG,
                        fps,
                        new OpenCvSharp.Size(exportW, exportH),
                        isColor: true);

                    if (!writers[pass].IsOpened())
                        throw new InvalidOperationException($"Failed to open AVI writer: {outAviPath}");
                }

                long totalWork = (long)frameCount * colours.Length;
                const int reportEveryNFrames = 200;

                dispatcher.BeginInvoke(new Action(() =>
                {
                    BackgroundColor = colours[0];
                    ForegroundColor = _palette.GetForeground(colours[0]);
                }));

                for (int frameIndex = 0; frameIndex < frameCount; frameIndex++)
                {
                    ct.ThrowIfCancellationRequested();

                    FrameData? frameData = null;
                    try
                    {
                        frameData = frameSource.GetFrameForExport(frameIndex);
                        ulong timestampNs = (ulong)frameIndex * nsPerFrame;

                        if (frameData?.Image == null || frameData.Image.Empty())
                        {
                            // Metadata once (zeros)
                            session.WriteFrame(
                                frameIndex: (ulong)frameIndex,
                                timestampNs: timestampNs,
                                corners: zeroCorners,
                                labels: zeroLabels,
                                labelMask: zeroMask);

                            // Write a black frame to every AVI to keep alignment
                            for (int pass = 0; pass < writers.Length; pass++)
                                writers[pass].Write(zeroBgr);

                            continue;
                        }

                        int yOffset = frameData.AuxImage?.Height ?? 0;
                        using var cropped = GetBelowDatabarRoi(frameData.Image, yOffset);

                        // ---- Metadata buffers ----
                        var tl = frameData.TopLeft;
                        var tr = frameData.TopRight;
                        var br = frameData.BottomRight;
                        var bl = frameData.BottomLeft;

                        cornersBuffer[0] = (float)(tl?.X ?? 0);
                        cornersBuffer[1] = (float)(tl?.Y ?? 0);
                        cornersBuffer[2] = (float)(tr?.X ?? 0);
                        cornersBuffer[3] = (float)(tr?.Y ?? 0);
                        cornersBuffer[4] = (float)(br?.X ?? 0);
                        cornersBuffer[5] = (float)(br?.Y ?? 0);
                        cornersBuffer[6] = (float)(bl?.X ?? 0);
                        cornersBuffer[7] = (float)(bl?.Y ?? 0);

                        labelMaskBuffer[0] = frameData.StylusX.HasValue ? (byte)1 : (byte)0;
                        labelsBuffer[0] = frameData.StylusX.GetValueOrDefault();

                        labelMaskBuffer[1] = frameData.StylusY.HasValue ? (byte)1 : (byte)0;
                        labelsBuffer[1] = frameData.StylusY.GetValueOrDefault();

                        labelMaskBuffer[2] = frameData.StylusPressure.HasValue ? (byte)1 : (byte)0;
                        labelsBuffer[2] = frameData.StylusPressure.GetValueOrDefault();

                        labelMaskBuffer[3] = frameData.StylusTiltX.HasValue ? (byte)1 : (byte)0;
                        labelsBuffer[3] = frameData.StylusTiltX.GetValueOrDefault();

                        labelMaskBuffer[4] = frameData.StylusTiltY.HasValue ? (byte)1 : (byte)0;
                        labelsBuffer[4] = frameData.StylusTiltY.GetValueOrDefault();

                        // Write metadata ONCE
                        session.WriteFrame(
                            frameIndex: (ulong)frameIndex,
                            timestampNs: timestampNs,
                            corners: cornersBuffer,
                            labels: labelsBuffer,
                            labelMask: labelMaskBuffer);

                        ComputeSurfaceAndKeyMasks(
                            cropped,
                            padX: 0,
                            frameData: frameData,
                            out var surfaceMask,
                            out var colorMask,
                            out var penMaskLab);

                        // ---- Apply each colour variant cheaply ----
                        for (int pass = 0; pass < colours.Length; pass++)
                        {
                            // Update UI only on first frame, optional
                            if (frameIndex == 0)
                            {
                                string colourName = colours[pass];
                                dispatcher.BeginInvoke(new Action(() =>
                                {
                                    BackgroundColor = colourName;
                                    ForegroundColor = _palette.GetForeground(colourName);
                                }));
                            }

                            var outFrame = outFrames[pass];

                            // Apply background replacement using the precomputed colorMask
                            ApplyBackgroundVariant(
                                dstBgr: outFrame,
                                srcBgr: cropped,
                                colorMask: colorMask,
                                backgroundScalar: bgScalars[pass],
                                frameIndex: frameData.FrameIndex);

                            // Apply pen overlay (same mask across variants)
                            ApplyPenOverlay(outFrame, penMaskLab);

                            writers[pass].Write(outFrame);
                        }
                    }
                    finally
                    {
                        frameData?.Image?.Dispose();
                        frameData?.AuxImage?.Dispose();
                    }

                    if (frameIndex == 0 || frameIndex == frameCount - 1 || (frameIndex % reportEveryNFrames == 0))
                    {
                        long done = (long)(frameIndex + 1) * colours.Length;
                        double pct = (totalWork <= 0) ? 1.0 : (done / (double)totalWork);
                        string status = $"Exporting {baseName}: frame {frameIndex + 1}/{frameCount} ({colours.Length} variants)";
                        progress?.Invoke(pct, status);
                    }
                }
            }
            finally
            {
                for (int i = 0; i < writers.Length; i++)
                    writers[i]?.Dispose();

                for (int i = 0; i < outFrames.Length; i++)
                    outFrames[i]?.Dispose();
            }
        }

        private void ExportImage()
        {
            if (_frameSource is null || FrameCount <= 0)
            {
                System.Windows.MessageBox.Show("No video is open.");
                return;
            }

            var dlg = new SaveFileDialog
            {
                Title = "Export current frame as PNG",
                Filter = "PNG Image|*.png|All files|*.*",
                AddExtension = true,
                DefaultExt = ".png",
                FileName = Path.GetFileNameWithoutExtension(CurrentFilePath) + $"_{CurrentFrameIndex}.png",
                InitialDirectory = string.IsNullOrWhiteSpace(CurrentFilePath) ? null : Path.GetDirectoryName(CurrentFilePath)
            };

            if (dlg.ShowDialog() != true)
                return;

            FrameData? frameData = null;

            try
            {
                frameData = _frameSource.GetFrame(CurrentFrameIndex);

                if (frameData?.Image == null || frameData.Image.Empty())
                {
                    System.Windows.MessageBox.Show("Could not read the current frame.");
                    return;
                }

                // IMPORTANT: FrameData.Image is already "minus data bar" per your note.
                using var outBgr = frameData.Image.Clone();

                // Apply mask exactly like the on-screen view (opaque)
                if (ShowSurfaceMask)
                {
                    // Use the same frameData coordinates you render with.
                    DrawSurfaceMask(outBgr, padX: 0, frameData);
                }

                Cv2.ImEncode(".png", outBgr, out byte[] buf);
                File.WriteAllBytes(dlg.FileName, buf);
            }
            catch (Exception ex)
            {
                System.Windows.MessageBox.Show("Export failed: " + ex.Message);
            }
            finally
            {
                frameData?.Image?.Dispose();
                frameData?.AuxImage?.Dispose();
            }
        }

        private void ExportDataset()
        {
            if (_frameSource is null || FrameCount <= 0)
            {
                System.Windows.MessageBox.Show("No video is open.");
                return;
            }

            if (_playCts != null)
            {
                var cts = _playCts;
                _playCts = null;
                try { cts.Cancel(); }
                catch { }
                finally { cts.Dispose(); }
            }

            var dlg = new SaveFileDialog
            {
                Title = "Export dataset (base filename)",
                Filter = "HDF5 files|*.h5;*.hdf5|All files|*.*",
                FileName = Path.GetFileNameWithoutExtension(CurrentFilePath) + "_dataset.h5",
                InitialDirectory = string.IsNullOrWhiteSpace(CurrentFilePath) ? null : Path.GetDirectoryName(CurrentFilePath),
                AddExtension = true,
                DefaultExt = ".h5",
                OverwritePrompt = true
            };

            if (dlg.ShowDialog() != true)
                return;

            var basePath = dlg.FileName;
            var baseDir = Path.GetDirectoryName(basePath) ?? Environment.CurrentDirectory;
            var baseName = Path.GetFileNameWithoutExtension(basePath);

            bool prevShowSurfaceMask = ShowSurfaceMask;
            ShowSurfaceMask = true;

            var prevBackgroundScalar = _backgroundScalar;
            var prevBackgroundColor = BackgroundColor;
            var prevForegroundColor = ForegroundColor;

            _exportCts?.Cancel();
            _exportCts?.Dispose();
            _exportCts = new CancellationTokenSource();
            var ct = _exportCts.Token;

            IsExporting = true;
            ExportProgress = 0;
            ExportStatus = "Starting export...";

            Task.Run(() =>
            {
                try
                {
                    ExportDatasetCore(_frameSource, FrameCount, baseDir, baseName, ct,
                        progress: (pct01, status) =>
                        {
                            System.Windows.Application.Current.Dispatcher.BeginInvoke(new Action(() =>
                            {
                                ExportProgress = pct01 * 100.0;
                                ExportStatus = status;
                            }));
                        });

                    System.Windows.Application.Current.Dispatcher.BeginInvoke(new Action(() =>
                    {
                        ExportProgress = 100;
                        ExportStatus = "Export complete.";
                    }));
                }
                catch (OperationCanceledException)
                {
                    System.Windows.Application.Current.Dispatcher.BeginInvoke(new Action(() =>
                    {
                        ExportStatus = "Export cancelled.";
                    }));
                }
                catch (Exception ex)
                {
                    System.Windows.Application.Current.Dispatcher.BeginInvoke(new Action(() =>
                    {
                        ExportStatus = "Export failed: " + ex.Message;
                        System.Windows.MessageBox.Show("Export failed: " + ex);
                    }));
                }
                finally
                {
                    System.Windows.Application.Current.Dispatcher.BeginInvoke(new Action(() =>
                    {
                        ShowSurfaceMask = prevShowSurfaceMask;
                        _backgroundScalar = prevBackgroundScalar;
                        BackgroundColor = prevBackgroundColor;
                        ForegroundColor = prevForegroundColor;
                        IsExporting = false;
                    }));
                }
            }, ct);
        }

        private static Mat GetBelowDatabarRoi(Mat src, int yOffset)
        {
            ArgumentNullException.ThrowIfNull(src);
            if (src.Empty()) throw new ArgumentException("src is empty", nameof(src));

            if (yOffset <= 0) return src;
            yOffset = Math.Clamp(yOffset, 0, src.Rows - 1);

            var rect = new Rect(0, yOffset, src.Cols, src.Rows - yOffset);
            return new Mat(src, rect);
        }

        private void LoadFrame(int index)
        {
            FrameData? frameData = null;

            try
            {
                if (_frameSource is null)
                    return;

                frameData = _frameSource.GetFrame(index);

                if (frameData is null || frameData.Image == null)
                {
                    CurrentFrameImage = null;
                    CurrentTextStripImage = null;
                    HasTextBar = false;
                    CurrentRawOcr = null;
                    OnPropertyChanged(nameof(CurrentRawOcr));
                    return;
                }

                CurrentTextStripImage = frameData.AuxBitmapSource;
                HasTextBar = frameData.HasStylusData;
                CurrentRawOcr = frameData.AdditionalText;
                OnPropertyChanged(nameof(CurrentRawOcr));

                if (frameData.TopRight is Point topRight)
                {
                    TopRightX = topRight.X;
                    TopRightY = topRight.Y;
                }
                if (frameData.TopLeft is Point topLeft)
                {
                    TopLeftX = topLeft.X;
                    TopLeftY = topLeft.Y;
                }
                if (frameData.BottomLeft is Point bottomLeft)
                {
                    BottomLeftX = bottomLeft.X;
                    BottomLeftY = bottomLeft.Y;
                }
                if (frameData.BottomRight is Point bottomRight)
                {
                    BottomRightX = bottomRight.X;
                    BottomRightY = bottomRight.Y;
                }

                StylusX = frameData.StylusX;
                StylusY = frameData.StylusY;
                StylusPressure = frameData.StylusPressure;
                StylusTiltX = frameData.StylusTiltX;
                StylusTiltY = frameData.StylusTiltY;

                // Decide what portion of the frame to show in the main viewport.
                // If a databar is present and we have a detected border height, show only the area below it.
                if (frameData.HasStylusData)
                {
                    // Create padded image: video centred with 300px bars left/right
                    using var paddedMat = AddHorizontalPadding(frameData.Image, HorizontalPadding);
                    var y = frameData.AuxImage?.Height ?? 0;

                    // Draw surface quad on the padded image
                    DrawSurfaceMask(paddedMat, HorizontalPadding, frameData);

                    var contentBmp = BitmapSourceConverter.ToBitmapSource(paddedMat);
                    contentBmp.Freeze();
                    CurrentFrameImage = contentBmp;
                }
                else
                {
                    // fallback: show full frame 
                    using var frameMat = frameData.Image.Clone();

                    var bmp = BitmapSourceConverter.ToBitmapSource(frameMat);
                    bmp.Freeze();
                    CurrentFrameImage = bmp;
                }
            }
            catch (Exception ex)
            {
                System.Diagnostics.Debug.WriteLine("LoadFrame error: " + ex);
            }
            finally
            {
                if (frameData != null)
                {
                    frameData.Image?.Dispose();
                    frameData.AuxImage?.Dispose();
                }
            }
        }

        /// <summary>
        /// Returns a new Mat with 'padX' pixels of padding on left and right,
        /// with 'src' copied into the centre.
        /// Caller is responsible for disposing the returned Mat.
        /// </summary>
        private static Mat AddHorizontalPadding(Mat src, int padX)
        {
            int newW = src.Width + 2 * padX;
            int newH = src.Height;

            // Create padded image (black background, same type as src)
            var padded = new Mat(newH, newW, src.Type(), Scalar.All(0));

            // Destination ROI where we copy the source frame
            using (var dstRoi = new Mat(padded, new Rect(padX, 0, src.Width, src.Height)))
            {
                src.CopyTo(dstRoi);
            }

            return padded;
        }

        private void GetH5Stats()
        {
            using var dlg = new System.Windows.Forms.FolderBrowserDialog
            {
                Description = "Select data folder",
                UseDescriptionForTitle = true,
                SelectedPath = Environment.CurrentDirectory,
                ShowNewFolderButton = true
            };

            if (dlg.ShowDialog() == System.Windows.Forms.DialogResult.OK)
            {                
                var dirStats = H5LabelStats.ScanDirectory(dlg.SelectedPath, recursive: false);
                
                // Totals
                Console.WriteLine($"Files: {dirStats.PerFile.Count}");
                Console.WriteLine($"Total frames: {dirStats.TotalFrames}");
                Console.WriteLine($"Frames with any labels: {dirStats.TotalFramesWithAnyLabels}");
                Console.WriteLine($"Frames with NO labels: {dirStats.TotalFramesWithNoLabels}");
                Console.WriteLine($"Labeled frames: pressure > 0: {dirStats.TotalLabeledPressureGt0}");
                Console.WriteLine($"Labeled frames: pressure == 0: {dirStats.TotalLabeledPressureEq0}");
                Console.WriteLine($"Labeled frames: pressure missing: {dirStats.TotalLabeledPressureMissing}");

                // Per file breakdown
                foreach (var f in dirStats.PerFile)
                {
                    Console.WriteLine($"{Path.GetFileName(f.Path)} : frames={f.FrameCount}, labeled={f.FramesWithAnyLabels}, unlabeled={f.FramesWithNoLabels}, p>0={f.LabeledFramesPressureGt0}, p=0={f.LabeledFramesPressureEq0}, p-missing={f.LabeledFramesPressureMissing}");
                }

            }
        }

        private static void MakeYellowPenMaskLab(Mat bgr,
                                                 Mat lab,
                                                 Mat labA,
                                                 Mat labB,
                                                 Mat bGeAMask,
                                                 Mat penMask)
        {
            const int PenBMin = 152; // derived from your samples

            // Convert to Lab
            Cv2.CvtColor(bgr, lab, ColorConversionCodes.BGR2Lab);
            Cv2.ExtractChannel(lab, labA, 1); // a
            Cv2.ExtractChannel(lab, labB, 2); // b

            // b >= 152
            Cv2.Compare(labB, PenBMin, penMask, CmpType.GE);

            // b >= a   (equivalent to (b - a) >= 0)
            Cv2.Compare(labB, labA, bGeAMask, CmpType.GE);

            // penMask = (b >= 152) & (b >= a)
            Cv2.BitwiseAnd(penMask, bGeAMask, penMask);
        }

        private Mat? _tubeMask, _tipCandMask, _ccLabels, _ccStats, _ccCentroids;
        private Mat? _h, _s, _v, _hueMask, _brightMask, _darkMask, _highSatMask, _darkHighSatMask, _valueCondMask;
        private static readonly Mat KernelGreen = Cv2.GetStructuringElement(MorphShapes.Ellipse, new Size(3, 3));
        private static readonly Mat KernelSkin = Cv2.GetStructuringElement(MorphShapes.Ellipse, new Size(1, 1));

        /// <summary>
        /// Compute the surface mask (area inside the page polygon) and the chroma-based
        /// colour mask used for background replacement. Optionally computes a pen mask
        /// when the yellow-background processing flag is set.
        ///
        /// This method reuses internal Mats for performance and MUST NOT modify the
        /// provided source image; it operates on copies/temporary buffers only.
        /// </summary>
        private bool ComputeSurfaceAndKeyMasks(Mat target,
                                               int padX,
                                               FrameData frameData,
                                               out Mat surfaceMask,
                                               out Mat colorMask,
                                               out Mat? penMaskLab)
        {
            penMaskLab = null;

            if (!ShowSurfaceMask)
            {
                EnsureSize(target, ref _surfaceMask, MatType.CV_8UC1);
                EnsureSize(target, ref _colorMask, MatType.CV_8UC1);
                _surfaceMask!.SetTo(Scalar.All(0));
                _colorMask!.SetTo(Scalar.All(0));
                surfaceMask = _surfaceMask!;
                colorMask = _colorMask!;
                return false;
            }

            if (frameData.TopLeft is null || frameData.TopRight is null ||
                frameData.BottomLeft is null || frameData.BottomRight is null)
            {
                EnsureSize(target, ref _surfaceMask, MatType.CV_8UC1);
                EnsureSize(target, ref _colorMask, MatType.CV_8UC1);
                _surfaceMask!.SetTo(Scalar.All(0));
                _colorMask!.SetTo(Scalar.All(0));
                surfaceMask = _surfaceMask!;
                colorMask = _colorMask!;
                return false;
            }

            EnsureSize(target, ref _hsv, MatType.CV_8UC3);
            EnsureSize(target, ref _ycrcb, MatType.CV_8UC3);

            EnsureSize(target, ref _surfaceMask, MatType.CV_8UC1);
            EnsureSize(target, ref _colorMask, MatType.CV_8UC1);
            EnsureSize(target, ref _tmpMask, MatType.CV_8UC1);
            EnsureSize(target, ref _skinMask, MatType.CV_8UC1);
            EnsureSize(target, ref _notSkinMask, MatType.CV_8UC1);
            EnsureSize(target, ref _pureGreenMask, MatType.CV_8UC1);

            EnsureSize(target, ref _h, MatType.CV_8UC1);
            EnsureSize(target, ref _s, MatType.CV_8UC1);
            EnsureSize(target, ref _v, MatType.CV_8UC1);
            EnsureSize(target, ref _hueMask, MatType.CV_8UC1);
            EnsureSize(target, ref _brightMask, MatType.CV_8UC1);
            EnsureSize(target, ref _darkMask, MatType.CV_8UC1);
            EnsureSize(target, ref _highSatMask, MatType.CV_8UC1);
            EnsureSize(target, ref _darkHighSatMask, MatType.CV_8UC1);
            EnsureSize(target, ref _valueCondMask, MatType.CV_8UC1);

            EnsureSize(target, ref _tubeMask, MatType.CV_8UC1);
            EnsureSize(target, ref _tipCandMask, MatType.CV_8UC1);
            EnsureSize(target, ref _ccLabels, MatType.CV_32SC1);
            EnsureSize(target, ref _ccStats, MatType.CV_32SC1);
            EnsureSize(target, ref _ccCentroids, MatType.CV_64FC1);

            EnsureSize(target, ref _lab, MatType.CV_8UC3);
            EnsureSize(target, ref _labA, MatType.CV_8UC1);
            EnsureSize(target, ref _labB, MatType.CV_8UC1);
            EnsureSize(target, ref _penMaskLab, MatType.CV_8UC1);
            EnsureSize(target, ref _bGeAMask, MatType.CV_8UC1);

            EnsureSize(target, ref _bgBgr, MatType.CV_8UC3);
            EnsureSize(target, ref _bg16, MatType.CV_16SC3);
            EnsureSize(target, ref _bgGrad16, MatType.CV_16SC3);
            EnsureSize(target, ref _noise8s, MatType.CV_8SC3);
            EnsureSize(target, ref _noise16s, MatType.CV_16SC3);
            EnsureSize(target, ref _col16, MatType.CV_16SC1);
            EnsureSize(target, ref _grad1_16, MatType.CV_16SC1);

            // --- polygon + clip ---
            var bl = Point2f.FromPoint(frameData.BottomLeft.Value) + new Point2f(padX, 0);
            var br = Point2f.FromPoint(frameData.BottomRight.Value) + new Point2f(padX, 0);
            var tr = Point2f.FromPoint(frameData.TopRight.Value) + new Point2f(padX, 0);
            var tl = Point2f.FromPoint(frameData.TopLeft.Value) + new Point2f(padX, 0);

            var poly = new List<Point2f>(4) { bl, br, tr, tl };
            var clipRect = new Rect2f(0, 0, target.Width, target.Height);
            var clipped = PolygonClipper.ClipToRect(poly, clipRect);
            if (clipped.Count < 3)
            {
                EnsureSize(target, ref _surfaceMask, MatType.CV_8UC1);
                EnsureSize(target, ref _colorMask, MatType.CV_8UC1);
                _surfaceMask!.SetTo(Scalar.All(0));
                _colorMask!.SetTo(Scalar.All(0));
                surfaceMask = _surfaceMask!;
                colorMask = _colorMask!;
                return false;
            }

            var pts = new Point[clipped.Count];
            for (int i = 0; i < clipped.Count; i++)
            {
                var p = clipped[i];
                pts[i] = new Point((int)Math.Round(p.X), (int)Math.Round(p.Y));
            }

            EnsureSurfaceMask(target, pts);

            // --- HSV chroma detection (your existing green-screen style mask) ---
            Cv2.CvtColor(target, _hsv!, ColorConversionCodes.BGR2HSV);

            Cv2.ExtractChannel(_hsv!, _h!, 0);
            Cv2.ExtractChannel(_hsv!, _s!, 1);
            Cv2.ExtractChannel(_hsv!, _v!, 2);

            // hueMask = inrange(H)
            Cv2.InRange(_h!, new Scalar(HueMin), new Scalar(HueMax), _hueMask!);

            // brightMask = V > 40
            Cv2.Threshold(_v!, _brightMask!, ValueThreshold, 255, ThresholdTypes.Binary);

            // darkMask = V <= 40
            Cv2.Threshold(_v!, _darkMask!, ValueThreshold, 255, ThresholdTypes.BinaryInv);

            // highSatMask = S > 27
            Cv2.Threshold(_s!, _highSatMask!, SaturationThreshold, 255, ThresholdTypes.Binary);

            // darkHighSatMask = darkMask & highSatMask
            Cv2.BitwiseAnd(_darkMask!, _highSatMask!, _darkHighSatMask!);

            // valueCondMask = brightMask | darkHighSatMask
            Cv2.BitwiseOr(_brightMask!, _darkHighSatMask!, _valueCondMask!);

            // colorMask = hueMask & valueCondMask
            Cv2.BitwiseAnd(_hueMask!, _valueCondMask!, _colorMask!);

            // restrict to polygon / surface
            Cv2.BitwiseAnd(_colorMask!, _surfaceMask!, _colorMask!);

            // Optional pen mask
            if (HasYellowBackground)
            {
                MakeYellowPenMaskLab(target, _lab!, _labA!, _labB!, _bGeAMask!, _penMaskLab!);

                // Restrict pen detection to the surface polygon first
                Cv2.BitwiseAnd(_penMaskLab!, _surfaceMask!, _penMaskLab!);

                // Build "available area" mask: inside surface AND NOT already in background colorMask
                // (_colorMask is already inside _surfaceMask, but ANDing with _surfaceMask is harmless + explicit)
                Cv2.BitwiseNot(_colorMask!, _tmpMask!);
                Cv2.BitwiseAnd(_tmpMask!, _surfaceMask!, _tmpMask!);

                // Keep yellow pen only where background mask hasn't been placed
                Cv2.BitwiseAnd(_penMaskLab!, _tmpMask!, _penMaskLab!);

                Cv2.MedianBlur(_penMaskLab!, _penMaskLab!, 3);
                penMaskLab = _penMaskLab!;
            }


            surfaceMask = _surfaceMask!;
            colorMask = _colorMask!;
            return true;
        }

        private void ApplyBackgroundVariantInPlace(Mat targetBgr,
                                                   Mat colorMask,
                                                   Scalar backgroundScalar,
                                                   int frameIndex)
       {
            EnsureSize(targetBgr, ref _bgBgr, MatType.CV_8UC3);
            EnsureSize(targetBgr, ref _bgNoise, MatType.CV_8SC3);

            BuildBackgroundFast(_bgBgr!, backgroundScalar, noiseAmp: 2, gradAmp: 6, frameIndex: frameIndex);
            _bgBgr!.CopyTo(targetBgr, colorMask);

            Cv2.InRange(targetBgr, new Scalar(0, 250, 0), new Scalar(15, 255, 25), _pureGreenMask!);
            Cv2.Dilate(_pureGreenMask!, _pureGreenMask!, KernelGreen, iterations: 1);
            targetBgr.SetTo(new Scalar(40, 40, 40), _pureGreenMask!);
        }

        private void ApplyBackgroundVariant(Mat dstBgr,
                                            Mat srcBgr,
                                            Mat colorMask,
                                            Scalar backgroundScalar,
                                            int frameIndex)
        {
            // Start from original frame
            srcBgr.CopyTo(dstBgr);

            // Build background once per variant (depends on scalar)
            EnsureSize(srcBgr, ref _bgBgr, MatType.CV_8UC3);
            EnsureSize(srcBgr, ref _bgNoise, MatType.CV_8SC3);

            BuildBackgroundFast(_bgBgr!, backgroundScalar, noiseAmp: 2, gradAmp: 6, frameIndex: frameIndex);

            // Composite background into masked pixels
            _bgBgr!.CopyTo(dstBgr, colorMask);

            // Optional cleanup pass (kept identical to your current behaviour)
            Cv2.InRange(dstBgr, new Scalar(0, 250, 0), new Scalar(15, 255, 25), _pureGreenMask!);
            Cv2.Dilate(_pureGreenMask!, _pureGreenMask!, KernelGreen, iterations: 1);
            dstBgr.SetTo(new Scalar(40, 40, 40), _pureGreenMask!);
        }

        // --- allocate / resize scratch Mats once ---
        static void EnsureSize(Mat target, ref Mat? m, MatType type)
        {
            if (m == null || m.Width != target.Width || m.Height != target.Height || m.Type() != type)
            {
                m?.Dispose();
                m = new Mat(target.Rows, target.Cols, type);
            }
        }

        private void DrawSurfaceMask(Mat target, int padX, FrameData frameData)
        {
            if (!ComputeSurfaceAndKeyMasks(target, padX, frameData, out var surfaceMask, out var colorMask, out var penMask))
                return;

            ApplyBackgroundVariantInPlace(target, colorMask, _backgroundScalar, frameData.FrameIndex);
            ApplyPenOverlay(target, penMask);
        }

        /// <summary>
        /// Desaturate and darken only where penMaskLab is non-zero.
        /// </summary>
        private void ApplyPenOverlay(Mat dstBgr, Mat? penMaskLab, double vScale = 0.85)
        {
            if (penMaskLab == null || penMaskLab.Empty())
                return;

            EnsureSize(dstBgr, ref _hsvOverlay, MatType.CV_8UC3);
            EnsureSize(dstBgr, ref _hCh, MatType.CV_8UC1);
            EnsureSize(dstBgr, ref _sCh, MatType.CV_8UC1);
            EnsureSize(dstBgr, ref _vCh, MatType.CV_8UC1);

            // Convert to HSV
            Cv2.CvtColor(dstBgr, _hsvOverlay!, ColorConversionCodes.BGR2HSV);

            // Split
            Cv2.ExtractChannel(_hsvOverlay!, _hCh!, 0);
            Cv2.ExtractChannel(_hsvOverlay!, _sCh!, 1);
            Cv2.ExtractChannel(_hsvOverlay!, _vCh!, 2);

            // S = 0 inside mask (desaturate)
            _sCh!.SetTo(Scalar.All(0), penMaskLab);

            // V = V * vScale inside mask (darken)
            // Multiply uses saturation arithmetic for 8U, which is fine here.
            Cv2.Multiply(_vCh!, new Scalar(vScale), _darkMask!);     // temp = V * scale
            _darkMask!.CopyTo(_vCh!, penMaskLab);                   // write back only in mask

            // Merge back
            Cv2.InsertChannel(_hCh!, _hsvOverlay!, 0);
            Cv2.InsertChannel(_sCh!, _hsvOverlay!, 1);
            Cv2.InsertChannel(_vCh!, _hsvOverlay!, 2);

            // Back to BGR
            Cv2.CvtColor(_hsvOverlay!, dstBgr, ColorConversionCodes.HSV2BGR);
        }

        private void EnsureNoiseBank(int rows, int cols, int noiseAmp)
        {
            if (_noise16Bank != null &&
                _noiseBankRows == rows &&
                _noiseBankCols == cols &&
                _noiseBankAmp == noiseAmp)
                return;

            if (_noise16Bank != null)
            {
                foreach (var m in _noise16Bank)
                    m?.Dispose();
            }

            _noise16Bank = new Mat[NoiseBankSize];

            using var tmp8s = new Mat(rows, cols, MatType.CV_8SC3);
            for (int i = 0; i < NoiseBankSize; i++)
            {
                Cv2.Randu(tmp8s,
                    new Scalar(-noiseAmp, -noiseAmp, -noiseAmp),
                    new Scalar(noiseAmp + 1, noiseAmp + 1, noiseAmp + 1));

                var n16 = new Mat(rows, cols, MatType.CV_16SC3);
                tmp8s.ConvertTo(n16, MatType.CV_16SC3);
                _noise16Bank[i] = n16;
            }

            _noiseBankRows = rows;
            _noiseBankCols = cols;
            _noiseBankAmp = noiseAmp;
        }

        private static bool ScalarEquals(Scalar a, Scalar b) => a.Val0 == b.Val0 && a.Val1 == b.Val1 && a.Val2 == b.Val2 && a.Val3 == b.Val3;

        private void EnsureBaseGrad(int rows, int cols, Scalar baseBgr, int gradAmp)
        {
            if (_bgBaseGrad16 != null &&
                _bgRowsCached == rows &&
                _bgColsCached == cols &&
                _bgGradAmpCached == gradAmp &&
                ScalarEquals(_bgBaseCached, baseBgr))
                return;

            _bgBaseGrad16?.Dispose();
            _bgBaseGrad16 = new Mat(rows, cols, MatType.CV_16SC3);

            // Prepare a constant base image and add a cached vertical gradient.
            _bgBaseGrad16.SetTo(baseBgr);

            EnsureGradient(rows, cols, gradAmp);

            Cv2.Add(_bgBaseGrad16, _bgGrad16!, _bgBaseGrad16);

            _bgRowsCached = rows;
            _bgColsCached = cols;
            _bgGradAmpCached = gradAmp;
            _bgBaseCached = baseBgr;
        }

        private void BuildBackgroundFast(Mat bgBgr8u, Scalar baseBgr, int noiseAmp, int gradAmp, int frameIndex)
        {
            int rows = bgBgr8u.Rows;
            int cols = bgBgr8u.Cols;

            if (_bg16 == null || _bg16.Rows != rows || _bg16.Cols != cols || _bg16.Type() != MatType.CV_16SC3)
            {
                _bg16?.Dispose();
                _bg16 = new Mat(rows, cols, MatType.CV_16SC3);
            }

            EnsureBaseGrad(rows, cols, baseBgr, gradAmp);

            _bgBaseGrad16!.CopyTo(_bg16);

            EnsureNoiseBank(rows, cols, noiseAmp);

            var noise16 = _noise16Bank![frameIndex % NoiseBankSize];
            Cv2.Add(_bg16!, noise16, _bg16!);

            _bg16!.ConvertTo(bgBgr8u, MatType.CV_8UC3);
        }

        private int _bgGradRowsCached = -1;
        private int _bgGradColsCached = -1;

        private void EnsureGradient(int rows, int cols, int gradAmp)
        {
            if (_bgGrad16 != null &&
                _bgGradRowsCached == rows &&
                _bgGradColsCached == cols &&
                _bgGradAmpCached == gradAmp)
                return;

            _bgGrad16?.Dispose();
            _bgGrad16 = new Mat(rows, cols, MatType.CV_16SC3);

            if (_col16 == null || _col16.Rows != rows || _col16.Cols != 1 || _col16.Type() != MatType.CV_16SC1)
            {
                _col16?.Dispose();
                _col16 = new Mat(rows, 1, MatType.CV_16SC1);
            }

            for (int y = 0; y < rows; y++)
            {
                double t = (rows <= 1) ? 0.0 : (double)y / (rows - 1);
                short offset = (short)Math.Round((t - 0.5) * gradAmp);
                _col16.Set(y, 0, offset);
            }

            if (_grad1_16 == null || _grad1_16.Rows != rows || _grad1_16.Cols != cols || _grad1_16.Type() != MatType.CV_16SC1)
            {
                _grad1_16?.Dispose();
                _grad1_16 = new Mat(rows, cols, MatType.CV_16SC1);
            }

            Cv2.Repeat(_col16, 1, cols, _grad1_16);

            Cv2.Merge([_grad1_16, _grad1_16, _grad1_16], _bgGrad16);

            _bgGradRowsCached = rows;
            _bgGradColsCached = cols;
            _bgGradAmpCached = gradAmp;
        }

        private void EnsureSurfaceMask(Mat paddedMat, Point[] surfacePoly)
        {
            // Rebuild if size changed or polygon changed
            bool needNew =
                _surfaceMask == null ||
                _surfaceMask.Width != paddedMat.Width ||
                _surfaceMask.Height != paddedMat.Height ||
                _lastSurfacePoly == null ||
                _lastSurfacePoly.Length != surfacePoly.Length;

            if (!needNew)
            {
                for (int i = 0; i < surfacePoly.Length; i++)
                {
                    if (_lastSurfacePoly![i] != surfacePoly[i])
                    {
                        needNew = true;
                        break;
                    }
                }
            }

            if (!needNew)
                return;

            _surfaceMask?.Dispose();
            _surfaceMask = new Mat(paddedMat.Rows, paddedMat.Cols, MatType.CV_8UC1, Scalar.All(0));

            // Fill polygon with 255 inside
            Cv2.FillConvexPoly(_surfaceMask, surfacePoly, new Scalar(255));

            _lastSurfacePoly = (Point[])surfacePoly.Clone();
        }

        /// <summary>
        /// Dispose the ViewModel and release all unmanaged resources.
        /// </summary>
        public void Dispose()
        {
            if (_disposed) return;
            _disposed = true;

            // 1) Stop playback
            var cts = _playCts;
            _playCts = null;
            if (cts != null)
            {
                try { cts.Cancel(); }
                catch { }
                finally { cts.Dispose(); }
            }

            // 2) Unsubscribe from event
            _videoService.FrameCountChanged -= VideoService_FrameCountChanged;

            // 3) Dispose frame source
            if (_frameSource is IDisposable fs)
            {
                fs.Dispose();
            }
            _frameSource = null;

            // 4) Dispose cached Mats
            _surfaceMask?.Dispose();
            _surfaceMask = null;

            _hsv?.Dispose();
            _hsv = null;

            _colorMask?.Dispose();
            _colorMask = null;

            _tmpMask?.Dispose();
            _tmpMask = null;

            _ycrcb?.Dispose();
            _ycrcb = null;

            _skinMask?.Dispose();
            _skinMask = null;

            _notSkinMask?.Dispose();
            _notSkinMask = null;

            _lastSurfacePoly = null;

            // 5) Dispose video service (if this VM owns it)
            if (_videoService is IDisposable d)
            {
                d.Dispose();
            }

            GC.SuppressFinalize(this);
        }
    }
}
