using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using InkMARC.Clean.Model;
using InkMARC.Clean.Services;
using InkMARC.Clean.Services.Interfaces;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Microsoft.Win32;
using OpenCvSharp;
using OpenCvSharp.WpfExtensions;
using System.Globalization;
using System.IO;
using System.Text.RegularExpressions;
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
            ExportFilledInkImageCommand = new RelayCommand(ExportFilledInkImage);
            ExportFilledInkImageWithMetricLabelsCommand = new RelayCommand(ExportFilledInkImageWithMetricLabels);
            CombineAnalysisResultsCommand = new RelayCommand(CombineAnalysisResults);
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
            _exportCts?.Cancel();
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

        //private HybridStylusInference GetOrCreateOnnx()
        private OnnxStylusInference GetOrCreateOnnx()
        {
            lock (_onnxLock)
            {
                if (_onnx != null) return _onnx;
                // Prefer model under resources/models/model.onnx relative to the app base directory
                var baseDir = AppDomain.CurrentDomain.BaseDirectory;
                var phase2bModelPath = Path.Combine(baseDir, "Resources", "Models", "phase2a_epoch1_int8_compat.onnx");
                var phase4bModelPath = Path.Combine(baseDir, "Resources", "Models", "phase2e_epoch1_fp32.onnx");

                // Fallback to legacy location next to exe
                if (!File.Exists(phase2bModelPath))
                {
                    var fallback = Path.Combine(baseDir, "model.onnx");
                    if (File.Exists(fallback))
                        phase2bModelPath = fallback;
                    else
                        throw new FileNotFoundException($"ONNX model not found. Expected at '{phase2bModelPath}' or '{fallback}'.");
                }

                // Fallback to legacy location next to exe
                if (!File.Exists(phase4bModelPath))
                {
                    var fallback = Path.Combine(baseDir, "phase2e_epoch1_fp32.onnx");
                    if (File.Exists(fallback))
                        phase4bModelPath = fallback;
                    else
                        throw new FileNotFoundException($"ONNX model not found. Expected at '{phase2bModelPath}' or '{fallback}'.");
                }

                _onnx = new OnnxStylusInference(phase2bModelPath, StylusModelKind.Phase2b);
                return _onnx;
            }
        }

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

        // Stylus coordinate system (surface size) — used to map normalized model outputs
        private const float STYLUS_MAX_X = 44800f;
        private const float STYLUS_MAX_Y = 29600f;
        private const int DefaultTaskExportWidth = 728;
        private const int DefaultTaskExportHeight = 481;
        private static readonly Regex TaskTokenRegex = new(@"(?i)(?<![a-z0-9])(T\d+[a-z]?)(?![a-z0-9])", RegexOptions.Compiled);
        private static readonly Regex ParticipantRegex = new(@"(?i)(?:^|_)participant[_-]?([^_]+)", RegexOptions.Compiled);
        private static readonly Regex SessionRegex = new(@"(?i)(?:^|_)session[_-]?([^_]+)", RegexOptions.Compiled);
        private static readonly Regex TrialRegex = new(@"(?i)(?:^|_)trial[_-]?([^_]+)", RegexOptions.Compiled);

        [ObservableProperty]
        private string backgroundColor = "White";
        [ObservableProperty]
        private string foregroundColor = "Black";

        [ObservableProperty]
        private bool hasYellowBackground = false;

        [ObservableProperty]
        private bool inference;

        [ObservableProperty]
        private float? predictedXNorm;   // 0..1

        [ObservableProperty]
        private float? predictedYNorm;   // 0..1

        [ObservableProperty]
        private int? predictedXPixel;    // in model input pixels (e.g., 480 wide)

        [ObservableProperty]
        private int? predictedYPixel;    // in model input pixels (e.g., 320 high)

        private Scalar _backgroundScalar = new(255, 255, 255);

        /// <summary>
        /// When true, the computed surface mask will be shown/used in rendering.
        /// </summary>
        public bool ShowSurfaceMask { get; set; }

        /// <summary>
        /// Raw text recognized from the frame's auxiliary strip (if any).
        /// </summary>
        public string? CurrentRawOcr { get; set; }

        private OnnxStylusInference? _onnx;
        private readonly object _onnxLock = new();

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
        /// <summary>Exports a flattened task image with replayed ink.</summary>
        public ICommand ExportFilledInkImageCommand { get; }
        /// <summary>Exports a flattened task image and labelled T1 diagnostic analysis.</summary>
        public ICommand ExportFilledInkImageWithMetricLabelsCommand { get; }
        /// <summary>Combines task analysis result CSV files from a directory into one tidy CSV.</summary>
        public ICommand CombineAnalysisResultsCommand { get; }
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

        private void CombineAnalysisResults()
        {
            var inputDir = PickAnalysisResultsDirectory();
            if (string.IsNullOrWhiteSpace(inputDir))
                return;

            var saveDlg = new SaveFileDialog
            {
                Title = "Save combined analysis CSV",
                Filter = "CSV file|*.csv|All files|*.*",
                AddExtension = true,
                DefaultExt = ".csv",
                InitialDirectory = inputDir,
                FileName = $"combined_task_analysis_{DateTime.Now:yyyyMMdd_HHmmss}.csv"
            };

            if (saveDlg.ShowDialog() != true)
                return;

            _exportCts?.Cancel();
            _exportCts?.Dispose();
            _exportCts = new CancellationTokenSource();
            var ct = _exportCts.Token;

            IsExporting = true;
            ExportProgress = 0;
            ExportStatus = "Combining analysis results...";

            Task.Run(() =>
            {
                try
                {
                    int rowsWritten = CombineAnalysisResultsCore(
                        inputDir,
                        saveDlg.FileName,
                        ct,
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
                        ExportStatus = $"Combined {rowsWritten} metric row(s).";
                        System.Windows.MessageBox.Show($"Combined {rowsWritten} metric row(s) into:\n{saveDlg.FileName}");
                    }));
                }
                catch (OperationCanceledException)
                {
                    System.Windows.Application.Current.Dispatcher.BeginInvoke(new Action(() =>
                    {
                        ExportStatus = "Combine results cancelled.";
                    }));
                }
                catch (Exception ex)
                {
                    System.Windows.Application.Current.Dispatcher.BeginInvoke(new Action(() =>
                    {
                        ExportStatus = "Combine results failed: " + ex.Message;
                        System.Windows.MessageBox.Show("Combine results failed: " + ex.Message);
                    }));
                }
                finally
                {
                    System.Windows.Application.Current.Dispatcher.BeginInvoke(new Action(() =>
                    {
                        IsExporting = false;
                    }));
                }
            }, ct);
        }

        private static string? PickAnalysisResultsDirectory()
        {
            using var dlg = new System.Windows.Forms.FolderBrowserDialog
            {
                Description = "Select a folder containing analysis result CSV files",
                UseDescriptionForTitle = true,
                SelectedPath = Environment.CurrentDirectory,
                ShowNewFolderButton = false
            };

            return dlg.ShowDialog() == System.Windows.Forms.DialogResult.OK
                ? dlg.SelectedPath
                : null;
        }

        private static int CombineAnalysisResultsCore(string inputDirectory,
                                                      string outputPath,
                                                      CancellationToken ct,
                                                      Action<double, string>? progress = null)
        {
            if (!Directory.Exists(inputDirectory))
                throw new DirectoryNotFoundException(inputDirectory);

            string resolvedOutputPath = Path.GetFullPath(outputPath);
            var csvFiles = Directory
                .EnumerateFiles(inputDirectory, "*.csv", SearchOption.AllDirectories)
                .Where(path => !Path.GetFullPath(path).Equals(resolvedOutputPath, StringComparison.OrdinalIgnoreCase))
                .Where(path => !Path.GetFileName(path).StartsWith("combined_task_analysis_", StringComparison.OrdinalIgnoreCase))
                .OrderBy(path => path, StringComparer.OrdinalIgnoreCase)
                .ToList();

            if (csvFiles.Count == 0)
                throw new InvalidOperationException("No CSV result files were found in the selected directory.");

            Directory.CreateDirectory(Path.GetDirectoryName(resolvedOutputPath) ?? inputDirectory);

            int rowsWritten = 0;
            using var writer = new StreamWriter(resolvedOutputPath);
            writer.WriteLine("source_csv,result_kind,source_row,participant_id,session_id,trial_id,task_id,metric,value");

            for (int fileIndex = 0; fileIndex < csvFiles.Count; fileIndex++)
            {
                ct.ThrowIfCancellationRequested();

                string csvPath = csvFiles[fileIndex];
                string resultKind = GuessAnalysisResultKind(csvPath);
                progress?.Invoke(fileIndex / (double)Math.Max(1, csvFiles.Count), $"Combining {Path.GetFileName(csvPath)}...");

                using var reader = new StreamReader(csvPath);
                string? headerLine = reader.ReadLine();
                if (string.IsNullOrWhiteSpace(headerLine))
                    continue;

                var headers = SplitCsvLine(headerLine);
                if (headers.Count == 0)
                    continue;

                var headerLookup = headers
                    .Select((header, index) => new { Header = header.Trim(), Index = index })
                    .GroupBy(item => item.Header, StringComparer.OrdinalIgnoreCase)
                    .ToDictionary(group => group.Key, group => group.First().Index, StringComparer.OrdinalIgnoreCase);

                int sourceRow = 0;
                string? line;
                while ((line = reader.ReadLine()) != null)
                {
                    ct.ThrowIfCancellationRequested();
                    sourceRow++;
                    if (string.IsNullOrWhiteSpace(line))
                        continue;

                    var values = SplitCsvLine(line);
                    string participantId = GetCsvValue(values, headerLookup, "participant_id");
                    string sessionId = GetCsvValue(values, headerLookup, "session_id");
                    string trialId = GetCsvValue(values, headerLookup, "trial_id");
                    string taskId = GetCsvValue(values, headerLookup, "task_id");

                    for (int columnIndex = 0; columnIndex < headers.Count; columnIndex++)
                    {
                        string metric = headers[columnIndex].Trim();
                        if (IsCombinedMetadataColumn(metric))
                            continue;

                        string value = columnIndex < values.Count ? values[columnIndex] : string.Empty;
                        if (string.IsNullOrWhiteSpace(value))
                            continue;

                        writer.WriteLine(string.Join(",", new[]
                        {
                            Csv(Path.GetRelativePath(inputDirectory, csvPath)),
                            Csv(resultKind),
                            Csv(sourceRow),
                            Csv(participantId),
                            Csv(sessionId),
                            Csv(trialId),
                            Csv(taskId),
                            Csv(metric),
                            Csv(value)
                        }));
                        rowsWritten++;
                    }
                }
            }

            progress?.Invoke(1.0, "Combined analysis results complete.");
            return rowsWritten;
        }

        private static bool IsCombinedMetadataColumn(string columnName)
        {
            return columnName.Equals("participant_id", StringComparison.OrdinalIgnoreCase)
                || columnName.Equals("session_id", StringComparison.OrdinalIgnoreCase)
                || columnName.Equals("trial_id", StringComparison.OrdinalIgnoreCase)
                || columnName.Equals("task_id", StringComparison.OrdinalIgnoreCase);
        }

        private static string GetCsvValue(IReadOnlyList<string> values, IReadOnlyDictionary<string, int> headerLookup, string columnName)
        {
            return headerLookup.TryGetValue(columnName, out int index) && index >= 0 && index < values.Count
                ? values[index]
                : string.Empty;
        }

        private static string GuessAnalysisResultKind(string csvPath)
        {
            string name = Path.GetFileNameWithoutExtension(csvPath);
            var markers = new[]
            {
                "_task_timing_summary",
                "_task_strokes",
                "_t1a_trials",
                "_t1a_summary",
                "_t1b_strokes",
                "_t1b_summary",
                "_t2_lines",
                "_t2_summary",
                "_t3a_paths",
                "_t3a_summary",
                "_t3b_shapes",
                "_t3b_summary"
            };

            foreach (var marker in markers)
            {
                if (name.EndsWith(marker, StringComparison.OrdinalIgnoreCase))
                    return marker.TrimStart('_');
            }

            int markerIndex = name.LastIndexOf('_');
            return markerIndex >= 0 && markerIndex + 1 < name.Length
                ? name[(markerIndex + 1)..]
                : name;
        }

        private static List<string> SplitCsvLine(string line)
        {
            var values = new List<string>();
            var current = new System.Text.StringBuilder();
            bool inQuotes = false;

            for (int i = 0; i < line.Length; i++)
            {
                char c = line[i];
                if (c == '"')
                {
                    if (inQuotes && i + 1 < line.Length && line[i + 1] == '"')
                    {
                        current.Append('"');
                        i++;
                    }
                    else
                    {
                        inQuotes = !inQuotes;
                    }
                }
                else if (c == ',' && !inQuotes)
                {
                    values.Add(current.ToString());
                    current.Clear();
                }
                else
                {
                    current.Append(c);
                }
            }

            values.Add(current.ToString());
            return values;
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
                    var skippedFiles = new List<string>();

                    for (int fileIdx = 0; fileIdx < files.Length; fileIdx++)
                    {
                        ct.ThrowIfCancellationRequested();

                        var inputPath = files[fileIdx];
                        var baseName = Path.GetFileNameWithoutExtension(inputPath);

                        try
                        {
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
                                exportColorVariants: false,
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
                        catch (OperationCanceledException)
                        {
                            throw;
                        }
                        catch (Exception ex)
                        {
                            skippedFiles.Add(baseName);
                            System.Diagnostics.Debug.WriteLine($"Skipping batch export for '{inputPath}': {ex}");

                            double overall01 = (fileIdx + 1) / Math.Max(1.0, files.Length);
                            dispatcher.BeginInvoke(new Action(() =>
                            {
                                ExportProgress = overall01 * 100.0;
                                ExportStatus = $"[{fileIdx + 1}/{files.Length}] Skipped {baseName}: {ex.Message}";
                            }));
                        }
                    }

                    System.Windows.Application.Current.Dispatcher.BeginInvoke(new Action(() =>
                    {
                        ExportProgress = 100;
                        ExportStatus = skippedFiles.Count == 0
                            ? "Batch export complete."
                            : $"Batch export complete with {skippedFiles.Count} skipped file(s).";
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
                                                    bool exportColorVariants = true,
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

            // 4) Output paths: ONE H5, optionally many colour AVIs, always one overlayless AVI
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

            int colourVariantCount = exportColorVariants ? colours.Length : 0;
            int overlaylessWriterIndex = colourVariantCount;

            // Open all AVI writers once
            var writers = new OpenCvSharp.VideoWriter[colourVariantCount + 1];

            // Preallocate per-variant output frames to avoid Clone() allocations
            var outFrames = new Mat[colourVariantCount];
            for (int i = 0; i < outFrames.Length; i++)
                outFrames[i] = new Mat(exportH, exportW, MatType.CV_8UC3);

            try
            {
                for (int pass = 0; pass < colourVariantCount; pass++)
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

                // Write non-masked                
                writers[overlaylessWriterIndex] = new OpenCvSharp.VideoWriter(
                    Path.Combine(baseDir, $"{baseName}.avi"),
                    OpenCvSharp.FourCC.MJPG,
                    fps,
                    new OpenCvSharp.Size(exportW, exportH),
                    isColor: true);

                if (!writers[overlaylessWriterIndex].IsOpened())
                    throw new InvalidOperationException($"Failed to open AVI writer: {Path.Combine(baseDir, $"{baseName}.avi")}");

                long totalWork = Math.Max(1, frameCount);
                const int reportEveryNFrames = 200;

                if (exportColorVariants)
                {
                    dispatcher.BeginInvoke(new Action(() =>
                    {
                        BackgroundColor = colours[0];
                        ForegroundColor = _palette.GetForeground(colours[0]);
                    }));
                }

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

                        if (exportColorVariants)
                        {
                            ComputeSurfaceAndKeyMasks(
                                cropped,
                                padX: 0,
                                frameData: frameData,
                                out var surfaceMask,
                                out var colorMask,
                                out var penMaskLab);

                            // ---- Apply each colour variant cheaply ----
                            for (int pass = 0; pass < colourVariantCount; pass++)
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

                        // ---- Save one non-masked variant ----
                        writers[overlaylessWriterIndex].Write(cropped);
                    }
                    finally
                    {
                        frameData?.Image?.Dispose();
                        frameData?.AuxImage?.Dispose();
                    }

                    if (frameIndex == 0 || frameIndex == frameCount - 1 || (frameIndex % reportEveryNFrames == 0))
                    {
                        long done = frameIndex + 1;
                        double pct = (totalWork <= 0) ? 1.0 : (done / (double)totalWork);
                        string status = exportColorVariants
                            ? $"Exporting {baseName}: frame {frameIndex + 1}/{frameCount} ({colourVariantCount} variants + overlayless)"
                            : $"Exporting {baseName}: frame {frameIndex + 1}/{frameCount} (overlayless + H5)";
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

        private void ExportFilledInkImage()
        {
            ExportFilledInkImage(includeMetricLabels: false);
        }

        private void ExportFilledInkImageWithMetricLabels()
        {
            ExportFilledInkImagesWithMetricLabelsBatch();
        }

        private void ExportFilledInkImagesWithMetricLabelsBatch()
        {
            if (_playCts != null)
            {
                var cts = _playCts;
                _playCts = null;
                try { cts.Cancel(); }
                catch { }
                finally { cts.Dispose(); }
            }

            var openDlg = new OpenFileDialog
            {
                Title = "Select videos to analyse",
                Multiselect = true,
                Filter = "Video/session files|*.mp4;*.mov;*.avi;*.mkv;*.wmv;*.m4v;*.h5;*.hdf5|All files|*.*"
            };

            if (openDlg.ShowDialog() != true || openDlg.FileNames.Length == 0)
                return;

            var initialDir = Path.GetDirectoryName(openDlg.FileNames[0]);
            var outDir = PickOutputDirectory(initialDir);
            if (string.IsNullOrWhiteSpace(outDir))
                return;

            _exportCts?.Cancel();
            _exportCts?.Dispose();
            _exportCts = new CancellationTokenSource();
            var ct = _exportCts.Token;

            var files = openDlg.FileNames;

            IsExporting = true;
            ExportProgress = 0;
            ExportStatus = "Starting task analysis batch export...";

            Task.Run(() =>
            {
                try
                {
                    var dispatcher = System.Windows.Application.Current.Dispatcher;
                    var skippedFiles = new List<string>();

                    for (int fileIdx = 0; fileIdx < files.Length; fileIdx++)
                    {
                        ct.ThrowIfCancellationRequested();

                        var inputPath = files[fileIdx];
                        var baseName = Path.GetFileNameWithoutExtension(inputPath);

                        try
                        {
                            string sessionPath = ResolveSessionPath(inputPath);
                            string sessionBaseName = Path.GetFileNameWithoutExtension(sessionPath);
                            string outputPath = GetUniqueOutputPath(outDir!, $"{sessionBaseName}_filled.png");

                            ExportFilledInkImageCore(
                                sessionPath,
                                outputPath,
                                includeMetricLabels: true,
                                ct,
                                progress: (pct01WithinFile, status) =>
                                {
                                    double overall01 = (fileIdx + pct01WithinFile) / Math.Max(1.0, files.Length);

                                    dispatcher.BeginInvoke(new Action(() =>
                                    {
                                        ExportProgress = overall01 * 100.0;
                                        ExportStatus = $"[{fileIdx + 1}/{files.Length}] {baseName}: {status}";
                                    }));
                                });
                        }
                        catch (OperationCanceledException)
                        {
                            throw;
                        }
                        catch (Exception ex)
                        {
                            skippedFiles.Add(baseName);
                            System.Diagnostics.Debug.WriteLine($"Skipping task analysis export for '{inputPath}': {ex}");

                            double overall01 = (fileIdx + 1) / Math.Max(1.0, files.Length);
                            dispatcher.BeginInvoke(new Action(() =>
                            {
                                ExportProgress = overall01 * 100.0;
                                ExportStatus = $"[{fileIdx + 1}/{files.Length}] Skipped {baseName}: {ex.Message}";
                            }));
                        }
                    }

                    System.Windows.Application.Current.Dispatcher.BeginInvoke(new Action(() =>
                    {
                        ExportProgress = 100;
                        ExportStatus = skippedFiles.Count == 0
                            ? "Task analysis batch export complete."
                            : $"Task analysis batch export complete with {skippedFiles.Count} skipped file(s).";
                    }));
                }
                catch (OperationCanceledException)
                {
                    System.Windows.Application.Current.Dispatcher.BeginInvoke(new Action(() =>
                    {
                        ExportStatus = "Task analysis batch export cancelled.";
                    }));
                }
                catch (Exception ex)
                {
                    System.Windows.Application.Current.Dispatcher.BeginInvoke(new Action(() =>
                    {
                        ExportStatus = "Task analysis batch export failed: " + ex.Message;
                        System.Windows.MessageBox.Show("Task analysis batch export failed: " + ex);
                    }));
                }
                finally
                {
                    System.Windows.Application.Current.Dispatcher.BeginInvoke(new Action(() =>
                    {
                        IsExporting = false;
                    }));
                }
            }, ct);
        }

        private void ExportFilledInkImage(bool includeMetricLabels)
        {
            if (FrameCount <= 0 || string.IsNullOrWhiteSpace(CurrentFilePath))
            {
                System.Windows.MessageBox.Show("No dataset is open.");
                return;
            }

            string sessionPath;
            try
            {
                sessionPath = ResolveSessionPath(CurrentFilePath);
            }
            catch (Exception ex)
            {
                System.Windows.MessageBox.Show("Could not resolve the session metadata file: " + ex.Message);
                return;
            }

            var dlg = new SaveFileDialog
            {
                Title = "Export filled task image as PNG",
                Filter = "PNG Image|*.png|All files|*.*",
                AddExtension = true,
                DefaultExt = ".png",
                FileName = Path.GetFileNameWithoutExtension(sessionPath) + "_filled.png",
                InitialDirectory = Path.GetDirectoryName(CurrentFilePath)
            };

            if (dlg.ShowDialog() != true)
                return;

            _exportCts?.Cancel();
            _exportCts?.Dispose();
            _exportCts = new CancellationTokenSource();
            var ct = _exportCts.Token;

            IsExporting = true;
            ExportProgress = 0;
            ExportStatus = "Replaying ink...";

            Task.Run(() =>
            {
                try
                {
                    ExportFilledInkImageCore(
                        sessionPath,
                        dlg.FileName,
                        includeMetricLabels,
                        ct,
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
                        ExportStatus = "Filled image export complete.";
                    }));
                }
                catch (OperationCanceledException)
                {
                    System.Windows.Application.Current.Dispatcher.BeginInvoke(new Action(() =>
                    {
                        ExportStatus = "Filled image export cancelled.";
                    }));
                }
                catch (Exception ex)
                {
                    System.Windows.Application.Current.Dispatcher.BeginInvoke(new Action(() =>
                    {
                        ExportStatus = "Filled image export failed: " + ex.Message;
                        System.Windows.MessageBox.Show("Filled image export failed: " + ex.Message);
                    }));
                }
                finally
                {
                    System.Windows.Application.Current.Dispatcher.BeginInvoke(new Action(() =>
                    {
                        IsExporting = false;
                    }));
                }
            }, ct);
        }

        private void ExportFilledInkImageCore(string sessionPath,
                                              string outputPath,
                                              bool includeMetricLabels,
                                              CancellationToken ct,
                                              Action<double, string>? progress = null)
        {
            using var session = SessionManager.OpenExisting(sessionPath, aviPathOverride: null, writeable: false);
            using var output = LoadTaskReplayBaseImage(sessionPath);

            var taskRect = new[]
            {
                new Point2f(0, 0),
                new Point2f(output.Width - 1, 0),
                new Point2f(output.Width - 1, output.Height - 1),
                new Point2f(0, output.Height - 1)
            };

            var stylusRect = new[]
            {
                new Point2f(0, 0),
                new Point2f(STYLUS_MAX_X - 1, 0),
                new Point2f(STYLUS_MAX_X - 1, STYLUS_MAX_Y - 1),
                new Point2f(0, STYLUS_MAX_Y - 1)
            };

            var corners = new float[8];
            var labels = new float[session.AttrCount];
            var labelMask = new byte[session.AttrCount];

            string taskId = TryExtractTaskToken(sessionPath, out var taskToken) ? taskToken : string.Empty;
            bool renderPressureWidth = taskId.Equals("T5a", StringComparison.OrdinalIgnoreCase);
            bool renderCalligraphyTilt = taskId.Equals("T5b", StringComparison.OrdinalIgnoreCase);
            var pressureRange = renderPressureWidth
                ? ComputePositivePressureRange(session, ct)
                : (Min: (double?)null, Max: (double?)null);

            Point? previousInkPoint = null;
            bool drewInk = false;
            int thickness = Math.Max(1, (int)Math.Round(output.Width / 300.0));
            int pointRadius = Math.Max(1, thickness / 2);
            const int reportEveryNFrames = 250;

            for (ulong frameIndex = 0; frameIndex < session.FrameCount; frameIndex++)
            {
                ct.ThrowIfCancellationRequested();

                session.ReadMetadata(frameIndex, out _, corners, labels, labelMask);

                bool hasStylusSample =
                    session.AttrCount >= 3 &&
                    labelMask[0] != 0 &&
                    labelMask[1] != 0 &&
                    labelMask[2] != 0;

                if (!hasStylusSample || !(labels[2] > 0f) || !TryGetFrameQuad(corners, out var frameQuad))
                {
                    previousInkPoint = null;
                }
                else
                {
                    using var stylusToFrame = Cv2.GetPerspectiveTransform(stylusRect, frameQuad);
                    using var frameToTask = Cv2.GetPerspectiveTransform(frameQuad, taskRect);

                    var stylusPoint = new Point2f(labels[0], labels[1]);
                    var framePoint = WarpPointByHomographyPrecise(stylusPoint, stylusToFrame);
                    var taskPoint = WarpPointByHomographyPrecise(framePoint, frameToTask);
                    var clampedPoint = ClampPointToImage(taskPoint, output.Width, output.Height);

                    if (renderCalligraphyTilt)
                    {
                        double tiltX = session.AttrCount >= 5 && labelMask[3] != 0 ? labels[3] : 0.0;
                        double tiltY = session.AttrCount >= 5 && labelMask[4] != 0 ? labels[4] : 1.0;
                        DrawT5bCalligraphyInk(output, previousInkPoint, clampedPoint, tiltX, tiltY);
                    }
                    else
                    {
                        int sampleThickness = renderPressureWidth
                            ? ComputeT5aPressureThickness(labels[2], pressureRange.Min, pressureRange.Max, output.Width)
                            : thickness;
                        int samplePointRadius = Math.Max(1, sampleThickness / 2);

                        if (previousInkPoint.HasValue)
                        {
                            Cv2.Line(output, previousInkPoint.Value, clampedPoint, Scalar.Black, sampleThickness, LineTypes.AntiAlias);
                        }
                        else
                        {
                            Cv2.Circle(output, clampedPoint, samplePointRadius, Scalar.Black, -1, LineTypes.AntiAlias);
                        }
                    }

                    previousInkPoint = clampedPoint;
                    drewInk = true;
                }

                if (frameIndex == 0 || frameIndex + 1 == session.FrameCount || (frameIndex % reportEveryNFrames == 0))
                {
                    double pct01 = (frameIndex + 1) / (double)Math.Max(1UL, session.FrameCount);
                    progress?.Invoke(pct01, $"Replaying ink: frame {frameIndex + 1}/{session.FrameCount}");
                }
            }

            if (!drewInk)
                throw new InvalidOperationException("No stylus samples with pressure greater than 0 were found.");

            progress?.Invoke(1.0, "Writing PNG...");
            Cv2.ImEncode(".png", output, out byte[] buffer);
            File.WriteAllBytes(outputPath, buffer);

            var taskWarnings = analyse_task(
                taskId: taskId,
                sessionPath: sessionPath,
                outputImagePath: outputPath,
                includeDiagnosticLabels: includeMetricLabels,
                ct: ct);

            foreach (var warning in taskWarnings)
                System.Diagnostics.Debug.WriteLine("Task analysis warning: " + warning);
        }

        private sealed record T1aTarget(
            int TargetId,
            double CenterX,
            double CenterY,
            double Radius,
            string TargetSizeClass);

        private sealed record T1aTapInput(
            int InputId,
            double X,
            double Y,
            ulong FirstTimestampNs,
            ulong LastTimestampNs);

        private sealed record TargetPixelCluster(
            double X,
            double Y,
            double Weight);

        private sealed record T1aTrialMetric(
            string? ParticipantId,
            string? SessionId,
            string? TrialId,
            string? TaskId,
            int? TargetId,
            double? TargetCenterX,
            double? TargetCenterY,
            double? TargetRadius,
            string? TargetSizeClass,
            double? InputX,
            double? InputY,
            double? DxPx,
            double? DyPx,
            double? RawErrorPx,
            double? NormalisedError,
            bool? Hit,
            double? CompletionTimeMs);

        private sealed record T1aAnalysisResult(
            string TrialsCsvPath,
            string SummaryCsvPath,
            string DiagnosticImagePath,
            string? WarningsPath,
            IReadOnlyList<string> Warnings);

        private sealed record T1bTargetStroke(
            int StrokeId,
            string? TargetColour,
            double StartX,
            double StartY,
            double EndX,
            double EndY,
            double DotRadius);

        private sealed record T1bUserStroke(
            int StrokeId,
            IReadOnlyList<Point2d> Points,
            ulong FirstTimestampNs,
            ulong LastTimestampNs);

        private sealed record StylusStrokeSummary(
            int StrokeId,
            ulong FirstTimestampNs,
            ulong LastTimestampNs,
            int NSamples,
            double? MeanPressure,
            double? MeanTiltMagnitude,
            double? MeanTiltDirectionDeg);

        private sealed record GeneralTaskTimingResult(
            string SummaryCsvPath,
            IReadOnlyList<string> Warnings);

        private sealed record T1bStrokeMetric(
            string? ParticipantId,
            string? SessionId,
            string? TrialId,
            string? TaskId,
            int? StrokeId,
            string? TargetColour,
            double? TargetStartX,
            double? TargetStartY,
            double? TargetEndX,
            double? TargetEndY,
            double? TargetLengthPx,
            double? TargetAngleDeg,
            double? UserStartX,
            double? UserStartY,
            double? UserEndX,
            double? UserEndY,
            double? UserPathLengthPx,
            double? UserAngleDeg,
            int? NSamplePoints,
            string? EndpointMatchDirection,
            double? StartErrorPx,
            double? EndErrorPx,
            double? MeanEndpointErrorPx,
            double? MaxEndpointErrorPx,
            double? MeanLineDeviationPx,
            double? MedianLineDeviationPx,
            double? MaxLineDeviationPx,
            double? RmseLineDeviationPx,
            double? PercentPointsWithinTolerance,
            double? LengthRatio,
            double? AngleErrorDeg,
            double? EndpointTolerancePx,
            double? LineTolerancePx,
            bool? Successful,
            double? CompletionTimeMs);

        private sealed record T1bAnalysisResult(
            string StrokesCsvPath,
            string SummaryCsvPath,
            string DiagnosticImagePath,
            string? WarningsPath,
            IReadOnlyList<string> Warnings);

        private sealed record T2LineReference(
            int LineId,
            string Orientation,
            Point2d Start,
            Point2d End);

        private sealed record T2LineMetric(
            string? ParticipantId,
            string? SessionId,
            string? TrialId,
            string? TaskId,
            int? LineId,
            string? Orientation,
            double? TargetStartX,
            double? TargetStartY,
            double? TargetEndX,
            double? TargetEndY,
            double? TargetLengthPx,
            double? TargetAngleDeg,
            double? UserStartX,
            double? UserStartY,
            double? UserEndX,
            double? UserEndY,
            double? UserPathLengthPx,
            double? UserDirectLengthPx,
            double? UserAngleDeg,
            int? NSamplePoints,
            string? EndpointMatchDirection,
            double? StartErrorPx,
            double? EndErrorPx,
            double? MeanEndpointErrorPx,
            double? MaxEndpointErrorPx,
            double? MeanLineDeviationPx,
            double? MedianLineDeviationPx,
            double? RmseLineDeviationPx,
            double? MaxLineDeviationPx,
            double? PercentPointsWithinTolerance,
            double? StraightnessRatio,
            double? LengthRatio,
            double? AngleErrorDeg,
            double? CompletionTimeMs,
            double? MeanSpeedPxPerS,
            bool? Successful);

        private sealed record T2AnalysisResult(
            string LinesCsvPath,
            string SummaryCsvPath,
            string DiagnosticImagePath,
            string? WarningsPath,
            IReadOnlyList<string> Warnings);

        private sealed record TracingReference(
            int Id,
            string ReferenceType,
            IReadOnlyList<Point2d> Points,
            Rect2d BoundingBox,
            bool IsClosed,
            double GuideRadiusPx);

        private sealed record TracingMetric(
            string? ParticipantId,
            string? SessionId,
            string? TrialId,
            string? TaskId,
            int? PathOrShapeId,
            string? ReferenceType,
            double? ReferencePathLengthPx,
            double? UserPathLengthPx,
            double? LengthRatio,
            int? NReferencePoints,
            int? NUserPoints,
            double? MeanPathDeviationPx,
            double? MedianPathDeviationPx,
            double? SdPathDeviationPx,
            double? RmsePathDeviationPx,
            double? MaxPathDeviationPx,
            double? PercentUserPointsWithinTolerance,
            double? MeanReferenceToUserDeviationPx,
            double? PercentReferenceCovered,
            double? LongestUncoveredGapPx,
            double? ClosureErrorPx,
            double? BoundingBoxSimilarity,
            double? CentroidOffsetPx,
            double? AreaDifferenceRatio,
            double? PercentPointsInsideCorridor,
            int? CorridorExits,
            double? MeanDistanceFromCorridorCenterlinePx,
            double? CompletionTimeMs,
            bool? Successful);

        private sealed record TracingAnalysisResult(
            string DetailCsvPath,
            string SummaryCsvPath,
            string DiagnosticImagePath,
            string? WarningsPath,
            IReadOnlyList<string> Warnings);

        private static IReadOnlyList<string> analyse_task(string taskId,
                                                          string sessionPath,
                                                          string outputImagePath,
                                                          bool includeDiagnosticLabels = false,
                                                          CancellationToken ct = default)
        {
            string normalizedTaskId = NormalizeTaskToken(taskId ?? string.Empty);
            var timingAnalysis = analyse_general_task_timing(
                taskId: normalizedTaskId,
                sessionPath: sessionPath,
                outputImagePath: outputImagePath,
                ct: ct);

            if (normalizedTaskId.Equals("T1a", StringComparison.OrdinalIgnoreCase))
            {
                var analysis = analyse_t1a_target_tapping(
                    sessionPath: sessionPath,
                    outputImagePath: outputImagePath,
                    includeMetricLabels: includeDiagnosticLabels,
                    ct: ct);
                return analysis.Warnings.Concat(timingAnalysis.Warnings).ToList();
            }

            if (normalizedTaskId.Equals("T1b", StringComparison.OrdinalIgnoreCase))
            {
                RemoveStaleTaskOutputs(outputImagePath, sessionPath, "_t1a_");

                var analysis = analyse_t1b_connect_dots(
                    sessionPath: sessionPath,
                    outputImagePath: outputImagePath,
                    includeDiagnosticLabels: includeDiagnosticLabels,
                    ct: ct);
                return analysis.Warnings.Concat(timingAnalysis.Warnings).ToList();
            }

            if (normalizedTaskId.Equals("T2", StringComparison.OrdinalIgnoreCase))
            {
                var analysis = analyse_t2_straight_lines(
                    sessionPath: sessionPath,
                    outputImagePath: outputImagePath,
                    includeDiagnosticLabels: includeDiagnosticLabels,
                    ct: ct);
                return analysis.Warnings.Concat(timingAnalysis.Warnings).ToList();
            }

            if (normalizedTaskId.Equals("T3a", StringComparison.OrdinalIgnoreCase))
            {
                var analysis = analyse_t3a_path_tracing(
                    sessionPath: sessionPath,
                    outputImagePath: outputImagePath,
                    includeDiagnosticLabels: includeDiagnosticLabels,
                    ct: ct);
                return analysis.Warnings.Concat(timingAnalysis.Warnings).ToList();
            }

            if (normalizedTaskId.Equals("T3b", StringComparison.OrdinalIgnoreCase))
            {
                var analysis = analyse_t3b_shape_tracing(
                    sessionPath: sessionPath,
                    outputImagePath: outputImagePath,
                    includeDiagnosticLabels: includeDiagnosticLabels,
                    ct: ct);
                return analysis.Warnings.Concat(timingAnalysis.Warnings).ToList();
            }

            if (IsTaskT1ThroughT7(normalizedTaskId))
            {
                var timingOnlyWarnings = new List<string>(timingAnalysis.Warnings)
                {
                    $"No task-specific geometric analysis is implemented for '{normalizedTaskId}'; wrote timing summary only."
                };

                var timingOnlyResultsDir = GetAnalysisResultsDirectory(outputImagePath, sessionPath);
                Directory.CreateDirectory(timingOnlyResultsDir);
                string timingOnlyBaseName = Path.GetFileNameWithoutExtension(sessionPath);
                File.WriteAllLines(Path.Combine(timingOnlyResultsDir, $"{timingOnlyBaseName}_{normalizedTaskId.ToLowerInvariant()}_warnings.txt"), timingOnlyWarnings);
                return timingOnlyWarnings;
            }

            var warnings = new List<string>
            {
                string.IsNullOrWhiteSpace(taskId)
                    ? "Could not identify task id; skipping task-specific analysis."
                    : $"Unknown task id '{taskId}'; skipping task-specific analysis."
            };
            warnings.AddRange(timingAnalysis.Warnings);

            var resultsDir = GetAnalysisResultsDirectory(outputImagePath, sessionPath);
            Directory.CreateDirectory(resultsDir);
            string baseName = Path.GetFileNameWithoutExtension(sessionPath);
            File.WriteAllLines(Path.Combine(resultsDir, $"{baseName}_task_analysis_warnings.txt"), warnings);
            return warnings;
        }

        private static bool IsTaskT1ThroughT7(string taskId)
        {
            if (string.IsNullOrWhiteSpace(taskId) || taskId.Length < 2 || char.ToUpperInvariant(taskId[0]) != 'T')
                return false;

            int i = 1;
            while (i < taskId.Length && char.IsDigit(taskId[i]))
                i++;

            return int.TryParse(taskId[1..i], NumberStyles.Integer, CultureInfo.InvariantCulture, out int taskNumber)
                && taskNumber >= 1
                && taskNumber <= 7;
        }

        private static GeneralTaskTimingResult analyse_general_task_timing(string taskId,
                                                                           string sessionPath,
                                                                           string outputImagePath,
                                                                           CancellationToken ct = default)
        {
            var warnings = new List<string>();
            using var session = SessionManager.OpenExisting(sessionPath, aviPathOverride: null, writeable: false);

            var strokes = ExtractStylusStrokeSummaries(session, warnings, ct);
            var metadata = ExtractAnalysisMetadata(sessionPath, taskId);
            var resultsDir = GetAnalysisResultsDirectory(outputImagePath, sessionPath);
            Directory.CreateDirectory(resultsDir);

            string baseName = Path.GetFileNameWithoutExtension(sessionPath);
            string summaryPath = Path.Combine(resultsDir, $"{baseName}_task_timing_summary.csv");
            string strokesPath = Path.Combine(resultsDir, $"{baseName}_task_strokes.csv");
            WriteGeneralTaskStrokesCsv(strokesPath, metadata, strokes, taskId);
            WriteGeneralTaskTimingSummaryCsv(summaryPath, metadata, strokes, taskId, warnings);

            if (warnings.Count > 0)
            {
                string warningsPath = Path.Combine(resultsDir, $"{baseName}_task_timing_warnings.txt");
                File.WriteAllLines(warningsPath, warnings);
            }

            return new GeneralTaskTimingResult(summaryPath, warnings);
        }

        private static List<StylusStrokeSummary> ExtractStylusStrokeSummaries(SessionManager session,
                                                                              List<string> warnings,
                                                                              CancellationToken ct)
        {
            var labels = new float[session.AttrCount];
            var labelMask = new byte[session.AttrCount];
            var strokes = new List<StylusStrokeSummary>();

            var pressureSamples = new List<double>();
            var tiltMagnitudeSamples = new List<double>();
            var tiltXSamples = new List<double>();
            var tiltYSamples = new List<double>();
            ulong firstTimestamp = 0;
            ulong lastTimestamp = 0;
            int sampleCount = 0;
            bool inStroke = false;

            void FinishStroke()
            {
                if (!inStroke || sampleCount == 0)
                    return;

                double? meanPressure = pressureSamples.Count > 0 ? pressureSamples.Average() : null;
                double? meanTiltMagnitude = tiltMagnitudeSamples.Count > 0 ? tiltMagnitudeSamples.Average() : null;
                double? meanTiltDirection = null;
                if (tiltXSamples.Count > 0 && tiltYSamples.Count > 0)
                {
                    double meanX = tiltXSamples.Average();
                    double meanY = tiltYSamples.Average();
                    meanTiltDirection = Math.Atan2(meanY, meanX) * 180.0 / Math.PI;
                }

                strokes.Add(new StylusStrokeSummary(
                    StrokeId: strokes.Count + 1,
                    FirstTimestampNs: firstTimestamp,
                    LastTimestampNs: lastTimestamp,
                    NSamples: sampleCount,
                    MeanPressure: meanPressure,
                    MeanTiltMagnitude: meanTiltMagnitude,
                    MeanTiltDirectionDeg: meanTiltDirection));

                pressureSamples.Clear();
                tiltMagnitudeSamples.Clear();
                tiltXSamples.Clear();
                tiltYSamples.Clear();
                firstTimestamp = 0;
                lastTimestamp = 0;
                sampleCount = 0;
                inStroke = false;
            }

            for (ulong frameIndex = 0; frameIndex < session.FrameCount; frameIndex++)
            {
                ct.ThrowIfCancellationRequested();
                session.ReadMetadata(frameIndex, out ulong timestampNs, corners: null, labels, labelMask);

                bool hasPressureSample = session.AttrCount >= 3 && labelMask[2] != 0;
                bool isTouching = hasPressureSample && labels[2] > 0f;
                if (!isTouching)
                {
                    FinishStroke();
                    continue;
                }

                if (!inStroke)
                {
                    inStroke = true;
                    firstTimestamp = timestampNs;
                }

                lastTimestamp = timestampNs;
                sampleCount++;
                pressureSamples.Add(labels[2]);

                bool hasTiltSample = session.AttrCount >= 5 && labelMask[3] != 0 && labelMask[4] != 0;
                if (hasTiltSample)
                {
                    double tiltX = labels[3];
                    double tiltY = labels[4];
                    tiltXSamples.Add(tiltX);
                    tiltYSamples.Add(tiltY);
                    tiltMagnitudeSamples.Add(Math.Sqrt((tiltX * tiltX) + (tiltY * tiltY)));
                }
            }

            FinishStroke();

            if (strokes.Count == 0)
                warnings.Add("No pressure-positive stylus strokes were found for task timing.");

            return strokes;
        }

        private static void WriteGeneralTaskTimingSummaryCsv(string path,
                                                             (string? ParticipantId, string? SessionId, string? TrialId, string? TaskId) metadata,
                                                             IReadOnlyList<StylusStrokeSummary> strokes,
                                                             string taskId,
                                                             List<string> warnings)
        {
            double? taskDurationMs = ComputeStylusTaskDurationMs(strokes);
            var strokeDurations = strokes
                .Select(s => s.LastTimestampNs >= s.FirstTimestampNs ? (double?)((s.LastTimestampNs - s.FirstTimestampNs) / 1_000_000.0) : null)
                .Where(v => v.HasValue)
                .Select(v => v!.Value)
                .ToList();

            const double pressureDistinctTolerance = 10.0;
            const double tiltDistinctTolerance = 5.0;
            int distinctPressureStrokes = CountDistinctValues(strokes.Select(s => s.MeanPressure), pressureDistinctTolerance);
            int distinctTiltStrokes = CountDistinctValues(strokes.Select(s => s.MeanTiltMagnitude), tiltDistinctTolerance);
            bool? pressureLadderPass = taskId.Equals("T5a", StringComparison.OrdinalIgnoreCase) ? distinctPressureStrokes >= 3 : null;
            bool? tiltLadderPass = taskId.Equals("T5b", StringComparison.OrdinalIgnoreCase) ? distinctTiltStrokes >= 3 : null;

            if (pressureLadderPass == false)
                warnings.Add($"T5a pressure ladder check failed: {distinctPressureStrokes} distinct stroke pressure level(s) found; expected at least 3.");

            if (tiltLadderPass == false)
                warnings.Add($"T5b tilt ladder check failed: {distinctTiltStrokes} distinct stroke tilt angle level(s) found; expected at least 3.");

            using var writer = new StreamWriter(path);
            writer.WriteLine("participant_id,session_id,trial_id,task_id,n_strokes,first_stroke_timestamp_ns,last_stroke_timestamp_ns,task_duration_ms,mean_stroke_duration_ms,sd_stroke_duration_ms,mean_stroke_pressure,sd_stroke_pressure,n_distinct_pressure_strokes,pressure_ladder_pass,mean_stroke_tilt_magnitude,sd_stroke_tilt_magnitude,n_distinct_tilt_strokes,tilt_ladder_pass");
            writer.WriteLine(string.Join(",", new[]
            {
                Csv(metadata.ParticipantId),
                Csv(metadata.SessionId),
                Csv(metadata.TrialId),
                Csv(taskId),
                Csv(strokes.Count),
                Csv(strokes.Count > 0 ? strokes.Min(s => s.FirstTimestampNs) : null),
                Csv(strokes.Count > 0 ? strokes.Max(s => s.LastTimestampNs) : null),
                Csv(taskDurationMs),
                Csv(Mean(strokeDurations)),
                Csv(StandardDeviation(strokeDurations)),
                Csv(Mean(strokes.Select(s => s.MeanPressure).Where(v => v.HasValue).Select(v => v!.Value).ToList())),
                Csv(StandardDeviation(strokes.Select(s => s.MeanPressure).Where(v => v.HasValue).Select(v => v!.Value).ToList())),
                Csv(distinctPressureStrokes),
                Csv(pressureLadderPass),
                Csv(Mean(strokes.Select(s => s.MeanTiltMagnitude).Where(v => v.HasValue).Select(v => v!.Value).ToList())),
                Csv(StandardDeviation(strokes.Select(s => s.MeanTiltMagnitude).Where(v => v.HasValue).Select(v => v!.Value).ToList())),
                Csv(distinctTiltStrokes),
                Csv(tiltLadderPass)
            }));
        }

        private static void WriteGeneralTaskStrokesCsv(string path,
                                                       (string? ParticipantId, string? SessionId, string? TrialId, string? TaskId) metadata,
                                                       IReadOnlyList<StylusStrokeSummary> strokes,
                                                       string taskId)
        {
            using var writer = new StreamWriter(path);
            writer.WriteLine("participant_id,session_id,trial_id,task_id,stroke_id,first_timestamp_ns,last_timestamp_ns,stroke_duration_ms,n_samples,mean_pressure,mean_tilt_magnitude,mean_tilt_direction_deg");

            foreach (var stroke in strokes)
            {
                double? strokeDurationMs = stroke.LastTimestampNs >= stroke.FirstTimestampNs
                    ? (stroke.LastTimestampNs - stroke.FirstTimestampNs) / 1_000_000.0
                    : null;

                writer.WriteLine(string.Join(",", new[]
                {
                    Csv(metadata.ParticipantId),
                    Csv(metadata.SessionId),
                    Csv(metadata.TrialId),
                    Csv(taskId),
                    Csv(stroke.StrokeId),
                    Csv(stroke.FirstTimestampNs),
                    Csv(stroke.LastTimestampNs),
                    Csv(strokeDurationMs),
                    Csv(stroke.NSamples),
                    Csv(stroke.MeanPressure),
                    Csv(stroke.MeanTiltMagnitude),
                    Csv(stroke.MeanTiltDirectionDeg)
                }));
            }
        }

        private static double? ComputeStylusTaskDurationMs(IReadOnlyList<StylusStrokeSummary> strokes)
        {
            if (strokes.Count == 0)
                return null;

            ulong first = strokes.Min(s => s.FirstTimestampNs);
            ulong last = strokes.Max(s => s.LastTimestampNs);
            return last >= first ? (last - first) / 1_000_000.0 : null;
        }

        private static int CountDistinctValues(IEnumerable<double?> values, double tolerance)
        {
            var sorted = values
                .Where(v => v.HasValue)
                .Select(v => v!.Value)
                .OrderBy(v => v)
                .ToList();

            if (sorted.Count == 0)
                return 0;

            int count = 1;
            double lastAccepted = sorted[0];
            for (int i = 1; i < sorted.Count; i++)
            {
                if (Math.Abs(sorted[i] - lastAccepted) >= tolerance)
                {
                    count++;
                    lastAccepted = sorted[i];
                }
            }

            return count;
        }

        private static string GetAnalysisResultsDirectory(string outputImagePath, string sessionPath)
        {
            return Path.Combine(
                Path.GetDirectoryName(outputImagePath) ?? Path.GetDirectoryName(sessionPath) ?? Environment.CurrentDirectory,
                "analysis",
                "results");
        }

        private static void RemoveStaleTaskOutputs(string outputImagePath, string sessionPath, string staleTaskMarker)
        {
            var resultsDir = GetAnalysisResultsDirectory(outputImagePath, sessionPath);
            if (!Directory.Exists(resultsDir))
                return;

            string baseName = Path.GetFileNameWithoutExtension(sessionPath);
            foreach (var path in Directory.EnumerateFiles(resultsDir, $"{baseName}{staleTaskMarker}*"))
            {
                try
                {
                    File.Delete(path);
                }
                catch (Exception ex)
                {
                    System.Diagnostics.Debug.WriteLine($"Could not delete stale analysis output '{path}': {ex.Message}");
                }
            }
        }

        private static T1aAnalysisResult compute_t1a_metrics(string sessionPath,
                                                             string outputImagePath,
                                                             bool includeMetricLabels = false,
                                                             CancellationToken ct = default)
        {
            return analyse_t1a_target_tapping(
                sessionPath: sessionPath,
                outputImagePath: outputImagePath,
                includeMetricLabels: includeMetricLabels,
                ct: ct);
        }

        private static T1aAnalysisResult analyse_t1a_target_tapping(string sessionPath,
                                                                    string outputImagePath,
                                                                    bool includeMetricLabels = false,
                                                                    CancellationToken ct = default)
        {
            var warnings = new List<string>();
            if (!TryExtractTaskToken(sessionPath, out var taskToken))
                taskToken = string.Empty;

            using var session = SessionManager.OpenExisting(sessionPath, aviPathOverride: null, writeable: false);
            using var baseImage = LoadTaskReplayBaseImage(sessionPath);

            var targets = DetectT1TargetsFromTemplate(sessionPath, warnings);
            var inputs = ExtractT1TapInputs(session, baseImage.Width, baseImage.Height, warnings, ct);
            double? completionTimeMs = ComputeCompletionTimeMs(inputs);

            if (targets.Count != inputs.Count)
                warnings.Add($"Target/input count mismatch: targets={targets.Count}, inputs={inputs.Count}.");

            var metadata = ExtractAnalysisMetadata(sessionPath, taskToken);
            var rows = BuildT1aMetricRows(targets, inputs, metadata, completionTimeMs, warnings, ct);

            var resultsDir = Path.Combine(
                Path.GetDirectoryName(outputImagePath) ?? Path.GetDirectoryName(sessionPath) ?? Environment.CurrentDirectory,
                "analysis",
                "results");
            Directory.CreateDirectory(resultsDir);

            string baseName = Path.GetFileNameWithoutExtension(sessionPath);
            string trialsPath = Path.Combine(resultsDir, $"{baseName}_t1a_trials.csv");
            string summaryPath = Path.Combine(resultsDir, $"{baseName}_t1a_summary.csv");
            string diagnosticPath = Path.Combine(resultsDir, $"{baseName}_t1a_diagnostic{(includeMetricLabels ? "_labelled" : string.Empty)}.png");
            string debugPath = Path.Combine(resultsDir, $"{baseName}_t1a_debug.csv");

            WriteT1aTrialsCsv(trialsPath, rows);
            WriteT1aSummaryCsv(summaryPath, rows);
            WriteT1aDiagnosticImage(diagnosticPath, baseImage, targets, inputs, rows, includeMetricLabels);
            WriteT1aDebugCsv(debugPath, rows);

            string? warningsPath = null;
            if (warnings.Count > 0)
            {
                warningsPath = Path.Combine(resultsDir, $"{baseName}_t1a_warnings.txt");
                File.WriteAllLines(warningsPath, warnings);
            }

            return new T1aAnalysisResult(trialsPath, summaryPath, diagnosticPath, warningsPath, warnings);
        }

        private static T1bAnalysisResult compute_t1b_stroke_metrics(string sessionPath,
                                                                    string outputImagePath,
                                                                    bool includeDiagnosticLabels = false,
                                                                    CancellationToken ct = default)
        {
            return analyse_t1b_connect_dots(
                sessionPath: sessionPath,
                outputImagePath: outputImagePath,
                includeDiagnosticLabels: includeDiagnosticLabels,
                ct: ct);
        }

        private static T1bAnalysisResult analyse_t1b_connect_dots(string sessionPath,
                                                                  string outputImagePath,
                                                                  bool includeDiagnosticLabels = false,
                                                                  double? endpointTolerancePx = null,
                                                                  double? lineTolerancePx = null,
                                                                  double requiredPercentWithinTolerance = 80.0,
                                                                  CancellationToken ct = default)
        {
            var warnings = new List<string>();
            if (!TryExtractTaskToken(sessionPath, out var taskToken))
                taskToken = "T1b";

            using var session = SessionManager.OpenExisting(sessionPath, aviPathOverride: null, writeable: false);
            using var baseImage = LoadTaskReplayBaseImage(sessionPath);

            var targets = DetectT1bTargetStrokePairs(sessionPath, warnings);
            var userStrokes = ExtractT1bUserStrokes(session, baseImage.Width, baseImage.Height, warnings, ct);
            var metadata = ExtractAnalysisMetadata(sessionPath, taskToken);
            double? completionTimeMs = ComputeT1bCompletionTimeMs(userStrokes);
            var rows = BuildT1bStrokeMetricRows(
                targets,
                userStrokes,
                metadata,
                completionTimeMs,
                endpointTolerancePx,
                lineTolerancePx,
                requiredPercentWithinTolerance,
                warnings,
                ct);

            var resultsDir = Path.Combine(
                Path.GetDirectoryName(outputImagePath) ?? Path.GetDirectoryName(sessionPath) ?? Environment.CurrentDirectory,
                "analysis",
                "results");
            Directory.CreateDirectory(resultsDir);

            string baseName = Path.GetFileNameWithoutExtension(sessionPath);
            string strokesPath = Path.Combine(resultsDir, $"{baseName}_t1b_strokes.csv");
            string summaryPath = Path.Combine(resultsDir, $"{baseName}_t1b_summary.csv");
            string diagnosticPath = Path.Combine(resultsDir, $"{baseName}_t1b_diagnostic{(includeDiagnosticLabels ? "_labelled" : string.Empty)}.png");

            write_t1b_analysis_outputs(strokesPath, summaryPath, diagnosticPath, baseImage, targets, userStrokes, rows, includeDiagnosticLabels);

            string? warningsPath = null;
            if (warnings.Count > 0)
            {
                warningsPath = Path.Combine(resultsDir, $"{baseName}_t1b_warnings.txt");
                File.WriteAllLines(warningsPath, warnings);
            }

            return new T1bAnalysisResult(strokesPath, summaryPath, diagnosticPath, warningsPath, warnings);
        }

        private static List<T1bTargetStroke> DetectT1bTargetStrokePairs(string sessionPath, List<string> warnings)
        {
            if (!TryExtractTaskToken(sessionPath, out var taskToken))
                taskToken = "T1b";

            string? templatePath = FindTaskTemplatePath(sessionPath, taskToken);
            if (string.IsNullOrWhiteSpace(templatePath))
            {
                warnings.Add($"Could not find task template PNG for '{taskToken}'.");
                return new List<T1bTargetStroke>();
            }

            using var raw = Cv2.ImRead(templatePath, ImreadModes.Unchanged);
            if (raw == null || raw.Empty())
            {
                warnings.Add($"Could not read task template PNG: {templatePath}");
                return new List<T1bTargetStroke>();
            }

            using var flipped = new Mat();
            Cv2.Flip(raw, flipped, FlipMode.XY);

            using var bgr = new Mat();
            if (flipped.Channels() == 4)
                Cv2.CvtColor(flipped, bgr, ColorConversionCodes.BGRA2BGR);
            else if (flipped.Channels() == 1)
                Cv2.CvtColor(flipped, bgr, ColorConversionCodes.GRAY2BGR);
            else
                flipped.CopyTo(bgr);

            using var hsv = new Mat();
            Cv2.CvtColor(bgr, hsv, ColorConversionCodes.BGR2HSV);

            using var mask = new Mat(bgr.Rows, bgr.Cols, MatType.CV_8UC1, Scalar.All(0));
            for (int y = 0; y < bgr.Rows; y++)
            {
                for (int x = 0; x < bgr.Cols; x++)
                {
                    var h = hsv.At<Vec3b>(y, x);
                    byte alpha = flipped.Channels() == 4 ? flipped.At<Vec4b>(y, x).Item3 : (byte)255;
                    if (alpha > 0 && h.Item1 >= 60 && h.Item2 >= 80)
                        mask.Set(y, x, 255);
                }
            }

            using var kernel = Cv2.GetStructuringElement(MorphShapes.Ellipse, new Size(3, 3));
            Cv2.MorphologyEx(mask, mask, MorphTypes.Close, kernel);
            using var componentLabels = new Mat();
            using var stats = new Mat();
            using var centroids = new Mat();
            int n = Cv2.ConnectedComponentsWithStats(mask, componentLabels, stats, centroids, PixelConnectivity.Connectivity8);

            var dots = new List<(Point2d Center, double Radius, string Colour)>();
            for (int i = 1; i < n; i++)
            {
                int area = stats.Get<int>(i, (int)ConnectedComponentsTypes.Area);
                if (area < 5)
                    continue;

                double cx = centroids.Get<double>(i, 0);
                double cy = centroids.Get<double>(i, 1);
                string colour = ClassifyDotColour(bgr, componentLabels, i);
                dots.Add((new Point2d(cx, cy), Math.Sqrt(area / Math.PI), colour));
            }

            var strokes = new List<T1bTargetStroke>();
            foreach (var group in dots.GroupBy(d => d.Colour).OrderBy(g => g.Key))
            {
                var groupDots = group.OrderBy(d => d.Center.X).ThenBy(d => d.Center.Y).ToList();
                if (groupDots.Count != 2)
                {
                    warnings.Add($"Colour group '{group.Key}' has {groupDots.Count} dots; expected 2. Skipping ambiguous group.");
                    continue;
                }

                strokes.Add(new T1bTargetStroke(
                    StrokeId: strokes.Count + 1,
                    TargetColour: group.Key,
                    StartX: groupDots[0].Center.X,
                    StartY: groupDots[0].Center.Y,
                    EndX: groupDots[1].Center.X,
                    EndY: groupDots[1].Center.Y,
                    DotRadius: groupDots.Average(d => d.Radius)));
            }

            if (strokes.Count == 0)
                warnings.Add("No coloured dot pairs were detected for T1b.");

            return strokes;
        }

        private static string ClassifyDotColour(Mat bgr, Mat labels, int label)
        {
            double sumB = 0;
            double sumG = 0;
            double sumR = 0;
            int count = 0;
            for (int y = 0; y < labels.Rows; y++)
            {
                for (int x = 0; x < labels.Cols; x++)
                {
                    if (labels.Get<int>(y, x) != label)
                        continue;

                    var c = bgr.At<Vec3b>(y, x);
                    sumB += c.Item0;
                    sumG += c.Item1;
                    sumR += c.Item2;
                    count++;
                }
            }

            if (count == 0)
                return "unknown";

            double b = sumB / count;
            double g = sumG / count;
            double r = sumR / count;

            if (r > 180 && g < 100 && b < 100) return "red";
            if (g > r && g > b) return "green";
            if (b > r && b > g) return "cyan";
            if (r > 120 && b > 100) return "magenta";
            return $"rgb_{(int)Math.Round(r)}_{(int)Math.Round(g)}_{(int)Math.Round(b)}";
        }

        private static List<T1bUserStroke> ExtractT1bUserStrokes(SessionManager session,
                                                                 int outputWidth,
                                                                 int outputHeight,
                                                                 List<string> warnings,
                                                                 CancellationToken ct)
        {
            var taskRect = new[]
            {
                new Point2f(0, 0),
                new Point2f(outputWidth - 1, 0),
                new Point2f(outputWidth - 1, outputHeight - 1),
                new Point2f(0, outputHeight - 1)
            };

            var stylusRect = new[]
            {
                new Point2f(0, 0),
                new Point2f(STYLUS_MAX_X - 1, 0),
                new Point2f(STYLUS_MAX_X - 1, STYLUS_MAX_Y - 1),
                new Point2f(0, STYLUS_MAX_Y - 1)
            };

            var corners = new float[8];
            var labels = new float[session.AttrCount];
            var labelMask = new byte[session.AttrCount];
            var strokes = new List<T1bUserStroke>();
            var points = new List<Point2d>();
            ulong firstTimestamp = 0;
            ulong lastTimestamp = 0;
            bool inStroke = false;

            void FinishStroke()
            {
                if (!inStroke || points.Count == 0)
                    return;

                strokes.Add(new T1bUserStroke(strokes.Count + 1, points.ToList(), firstTimestamp, lastTimestamp));
                points.Clear();
                firstTimestamp = 0;
                lastTimestamp = 0;
                inStroke = false;
            }

            for (ulong frameIndex = 0; frameIndex < session.FrameCount; frameIndex++)
            {
                ct.ThrowIfCancellationRequested();
                session.ReadMetadata(frameIndex, out ulong timestampNs, corners, labels, labelMask);
                bool hasSample = session.AttrCount >= 3 && labelMask[0] != 0 && labelMask[1] != 0 && labelMask[2] != 0;

                if (!hasSample || !(labels[2] > 0f))
                {
                    FinishStroke();
                    continue;
                }

                if (!TryGetFrameQuad(corners, out var frameQuad))
                {
                    warnings.Add($"Frame {frameIndex} has stroke data but missing/invalid corners; sample skipped.");
                    continue;
                }

                using var stylusToFrame = Cv2.GetPerspectiveTransform(stylusRect, frameQuad);
                using var frameToTask = Cv2.GetPerspectiveTransform(frameQuad, taskRect);
                var framePoint = WarpPointByHomographyPrecise(new Point2f(labels[0], labels[1]), stylusToFrame);
                var taskPoint = WarpPointByHomographyPrecise(framePoint, frameToTask);
                var clamped = ClampPointToImage(taskPoint, outputWidth, outputHeight);

                if (!inStroke)
                {
                    inStroke = true;
                    firstTimestamp = timestampNs;
                }

                points.Add(new Point2d(clamped.X, clamped.Y));
                lastTimestamp = timestampNs;
            }

            FinishStroke();
            if (strokes.Count == 0)
                warnings.Add("No T1b user strokes were found from pressure > 0 samples.");

            return strokes;
        }

        private static List<T1bStrokeMetric> BuildT1bStrokeMetricRows(IReadOnlyList<T1bTargetStroke> targets,
                                                                      IReadOnlyList<T1bUserStroke> userStrokes,
                                                                      (string? ParticipantId, string? SessionId, string? TrialId, string? TaskId) metadata,
                                                                      double? completionTimeMs,
                                                                      double? configuredEndpointTolerancePx,
                                                                      double? configuredLineTolerancePx,
                                                                      double requiredPercentWithinTolerance,
                                                                      List<string> warnings,
                                                                      CancellationToken ct)
        {
            var rows = new List<T1bStrokeMetric>(Math.Max(targets.Count, userStrokes.Count));
            var unusedStrokeIndices = new HashSet<int>(Enumerable.Range(0, userStrokes.Count));

            foreach (var target in targets.OrderBy(t => t.StrokeId))
            {
                ct.ThrowIfCancellationRequested();
                int? bestIndex = null;
                double bestCost = double.PositiveInfinity;

                foreach (int strokeIndex in unusedStrokeIndices)
                {
                    double cost = ComputeT1bMatchingCost(target, userStrokes[strokeIndex]);
                    if (cost < bestCost)
                    {
                        bestCost = cost;
                        bestIndex = strokeIndex;
                    }
                }

                T1bUserStroke? userStroke = null;
                if (bestIndex.HasValue)
                {
                    userStroke = userStrokes[bestIndex.Value];
                    unusedStrokeIndices.Remove(bestIndex.Value);
                }
                else
                {
                    warnings.Add($"Target stroke {target.StrokeId} has no matching user stroke.");
                }

                rows.Add(CreateT1bStrokeMetricRow(
                    target,
                    userStroke,
                    metadata,
                    completionTimeMs,
                    configuredEndpointTolerancePx,
                    configuredLineTolerancePx,
                    requiredPercentWithinTolerance,
                    warnings));
            }

            foreach (int strokeIndex in unusedStrokeIndices.OrderBy(i => userStrokes[i].StrokeId))
            {
                warnings.Add($"User stroke {userStrokes[strokeIndex].StrokeId} has no matching target pair.");
                rows.Add(CreateT1bStrokeMetricRow(null, userStrokes[strokeIndex], metadata, completionTimeMs, null, null, requiredPercentWithinTolerance, warnings));
            }

            return rows;
        }

        private static T1bStrokeMetric CreateT1bStrokeMetricRow(T1bTargetStroke? target,
                                                                T1bUserStroke? userStroke,
                                                                (string? ParticipantId, string? SessionId, string? TrialId, string? TaskId) metadata,
                                                                double? completionTimeMs,
                                                                double? configuredEndpointTolerancePx,
                                                                double? configuredLineTolerancePx,
                                                                double requiredPercentWithinTolerance,
                                                                List<string> warnings)
        {
            double? targetLength = target != null ? Distance(target.StartX, target.StartY, target.EndX, target.EndY) : null;
            double? targetAngle = target != null ? ComputeAngleDeg(target.StartX, target.StartY, target.EndX, target.EndY) : null;
            Point2d? userStart = userStroke?.Points.Count > 0 ? userStroke.Points[0] : null;
            Point2d? userEnd = userStroke?.Points.Count > 0 ? userStroke.Points[^1] : null;
            double? userPathLength = userStroke != null ? ComputePathLength(userStroke.Points) : null;
            double? userAngle = userStart.HasValue && userEnd.HasValue ? ComputeAngleDeg(userStart.Value.X, userStart.Value.Y, userEnd.Value.X, userEnd.Value.Y) : null;

            string? direction = null;
            double? startError = null;
            double? endError = null;
            double? meanEndpointError = null;
            double? maxEndpointError = null;
            double? meanDeviation = null;
            double? medianDeviation = null;
            double? maxDeviation = null;
            double? rmseDeviation = null;
            double? percentWithin = null;
            double? lengthRatio = null;
            double? angleError = null;
            bool? successful = null;
            double? endpointTolerance = null;
            double? lineTolerance = null;

            if (target != null)
            {
                endpointTolerance = configuredEndpointTolerancePx ?? (target.DotRadius > 0 ? target.DotRadius * 1.5 : null);
                lineTolerance = configuredLineTolerancePx ?? (target.DotRadius > 0 ? target.DotRadius * 1.5 : null);
                if (!endpointTolerance.HasValue || !lineTolerance.HasValue)
                    warnings.Add($"Target stroke {target.StrokeId} has unreliable dot radius; tolerance metrics left blank.");
            }

            if (target != null && userStroke != null && userStart.HasValue && userEnd.HasValue && targetLength > 0)
            {
                var endpointErrors = compute_orientation_invariant_endpoint_errors(
                    userStart.Value,
                    userEnd.Value,
                    new Point2d(target.StartX, target.StartY),
                    new Point2d(target.EndX, target.EndY));

                direction = endpointErrors.Direction;
                startError = endpointErrors.StartError;
                endError = endpointErrors.EndError;
                meanEndpointError = (startError + endError) / 2.0;
                maxEndpointError = Math.Max(startError.Value, endError.Value);

                var deviations = userStroke.Points
                    .Select(point => compute_point_to_segment_distance(point, new Point2d(target.StartX, target.StartY), new Point2d(target.EndX, target.EndY)))
                    .ToList();

                meanDeviation = Mean(deviations);
                medianDeviation = Median(deviations);
                maxDeviation = deviations.Count > 0 ? deviations.Max() : null;
                rmseDeviation = deviations.Count > 0 ? Math.Sqrt(deviations.Average(d => d * d)) : null;
                percentWithin = lineTolerance.HasValue && deviations.Count > 0
                    ? deviations.Count(d => d <= lineTolerance.Value) * 100.0 / deviations.Count
                    : null;

                lengthRatio = userPathLength.HasValue ? userPathLength / targetLength : null;
                angleError = userAngle.HasValue && targetAngle.HasValue
                    ? compute_direction_invariant_angle_error(userAngle.Value, targetAngle.Value)
                    : null;

                successful =
                    endpointTolerance.HasValue &&
                    lineTolerance.HasValue &&
                    startError <= endpointTolerance &&
                    endError <= endpointTolerance &&
                    percentWithin >= requiredPercentWithinTolerance;
            }
            else if (target != null && targetLength <= 0)
            {
                warnings.Add($"Target stroke {target.StrokeId} has zero-length target segment; metrics left blank.");
            }

            return new T1bStrokeMetric(
                ParticipantId: metadata.ParticipantId,
                SessionId: metadata.SessionId,
                TrialId: metadata.TrialId,
                TaskId: metadata.TaskId,
                StrokeId: target?.StrokeId,
                TargetColour: target?.TargetColour,
                TargetStartX: target?.StartX,
                TargetStartY: target?.StartY,
                TargetEndX: target?.EndX,
                TargetEndY: target?.EndY,
                TargetLengthPx: targetLength,
                TargetAngleDeg: targetAngle,
                UserStartX: userStart?.X,
                UserStartY: userStart?.Y,
                UserEndX: userEnd?.X,
                UserEndY: userEnd?.Y,
                UserPathLengthPx: userPathLength,
                UserAngleDeg: userAngle,
                NSamplePoints: userStroke?.Points.Count,
                EndpointMatchDirection: direction,
                StartErrorPx: startError,
                EndErrorPx: endError,
                MeanEndpointErrorPx: meanEndpointError,
                MaxEndpointErrorPx: maxEndpointError,
                MeanLineDeviationPx: meanDeviation,
                MedianLineDeviationPx: medianDeviation,
                MaxLineDeviationPx: maxDeviation,
                RmseLineDeviationPx: rmseDeviation,
                PercentPointsWithinTolerance: percentWithin,
                LengthRatio: lengthRatio,
                AngleErrorDeg: angleError,
                EndpointTolerancePx: endpointTolerance,
                LineTolerancePx: lineTolerance,
                Successful: successful,
                CompletionTimeMs: completionTimeMs);
        }

        private static double ComputeT1bMatchingCost(T1bTargetStroke target, T1bUserStroke userStroke)
        {
            if (userStroke.Points.Count == 0)
                return double.PositiveInfinity;

            var endpointErrors = compute_orientation_invariant_endpoint_errors(
                userStroke.Points[0],
                userStroke.Points[^1],
                new Point2d(target.StartX, target.StartY),
                new Point2d(target.EndX, target.EndY));

            double endpointCost = endpointErrors.StartError + endpointErrors.EndError;
            double lineCost = userStroke.Points.Average(point => compute_point_to_segment_distance(point, new Point2d(target.StartX, target.StartY), new Point2d(target.EndX, target.EndY)));
            return endpointCost + lineCost;
        }

        private static (string Direction, double StartError, double EndError) compute_orientation_invariant_endpoint_errors(Point2d userStart,
                                                                                                                          Point2d userEnd,
                                                                                                                          Point2d targetStart,
                                                                                                                          Point2d targetEnd)
        {
            double forwardStart = Distance(userStart.X, userStart.Y, targetStart.X, targetStart.Y);
            double forwardEnd = Distance(userEnd.X, userEnd.Y, targetEnd.X, targetEnd.Y);
            double reverseStart = Distance(userStart.X, userStart.Y, targetEnd.X, targetEnd.Y);
            double reverseEnd = Distance(userEnd.X, userEnd.Y, targetStart.X, targetStart.Y);

            return forwardStart + forwardEnd <= reverseStart + reverseEnd
                ? ("forward", forwardStart, forwardEnd)
                : ("reverse", reverseStart, reverseEnd);
        }

        private static double compute_point_to_segment_distance(Point2d point, Point2d segmentStart, Point2d segmentEnd)
        {
            double vx = segmentEnd.X - segmentStart.X;
            double vy = segmentEnd.Y - segmentStart.Y;
            double wx = point.X - segmentStart.X;
            double wy = point.Y - segmentStart.Y;
            double len2 = (vx * vx) + (vy * vy);

            if (len2 <= double.Epsilon)
                return Distance(point.X, point.Y, segmentStart.X, segmentStart.Y);

            double t = Math.Clamp(((wx * vx) + (wy * vy)) / len2, 0.0, 1.0);
            return Distance(point.X, point.Y, segmentStart.X + (t * vx), segmentStart.Y + (t * vy));
        }

        private static double compute_direction_invariant_angle_error(double userAngleDeg, double targetAngleDeg)
        {
            double diff = Math.Abs(NormalizeAngleDeg(userAngleDeg - targetAngleDeg));
            if (diff > 180.0)
                diff = 360.0 - diff;
            return diff > 90.0 ? 180.0 - diff : diff;
        }

        private static string OrientationCategory(double? targetAngleDeg)
        {
            if (!targetAngleDeg.HasValue)
                return string.Empty;

            double angle = Math.Abs(NormalizeAngleDeg(targetAngleDeg.Value));
            if (angle > 90.0)
                angle = 180.0 - angle;
            if (angle < 22.5) return "horizontal";
            if (angle > 67.5) return "vertical";
            return "diagonal";
        }

        private static double NormalizeAngleDeg(double angle)
        {
            angle %= 360.0;
            if (angle < -180.0) angle += 360.0;
            if (angle > 180.0) angle -= 360.0;
            return angle;
        }

        private static double ComputeAngleDeg(double x1, double y1, double x2, double y2)
            => Math.Atan2(y2 - y1, x2 - x1) * 180.0 / Math.PI;

        private static double ComputePathLength(IReadOnlyList<Point2d> points)
        {
            double length = 0;
            for (int i = 1; i < points.Count; i++)
                length += Distance(points[i - 1].X, points[i - 1].Y, points[i].X, points[i].Y);
            return length;
        }

        private static double Distance(double x1, double y1, double x2, double y2)
        {
            double dx = x2 - x1;
            double dy = y2 - y1;
            return Math.Sqrt((dx * dx) + (dy * dy));
        }

        private static double? ComputeT1bCompletionTimeMs(IReadOnlyList<T1bUserStroke> strokes)
        {
            if (strokes.Count == 0)
                return null;

            ulong first = strokes.Min(s => s.FirstTimestampNs);
            ulong last = strokes.Max(s => s.LastTimestampNs);
            return last >= first ? (last - first) / 1_000_000.0 : null;
        }

        private static List<T1aTrialMetric> BuildT1aMetricRows(IReadOnlyList<T1aTarget> targets,
                                                               IReadOnlyList<T1aTapInput> inputs,
                                                               (string? ParticipantId, string? SessionId, string? TrialId, string? TaskId) metadata,
                                                               double? completionTimeMs,
                                                               List<string> warnings,
                                                               CancellationToken ct)
        {
            var rows = new List<T1aTrialMetric>(Math.Max(targets.Count, inputs.Count));
            var unusedInputIndices = new HashSet<int>(Enumerable.Range(0, inputs.Count));

            foreach (var target in targets.OrderBy(t => t.TargetId))
            {
                ct.ThrowIfCancellationRequested();

                int? inputIndex = null;
                double bestDistance = double.PositiveInfinity;

                foreach (int candidateIndex in unusedInputIndices)
                {
                    var candidate = inputs[candidateIndex];
                    double dxCandidate = candidate.X - target.CenterX;
                    double dyCandidate = candidate.Y - target.CenterY;
                    double distance = Math.Sqrt((dxCandidate * dxCandidate) + (dyCandidate * dyCandidate));
                    if (distance < bestDistance)
                    {
                        bestDistance = distance;
                        inputIndex = candidateIndex;
                    }
                }

                T1aTapInput? input = null;
                if (inputIndex.HasValue)
                {
                    input = inputs[inputIndex.Value];
                    unusedInputIndices.Remove(inputIndex.Value);
                }
                else
                {
                    warnings.Add($"Target {target.TargetId} has no matching input point.");
                }

                rows.Add(CreateT1aMetricRow(target, input, metadata, completionTimeMs, warnings));
            }

            foreach (int inputIndex in unusedInputIndices.OrderBy(i => inputs[i].InputId))
            {
                ct.ThrowIfCancellationRequested();
                warnings.Add($"Input {inputs[inputIndex].InputId} has no matching target.");
                rows.Add(CreateT1aMetricRow(null, inputs[inputIndex], metadata, completionTimeMs, warnings));
            }

            return rows;
        }

        private static T1aTrialMetric CreateT1aMetricRow(T1aTarget? target,
                                                         T1aTapInput? input,
                                                         (string? ParticipantId, string? SessionId, string? TrialId, string? TaskId) metadata,
                                                         double? completionTimeMs,
                                                         List<string> warnings)
        {
            double? dx = null;
            double? dy = null;
            double? rawError = null;
            double? normalisedError = null;
            bool? hit = null;

            if (target != null && input != null && target.Radius > 0)
            {
                dx = input.X - target.CenterX;
                dy = input.Y - target.CenterY;

                // raw_error_px is Euclidean distance from the target centre to the recorded input point.
                rawError = Math.Sqrt((dx.Value * dx.Value) + (dy.Value * dy.Value));

                // normalised_error is raw_error_px divided by target_radius.
                normalisedError = rawError.Value / target.Radius;

                // hit is true when raw_error_px <= target_radius.
                hit = rawError.Value <= target.Radius;
            }
            else if (target != null && input != null && target.Radius <= 0)
            {
                warnings.Add($"Target {target.TargetId} has missing or invalid radius; metrics left blank.");
            }

            return new T1aTrialMetric(
                ParticipantId: metadata.ParticipantId,
                SessionId: metadata.SessionId,
                TrialId: metadata.TrialId,
                TaskId: metadata.TaskId,
                TargetId: target?.TargetId,
                TargetCenterX: target?.CenterX,
                TargetCenterY: target?.CenterY,
                TargetRadius: target?.Radius,
                TargetSizeClass: target?.TargetSizeClass,
                InputX: input?.X,
                InputY: input?.Y,
                DxPx: dx,
                DyPx: dy,
                RawErrorPx: rawError,
                NormalisedError: normalisedError,
                Hit: hit,
                CompletionTimeMs: completionTimeMs);
        }

        private static List<T1aTarget> DetectT1TargetsFromTemplate(string sessionPath, List<string> warnings)
        {
            if (!TryExtractTaskToken(sessionPath, out var taskToken))
            {
                warnings.Add("Could not infer task id from filename; no target template was loaded.");
                return new List<T1aTarget>();
            }

            string? templatePath = FindTaskTemplatePath(sessionPath, taskToken);
            if (string.IsNullOrWhiteSpace(templatePath))
            {
                warnings.Add($"Could not find task template PNG for '{taskToken}'.");
                return new List<T1aTarget>();
            }

            using var raw = Cv2.ImRead(templatePath, ImreadModes.Unchanged);
            if (raw == null || raw.Empty())
            {
                warnings.Add($"Could not read task template PNG: {templatePath}");
                return new List<T1aTarget>();
            }

            using var flipped = new Mat();
            Cv2.Flip(raw, flipped, FlipMode.XY);

            using var bgr = new Mat();
            if (flipped.Channels() == 4)
                Cv2.CvtColor(flipped, bgr, ColorConversionCodes.BGRA2BGR);
            else if (flipped.Channels() == 1)
                Cv2.CvtColor(flipped, bgr, ColorConversionCodes.GRAY2BGR);
            else
                flipped.CopyTo(bgr);

            using var redMask = new Mat();
            Cv2.InRange(bgr, new Scalar(0, 0, 100), new Scalar(130, 130, 255), redMask);

            if (flipped.Channels() == 4)
            {
                using var alpha = new Mat();
                Cv2.ExtractChannel(flipped, alpha, 3);
                Cv2.BitwiseAnd(redMask, alpha, redMask);
            }

            var redPixels = GetMaskPixels(redMask);
            Cv2.FindContours(redMask, out Point[][] contours, out _, RetrievalModes.External, ContourApproximationModes.ApproxSimple);

            var candidates = new List<TargetPixelCluster>();
            foreach (var contour in contours)
            {
                double area = Cv2.ContourArea(contour);
                if (area < 8)
                    continue;

                var moments = Cv2.Moments(contour);
                if (Math.Abs(moments.M00) < double.Epsilon)
                    continue;

                candidates.Add(new TargetPixelCluster(
                    X: moments.M10 / moments.M00,
                    Y: moments.M01 / moments.M00,
                    Weight: area));
            }

            int? expectedTargetCount = GetExpectedTargetCount(taskToken);
            var circles = MergeTargetPixelClusters(candidates, redPixels, expectedTargetCount, warnings);

            circles = circles
                .OrderBy(c => c.Y)
                .ThenBy(c => c.X)
                .ToList();

            if (circles.Count == 0)
                warnings.Add("No red target circles were detected in the task template.");

            var sizeClasses = ClassifyTargetSizes(circles.Select(c => c.Radius).ToList());
            var targets = new List<T1aTarget>(circles.Count);
            for (int i = 0; i < circles.Count; i++)
            {
                var circle = circles[i];
                targets.Add(new T1aTarget(
                    TargetId: i + 1,
                    CenterX: circle.X,
                    CenterY: circle.Y,
                    Radius: circle.Radius,
                    TargetSizeClass: sizeClasses[i]));
            }

            return targets;
        }

        private static int? GetExpectedTargetCount(string taskToken)
            => taskToken.StartsWith("T1", StringComparison.OrdinalIgnoreCase) ? 8 : null;

        private static List<Point2f> GetMaskPixels(Mat mask)
        {
            var pixels = new List<Point2f>();
            for (int y = 0; y < mask.Rows; y++)
            {
                for (int x = 0; x < mask.Cols; x++)
                {
                    if (mask.At<byte>(y, x) != 0)
                        pixels.Add(new Point2f(x, y));
                }
            }

            return pixels;
        }

        private static List<(double X, double Y, double Radius)> MergeTargetPixelClusters(
            List<TargetPixelCluster> candidates,
            IReadOnlyList<Point2f> redPixels,
            int? expectedTargetCount,
            List<string> warnings)
        {
            if (candidates.Count == 0)
                return new List<(double X, double Y, double Radius)>();

            if (!expectedTargetCount.HasValue)
                return RefineTargetCirclesFromRedPixels(candidates, redPixels, warnings);

            if (candidates.Count <= expectedTargetCount.Value)
                return RefineTargetCirclesFromRedPixels(candidates, redPixels, warnings);

            warnings.Add($"Detected {candidates.Count} red target fragments; merging to expected target count {expectedTargetCount.Value}.");

            var clusters = candidates
                .ToList();

            while (clusters.Count > expectedTargetCount.Value)
            {
                int bestA = -1;
                int bestB = -1;
                double bestDistance = double.PositiveInfinity;

                for (int i = 0; i < clusters.Count; i++)
                {
                    for (int j = i + 1; j < clusters.Count; j++)
                    {
                        double dx = clusters[i].X - clusters[j].X;
                        double dy = clusters[i].Y - clusters[j].Y;
                        double distance = Math.Sqrt((dx * dx) + (dy * dy));
                        if (distance < bestDistance)
                        {
                            bestDistance = distance;
                            bestA = i;
                            bestB = j;
                        }
                    }
                }

                if (bestA < 0 || bestB < 0)
                    break;

                var a = clusters[bestA];
                var b = clusters[bestB];
                double mergedWeight = a.Weight + b.Weight;
                double mergedX = ((a.X * a.Weight) + (b.X * b.Weight)) / mergedWeight;
                double mergedY = ((a.Y * a.Weight) + (b.Y * b.Weight)) / mergedWeight;

                clusters[bestA] = new TargetPixelCluster(mergedX, mergedY, mergedWeight);
                clusters.RemoveAt(bestB);
            }

            if (clusters.Count != expectedTargetCount.Value)
                warnings.Add($"Expected {expectedTargetCount.Value} targets but detected {clusters.Count} after merging.");

            return RefineTargetCirclesFromRedPixels(clusters, redPixels, warnings);
        }

        private static List<(double X, double Y, double Radius)> RefineTargetCirclesFromRedPixels(
            IReadOnlyList<TargetPixelCluster> seedClusters,
            IReadOnlyList<Point2f> redPixels,
            List<string> warnings)
        {
            if (redPixels.Count == 0)
            {
                warnings.Add("No red pixels were available for target radius refinement.");
                return seedClusters.Select(c => (c.X, c.Y, Radius: 0.0)).ToList();
            }

            var centers = seedClusters.Select(c => new Point2d(c.X, c.Y)).ToArray();
            List<Point2f>[] groups = Array.Empty<List<Point2f>>();

            for (int iteration = 0; iteration < 6; iteration++)
            {
                groups = Enumerable.Range(0, centers.Length).Select(_ => new List<Point2f>()).ToArray();

                foreach (var pixel in redPixels)
                {
                    int bestIndex = 0;
                    double bestDistance = double.PositiveInfinity;

                    for (int i = 0; i < centers.Length; i++)
                    {
                        double dx = pixel.X - centers[i].X;
                        double dy = pixel.Y - centers[i].Y;
                        double distance = (dx * dx) + (dy * dy);
                        if (distance < bestDistance)
                        {
                            bestDistance = distance;
                            bestIndex = i;
                        }
                    }

                    groups[bestIndex].Add(pixel);
                }

                for (int i = 0; i < groups.Length; i++)
                {
                    if (groups[i].Count == 0)
                        continue;

                    centers[i] = new Point2d(
                        groups[i].Average(p => p.X),
                        groups[i].Average(p => p.Y));
                }
            }

            var circles = new List<(double X, double Y, double Radius)>(centers.Length);
            for (int i = 0; i < centers.Length; i++)
            {
                if (groups.Length <= i || groups[i].Count == 0)
                {
                    warnings.Add($"Target cluster {i + 1} has no red pixels after refinement; radius set to 0.");
                    circles.Add((centers[i].X, centers[i].Y, 0));
                    continue;
                }

                var distances = groups[i]
                    .Select(p =>
                    {
                        double dx = p.X - centers[i].X;
                        double dy = p.Y - centers[i].Y;
                        return Math.Sqrt((dx * dx) + (dy * dy));
                    })
                    .OrderBy(d => d)
                    .ToArray();

                circles.Add((centers[i].X, centers[i].Y, MedianSorted(distances)));
            }

            return circles;
        }

        private static List<string> ClassifyTargetSizes(IReadOnlyList<double> radii)
        {
            if (radii.Count == 0)
                return new List<string>();

            var distinct = radii
                .OrderBy(r => r)
                .Aggregate(new List<double>(), (acc, radius) =>
                {
                    if (acc.Count == 0 || Math.Abs(acc[^1] - radius) > 1.5)
                        acc.Add(radius);
                    return acc;
                });

            if (distinct.Count == 1)
                return radii.Select(_ => "large").ToList();

            double split = distinct[distinct.Count / 2];
            return radii.Select(radius => radius < split ? "small" : "large").ToList();
        }

        private static List<T1aTapInput> ExtractT1TapInputs(SessionManager session,
                                                            int outputWidth,
                                                            int outputHeight,
                                                            List<string> warnings,
                                                            CancellationToken ct)
        {
            var taskRect = new[]
            {
                new Point2f(0, 0),
                new Point2f(outputWidth - 1, 0),
                new Point2f(outputWidth - 1, outputHeight - 1),
                new Point2f(0, outputHeight - 1)
            };

            var stylusRect = new[]
            {
                new Point2f(0, 0),
                new Point2f(STYLUS_MAX_X - 1, 0),
                new Point2f(STYLUS_MAX_X - 1, STYLUS_MAX_Y - 1),
                new Point2f(0, STYLUS_MAX_Y - 1)
            };

            var corners = new float[8];
            var labels = new float[session.AttrCount];
            var labelMask = new byte[session.AttrCount];
            var inputs = new List<T1aTapInput>();

            bool inTouch = false;
            double sumX = 0;
            double sumY = 0;
            int sampleCount = 0;
            ulong firstTimestamp = 0;
            ulong lastTimestamp = 0;

            void FinishTouch()
            {
                if (!inTouch || sampleCount == 0)
                    return;

                inputs.Add(new T1aTapInput(
                    InputId: inputs.Count + 1,
                    X: sumX / sampleCount,
                    Y: sumY / sampleCount,
                    FirstTimestampNs: firstTimestamp,
                    LastTimestampNs: lastTimestamp));

                inTouch = false;
                sumX = 0;
                sumY = 0;
                sampleCount = 0;
                firstTimestamp = 0;
                lastTimestamp = 0;
            }

            for (ulong frameIndex = 0; frameIndex < session.FrameCount; frameIndex++)
            {
                ct.ThrowIfCancellationRequested();

                session.ReadMetadata(frameIndex, out ulong timestampNs, corners, labels, labelMask);
                bool hasStylusSample =
                    session.AttrCount >= 3 &&
                    labelMask[0] != 0 &&
                    labelMask[1] != 0 &&
                    labelMask[2] != 0;

                if (!hasStylusSample || !(labels[2] > 0f))
                {
                    FinishTouch();
                    continue;
                }

                if (!TryGetFrameQuad(corners, out var frameQuad))
                {
                    warnings.Add($"Frame {frameIndex} has touch data but missing/invalid corners; sample skipped.");
                    continue;
                }

                using var stylusToFrame = Cv2.GetPerspectiveTransform(stylusRect, frameQuad);
                using var frameToTask = Cv2.GetPerspectiveTransform(frameQuad, taskRect);
                var stylusPoint = new Point2f(labels[0], labels[1]);
                var framePoint = WarpPointByHomographyPrecise(stylusPoint, stylusToFrame);
                var taskPoint = WarpPointByHomographyPrecise(framePoint, frameToTask);
                var clampedPoint = ClampPointToImage(taskPoint, outputWidth, outputHeight);

                if (!inTouch)
                {
                    inTouch = true;
                    firstTimestamp = timestampNs;
                }

                sumX += clampedPoint.X;
                sumY += clampedPoint.Y;
                sampleCount++;
                lastTimestamp = timestampNs;
            }

            FinishTouch();

            if (inputs.Count == 0)
                warnings.Add("No tap inputs were found from pressure > 0 samples.");

            return inputs;
        }

        private static double? ComputeCompletionTimeMs(IReadOnlyList<T1aTapInput> inputs)
        {
            if (inputs.Count == 0)
                return null;

            ulong first = inputs.Min(i => i.FirstTimestampNs);
            ulong last = inputs.Max(i => i.LastTimestampNs);
            return last >= first ? (last - first) / 1_000_000.0 : null;
        }

        private static (string? ParticipantId, string? SessionId, string? TrialId, string? TaskId) ExtractAnalysisMetadata(string sessionPath, string taskToken)
        {
            string stem = Path.GetFileNameWithoutExtension(sessionPath) ?? string.Empty;
            string? participantId = MatchValue(ParticipantRegex, stem);
            string? sessionId = MatchValue(SessionRegex, stem);
            string? trialId = MatchValue(TrialRegex, stem) ?? stem;

            return (
                ParticipantId: participantId,
                SessionId: sessionId,
                TrialId: trialId,
                TaskId: string.IsNullOrWhiteSpace(taskToken) ? null : taskToken);

            static string? MatchValue(Regex regex, string value)
            {
                var match = regex.Match(value);
                return match.Success ? match.Groups[1].Value : null;
            }
        }

        private static void WriteT1aTrialsCsv(string path, IReadOnlyList<T1aTrialMetric> rows)
        {
            using var writer = new StreamWriter(path);
            writer.WriteLine("participant_id,session_id,trial_id,task_id,target_id,target_center_x,target_center_y,target_radius,target_size_class,input_x,input_y,dx_px,dy_px,raw_error_px,normalised_error,hit,completion_time_ms");

            foreach (var row in rows)
            {
                writer.WriteLine(string.Join(",", new[]
                {
                    Csv(row.ParticipantId),
                    Csv(row.SessionId),
                    Csv(row.TrialId),
                    Csv(row.TaskId),
                    Csv(row.TargetId),
                    Csv(row.TargetCenterX),
                    Csv(row.TargetCenterY),
                    Csv(row.TargetRadius),
                    Csv(row.TargetSizeClass),
                    Csv(row.InputX),
                    Csv(row.InputY),
                    Csv(row.DxPx),
                    Csv(row.DyPx),
                    Csv(row.RawErrorPx),
                    Csv(row.NormalisedError),
                    Csv(row.Hit),
                    Csv(row.CompletionTimeMs)
                }));
            }
        }

        private static void write_t1b_analysis_outputs(string strokesPath,
                                                       string summaryPath,
                                                       string diagnosticPath,
                                                       Mat baseImage,
                                                       IReadOnlyList<T1bTargetStroke> targets,
                                                       IReadOnlyList<T1bUserStroke> userStrokes,
                                                       IReadOnlyList<T1bStrokeMetric> rows,
                                                       bool includeDiagnosticLabels)
        {
            WriteT1bStrokesCsv(strokesPath, rows);
            WriteT1bSummaryCsv(summaryPath, rows);
            WriteT1bDiagnosticImage(diagnosticPath, baseImage, targets, userStrokes, rows, includeDiagnosticLabels);
        }

        private static void WriteT1bStrokesCsv(string path, IReadOnlyList<T1bStrokeMetric> rows)
        {
            using var writer = new StreamWriter(path);
            writer.WriteLine("participant_id,session_id,trial_id,task_id,stroke_id,target_colour,target_start_x,target_start_y,target_end_x,target_end_y,target_length_px,target_angle_deg,user_start_x,user_start_y,user_end_x,user_end_y,user_path_length_px,user_angle_deg,n_sample_points,endpoint_match_direction,start_error_px,end_error_px,mean_endpoint_error_px,max_endpoint_error_px,mean_line_deviation_px,median_line_deviation_px,max_line_deviation_px,rmse_line_deviation_px,percent_points_within_tolerance,length_ratio,angle_error_deg,endpoint_tolerance_px,line_tolerance_px,successful");

            foreach (var row in rows)
            {
                writer.WriteLine(string.Join(",", new[]
                {
                    Csv(row.ParticipantId),
                    Csv(row.SessionId),
                    Csv(row.TrialId),
                    Csv(row.TaskId),
                    Csv(row.StrokeId),
                    Csv(row.TargetColour),
                    Csv(row.TargetStartX),
                    Csv(row.TargetStartY),
                    Csv(row.TargetEndX),
                    Csv(row.TargetEndY),
                    Csv(row.TargetLengthPx),
                    Csv(row.TargetAngleDeg),
                    Csv(row.UserStartX),
                    Csv(row.UserStartY),
                    Csv(row.UserEndX),
                    Csv(row.UserEndY),
                    Csv(row.UserPathLengthPx),
                    Csv(row.UserAngleDeg),
                    Csv(row.NSamplePoints),
                    Csv(row.EndpointMatchDirection),
                    Csv(row.StartErrorPx),
                    Csv(row.EndErrorPx),
                    Csv(row.MeanEndpointErrorPx),
                    Csv(row.MaxEndpointErrorPx),
                    Csv(row.MeanLineDeviationPx),
                    Csv(row.MedianLineDeviationPx),
                    Csv(row.MaxLineDeviationPx),
                    Csv(row.RmseLineDeviationPx),
                    Csv(row.PercentPointsWithinTolerance),
                    Csv(row.LengthRatio),
                    Csv(row.AngleErrorDeg),
                    Csv(row.EndpointTolerancePx),
                    Csv(row.LineTolerancePx),
                    Csv(row.Successful)
                }));
            }
        }

        private static void WriteT1bSummaryCsv(string path, IReadOnlyList<T1bStrokeMetric> rows)
        {
            using var writer = new StreamWriter(path);
            bool hasCompletionTimes = rows.Any(r => r.CompletionTimeMs.HasValue);
            string header = "participant_id,task_id,orientation_category,n_strokes,n_successful,success_rate,mean_start_error_px,mean_end_error_px,mean_endpoint_error_px,sd_endpoint_error_px,mean_line_deviation_px,sd_line_deviation_px,mean_rmse_line_deviation_px,mean_percent_points_within_tolerance,mean_length_ratio,sd_length_ratio,mean_angle_error_deg,sd_angle_error_deg";
            if (hasCompletionTimes)
                header += ",mean_completion_time_ms,sd_completion_time_ms";
            writer.WriteLine(header);

            var groups = rows
                .GroupBy(row => new
                {
                    ParticipantId = row.ParticipantId ?? string.Empty,
                    TaskId = row.TaskId ?? string.Empty,
                    Orientation = OrientationCategory(row.TargetAngleDeg)
                })
                .OrderBy(g => g.Key.ParticipantId)
                .ThenBy(g => g.Key.Orientation);

            foreach (var group in groups)
            {
                var startErrors = Values(group.Select(r => r.StartErrorPx));
                var endErrors = Values(group.Select(r => r.EndErrorPx));
                var endpointErrors = Values(group.Select(r => r.MeanEndpointErrorPx));
                var lineErrors = Values(group.Select(r => r.MeanLineDeviationPx));
                var rmses = Values(group.Select(r => r.RmseLineDeviationPx));
                var percents = Values(group.Select(r => r.PercentPointsWithinTolerance));
                var lengthRatios = Values(group.Select(r => r.LengthRatio));
                var angleErrors = Values(group.Select(r => r.AngleErrorDeg));
                var completionTimes = Values(group.Select(r => r.CompletionTimeMs)).Distinct().ToList();
                int nStrokes = group.Count(r => r.TargetLengthPx.HasValue);
                int nSuccessful = group.Count(r => r.Successful == true);

                var fields = new List<string>
                {
                    Csv(group.Key.ParticipantId),
                    Csv(group.Key.TaskId),
                    Csv(group.Key.Orientation),
                    Csv(nStrokes),
                    Csv(nSuccessful),
                    Csv(nStrokes > 0 ? nSuccessful / (double)nStrokes : null),
                    Csv(Mean(startErrors)),
                    Csv(Mean(endErrors)),
                    Csv(Mean(endpointErrors)),
                    Csv(StandardDeviation(endpointErrors)),
                    Csv(Mean(lineErrors)),
                    Csv(StandardDeviation(lineErrors)),
                    Csv(Mean(rmses)),
                    Csv(Mean(percents)),
                    Csv(Mean(lengthRatios)),
                    Csv(StandardDeviation(lengthRatios)),
                    Csv(Mean(angleErrors)),
                    Csv(StandardDeviation(angleErrors))
                };

                if (hasCompletionTimes)
                {
                    fields.Add(Csv(Mean(completionTimes)));
                    fields.Add(Csv(StandardDeviation(completionTimes)));
                }

                writer.WriteLine(string.Join(",", fields));
            }

            static List<double> Values(IEnumerable<double?> source)
                => source.Where(v => v.HasValue).Select(v => v!.Value).ToList();
        }

        private static void WriteT1bDiagnosticImage(string path,
                                                    Mat baseImage,
                                                    IReadOnlyList<T1bTargetStroke> targets,
                                                    IReadOnlyList<T1bUserStroke> userStrokes,
                                                    IReadOnlyList<T1bStrokeMetric> rows,
                                                    bool includeDiagnosticLabels)
        {
            using var diagnostic = baseImage.Clone();

            foreach (var target in targets)
            {
                Cv2.Line(diagnostic, new Point(target.StartX, target.StartY), new Point(target.EndX, target.EndY), new Scalar(0, 0, 255), 1, LineTypes.AntiAlias);
                Cv2.Circle(diagnostic, new Point(target.StartX, target.StartY), 4, new Scalar(0, 0, 255), -1, LineTypes.AntiAlias);
                Cv2.Circle(diagnostic, new Point(target.EndX, target.EndY), 4, new Scalar(0, 0, 255), -1, LineTypes.AntiAlias);
            }

            foreach (var stroke in userStrokes)
            {
                for (int i = 1; i < stroke.Points.Count; i++)
                    Cv2.Line(diagnostic, new Point(stroke.Points[i - 1].X, stroke.Points[i - 1].Y), new Point(stroke.Points[i].X, stroke.Points[i].Y), Scalar.Black, 2, LineTypes.AntiAlias);

                if (stroke.Points.Count > 0)
                {
                    Cv2.Circle(diagnostic, new Point(stroke.Points[0].X, stroke.Points[0].Y), 3, new Scalar(255, 0, 0), -1, LineTypes.AntiAlias);
                    Cv2.Circle(diagnostic, new Point(stroke.Points[^1].X, stroke.Points[^1].Y), 3, new Scalar(0, 160, 0), -1, LineTypes.AntiAlias);
                }
            }

            if (includeDiagnosticLabels)
            {
                foreach (var row in rows)
                {
                    if (!row.StrokeId.HasValue || !row.TargetStartX.HasValue || !row.TargetStartY.HasValue || !row.TargetEndX.HasValue || !row.TargetEndY.HasValue)
                        continue;

                    string label = $"{row.StrokeId.Value}";
                    if (row.MeanEndpointErrorPx.HasValue && row.MeanLineDeviationPx.HasValue && row.Successful.HasValue)
                        label += $" ep{row.MeanEndpointErrorPx.Value:0.0} ln{row.MeanLineDeviationPx.Value:0.0} {(row.Successful.Value ? "ok" : "fail")}";

                    var p = new Point(
                        (int)Math.Round((row.TargetStartX.Value + row.TargetEndX.Value) / 2.0 + 8),
                        (int)Math.Round((row.TargetStartY.Value + row.TargetEndY.Value) / 2.0 - 8));
                    Cv2.PutText(diagnostic, label, p, HersheyFonts.HersheySimplex, 0.32, Scalar.Black, 1, LineTypes.AntiAlias);
                }
            }

            Cv2.ImEncode(".png", diagnostic, out byte[] buffer);
            File.WriteAllBytes(path, buffer);
        }

        private static T2AnalysisResult analyse_t2_straight_lines(string sessionPath,
                                                                  string outputImagePath,
                                                                  bool includeDiagnosticLabels = false,
                                                                  double endpointTolerancePx = 20.0,
                                                                  double lineTolerancePx = 10.0,
                                                                  double requiredPercentPointsWithinTolerance = 80.0,
                                                                  double minimumStraightnessRatio = 0.90,
                                                                  double minimumLengthRatio = 0.75,
                                                                  double maximumLengthRatio = 1.35,
                                                                  double maximumAngleErrorDeg = 10.0,
                                                                  CancellationToken ct = default)
        {
            var warnings = new List<string>();
            using var session = SessionManager.OpenExisting(sessionPath, aviPathOverride: null, writeable: false);
            using var baseImage = LoadTaskReplayBaseImage(sessionPath);

            var userStrokes = segment_user_strokes(session, baseImage.Width, baseImage.Height, outputImagePath, warnings, ct);
            var references = BuildT2ReferenceLines(userStrokes, baseImage.Width, baseImage.Height, warnings);
            var matches = MatchT2UserStrokesToReferences(references, userStrokes, warnings);
            var metadata = ExtractAnalysisMetadata(sessionPath, "T2");
            double? completionTimeMs = ComputeT1bCompletionTimeMs(userStrokes);

            var rows = new List<T2LineMetric>();
            foreach (var reference in references)
            {
                T1bUserStroke? userStroke = matches.TryGetValue(reference.LineId, out int strokeIndex) ? userStrokes[strokeIndex] : null;
                if (userStroke == null)
                    warnings.Add($"T2 {reference.Orientation} line has no matching user stroke.");

                rows.Add(ComputeT2LineMetric(
                    metadata,
                    reference,
                    userStroke,
                    completionTimeMs,
                    endpointTolerancePx,
                    lineTolerancePx,
                    requiredPercentPointsWithinTolerance,
                    minimumStraightnessRatio,
                    minimumLengthRatio,
                    maximumLengthRatio,
                    maximumAngleErrorDeg,
                    warnings));
            }

            foreach (var extraStroke in userStrokes.Where((_, i) => !matches.Values.Contains(i)))
                warnings.Add($"T2 user stroke {extraStroke.StrokeId} was not matched to an intended line.");

            var resultsDir = GetAnalysisResultsDirectory(outputImagePath, sessionPath);
            Directory.CreateDirectory(resultsDir);
            string baseName = Path.GetFileNameWithoutExtension(sessionPath);
            string linesPath = Path.Combine(resultsDir, $"{baseName}_t2_lines.csv");
            string summaryPath = Path.Combine(resultsDir, $"{baseName}_t2_summary.csv");
            string diagnosticPath = Path.Combine(resultsDir, $"{baseName}_t2_diagnostic{(includeDiagnosticLabels ? "_labelled" : string.Empty)}.png");

            WriteT2AnalysisOutputs(linesPath, summaryPath, diagnosticPath, baseImage, references, userStrokes, matches, rows, includeDiagnosticLabels);

            string? warningsPath = null;
            if (warnings.Count > 0)
            {
                warningsPath = Path.Combine(resultsDir, $"{baseName}_t2_warnings.txt");
                File.WriteAllLines(warningsPath, warnings);
            }

            return new T2AnalysisResult(linesPath, summaryPath, diagnosticPath, warningsPath, warnings);
        }

        private static List<T2LineReference> BuildT2ReferenceLines(IReadOnlyList<T1bUserStroke> userStrokes, int imageWidth, int imageHeight, List<string> warnings)
        {
            string[] orientations = { "horizontal", "vertical", "diagonal" };
            var references = new List<T2LineReference>();
            var used = new HashSet<int>();

            foreach (var orientation in orientations)
            {
                int? strokeIndex = FindBestStrokeForOrientation(userStrokes, orientation, used);
                if (strokeIndex.HasValue)
                {
                    used.Add(strokeIndex.Value);
                    references.Add(CreateT2ReferenceFromStroke(references.Count + 1, orientation, userStrokes[strokeIndex.Value]));
                }
                else
                {
                    warnings.Add($"Could not infer T2 {orientation} reference from user stroke data; using default image-position reference.");
                    references.Add(CreateDefaultT2Reference(references.Count + 1, orientation, imageWidth, imageHeight));
                }
            }

            warnings.Add("No T2 template PNG was found; reference lines were inferred from stroke orientation/position where possible.");
            return references;
        }

        private static int? FindBestStrokeForOrientation(IReadOnlyList<T1bUserStroke> userStrokes, string orientation, HashSet<int> used)
        {
            double targetAngle = orientation switch
            {
                "vertical" => 90.0,
                "diagonal" => 45.0,
                _ => 0.0
            };

            int? bestIndex = null;
            double bestCost = double.PositiveInfinity;
            for (int i = 0; i < userStrokes.Count; i++)
            {
                if (used.Contains(i) || userStrokes[i].Points.Count < 2)
                    continue;

                var points = userStrokes[i].Points;
                double directLength = Distance(points[0].X, points[0].Y, points[^1].X, points[^1].Y);
                if (directLength <= 1.0)
                    continue;

                double angle = ComputeAngleDeg(points[0].X, points[0].Y, points[^1].X, points[^1].Y);
                double cost = compute_direction_invariant_angle_error(angle, targetAngle);
                if (cost < bestCost)
                {
                    bestCost = cost;
                    bestIndex = i;
                }
            }

            return bestIndex;
        }

        private static T2LineReference CreateT2ReferenceFromStroke(int lineId, string orientation, T1bUserStroke stroke)
        {
            var start = stroke.Points[0];
            var end = stroke.Points[^1];
            double length = Math.Max(1.0, Distance(start.X, start.Y, end.X, end.Y));
            var center = new Point2d((start.X + end.X) / 2.0, (start.Y + end.Y) / 2.0);
            double angleRad = orientation switch
            {
                "vertical" => Math.PI / 2.0,
                "diagonal" => Math.PI / 4.0,
                _ => 0.0
            };

            double dx = Math.Cos(angleRad) * length / 2.0;
            double dy = Math.Sin(angleRad) * length / 2.0;
            return new T2LineReference(lineId, orientation, new Point2d(center.X - dx, center.Y - dy), new Point2d(center.X + dx, center.Y + dy));
        }

        private static T2LineReference CreateDefaultT2Reference(int lineId, string orientation, int imageWidth, int imageHeight)
        {
            double w = imageWidth - 1;
            double h = imageHeight - 1;
            return orientation switch
            {
                "vertical" => new T2LineReference(lineId, orientation, new Point2d(w * 0.72, h * 0.20), new Point2d(w * 0.72, h * 0.80)),
                "diagonal" => new T2LineReference(lineId, orientation, new Point2d(w * 0.25, h * 0.72), new Point2d(w * 0.55, h * 0.42)),
                _ => new T2LineReference(lineId, orientation, new Point2d(w * 0.20, h * 0.25), new Point2d(w * 0.80, h * 0.25))
            };
        }

        private static Dictionary<int, int> MatchT2UserStrokesToReferences(IReadOnlyList<T2LineReference> references, IReadOnlyList<T1bUserStroke> userStrokes, List<string> warnings)
        {
            var matches = new Dictionary<int, int>();
            var unused = Enumerable.Range(0, userStrokes.Count).ToHashSet();

            foreach (var reference in references)
            {
                int? bestIndex = null;
                double bestCost = double.PositiveInfinity;
                foreach (int strokeIndex in unused)
                {
                    var stroke = userStrokes[strokeIndex];
                    if (stroke.Points.Count < 2)
                        continue;

                    double cost = ComputeT2MatchingCost(reference, stroke);
                    if (cost < bestCost)
                    {
                        bestCost = cost;
                        bestIndex = strokeIndex;
                    }
                }

                if (bestIndex.HasValue)
                {
                    matches[reference.LineId] = bestIndex.Value;
                    unused.Remove(bestIndex.Value);
                }
            }

            if (userStrokes.Count == 0)
                warnings.Add("No T2 user strokes were found.");
            return matches;
        }

        private static double ComputeT2MatchingCost(T2LineReference reference, T1bUserStroke stroke)
        {
            var points = stroke.Points;
            var endpointErrors = compute_orientation_invariant_endpoint_errors(points[0], points[^1], reference.Start, reference.End);
            double targetAngle = ComputeAngleDeg(reference.Start.X, reference.Start.Y, reference.End.X, reference.End.Y);
            double userAngle = ComputeAngleDeg(points[0].X, points[0].Y, points[^1].X, points[^1].Y);
            double angleCost = compute_direction_invariant_angle_error(userAngle, targetAngle) * 4.0;
            double lineCost = points.Average(p => compute_point_to_segment_distance(p, reference.Start, reference.End));
            return endpointErrors.StartError + endpointErrors.EndError + angleCost + lineCost;
        }

        private static T2LineMetric ComputeT2LineMetric((string? ParticipantId, string? SessionId, string? TrialId, string? TaskId) metadata,
                                                        T2LineReference reference,
                                                        T1bUserStroke? userStroke,
                                                        double? completionTimeMs,
                                                        double endpointTolerancePx,
                                                        double lineTolerancePx,
                                                        double requiredPercentPointsWithinTolerance,
                                                        double minimumStraightnessRatio,
                                                        double minimumLengthRatio,
                                                        double maximumLengthRatio,
                                                        double maximumAngleErrorDeg,
                                                        List<string> warnings)
        {
            double targetLength = Distance(reference.Start.X, reference.Start.Y, reference.End.X, reference.End.Y);
            double targetAngle = ComputeAngleDeg(reference.Start.X, reference.Start.Y, reference.End.X, reference.End.Y);

            if (targetLength <= 0)
            {
                warnings.Add($"T2 {reference.Orientation} reference line has zero length; metrics left blank.");
                targetLength = double.NaN;
            }

            Point2d? userStart = userStroke?.Points.Count > 0 ? userStroke.Points[0] : null;
            Point2d? userEnd = userStroke?.Points.Count > 0 ? userStroke.Points[^1] : null;
            double? userPathLength = userStroke != null ? compute_polyline_length(userStroke.Points) : null;
            double? userDirectLength = userStart.HasValue && userEnd.HasValue ? Distance(userStart.Value.X, userStart.Value.Y, userEnd.Value.X, userEnd.Value.Y) : null;
            double? userAngle = userStart.HasValue && userEnd.HasValue ? ComputeAngleDeg(userStart.Value.X, userStart.Value.Y, userEnd.Value.X, userEnd.Value.Y) : null;
            double? straightnessRatio = userDirectLength.HasValue && userPathLength > 0 ? userDirectLength / userPathLength : null;
            double? lengthRatio = userPathLength.HasValue && targetLength > 0 ? userPathLength / targetLength : null;
            double? meanSpeed = userPathLength.HasValue && completionTimeMs > 0 ? userPathLength / (completionTimeMs.Value / 1000.0) : null;

            string? direction = null;
            double? startError = null;
            double? endError = null;
            double? meanEndpointError = null;
            double? maxEndpointError = null;
            double? meanDeviation = null;
            double? medianDeviation = null;
            double? rmseDeviation = null;
            double? maxDeviation = null;
            double? percentWithin = null;
            double? angleError = null;
            bool? successful = null;

            if (userStroke != null && userStart.HasValue && userEnd.HasValue && targetLength > 0)
            {
                var endpointErrors = compute_orientation_invariant_endpoint_errors(userStart.Value, userEnd.Value, reference.Start, reference.End);
                direction = endpointErrors.Direction;
                startError = endpointErrors.StartError;
                endError = endpointErrors.EndError;
                meanEndpointError = (startError + endError) / 2.0;
                maxEndpointError = Math.Max(startError.Value, endError.Value);

                var deviations = userStroke.Points.Select(p => compute_point_to_segment_distance(p, reference.Start, reference.End)).ToList();
                meanDeviation = Mean(deviations);
                medianDeviation = Median(deviations);
                rmseDeviation = deviations.Count > 0 ? Math.Sqrt(deviations.Average(d => d * d)) : null;
                maxDeviation = deviations.Count > 0 ? deviations.Max() : null;
                percentWithin = deviations.Count > 0 ? deviations.Count(d => d <= lineTolerancePx) * 100.0 / deviations.Count : null;
                angleError = userAngle.HasValue ? compute_direction_invariant_angle_error(userAngle.Value, targetAngle) : null;

                successful =
                    startError <= endpointTolerancePx &&
                    endError <= endpointTolerancePx &&
                    percentWithin >= requiredPercentPointsWithinTolerance &&
                    straightnessRatio >= minimumStraightnessRatio &&
                    lengthRatio >= minimumLengthRatio &&
                    lengthRatio <= maximumLengthRatio &&
                    angleError <= maximumAngleErrorDeg;

                if (userDirectLength <= 1.0)
                    warnings.Add($"T2 {reference.Orientation} user stroke is zero-length or near zero-length.");
            }

            return new T2LineMetric(
                ParticipantId: metadata.ParticipantId,
                SessionId: metadata.SessionId,
                TrialId: metadata.TrialId,
                TaskId: metadata.TaskId,
                LineId: reference.LineId,
                Orientation: reference.Orientation,
                TargetStartX: reference.Start.X,
                TargetStartY: reference.Start.Y,
                TargetEndX: reference.End.X,
                TargetEndY: reference.End.Y,
                TargetLengthPx: double.IsNaN(targetLength) ? null : targetLength,
                TargetAngleDeg: double.IsNaN(targetLength) ? null : targetAngle,
                UserStartX: userStart?.X,
                UserStartY: userStart?.Y,
                UserEndX: userEnd?.X,
                UserEndY: userEnd?.Y,
                UserPathLengthPx: userPathLength,
                UserDirectLengthPx: userDirectLength,
                UserAngleDeg: userAngle,
                NSamplePoints: userStroke?.Points.Count,
                EndpointMatchDirection: direction,
                StartErrorPx: startError,
                EndErrorPx: endError,
                MeanEndpointErrorPx: meanEndpointError,
                MaxEndpointErrorPx: maxEndpointError,
                MeanLineDeviationPx: meanDeviation,
                MedianLineDeviationPx: medianDeviation,
                RmseLineDeviationPx: rmseDeviation,
                MaxLineDeviationPx: maxDeviation,
                PercentPointsWithinTolerance: percentWithin,
                StraightnessRatio: straightnessRatio,
                LengthRatio: lengthRatio,
                AngleErrorDeg: angleError,
                CompletionTimeMs: completionTimeMs,
                MeanSpeedPxPerS: meanSpeed,
                Successful: successful);
        }

        private static void WriteT2AnalysisOutputs(string linesPath,
                                                   string summaryPath,
                                                   string diagnosticPath,
                                                   Mat baseImage,
                                                   IReadOnlyList<T2LineReference> references,
                                                   IReadOnlyList<T1bUserStroke> userStrokes,
                                                   IReadOnlyDictionary<int, int> matches,
                                                   IReadOnlyList<T2LineMetric> rows,
                                                   bool includeDiagnosticLabels)
        {
            WriteT2LinesCsv(linesPath, rows);
            WriteT2SummaryCsv(summaryPath, rows);
            WriteT2DiagnosticImage(diagnosticPath, baseImage, references, userStrokes, matches, rows, includeDiagnosticLabels);
        }

        private static void WriteT2LinesCsv(string path, IReadOnlyList<T2LineMetric> rows)
        {
            using var writer = new StreamWriter(path);
            writer.WriteLine("participant_id,session_id,trial_id,task_id,line_id,orientation,target_start_x,target_start_y,target_end_x,target_end_y,target_length_px,target_angle_deg,user_start_x,user_start_y,user_end_x,user_end_y,user_path_length_px,user_direct_length_px,user_angle_deg,n_sample_points,endpoint_match_direction,start_error_px,end_error_px,mean_endpoint_error_px,max_endpoint_error_px,mean_line_deviation_px,median_line_deviation_px,rmse_line_deviation_px,max_line_deviation_px,percent_points_within_tolerance,straightness_ratio,length_ratio,angle_error_deg,completion_time_ms,mean_speed_px_per_s,successful");

            foreach (var row in rows)
            {
                writer.WriteLine(string.Join(",", new[]
                {
                    Csv(row.ParticipantId),
                    Csv(row.SessionId),
                    Csv(row.TrialId),
                    Csv(row.TaskId),
                    Csv(row.LineId),
                    Csv(row.Orientation),
                    Csv(row.TargetStartX),
                    Csv(row.TargetStartY),
                    Csv(row.TargetEndX),
                    Csv(row.TargetEndY),
                    Csv(row.TargetLengthPx),
                    Csv(row.TargetAngleDeg),
                    Csv(row.UserStartX),
                    Csv(row.UserStartY),
                    Csv(row.UserEndX),
                    Csv(row.UserEndY),
                    Csv(row.UserPathLengthPx),
                    Csv(row.UserDirectLengthPx),
                    Csv(row.UserAngleDeg),
                    Csv(row.NSamplePoints),
                    Csv(row.EndpointMatchDirection),
                    Csv(row.StartErrorPx),
                    Csv(row.EndErrorPx),
                    Csv(row.MeanEndpointErrorPx),
                    Csv(row.MaxEndpointErrorPx),
                    Csv(row.MeanLineDeviationPx),
                    Csv(row.MedianLineDeviationPx),
                    Csv(row.RmseLineDeviationPx),
                    Csv(row.MaxLineDeviationPx),
                    Csv(row.PercentPointsWithinTolerance),
                    Csv(row.StraightnessRatio),
                    Csv(row.LengthRatio),
                    Csv(row.AngleErrorDeg),
                    Csv(row.CompletionTimeMs),
                    Csv(row.MeanSpeedPxPerS),
                    Csv(row.Successful)
                }));
            }
        }

        private static void WriteT2SummaryCsv(string path, IReadOnlyList<T2LineMetric> rows)
        {
            using var writer = new StreamWriter(path);
            writer.WriteLine("participant_id,task_id,n_lines,n_successful,success_rate,mean_endpoint_error_px,sd_endpoint_error_px,mean_line_deviation_px,sd_line_deviation_px,mean_rmse_line_deviation_px,mean_percent_points_within_tolerance,mean_straightness_ratio,mean_length_ratio,sd_length_ratio,mean_angle_error_deg,sd_angle_error_deg,mean_completion_time_ms,mean_speed_px_per_s");

            foreach (var group in rows.GroupBy(r => new { ParticipantId = r.ParticipantId ?? string.Empty, TaskId = r.TaskId ?? string.Empty }))
            {
                var endpointErrors = Values(group.Select(r => r.MeanEndpointErrorPx));
                var lineDevs = Values(group.Select(r => r.MeanLineDeviationPx));
                var rmses = Values(group.Select(r => r.RmseLineDeviationPx));
                var within = Values(group.Select(r => r.PercentPointsWithinTolerance));
                var straightness = Values(group.Select(r => r.StraightnessRatio));
                var lengths = Values(group.Select(r => r.LengthRatio));
                var angles = Values(group.Select(r => r.AngleErrorDeg));
                var times = Values(group.Select(r => r.CompletionTimeMs)).Distinct().ToList();
                var speeds = Values(group.Select(r => r.MeanSpeedPxPerS));
                int n = group.Count(r => r.LineId.HasValue);
                int successful = group.Count(r => r.Successful == true);

                writer.WriteLine(string.Join(",", new[]
                {
                    Csv(group.Key.ParticipantId),
                    Csv(group.Key.TaskId),
                    Csv(n),
                    Csv(successful),
                    Csv(n > 0 ? successful / (double)n : null),
                    Csv(Mean(endpointErrors)),
                    Csv(StandardDeviation(endpointErrors)),
                    Csv(Mean(lineDevs)),
                    Csv(StandardDeviation(lineDevs)),
                    Csv(Mean(rmses)),
                    Csv(Mean(within)),
                    Csv(Mean(straightness)),
                    Csv(Mean(lengths)),
                    Csv(StandardDeviation(lengths)),
                    Csv(Mean(angles)),
                    Csv(StandardDeviation(angles)),
                    Csv(Mean(times)),
                    Csv(Mean(speeds))
                }));
            }

            static List<double> Values(IEnumerable<double?> source)
                => source.Where(v => v.HasValue).Select(v => v!.Value).ToList();
        }

        private static void WriteT2DiagnosticImage(string path,
                                                   Mat baseImage,
                                                   IReadOnlyList<T2LineReference> references,
                                                   IReadOnlyList<T1bUserStroke> userStrokes,
                                                   IReadOnlyDictionary<int, int> matches,
                                                   IReadOnlyList<T2LineMetric> rows,
                                                   bool includeDiagnosticLabels)
        {
            using var diagnostic = baseImage.Clone();
            foreach (var reference in references)
            {
                Cv2.Line(
                    diagnostic,
                    new Point((int)Math.Round(reference.Start.X), (int)Math.Round(reference.Start.Y)),
                    new Point((int)Math.Round(reference.End.X), (int)Math.Round(reference.End.Y)),
                    new Scalar(0, 0, 255),
                    1,
                    LineTypes.AntiAlias);
                Cv2.Circle(diagnostic, new Point((int)Math.Round(reference.Start.X), (int)Math.Round(reference.Start.Y)), 4, new Scalar(255, 0, 0), -1, LineTypes.AntiAlias);
                Cv2.Circle(diagnostic, new Point((int)Math.Round(reference.End.X), (int)Math.Round(reference.End.Y)), 4, new Scalar(0, 128, 255), -1, LineTypes.AntiAlias);
            }

            foreach (var stroke in userStrokes)
            {
                DrawPolyline(diagnostic, stroke.Points, Scalar.Black, 2);
                if (stroke.Points.Count > 0)
                {
                    Cv2.Circle(diagnostic, new Point((int)Math.Round(stroke.Points[0].X), (int)Math.Round(stroke.Points[0].Y)), 3, new Scalar(255, 0, 255), -1, LineTypes.AntiAlias);
                    Cv2.Circle(diagnostic, new Point((int)Math.Round(stroke.Points[^1].X), (int)Math.Round(stroke.Points[^1].Y)), 3, new Scalar(0, 160, 0), -1, LineTypes.AntiAlias);
                }
            }

            if (includeDiagnosticLabels)
            {
                foreach (var row in rows)
                {
                    if (!row.LineId.HasValue)
                        continue;

                    var reference = references.FirstOrDefault(r => r.LineId == row.LineId.Value);
                    if (reference == null)
                        continue;

                    var labelPoint = new Point(
                        (int)Math.Round((reference.Start.X + reference.End.X) / 2.0),
                        (int)Math.Round((reference.Start.Y + reference.End.Y) / 2.0));
                    string label = row.Orientation ?? row.LineId.Value.ToString(CultureInfo.InvariantCulture);
                    if (row.MeanLineDeviationPx.HasValue)
                        label += $" dev={row.MeanLineDeviationPx.Value:0.0}px";
                    if (row.StraightnessRatio.HasValue)
                        label += $" str={row.StraightnessRatio.Value:0.00}";
                    if (row.AngleErrorDeg.HasValue)
                        label += $" angle={row.AngleErrorDeg.Value:0.0}";
                    if (row.Successful.HasValue)
                        label += $" {(row.Successful.Value ? "ok" : "fail")}";

                    Cv2.PutText(diagnostic, label, labelPoint + new Point(6, -6), HersheyFonts.HersheySimplex, 0.34, Scalar.Black, 1, LineTypes.AntiAlias);
                }
            }

            Cv2.ImEncode(".png", diagnostic, out byte[] buffer);
            File.WriteAllBytes(path, buffer);
        }

        private static TracingAnalysisResult analyse_t3a_path_tracing(string sessionPath,
                                                                      string outputImagePath,
                                                                      bool includeDiagnosticLabels = false,
                                                                      double pathTolerancePx = 8.0,
                                                                      double requiredPercentUserPointsWithinTolerance = 80.0,
                                                                      double requiredPercentReferenceCovered = 80.0,
                                                                      double minimumLengthRatio = 0.70,
                                                                      double maximumLengthRatio = 1.50,
                                                                      CancellationToken ct = default)
        {
            return AnalyseTracingTask(
                taskId: "T3a",
                sessionPath: sessionPath,
                outputImagePath: outputImagePath,
                includeDiagnosticLabels: includeDiagnosticLabels,
                pathTolerancePx: pathTolerancePx,
                requiredPercentUserPointsWithinTolerance: requiredPercentUserPointsWithinTolerance,
                requiredPercentReferenceCovered: requiredPercentReferenceCovered,
                minimumLengthRatio: minimumLengthRatio,
                maximumLengthRatio: maximumLengthRatio,
                ct: ct);
        }

        private static TracingAnalysisResult analyse_t3b_shape_tracing(string sessionPath,
                                                                      string outputImagePath,
                                                                      bool includeDiagnosticLabels = false,
                                                                      double pathTolerancePx = 8.0,
                                                                      double requiredPercentUserPointsWithinTolerance = 80.0,
                                                                      double requiredPercentReferenceCovered = 80.0,
                                                                      double minimumLengthRatio = 0.70,
                                                                      double maximumLengthRatio = 1.50,
                                                                      CancellationToken ct = default)
        {
            return AnalyseTracingTask(
                taskId: "T3b",
                sessionPath: sessionPath,
                outputImagePath: outputImagePath,
                includeDiagnosticLabels: includeDiagnosticLabels,
                pathTolerancePx: pathTolerancePx,
                requiredPercentUserPointsWithinTolerance: requiredPercentUserPointsWithinTolerance,
                requiredPercentReferenceCovered: requiredPercentReferenceCovered,
                minimumLengthRatio: minimumLengthRatio,
                maximumLengthRatio: maximumLengthRatio,
                ct: ct);
        }

        private static TracingAnalysisResult AnalyseTracingTask(string taskId,
                                                                string sessionPath,
                                                                string outputImagePath,
                                                                bool includeDiagnosticLabels,
                                                                double pathTolerancePx,
                                                                double requiredPercentUserPointsWithinTolerance,
                                                                double requiredPercentReferenceCovered,
                                                                double minimumLengthRatio,
                                                                double maximumLengthRatio,
                                                                CancellationToken ct)
        {
            var warnings = new List<string>();
            using var session = SessionManager.OpenExisting(sessionPath, aviPathOverride: null, writeable: false);
            using var baseImage = LoadTaskReplayBaseImage(sessionPath);

            var references = extract_reference_polylines(sessionPath, taskId, warnings);
            var userStrokes = segment_user_strokes(session, baseImage.Width, baseImage.Height, outputImagePath, warnings, ct);
            var matches = match_user_strokes_to_references(references, userStrokes, warnings);
            var metadata = ExtractAnalysisMetadata(sessionPath, taskId);
            double? completionTimeMs = ComputeT1bCompletionTimeMs(userStrokes);

            var rows = new List<TracingMetric>(Math.Max(references.Count, userStrokes.Count));
            foreach (var reference in references)
            {
                ct.ThrowIfCancellationRequested();
                T1bUserStroke? userStroke = matches.TryGetValue(reference.Id, out int strokeIndex) ? userStrokes[strokeIndex] : null;
                if (userStroke == null)
                    warnings.Add($"{taskId} reference {reference.Id} has no matching user stroke.");

                rows.Add(CreateTracingMetricRow(
                    metadata,
                    taskId,
                    reference,
                    userStroke,
                    completionTimeMs,
                    pathTolerancePx,
                    requiredPercentUserPointsWithinTolerance,
                    requiredPercentReferenceCovered,
                    minimumLengthRatio,
                    maximumLengthRatio));
            }

            var matchedStrokeIndices = matches.Values.ToHashSet();
            foreach (var extraStroke in userStrokes.Where((_, i) => !matchedStrokeIndices.Contains(i)))
            {
                warnings.Add($"{taskId} user stroke {extraStroke.StrokeId} has no matching reference.");
                rows.Add(CreateTracingMetricRow(metadata, taskId, null, extraStroke, completionTimeMs, pathTolerancePx, requiredPercentUserPointsWithinTolerance, requiredPercentReferenceCovered, minimumLengthRatio, maximumLengthRatio));
            }

            var resultsDir = GetAnalysisResultsDirectory(outputImagePath, sessionPath);
            Directory.CreateDirectory(resultsDir);
            string baseName = Path.GetFileNameWithoutExtension(sessionPath);
            string suffix = taskId.Equals("T3a", StringComparison.OrdinalIgnoreCase) ? "t3a_paths" : "t3b_shapes";
            string detailPath = Path.Combine(resultsDir, $"{baseName}_{suffix}.csv");
            string summaryPath = Path.Combine(resultsDir, $"{baseName}_{(taskId.Equals("T3a", StringComparison.OrdinalIgnoreCase) ? "t3a" : "t3b")}_summary.csv");
            string diagnosticPath = Path.Combine(resultsDir, $"{baseName}_{(taskId.Equals("T3a", StringComparison.OrdinalIgnoreCase) ? "t3a" : "t3b")}_diagnostic{(includeDiagnosticLabels ? "_labelled" : string.Empty)}.png");

            write_tracing_analysis_outputs(detailPath, summaryPath, diagnosticPath, baseImage, references, userStrokes, matches, rows, includeDiagnosticLabels);

            string? warningsPath = null;
            if (warnings.Count > 0)
            {
                warningsPath = Path.Combine(resultsDir, $"{baseName}_{(taskId.Equals("T3a", StringComparison.OrdinalIgnoreCase) ? "t3a" : "t3b")}_warnings.txt");
                File.WriteAllLines(warningsPath, warnings);
            }

            return new TracingAnalysisResult(detailPath, summaryPath, diagnosticPath, warningsPath, warnings);
        }

        private static List<TracingReference> extract_reference_polylines(string sessionPath, string taskId, List<string> warnings)
        {
            string? templatePath = FindTaskTemplatePath(sessionPath, taskId);
            if (string.IsNullOrWhiteSpace(templatePath))
            {
                warnings.Add($"Could not find task template PNG for '{taskId}'.");
                return new List<TracingReference>();
            }

            using var raw = Cv2.ImRead(templatePath, ImreadModes.Unchanged);
            if (raw == null || raw.Empty())
            {
                warnings.Add($"Could not read task template PNG: {templatePath}");
                return new List<TracingReference>();
            }

            using var flipped = new Mat();
            Cv2.Flip(raw, flipped, FlipMode.XY);
            using var bgr = new Mat();
            if (flipped.Channels() == 4)
                Cv2.CvtColor(flipped, bgr, ColorConversionCodes.BGRA2BGR);
            else if (flipped.Channels() == 1)
                Cv2.CvtColor(flipped, bgr, ColorConversionCodes.GRAY2BGR);
            else
                flipped.CopyTo(bgr);

            using var mask = BuildTracingReferenceMask(bgr, flipped, taskId);
            using var labels = new Mat();
            using var stats = new Mat();
            using var centroids = new Mat();
            int n = Cv2.ConnectedComponentsWithStats(mask, labels, stats, centroids, PixelConnectivity.Connectivity8);

            var references = new List<TracingReference>();
            for (int label = 1; label < n; label++)
            {
                int area = stats.Get<int>(label, (int)ConnectedComponentsTypes.Area);
                if (area < 20)
                    continue;

                int x = stats.Get<int>(label, (int)ConnectedComponentsTypes.Left);
                int y = stats.Get<int>(label, (int)ConnectedComponentsTypes.Top);
                int w = stats.Get<int>(label, (int)ConnectedComponentsTypes.Width);
                int h = stats.Get<int>(label, (int)ConnectedComponentsTypes.Height);
                var pixels = GetComponentPixels(labels, label);
                IReadOnlyList<Point2d> points = taskId.Equals("T3b", StringComparison.OrdinalIgnoreCase)
                    ? ExtractClosedContourPolyline(mask, label, new Rect(x, y, w, h))
                    : BuildCenterlinePolyline(pixels);

                if (points.Count < 2)
                {
                    warnings.Add($"{taskId} reference component {label} has too few points; skipped.");
                    continue;
                }

                var bbox = new Rect2d(x, y, w, h);
                double guideRadius = Math.Max(1.0, Math.Sqrt(area / Math.PI) / Math.Max(1.0, points.Count));
                string referenceType = taskId.Equals("T3b", StringComparison.OrdinalIgnoreCase)
                    ? ClassifyT3bShape(points, bbox)
                    : ClassifyT3aPath(points, bbox);

                references.Add(new TracingReference(
                    Id: references.Count + 1,
                    ReferenceType: referenceType,
                    Points: ResamplePolyline(points, 2.0, taskId.Equals("T3b", StringComparison.OrdinalIgnoreCase)),
                    BoundingBox: bbox,
                    IsClosed: taskId.Equals("T3b", StringComparison.OrdinalIgnoreCase),
                    GuideRadiusPx: guideRadius));
            }

            if (references.Count == 0)
                warnings.Add($"No {taskId} reference paths/shapes were detected.");

            if (taskId.Equals("T3a", StringComparison.OrdinalIgnoreCase))
                references = BuildT3aCorridorReferences(references, warnings);

            var orderedReferences = references
                .OrderBy(r => r.BoundingBox.Y)
                .ThenBy(r => r.BoundingBox.X)
                .Select((r, i) => r with { Id = i + 1 })
                .ToList();

            if (taskId.Equals("T3b", StringComparison.OrdinalIgnoreCase))
                return ApplyKnownT3bShapeLabels(orderedReferences, warnings);

            return orderedReferences;
        }

        private static List<TracingReference> BuildT3aCorridorReferences(IReadOnlyList<TracingReference> guideLines, List<string> warnings)
        {
            if (guideLines.Count < 2)
                return guideLines.ToList();

            var unused = guideLines.ToList();
            var corridors = new List<TracingReference>();
            while (unused.Count >= 2)
            {
                var first = unused[0];
                unused.RemoveAt(0);

                int bestIndex = -1;
                double bestCost = double.PositiveInfinity;
                for (int i = 0; i < unused.Count; i++)
                {
                    double cost = ComputeT3aGuidePairCost(first, unused[i]);
                    if (cost < bestCost)
                    {
                        bestCost = cost;
                        bestIndex = i;
                    }
                }

                if (bestIndex < 0)
                {
                    warnings.Add($"T3a guide line {first.Id} could not be paired into a corridor.");
                    corridors.Add(first);
                    continue;
                }

                var second = unused[bestIndex];
                unused.RemoveAt(bestIndex);
                corridors.Add(CreateT3aCorridorReference(corridors.Count + 1, first, second));
            }

            if (unused.Count > 0)
            {
                warnings.Add($"T3a guide line {unused[0].Id} was left unpaired; keeping it as a fallback reference.");
                corridors.Add(unused[0] with { Id = corridors.Count + 1 });
            }

            return corridors;
        }

        private static double ComputeT3aGuidePairCost(TracingReference a, TracingReference b)
        {
            var deviation = compute_bidirectional_polyline_deviation(a.Points, b.Points);
            double lengthA = compute_polyline_length(a.Points);
            double lengthB = compute_polyline_length(b.Points);
            double lengthCost = Math.Abs(lengthA - lengthB);
            double angleCost = compute_direction_invariant_angle_error(GetPolylineAngleDeg(a.Points), GetPolylineAngleDeg(b.Points)) * 2.0;
            double overlapCost = (1.0 - ComputeBoundingBoxSimilarity(a.BoundingBox, b.BoundingBox)) * 20.0;
            return deviation.MeanBidirectionalDeviation + lengthCost * 0.15 + angleCost + overlapCost;
        }

        private static TracingReference CreateT3aCorridorReference(int id, TracingReference first, TracingReference second)
        {
            var a = ResamplePolyline(first.Points, 2.0, closed: false).ToList();
            var b = ResamplePolyline(second.Points, 2.0, closed: false).ToList();
            if (a.Count == 0 || b.Count == 0)
                return first with { Id = id };

            if (Distance(a[0].X, a[0].Y, b[0].X, b[0].Y) + Distance(a[^1].X, a[^1].Y, b[^1].X, b[^1].Y) >
                Distance(a[0].X, a[0].Y, b[^1].X, b[^1].Y) + Distance(a[^1].X, a[^1].Y, b[0].X, b[0].Y))
            {
                b.Reverse();
            }

            int n = Math.Min(a.Count, b.Count);
            var centerline = new List<Point2d>(n);
            var separations = new List<double>(n);
            for (int i = 0; i < n; i++)
            {
                centerline.Add(new Point2d((a[i].X + b[i].X) / 2.0, (a[i].Y + b[i].Y) / 2.0));
                separations.Add(Distance(a[i].X, a[i].Y, b[i].X, b[i].Y));
            }

            var bbox = Union(first.BoundingBox, second.BoundingBox);
            double halfWidth = Math.Max(1.0, (Median(separations) ?? 0.0) / 2.0);
            return new TracingReference(
                Id: id,
                ReferenceType: ClassifyT3aPath(centerline, bbox),
                Points: centerline,
                BoundingBox: bbox,
                IsClosed: false,
                GuideRadiusPx: halfWidth);
        }

        private static double GetPolylineAngleDeg(IReadOnlyList<Point2d> points)
        {
            if (points.Count < 2)
                return 0.0;
            return ComputeAngleDeg(points[0].X, points[0].Y, points[^1].X, points[^1].Y);
        }

        private static Rect2d Union(Rect2d a, Rect2d b)
        {
            double left = Math.Min(a.X, b.X);
            double top = Math.Min(a.Y, b.Y);
            double right = Math.Max(a.X + a.Width, b.X + b.Width);
            double bottom = Math.Max(a.Y + a.Height, b.Y + b.Height);
            return new Rect2d(left, top, right - left, bottom - top);
        }

        private static List<TracingReference> ApplyKnownT3bShapeLabels(IReadOnlyList<TracingReference> references, List<string> warnings)
        {
            string[] knownShapeOrder = { "square", "star", "triangle", "circle", "infinity" };
            if (references.Count != knownShapeOrder.Length)
                warnings.Add($"Expected {knownShapeOrder.Length} T3b reference shapes, but detected {references.Count}; applying known labels to detected order where possible.");

            return references
                .Select((reference, index) => reference with
                {
                    Id = index + 1,
                    ReferenceType = index < knownShapeOrder.Length ? knownShapeOrder[index] : reference.ReferenceType
                })
                .ToList();
        }

        private static Mat BuildTracingReferenceMask(Mat bgr, Mat sourceWithAlpha, string taskId)
        {
            var mask = new Mat(bgr.Rows, bgr.Cols, MatType.CV_8UC1, Scalar.All(0));
            for (int y = 0; y < bgr.Rows; y++)
            {
                for (int x = 0; x < bgr.Cols; x++)
                {
                    byte alpha = sourceWithAlpha.Channels() == 4 ? sourceWithAlpha.At<Vec4b>(y, x).Item3 : (byte)255;
                    if (alpha == 0)
                        continue;

                    var c = bgr.At<Vec3b>(y, x);
                    bool keep = taskId.Equals("T3b", StringComparison.OrdinalIgnoreCase)
                        ? c.Item2 > 120 && c.Item1 < 120 && c.Item0 < 120
                        : c.Item0 > 120 && c.Item1 > 80 && c.Item2 < 120;
                    if (keep)
                        mask.Set(y, x, 255);
                }
            }

            using var kernel = Cv2.GetStructuringElement(MorphShapes.Ellipse, new Size(3, 3));
            Cv2.MorphologyEx(mask, mask, MorphTypes.Close, kernel);
            return mask;
        }

        private static List<Point2d> GetComponentPixels(Mat labels, int componentLabel)
        {
            var pixels = new List<Point2d>();
            for (int y = 0; y < labels.Rows; y++)
            {
                for (int x = 0; x < labels.Cols; x++)
                {
                    if (labels.Get<int>(y, x) == componentLabel)
                        pixels.Add(new Point2d(x, y));
                }
            }

            return pixels;
        }

        private static IReadOnlyList<Point2d> BuildCenterlinePolyline(IReadOnlyList<Point2d> pixels)
        {
            if (pixels.Count == 0)
                return Array.Empty<Point2d>();

            double minX = pixels.Min(p => p.X);
            double maxX = pixels.Max(p => p.X);
            double minY = pixels.Min(p => p.Y);
            double maxY = pixels.Max(p => p.Y);
            bool sortByX = (maxX - minX) >= (maxY - minY);

            return pixels
                .GroupBy(p => sortByX ? (int)Math.Round(p.X) : (int)Math.Round(p.Y))
                .OrderBy(g => g.Key)
                .Select(g => sortByX
                    ? new Point2d(g.Key, g.Average(p => p.Y))
                    : new Point2d(g.Average(p => p.X), g.Key))
                .ToList();
        }

        private static IReadOnlyList<Point2d> ExtractClosedContourPolyline(Mat fullMask, int componentLabel, Rect bbox)
        {
            using var component = new Mat(fullMask.Rows, fullMask.Cols, MatType.CV_8UC1, Scalar.All(0));
            using var labels = new Mat();
            Cv2.ConnectedComponents(fullMask, labels, PixelConnectivity.Connectivity8);
            for (int y = bbox.Top; y < bbox.Bottom; y++)
            {
                for (int x = bbox.Left; x < bbox.Right; x++)
                {
                    if (labels.Get<int>(y, x) == componentLabel)
                        component.Set(y, x, 255);
                }
            }

            Cv2.FindContours(component, out Point[][] contours, out _, RetrievalModes.External, ContourApproximationModes.ApproxNone);
            var contour = contours.OrderByDescending(c => Cv2.ContourArea(c)).FirstOrDefault();
            if (contour == null || contour.Length == 0)
                return Array.Empty<Point2d>();

            var points = contour.Select(p => new Point2d(p.X, p.Y)).ToList();
            points.Add(points[0]);
            return points;
        }

        private static string ClassifyT3aPath(IReadOnlyList<Point2d> points, Rect2d bbox)
        {
            if (bbox.Height > bbox.Width * 2.0) return "vertical_corridor";
            if (bbox.Width > bbox.Height * 2.0)
            {
                int turns = CountDirectionChanges(points);
                if (turns >= 8) return "wavy";
                if (turns >= 3) return "zigzag";
                return "horizontal";
            }
            return "path";
        }

        private static string ClassifyT3bShape(IReadOnlyList<Point2d> points, Rect2d bbox)
        {
            double area = Math.Abs(PolygonArea(points));
            double perimeter = compute_polyline_length(points);
            double circularity = perimeter > 0 ? 4.0 * Math.PI * area / (perimeter * perimeter) : 0;
            if (circularity > 0.72) return "circle";
            if (bbox.Width / Math.Max(1.0, bbox.Height) > 1.6) return "figure8";
            if (points.Count < 80) return "triangle";
            if (circularity < 0.35) return "star";
            return "square";
        }

        private static int CountDirectionChanges(IReadOnlyList<Point2d> points)
        {
            int changes = 0;
            double? previousSign = null;
            for (int i = 2; i < points.Count; i++)
            {
                double dy = points[i].Y - points[i - 1].Y;
                double sign = Math.Sign(dy);
                if (Math.Abs(sign) < double.Epsilon)
                    continue;
                if (previousSign.HasValue && Math.Abs(sign - previousSign.Value) > double.Epsilon)
                    changes++;
                previousSign = sign;
            }
            return changes;
        }

        private static IReadOnlyList<T1bUserStroke> segment_user_strokes(SessionManager session,
                                                                         int outputWidth,
                                                                         int outputHeight,
                                                                         string? fallbackImagePath,
                                                                         List<string> warnings,
                                                                         CancellationToken ct)
        {
            var strokes = ExtractT1bUserStrokes(session, outputWidth, outputHeight, warnings, ct);
            if (strokes.Count > 0)
                return strokes;

            if (string.IsNullOrWhiteSpace(fallbackImagePath) || !File.Exists(fallbackImagePath))
            {
                warnings.Add("No rendered image was available for fallback black-stroke extraction.");
                return strokes;
            }

            warnings.Add("Falling back to image-based extraction of black user strokes because no pressure-positive raw samples were found.");
            return ExtractUserStrokesFromBlackPixels(fallbackImagePath, warnings);
        }

        private static IReadOnlyList<T1bUserStroke> ExtractUserStrokesFromBlackPixels(string imagePath, List<string> warnings)
        {
            using var image = Cv2.ImRead(imagePath, ImreadModes.Color);
            if (image.Empty())
            {
                warnings.Add($"Could not read rendered image '{imagePath}' for fallback stroke extraction.");
                return Array.Empty<T1bUserStroke>();
            }

            using var mask = new Mat(image.Rows, image.Cols, MatType.CV_8UC1, Scalar.All(0));
            for (int y = 0; y < image.Rows; y++)
            {
                for (int x = 0; x < image.Cols; x++)
                {
                    var c = image.Get<Vec3b>(y, x);
                    if (c.Item0 < 70 && c.Item1 < 70 && c.Item2 < 70)
                        mask.Set(y, x, 255);
                }
            }

            using var labels = new Mat();
            using var stats = new Mat();
            using var centroids = new Mat();
            Cv2.ConnectedComponentsWithStats(mask, labels, stats, centroids, PixelConnectivity.Connectivity8);

            var strokes = new List<T1bUserStroke>();
            for (int label = 1; label < stats.Rows; label++)
            {
                int area = stats.Get<int>(label, (int)ConnectedComponentsTypes.Area);
                if (area < 8)
                    continue;

                var pixels = GetComponentPixels(labels, label);
                var polyline = OrderStrokePixelsByNearestNeighbour(pixels);
                if (polyline.Count > 1)
                    strokes.Add(new T1bUserStroke(strokes.Count + 1, polyline, 0, 0));
            }

            if (strokes.Count == 0)
                warnings.Add("Fallback black-stroke extraction found no connected stroke components.");
            return strokes;
        }

        private static IReadOnlyList<Point2d> OrderStrokePixelsByNearestNeighbour(IReadOnlyList<Point2d> pixels)
        {
            if (pixels.Count <= 1)
                return pixels;

            var remaining = pixels.ToList();
            int startIndex = 0;
            for (int i = 1; i < remaining.Count; i++)
            {
                if (remaining[i].X + remaining[i].Y < remaining[startIndex].X + remaining[startIndex].Y)
                    startIndex = i;
            }

            var ordered = new List<Point2d>(remaining.Count) { remaining[startIndex] };
            remaining.RemoveAt(startIndex);
            while (remaining.Count > 0)
            {
                var last = ordered[^1];
                int nextIndex = 0;
                double nextDistance = double.PositiveInfinity;
                for (int i = 0; i < remaining.Count; i++)
                {
                    double distance = Distance(last.X, last.Y, remaining[i].X, remaining[i].Y);
                    if (distance < nextDistance)
                    {
                        nextDistance = distance;
                        nextIndex = i;
                    }
                }

                ordered.Add(remaining[nextIndex]);
                remaining.RemoveAt(nextIndex);
            }

            return ResamplePolyline(ordered, 2.0, closed: false);
        }

        private static Dictionary<int, int> match_user_strokes_to_references(IReadOnlyList<TracingReference> references, IReadOnlyList<T1bUserStroke> userStrokes, List<string> warnings)
        {
            var matches = new Dictionary<int, int>();
            var unused = new HashSet<int>(Enumerable.Range(0, userStrokes.Count));
            warnings.Add("No explicit T3 stroke mapping metadata was found; using combined bounding-box, bidirectional deviation, length, and overlap cost matching.");

            foreach (var reference in references)
            {
                int? bestIndex = null;
                double bestCost = double.PositiveInfinity;
                foreach (int strokeIndex in unused)
                {
                    double cost = ComputeTracingMatchCost(reference, userStrokes[strokeIndex]);
                    if (cost < bestCost)
                    {
                        bestCost = cost;
                        bestIndex = strokeIndex;
                    }
                }

                if (bestIndex.HasValue)
                {
                    matches[reference.Id] = bestIndex.Value;
                    unused.Remove(bestIndex.Value);
                }
            }

            return matches;
        }

        private static double ComputeTracingMatchCost(TracingReference reference, T1bUserStroke stroke)
        {
            if (stroke.Points.Count < 2 || reference.Points.Count < 2)
                return double.PositiveInfinity;

            var userBox = ComputeBoundingBox(stroke.Points);
            double centerCost = Distance(reference.BoundingBox.X + reference.BoundingBox.Width / 2.0, reference.BoundingBox.Y + reference.BoundingBox.Height / 2.0, userBox.X + userBox.Width / 2.0, userBox.Y + userBox.Height / 2.0);
            var deviation = compute_bidirectional_polyline_deviation(reference.Points, stroke.Points);
            double refLength = compute_polyline_length(reference.Points);
            double userLength = compute_polyline_length(stroke.Points);
            double lengthCost = refLength > 0 ? Math.Abs(userLength / refLength - 1.0) * 20.0 : 1000.0;
            double overlapCost = (1.0 - ComputeBoundingBoxSimilarity(reference.BoundingBox, userBox)) * 20.0;
            return centerCost + deviation.MeanBidirectionalDeviation + lengthCost + overlapCost;
        }

        private static TracingMetric CreateTracingMetricRow((string? ParticipantId, string? SessionId, string? TrialId, string? TaskId) metadata,
                                                            string taskId,
                                                            TracingReference? reference,
                                                            T1bUserStroke? userStroke,
                                                            double? completionTimeMs,
                                                            double pathTolerancePx,
                                                            double requiredPercentUserPointsWithinTolerance,
                                                            double requiredPercentReferenceCovered,
                                                            double minimumLengthRatio,
                                                            double maximumLengthRatio)
        {
            double? referenceLength = reference != null ? compute_polyline_length(reference.Points) : null;
            double? userLength = userStroke != null ? compute_polyline_length(userStroke.Points) : null;
            var userBox = userStroke != null && userStroke.Points.Count > 0 ? ComputeBoundingBox(userStroke.Points) : (Rect2d?)null;
            var userStart = userStroke?.Points.Count > 0 ? userStroke.Points[0] : (Point2d?)null;
            var userEnd = userStroke?.Points.Count > 0 ? userStroke.Points[^1] : (Point2d?)null;

            double? lengthRatio = referenceLength > 0 && userLength.HasValue ? userLength / referenceLength : null;
            double effectiveTolerancePx = taskId.Equals("T3a", StringComparison.OrdinalIgnoreCase) && reference != null
                ? Math.Max(pathTolerancePx, reference.GuideRadiusPx)
                : pathTolerancePx;
            List<double> userToRef = reference != null && userStroke != null
                ? userStroke.Points.Select(p => compute_point_to_polyline_distance(p, reference.Points)).ToList()
                : new List<double>();
            List<double> refToUser = reference != null && userStroke != null
                ? reference.Points.Select(p => compute_point_to_polyline_distance(p, userStroke.Points)).ToList()
                : new List<double>();

            var coverage = reference != null && userStroke != null
                ? compute_reference_coverage(reference.Points, userStroke.Points, effectiveTolerancePx)
                : (MeanReferenceToUserDeviation: (double?)null, PercentReferenceCovered: (double?)null, LongestUncoveredGapPx: (double?)null);

            var closed = reference != null && userStroke != null && reference.IsClosed
                ? compute_closed_shape_metrics(reference.Points, userStroke.Points)
                : (ClosureErrorPx: (double?)null, BoundingBoxSimilarity: (double?)null, CentroidOffsetPx: (double?)null, AreaDifferenceRatio: (double?)null);

            double? percentWithin = userToRef.Count > 0 ? userToRef.Count(d => d <= effectiveTolerancePx) * 100.0 / userToRef.Count : null;
            bool? successful = null;
            if (!taskId.Equals("T3a", StringComparison.OrdinalIgnoreCase))
            {
                successful =
                    percentWithin.HasValue &&
                    coverage.PercentReferenceCovered.HasValue &&
                    lengthRatio.HasValue &&
                    percentWithin >= requiredPercentUserPointsWithinTolerance &&
                    coverage.PercentReferenceCovered >= requiredPercentReferenceCovered &&
                    lengthRatio >= minimumLengthRatio &&
                    lengthRatio <= maximumLengthRatio;
            }

            double? percentInsideCorridor = null;
            int? corridorExits = null;
            double? meanDistanceFromCenterline = null;
            if (taskId.Equals("T3a", StringComparison.OrdinalIgnoreCase) && reference != null && userStroke != null)
            {
                percentInsideCorridor = userToRef.Count > 0 ? userToRef.Count(d => d <= effectiveTolerancePx) * 100.0 / userToRef.Count : null;
                corridorExits = CountCorridorExits(userToRef, effectiveTolerancePx);
                meanDistanceFromCenterline = Mean(userToRef);
                successful =
                    coverage.PercentReferenceCovered.HasValue &&
                    meanDistanceFromCenterline.HasValue &&
                    coverage.PercentReferenceCovered >= requiredPercentReferenceCovered &&
                    meanDistanceFromCenterline <= effectiveTolerancePx;
            }

            return new TracingMetric(
                ParticipantId: metadata.ParticipantId,
                SessionId: metadata.SessionId,
                TrialId: metadata.TrialId,
                TaskId: metadata.TaskId,
                PathOrShapeId: reference?.Id,
                ReferenceType: reference?.ReferenceType,
                ReferencePathLengthPx: referenceLength,
                UserPathLengthPx: userLength,
                LengthRatio: lengthRatio,
                NReferencePoints: reference?.Points.Count,
                NUserPoints: userStroke?.Points.Count,
                MeanPathDeviationPx: Mean(userToRef),
                MedianPathDeviationPx: Median(userToRef),
                SdPathDeviationPx: StandardDeviation(userToRef),
                RmsePathDeviationPx: userToRef.Count > 0 ? Math.Sqrt(userToRef.Average(d => d * d)) : null,
                MaxPathDeviationPx: userToRef.Count > 0 ? userToRef.Max() : null,
                PercentUserPointsWithinTolerance: percentWithin,
                MeanReferenceToUserDeviationPx: coverage.MeanReferenceToUserDeviation,
                PercentReferenceCovered: coverage.PercentReferenceCovered,
                LongestUncoveredGapPx: coverage.LongestUncoveredGapPx,
                ClosureErrorPx: closed.ClosureErrorPx,
                BoundingBoxSimilarity: closed.BoundingBoxSimilarity ?? (reference != null && userBox.HasValue ? ComputeBoundingBoxSimilarity(reference.BoundingBox, userBox.Value) : null),
                CentroidOffsetPx: closed.CentroidOffsetPx,
                AreaDifferenceRatio: closed.AreaDifferenceRatio,
                PercentPointsInsideCorridor: percentInsideCorridor,
                CorridorExits: corridorExits,
                MeanDistanceFromCorridorCenterlinePx: meanDistanceFromCenterline,
                CompletionTimeMs: completionTimeMs,
                Successful: successful);
        }

        private static double compute_polyline_length(IReadOnlyList<Point2d> points)
            => ComputePathLength(points);

        private static double compute_point_to_polyline_distance(Point2d point, IReadOnlyList<Point2d> polyline)
        {
            if (polyline.Count == 0)
                return double.NaN;
            if (polyline.Count == 1)
                return Distance(point.X, point.Y, polyline[0].X, polyline[0].Y);

            double min = double.PositiveInfinity;
            for (int i = 1; i < polyline.Count; i++)
                min = Math.Min(min, compute_point_to_segment_distance(point, polyline[i - 1], polyline[i]));
            return min;
        }

        private static (double MeanReferenceToUserDeviation, double MeanUserToReferenceDeviation, double MeanBidirectionalDeviation) compute_bidirectional_polyline_deviation(IReadOnlyList<Point2d> reference, IReadOnlyList<Point2d> user)
        {
            double refToUser = reference.Count > 0 ? reference.Average(p => compute_point_to_polyline_distance(p, user)) : double.NaN;
            double userToRef = user.Count > 0 ? user.Average(p => compute_point_to_polyline_distance(p, reference)) : double.NaN;
            return (refToUser, userToRef, (refToUser + userToRef) / 2.0);
        }

        private static (double? MeanReferenceToUserDeviation, double? PercentReferenceCovered, double? LongestUncoveredGapPx) compute_reference_coverage(IReadOnlyList<Point2d> reference, IReadOnlyList<Point2d> user, double coverageTolerancePx)
        {
            if (reference.Count == 0 || user.Count == 0)
                return (null, null, null);

            var distances = reference.Select(p => compute_point_to_polyline_distance(p, user)).ToList();
            double percentCovered = distances.Count(d => d <= coverageTolerancePx) * 100.0 / distances.Count;
            double longestGap = 0;
            double currentGap = 0;
            for (int i = 1; i < reference.Count; i++)
            {
                double segmentLength = Distance(reference[i - 1].X, reference[i - 1].Y, reference[i].X, reference[i].Y);
                if (distances[i] > coverageTolerancePx)
                    currentGap += segmentLength;
                else
                {
                    longestGap = Math.Max(longestGap, currentGap);
                    currentGap = 0;
                }
            }
            longestGap = Math.Max(longestGap, currentGap);
            return (distances.Average(), percentCovered, longestGap);
        }

        private static (double? ClosureErrorPx, double? BoundingBoxSimilarity, double? CentroidOffsetPx, double? AreaDifferenceRatio) compute_closed_shape_metrics(IReadOnlyList<Point2d> reference, IReadOnlyList<Point2d> user)
        {
            if (reference.Count < 3 || user.Count < 2)
                return (null, null, null, null);

            double closure = Distance(user[0].X, user[0].Y, user[^1].X, user[^1].Y);
            var referenceBox = ComputeBoundingBox(reference);
            var userBox = ComputeBoundingBox(user);
            double bboxSimilarity = ComputeBoundingBoxSimilarity(referenceBox, userBox);
            var referenceCentroid = ComputeCentroid(reference);
            var userCentroid = ComputeCentroid(user);
            double centroidOffset = Distance(referenceCentroid.X, referenceCentroid.Y, userCentroid.X, userCentroid.Y);
            double referenceArea = Math.Abs(PolygonArea(reference));
            double userArea = Math.Abs(PolygonArea(user));
            double? areaDiff = referenceArea > 0 ? Math.Abs(userArea - referenceArea) / referenceArea : null;
            return (closure, bboxSimilarity, centroidOffset, areaDiff);
        }

        private static void write_tracing_analysis_outputs(string detailPath,
                                                           string summaryPath,
                                                           string diagnosticPath,
                                                           Mat baseImage,
                                                           IReadOnlyList<TracingReference> references,
                                                           IReadOnlyList<T1bUserStroke> userStrokes,
                                                           IReadOnlyDictionary<int, int> matches,
                                                           IReadOnlyList<TracingMetric> rows,
                                                           bool includeDiagnosticLabels)
        {
            WriteTracingDetailCsv(detailPath, rows);
            WriteTracingSummaryCsv(summaryPath, rows);
            WriteTracingDiagnosticImage(diagnosticPath, baseImage, references, userStrokes, matches, rows, includeDiagnosticLabels);
        }

        private static void WriteTracingDetailCsv(string path, IReadOnlyList<TracingMetric> rows)
        {
            using var writer = new StreamWriter(path);
            writer.WriteLine("participant_id,session_id,trial_id,task_id,path_or_shape_id,reference_type,reference_path_length_px,user_path_length_px,length_ratio,n_reference_points,n_user_points,mean_path_deviation_px,median_path_deviation_px,sd_path_deviation_px,rmse_path_deviation_px,max_path_deviation_px,percent_user_points_within_tolerance,mean_reference_to_user_deviation_px,percent_reference_covered,longest_uncovered_gap_px,closure_error_px,bounding_box_similarity,centroid_offset_px,area_difference_ratio,percent_points_inside_corridor,corridor_exits,mean_distance_from_corridor_centerline_px,completion_time_ms,successful");

            foreach (var row in rows)
            {
                writer.WriteLine(string.Join(",", new[]
                {
                    Csv(row.ParticipantId),
                    Csv(row.SessionId),
                    Csv(row.TrialId),
                    Csv(row.TaskId),
                    Csv(row.PathOrShapeId),
                    Csv(row.ReferenceType),
                    Csv(row.ReferencePathLengthPx),
                    Csv(row.UserPathLengthPx),
                    Csv(row.LengthRatio),
                    Csv(row.NReferencePoints),
                    Csv(row.NUserPoints),
                    Csv(row.MeanPathDeviationPx),
                    Csv(row.MedianPathDeviationPx),
                    Csv(row.SdPathDeviationPx),
                    Csv(row.RmsePathDeviationPx),
                    Csv(row.MaxPathDeviationPx),
                    Csv(row.PercentUserPointsWithinTolerance),
                    Csv(row.MeanReferenceToUserDeviationPx),
                    Csv(row.PercentReferenceCovered),
                    Csv(row.LongestUncoveredGapPx),
                    Csv(row.ClosureErrorPx),
                    Csv(row.BoundingBoxSimilarity),
                    Csv(row.CentroidOffsetPx),
                    Csv(row.AreaDifferenceRatio),
                    Csv(row.PercentPointsInsideCorridor),
                    Csv(row.CorridorExits),
                    Csv(row.MeanDistanceFromCorridorCenterlinePx),
                    Csv(row.CompletionTimeMs),
                    Csv(row.Successful)
                }));
            }
        }

        private static void WriteTracingSummaryCsv(string path, IReadOnlyList<TracingMetric> rows)
        {
            using var writer = new StreamWriter(path);
            writer.WriteLine("participant_id,task_id,reference_type,n_paths_or_shapes,n_successful,success_rate,mean_path_deviation_px,sd_path_deviation_px,mean_rmse_path_deviation_px,mean_percent_user_points_within_tolerance,mean_percent_reference_covered,mean_length_ratio,sd_length_ratio,mean_closure_error_px,mean_completion_time_ms,sd_completion_time_ms");

            foreach (var group in rows.GroupBy(r => new
            {
                ParticipantId = r.ParticipantId ?? string.Empty,
                TaskId = r.TaskId ?? string.Empty,
                ReferenceType = r.ReferenceType ?? string.Empty
            }).OrderBy(g => g.Key.ParticipantId).ThenBy(g => g.Key.TaskId).ThenBy(g => g.Key.ReferenceType))
            {
                var pathDevs = Values(group.Select(r => r.MeanPathDeviationPx));
                var rmses = Values(group.Select(r => r.RmsePathDeviationPx));
                var within = Values(group.Select(r => r.PercentUserPointsWithinTolerance));
                var covered = Values(group.Select(r => r.PercentReferenceCovered));
                var lengths = Values(group.Select(r => r.LengthRatio));
                var closures = Values(group.Select(r => r.ClosureErrorPx));
                var times = Values(group.Select(r => r.CompletionTimeMs)).Distinct().ToList();
                int n = group.Count(r => r.ReferencePathLengthPx.HasValue);
                int successful = group.Count(r => r.Successful == true);

                writer.WriteLine(string.Join(",", new[]
                {
                    Csv(group.Key.ParticipantId),
                    Csv(group.Key.TaskId),
                    Csv(group.Key.ReferenceType),
                    Csv(n),
                    Csv(successful),
                    Csv(n > 0 ? successful / (double)n : null),
                    Csv(Mean(pathDevs)),
                    Csv(StandardDeviation(pathDevs)),
                    Csv(Mean(rmses)),
                    Csv(Mean(within)),
                    Csv(Mean(covered)),
                    Csv(Mean(lengths)),
                    Csv(StandardDeviation(lengths)),
                    Csv(Mean(closures)),
                    Csv(Mean(times)),
                    Csv(StandardDeviation(times))
                }));
            }

            static List<double> Values(IEnumerable<double?> source)
                => source.Where(v => v.HasValue).Select(v => v!.Value).ToList();
        }

        private static void WriteTracingDiagnosticImage(string path,
                                                        Mat baseImage,
                                                        IReadOnlyList<TracingReference> references,
                                                        IReadOnlyList<T1bUserStroke> userStrokes,
                                                        IReadOnlyDictionary<int, int> matches,
                                                        IReadOnlyList<TracingMetric> rows,
                                                        bool includeDiagnosticLabels)
        {
            using var diagnostic = baseImage.Clone();
            foreach (var reference in references)
            {
                DrawPolyline(diagnostic, reference.Points, new Scalar(0, 0, 255), 1);
            }

            foreach (var stroke in userStrokes)
            {
                DrawPolyline(diagnostic, stroke.Points, Scalar.Black, 2);
            }

            if (includeDiagnosticLabels)
            {
                foreach (var row in rows)
                {
                    if (!row.PathOrShapeId.HasValue)
                        continue;

                    var reference = references.FirstOrDefault(r => r.Id == row.PathOrShapeId.Value);
                    if (reference == null)
                        continue;

                    var center = new Point(
                        (int)Math.Round(reference.BoundingBox.X + reference.BoundingBox.Width / 2.0),
                        (int)Math.Round(reference.BoundingBox.Y + reference.BoundingBox.Height / 2.0));
                    string label = $"{row.PathOrShapeId.Value}";
                    if (row.MeanPathDeviationPx.HasValue && row.PercentReferenceCovered.HasValue && row.Successful.HasValue)
                        label += $" d{row.MeanPathDeviationPx.Value:0.0} c{row.PercentReferenceCovered.Value:0} {(row.Successful.Value ? "ok" : "fail")}";
                    Cv2.PutText(diagnostic, label, center + new Point(6, -6), HersheyFonts.HersheySimplex, 0.34, Scalar.Black, 1, LineTypes.AntiAlias);
                }
            }

            Cv2.ImEncode(".png", diagnostic, out byte[] buffer);
            File.WriteAllBytes(path, buffer);
        }

        private static void DrawPolyline(Mat image, IReadOnlyList<Point2d> points, Scalar color, int thickness)
        {
            for (int i = 1; i < points.Count; i++)
            {
                Cv2.Line(
                    image,
                    new Point((int)Math.Round(points[i - 1].X), (int)Math.Round(points[i - 1].Y)),
                    new Point((int)Math.Round(points[i].X), (int)Math.Round(points[i].Y)),
                    color,
                    thickness,
                    LineTypes.AntiAlias);
            }
        }

        private static IReadOnlyList<Point2d> ResamplePolyline(IReadOnlyList<Point2d> points, double spacingPx, bool closed)
        {
            if (points.Count < 2)
                return points.ToList();

            var source = points.ToList();
            if (closed && Distance(source[0].X, source[0].Y, source[^1].X, source[^1].Y) > 0.001)
                source.Add(source[0]);

            var result = new List<Point2d> { source[0] };
            double carry = 0;
            for (int i = 1; i < source.Count; i++)
            {
                var a = source[i - 1];
                var b = source[i];
                double segment = Distance(a.X, a.Y, b.X, b.Y);
                if (segment <= double.Epsilon)
                    continue;

                double d = spacingPx - carry;
                while (d <= segment)
                {
                    double t = d / segment;
                    result.Add(new Point2d(a.X + ((b.X - a.X) * t), a.Y + ((b.Y - a.Y) * t)));
                    d += spacingPx;
                }

                carry = segment - (d - spacingPx);
            }

            if (!closed && Distance(result[^1].X, result[^1].Y, source[^1].X, source[^1].Y) > 0.001)
                result.Add(source[^1]);

            return result;
        }

        private static Rect2d ComputeBoundingBox(IReadOnlyList<Point2d> points)
        {
            if (points.Count == 0)
                return new Rect2d();

            double minX = points.Min(p => p.X);
            double minY = points.Min(p => p.Y);
            double maxX = points.Max(p => p.X);
            double maxY = points.Max(p => p.Y);
            return new Rect2d(minX, minY, maxX - minX, maxY - minY);
        }

        private static double ComputeBoundingBoxSimilarity(Rect2d a, Rect2d b)
        {
            double x1 = Math.Max(a.X, b.X);
            double y1 = Math.Max(a.Y, b.Y);
            double x2 = Math.Min(a.X + a.Width, b.X + b.Width);
            double y2 = Math.Min(a.Y + a.Height, b.Y + b.Height);
            double intersection = Math.Max(0, x2 - x1) * Math.Max(0, y2 - y1);
            double union = (a.Width * a.Height) + (b.Width * b.Height) - intersection;
            return union > 0 ? intersection / union : 0;
        }

        private static Point2d ComputeCentroid(IReadOnlyList<Point2d> points)
            => points.Count == 0 ? new Point2d() : new Point2d(points.Average(p => p.X), points.Average(p => p.Y));

        private static double PolygonArea(IReadOnlyList<Point2d> points)
        {
            if (points.Count < 3)
                return 0;

            double area = 0;
            for (int i = 0; i < points.Count; i++)
            {
                var a = points[i];
                var b = points[(i + 1) % points.Count];
                area += (a.X * b.Y) - (b.X * a.Y);
            }
            return area / 2.0;
        }

        private static int CountCorridorExits(IReadOnlyList<double> distances, double tolerance)
        {
            int exits = 0;
            bool wasInside = true;
            foreach (double distance in distances)
            {
                bool inside = distance <= tolerance;
                if (wasInside && !inside)
                    exits++;
                wasInside = inside;
            }
            return exits;
        }

        private static void WriteT1aSummaryCsv(string path, IReadOnlyList<T1aTrialMetric> rows)
        {
            using var writer = new StreamWriter(path);
            writer.WriteLine("participant_id,target_size_class,n_targets,n_hits,hit_rate,mean_raw_error_px,sd_raw_error_px,median_raw_error_px,mean_normalised_error,sd_normalised_error,median_normalised_error,mean_completion_time_ms,sd_completion_time_ms");

            var groups = rows
                .GroupBy(row => new
                {
                    ParticipantId = row.ParticipantId ?? string.Empty,
                    TargetSizeClass = row.TargetSizeClass ?? string.Empty
                })
                .OrderBy(g => g.Key.ParticipantId)
                .ThenBy(g => g.Key.TargetSizeClass);

            foreach (var group in groups)
            {
                var rawErrors = group.Select(r => r.RawErrorPx).Where(v => v.HasValue).Select(v => v!.Value).ToList();
                var normalisedErrors = group.Select(r => r.NormalisedError).Where(v => v.HasValue).Select(v => v!.Value).ToList();
                var completionTimes = group.Select(r => r.CompletionTimeMs).Where(v => v.HasValue).Select(v => v!.Value).Distinct().ToList();
                int nTargets = group.Count(r => r.TargetRadius.HasValue);
                int nHits = group.Count(r => r.Hit == true);

                writer.WriteLine(string.Join(",", new[]
                {
                    Csv(group.Key.ParticipantId),
                    Csv(group.Key.TargetSizeClass),
                    Csv(nTargets),
                    Csv(nHits),
                    Csv(nTargets > 0 ? nHits / (double)nTargets : null),
                    Csv(Mean(rawErrors)),
                    Csv(StandardDeviation(rawErrors)),
                    Csv(Median(rawErrors)),
                    Csv(Mean(normalisedErrors)),
                    Csv(StandardDeviation(normalisedErrors)),
                    Csv(Median(normalisedErrors)),
                    Csv(Mean(completionTimes)),
                    Csv(StandardDeviation(completionTimes))
                }));
            }
        }

        private static void WriteT1aDebugCsv(string path, IReadOnlyList<T1aTrialMetric> rows)
        {
            using var writer = new StreamWriter(path);
            writer.WriteLine("target_id,target_center_x,target_center_y,target_radius,input_x,input_y,raw_error_px,normalised_error,hit");

            foreach (var row in rows.OrderBy(r => r.TargetId ?? int.MaxValue))
            {
                string line = string.Join(",", new[]
                {
                    Csv(row.TargetId),
                    Csv(row.TargetCenterX),
                    Csv(row.TargetCenterY),
                    Csv(row.TargetRadius),
                    Csv(row.InputX),
                    Csv(row.InputY),
                    Csv(row.RawErrorPx),
                    Csv(row.NormalisedError),
                    Csv(row.Hit)
                });

                writer.WriteLine(line);
                System.Diagnostics.Debug.WriteLine(
                    $"T1 target={Csv(row.TargetId)} center=({Csv(row.TargetCenterX)},{Csv(row.TargetCenterY)}) radius={Csv(row.TargetRadius)} input=({Csv(row.InputX)},{Csv(row.InputY)}) raw_error_px={Csv(row.RawErrorPx)} normalised_error={Csv(row.NormalisedError)} hit={Csv(row.Hit)}");
            }
        }

        private static void WriteT1aDiagnosticImage(string path,
                                                    Mat baseImage,
                                                    IReadOnlyList<T1aTarget> targets,
                                                    IReadOnlyList<T1aTapInput> inputs,
                                                    IReadOnlyList<T1aTrialMetric> rows,
                                                    bool includeMetricLabels)
        {
            using var diagnostic = baseImage.Clone();

            foreach (var target in targets)
            {
                Cv2.Circle(
                    diagnostic,
                    new Point((int)Math.Round(target.CenterX), (int)Math.Round(target.CenterY)),
                    (int)Math.Round(target.Radius),
                    new Scalar(0, 0, 255),
                    2,
                    LineTypes.AntiAlias);
            }

            foreach (var input in inputs)
            {
                Cv2.Circle(
                    diagnostic,
                    new Point((int)Math.Round(input.X), (int)Math.Round(input.Y)),
                    4,
                    Scalar.Black,
                    -1,
                    LineTypes.AntiAlias);
            }

            if (includeMetricLabels)
            {
                foreach (var row in rows)
                {
                    if (!row.TargetId.HasValue || !row.TargetCenterX.HasValue || !row.TargetCenterY.HasValue)
                        continue;

                    string label = $"{row.TargetId.Value}";
                    if (row.RawErrorPx.HasValue && row.NormalisedError.HasValue && row.Hit.HasValue)
                        label += $" e{row.RawErrorPx.Value:0} n{row.NormalisedError.Value:0.0} {(row.Hit.Value ? "H" : "M")}";

                    double fromCenterX = row.TargetCenterX.Value - (baseImage.Width / 2.0);
                    double fromCenterY = row.TargetCenterY.Value - (baseImage.Height / 2.0);
                    double length = Math.Sqrt((fromCenterX * fromCenterX) + (fromCenterY * fromCenterY));
                    if (length < 1)
                        length = 1;

                    double offset = (row.TargetRadius ?? 8) + 12;
                    int labelX = (int)Math.Round(row.TargetCenterX.Value + (fromCenterX / length * offset));
                    int labelY = (int)Math.Round(row.TargetCenterY.Value + (fromCenterY / length * offset));
                    labelX = Math.Clamp(labelX, 2, Math.Max(2, baseImage.Width - 90));
                    labelY = Math.Clamp(labelY, 12, Math.Max(12, baseImage.Height - 4));

                    var labelPoint = new Point(labelX, labelY);
                    var textSize = Cv2.GetTextSize(label, HersheyFonts.HersheySimplex, 0.28, 1, out int baseline);
                    var background = new Rect(
                        Math.Max(0, labelPoint.X - 2),
                        Math.Max(0, labelPoint.Y - textSize.Height - 2),
                        Math.Min(textSize.Width + 4, baseImage.Width - Math.Max(0, labelPoint.X - 2)),
                        Math.Min(textSize.Height + baseline + 4, baseImage.Height - Math.Max(0, labelPoint.Y - textSize.Height - 2)));

                    if (background.Width > 0 && background.Height > 0)
                        Cv2.Rectangle(diagnostic, background, Scalar.White, -1);

                    Cv2.PutText(diagnostic, label, labelPoint, HersheyFonts.HersheySimplex, 0.28, Scalar.Black, 1, LineTypes.AntiAlias);
                }
            }

            Cv2.ImEncode(".png", diagnostic, out byte[] buffer);
            File.WriteAllBytes(path, buffer);
        }

        private static double? Mean(IReadOnlyList<double> values)
            => values.Count == 0 ? null : values.Average();

        private static double? Median(IReadOnlyList<double> values)
        {
            if (values.Count == 0)
                return null;

            var sorted = values.OrderBy(v => v).ToArray();
            int mid = sorted.Length / 2;
            return sorted.Length % 2 == 0
                ? (sorted[mid - 1] + sorted[mid]) / 2.0
                : sorted[mid];
        }

        private static double MedianSorted(IReadOnlyList<double> sortedValues)
        {
            int mid = sortedValues.Count / 2;
            return sortedValues.Count % 2 == 0
                ? (sortedValues[mid - 1] + sortedValues[mid]) / 2.0
                : sortedValues[mid];
        }

        private static double? StandardDeviation(IReadOnlyList<double> values)
        {
            if (values.Count < 2)
                return null;

            double mean = values.Average();
            double sumSquares = values.Sum(v => (v - mean) * (v - mean));
            return Math.Sqrt(sumSquares / (values.Count - 1));
        }

        private static string Csv(string? value)
        {
            if (string.IsNullOrEmpty(value))
                return string.Empty;

            if (value.Contains(',') || value.Contains('"') || value.Contains('\n') || value.Contains('\r'))
                return "\"" + value.Replace("\"", "\"\"") + "\"";

            return value;
        }

        private static string Csv(int? value)
            => value.HasValue ? value.Value.ToString(CultureInfo.InvariantCulture) : string.Empty;

        private static string Csv(double? value)
            => value.HasValue ? value.Value.ToString("0.######", CultureInfo.InvariantCulture) : string.Empty;

        private static string Csv(bool? value)
            => value.HasValue ? (value.Value ? "true" : "false") : string.Empty;

        private static string ResolveSessionPath(string currentPath)
        {
            if (string.IsNullOrWhiteSpace(currentPath))
                throw new ArgumentException("Path is null or empty.", nameof(currentPath));

            string ext = Path.GetExtension(currentPath);
            if (ext.Equals(".h5", StringComparison.OrdinalIgnoreCase) ||
                ext.Equals(".hdf5", StringComparison.OrdinalIgnoreCase))
            {
                return currentPath;
            }

            return Hdf5SessionFrameSource.DeriveH5PathFromVideoPath(currentPath);
        }

        private static string GetUniqueOutputPath(string directory, string fileName)
        {
            Directory.CreateDirectory(directory);

            string baseName = Path.GetFileNameWithoutExtension(fileName);
            string extension = Path.GetExtension(fileName);
            string candidate = Path.Combine(directory, fileName);

            for (int suffix = 2; File.Exists(candidate); suffix++)
            {
                candidate = Path.Combine(directory, $"{baseName}_{suffix}{extension}");
            }

            return candidate;
        }

        private static Mat LoadTaskReplayBaseImage(string sessionPath)
        {
            if (TryExtractTaskToken(sessionPath, out var taskToken))
            {
                string? templatePath = FindTaskTemplatePath(sessionPath, taskToken);
                if (!string.IsNullOrWhiteSpace(templatePath))
                {
                    var template = LoadTaskTemplateOnWhite(templatePath);
                    if (template != null && !template.Empty())
                        return template;
                }
            }

            return new Mat(DefaultTaskExportHeight, DefaultTaskExportWidth, MatType.CV_8UC3, new Scalar(255, 255, 255));
        }

        private static Mat? LoadTaskTemplateOnWhite(string templatePath)
        {
            using var raw = Cv2.ImRead(templatePath, ImreadModes.Unchanged);
            if (raw == null || raw.Empty())
                return null;

            using var flipped = new Mat();
            Cv2.Flip(raw, flipped, FlipMode.XY);

            if (flipped.Channels() == 4)
            {
                using var white = new Mat(flipped.Rows, flipped.Cols, MatType.CV_8UC3, new Scalar(255, 255, 255));
                using var bgr = new Mat();
                using var alpha = new Mat();

                Cv2.CvtColor(flipped, bgr, ColorConversionCodes.BGRA2BGR);
                Cv2.ExtractChannel(flipped, alpha, 3);
                bgr.CopyTo(white, alpha);
                return white.Clone();
            }

            if (flipped.Channels() == 1)
            {
                var bgr = new Mat();
                Cv2.CvtColor(flipped, bgr, ColorConversionCodes.GRAY2BGR);
                return bgr;
            }

            if (flipped.Channels() == 3)
                return flipped.Clone();

            return null;
        }

        private static bool TryExtractTaskToken(string path, out string taskToken)
        {
            taskToken = string.Empty;

            string stem = Path.GetFileNameWithoutExtension(path) ?? string.Empty;
            var match = TaskTokenRegex.Match(stem);
            if (!match.Success)
                return false;

            taskToken = NormalizeTaskToken(match.Groups[1].Value);
            return true;
        }

        private static string NormalizeTaskToken(string token)
        {
            token = token.Trim();
            if (token.Length == 0)
                return token;

            char prefix = char.ToUpperInvariant(token[0]);
            string body = token.Length > 1 ? token[1..] : string.Empty;

            if (body.Length > 0 && char.IsLetter(body[^1]))
                return prefix + body[..^1] + char.ToLowerInvariant(body[^1]);

            return prefix + body;
        }

        private static string? FindTaskTemplatePath(string sessionPath, string taskToken)
        {
            string fileName = taskToken + ".png";
            var candidates = new[]
            {
                Path.Combine(Path.GetDirectoryName(sessionPath) ?? string.Empty, "Resources", "Images", fileName),
                Path.Combine(Path.GetDirectoryName(sessionPath) ?? string.Empty, "resources", "images", fileName),
                Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "Resources", "Images", fileName),
                Path.Combine(Directory.GetCurrentDirectory(), "Resources", "Images", fileName)
            };

            foreach (var candidate in candidates)
            {
                if (File.Exists(candidate))
                    return candidate;
            }

            return null;
        }

        private static bool TryGetFrameQuad(float[] corners, out Point2f[] quad)
        {
            quad = Array.Empty<Point2f>();

            if (corners == null || corners.Length != 8)
                return false;

            var candidate = new[]
            {
                new Point2f(corners[0], corners[1]),
                new Point2f(corners[2], corners[3]),
                new Point2f(corners[4], corners[5]),
                new Point2f(corners[6], corners[7])
            };

            foreach (var point in candidate)
            {
                if (float.IsNaN(point.X) || float.IsNaN(point.Y) || float.IsInfinity(point.X) || float.IsInfinity(point.Y))
                    return false;
            }

            double signedArea2 = 0;
            for (int i = 0; i < candidate.Length; i++)
            {
                var a = candidate[i];
                var b = candidate[(i + 1) % candidate.Length];
                signedArea2 += (a.X * b.Y) - (b.X * a.Y);
            }

            if (Math.Abs(signedArea2) < 1.0)
                return false;

            quad = candidate;
            return true;
        }

        private static Point ClampPointToImage(Point2f point, int width, int height)
        {
            int x = (int)Math.Round(point.X);
            int y = (int)Math.Round(point.Y);

            x = Math.Clamp(x, 0, Math.Max(0, width - 1));
            y = Math.Clamp(y, 0, Math.Max(0, height - 1));

            return new Point(x, y);
        }

        private static (double? Min, double? Max) ComputePositivePressureRange(SessionManager session, CancellationToken ct)
        {
            if (session.AttrCount < 3)
                return (null, null);

            var labels = new float[session.AttrCount];
            var labelMask = new byte[session.AttrCount];
            double min = double.PositiveInfinity;
            double max = double.NegativeInfinity;

            for (ulong frameIndex = 0; frameIndex < session.FrameCount; frameIndex++)
            {
                ct.ThrowIfCancellationRequested();
                session.ReadMetadata(frameIndex, out _, corners: null, labels, labelMask);
                if (labelMask[2] == 0 || !(labels[2] > 0f))
                    continue;

                min = Math.Min(min, labels[2]);
                max = Math.Max(max, labels[2]);
            }

            return double.IsInfinity(min) || double.IsInfinity(max) ? (null, null) : (min, max);
        }

        private static int ComputeT5aPressureThickness(double pressure, double? minPressure, double? maxPressure, int outputWidth)
        {
            int minThickness = Math.Max(1, (int)Math.Round(outputWidth / 550.0));
            int maxThickness = Math.Max(minThickness + 1, (int)Math.Round(outputWidth / 95.0));

            if (!minPressure.HasValue || !maxPressure.HasValue || maxPressure.Value <= minPressure.Value)
                return Math.Max(1, (int)Math.Round(outputWidth / 300.0));

            double t = (pressure - minPressure.Value) / (maxPressure.Value - minPressure.Value);
            t = Math.Clamp(t, 0.0, 1.0);
            return (int)Math.Round(minThickness + ((maxThickness - minThickness) * t));
        }

        private static void DrawT5bCalligraphyInk(Mat output, Point? previousPoint, Point currentPoint, double tiltX, double tiltY)
        {
            double tiltMagnitude = Math.Sqrt((tiltX * tiltX) + (tiltY * tiltY));
            double angleRad = tiltMagnitude > 0.001 ? Math.Atan2(tiltY, tiltX) : Math.PI / 2.0;

            int minLength = Math.Max(4, (int)Math.Round(output.Width / 80.0));
            int maxLength = Math.Max(minLength + 2, (int)Math.Round(output.Width / 25.0));
            double t = Math.Clamp(tiltMagnitude / 60.0, 0.0, 1.0);
            double nibLength = minLength + ((maxLength - minLength) * t);
            int nibThickness = Math.Max(1, (int)Math.Round(output.Width / 500.0));

            if (previousPoint.HasValue)
            {
                double distance = Distance(previousPoint.Value.X, previousPoint.Value.Y, currentPoint.X, currentPoint.Y);
                int steps = Math.Max(1, (int)Math.Ceiling(distance / Math.Max(1.0, nibThickness * 2.0)));
                for (int step = 1; step <= steps; step++)
                {
                    double u = step / (double)steps;
                    var interpolated = new Point(
                        (int)Math.Round(previousPoint.Value.X + ((currentPoint.X - previousPoint.Value.X) * u)),
                        (int)Math.Round(previousPoint.Value.Y + ((currentPoint.Y - previousPoint.Value.Y) * u)));
                    DrawCalligraphyNib(output, interpolated, angleRad, nibLength, nibThickness);
                }
            }
            else
            {
                DrawCalligraphyNib(output, currentPoint, angleRad, nibLength, nibThickness);
            }
        }

        private static void DrawCalligraphyNib(Mat output, Point center, double angleRad, double length, int thickness)
        {
            double halfLength = length / 2.0;
            double dx = Math.Cos(angleRad) * halfLength;
            double dy = Math.Sin(angleRad) * halfLength;

            var start = new Point(
                Math.Clamp((int)Math.Round(center.X - dx), 0, Math.Max(0, output.Width - 1)),
                Math.Clamp((int)Math.Round(center.Y - dy), 0, Math.Max(0, output.Height - 1)));
            var end = new Point(
                Math.Clamp((int)Math.Round(center.X + dx), 0, Math.Max(0, output.Width - 1)),
                Math.Clamp((int)Math.Round(center.Y + dy), 0, Math.Max(0, output.Height - 1)));

            Cv2.Line(output, start, end, Scalar.Black, thickness, LineTypes.AntiAlias);
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

        //private void LoadFrame(int index)
        //{
        //    FrameData? frameData = null;

        //    try
        //    {
        //        if (_frameSource is null)
        //            return;

        //        frameData = _frameSource.GetFrame(index);

        //        if (frameData is null || frameData.Image == null)
        //        {
        //            CurrentFrameImage = null;
        //            CurrentTextStripImage = null;
        //            HasTextBar = false;
        //            CurrentRawOcr = null;
        //            OnPropertyChanged(nameof(CurrentRawOcr));
        //            return;
        //        }

        //        CurrentTextStripImage = frameData.AuxBitmapSource;
        //        HasTextBar = frameData.HasStylusData;
        //        CurrentRawOcr = frameData.AdditionalText;
        //        OnPropertyChanged(nameof(CurrentRawOcr));

        //        if (frameData.TopRight is Point topRight)
        //        {
        //            TopRightX = topRight.X;
        //            TopRightY = topRight.Y;
        //        }
        //        if (frameData.TopLeft is Point topLeft)
        //        {
        //            TopLeftX = topLeft.X;
        //            TopLeftY = topLeft.Y;
        //        }
        //        if (frameData.BottomLeft is Point bottomLeft)
        //        {
        //            BottomLeftX = bottomLeft.X;
        //            BottomLeftY = bottomLeft.Y;
        //        }
        //        if (frameData.BottomRight is Point bottomRight)
        //        {
        //            BottomRightX = bottomRight.X;
        //            BottomRightY = bottomRight.Y;
        //        }

        //        StylusX = frameData.StylusX;
        //        StylusY = frameData.StylusY;
        //        StylusPressure = frameData.StylusPressure;
        //        StylusTiltX = frameData.StylusTiltX;
        //        StylusTiltY = frameData.StylusTiltY;

        //        // Decide what portion of the frame to show in the main viewport.
        //        // If a databar is present and we have a detected border height, show only the area below it.
        //        if (frameData.HasStylusData)
        //        {
        //            // Create padded image: video centred with 300px bars left/right
        //            using var paddedMat = AddHorizontalPadding(frameData.Image, HorizontalPadding);
        //            var y = frameData.AuxImage?.Height ?? 0;

        //            // Draw surface quad on the padded image
        //            DrawSurfaceMask(paddedMat, HorizontalPadding, frameData);

        //            // Draw actual stylus position (red X) and inferred position (blue X) if available.
        //            try
        //            {
        //                // Only draw markers if we have the four surface corners
        //                if (frameData.TopLeft is Point tl && frameData.TopRight is Point tr && frameData.BottomRight is Point br && frameData.BottomLeft is Point bl)
        //                {
        //                    // Prepare homography that maps model-destination coords -> image paddedMat coords
        //                    var onnx = _onnx; // may be null
        //                    int modelW = onnx?.ModelW ?? 480;
        //                    int modelH = onnx?.ModelH ?? 320;

        //                    var dst = new[]
        //                    {
        //                        new Point2f(0, 0),
        //                        new Point2f(modelW - 1, 0),
        //                        new Point2f(modelW - 1, modelH - 1),
        //                        new Point2f(0, modelH - 1)
        //                    };

        //                    var quad = new[]
        //                    {
        //                        new Point2f((float)(tl.X + HorizontalPadding), (float)tl.Y),
        //                        new Point2f((float)(tr.X + HorizontalPadding), (float)tr.Y),
        //                        new Point2f((float)(br.X + HorizontalPadding), (float)br.Y),
        //                        new Point2f((float)(bl.X + HorizontalPadding), (float)bl.Y),
        //                    };

        //                    var Hinv = Cv2.GetPerspectiveTransform(dst, quad);

        //                    // Draw actual stylus location (if available in frameData)
        //                    if (frameData.StylusX.HasValue && frameData.StylusY.HasValue)
        //                    {
        //                        float nx = (float)(frameData.StylusX.Value / STYLUS_MAX_X);
        //                        float ny = (float)(frameData.StylusY.Value / STYLUS_MAX_Y);
        //                        var dstPt = new Point2f(nx * (modelW - 1), ny * (modelH - 1));
        //                        var imgPt = WarpPointByHomography(dstPt, Hinv);
        //                        DrawCross(paddedMat, imgPt, new Scalar(0, 0, 255)); // red (B,G,R)
        //                    }

        //                    // Draw inferred position if we already have a prediction for this view
        //                    if (PredictedXNorm.HasValue && PredictedYNorm.HasValue)
        //                    {
        //                        float nx = PredictedXNorm.Value;
        //                        float ny = PredictedYNorm.Value;
        //                        var dstPt = new Point2f(nx * (modelW - 1), ny * (modelH - 1));
        //                        var imgPt = WarpPointByHomography(dstPt, Hinv);
        //                        DrawCross(paddedMat, imgPt, new Scalar(255, 0, 0)); // blue
        //                    }
        //                }
        //            }
        //            catch (Exception ex)
        //            {
        //                System.Diagnostics.Debug.WriteLine("Draw markers error: " + ex);
        //            }

        //            var contentBmp = BitmapSourceConverter.ToBitmapSource(paddedMat);
        //            contentBmp.Freeze();
        //            CurrentFrameImage = contentBmp;
        //            if (Inference)
        //            {
        //                // Only if all corners exist
        //                if (frameData.TopLeft is Point tl &&
        //                    frameData.TopRight is Point tr &&
        //                    frameData.BottomRight is Point br &&
        //                    frameData.BottomLeft is Point bl)
        //                {
        //                    // Make quad in paddedMat coordinates (shift X by HorizontalPadding)
        //                    var quad = new[]
        //                    {
        //                        new Point2f((float)(tl.X + HorizontalPadding), (float)tl.Y),
        //                        new Point2f((float)(tr.X + HorizontalPadding), (float)tr.Y),
        //                        new Point2f((float)(br.X + HorizontalPadding), (float)br.Y),
        //                        new Point2f((float)(bl.X + HorizontalPadding), (float)bl.Y),
        //                    };

        //                    // Run off UI thread so frame scrubbing stays responsive
        //                    var matForInference = paddedMat.Clone(); // protect against disposal
        //                    _ = Task.Run(() =>
        //                    {
        //                        try
        //                        {
        //                            var onnx = GetOrCreateOnnx();
        //                            var (xNorm, yNorm) = onnx.Predict(matForInference, quad);

        //                            // Map normalized [0,1] model outputs to stylus surface coordinates
        //                            int xPx = (int)Math.Round(xNorm * (STYLUS_MAX_X - 1));
        //                            int yPx = (int)Math.Round(yNorm * (STYLUS_MAX_Y - 1));

        //                            System.Windows.Application.Current.Dispatcher.BeginInvoke(new Action(() =>
        //                            {
        //                                PredictedXNorm = xNorm;
        //                                PredictedYNorm = yNorm;
        //                                PredictedXPixel = xPx;
        //                                PredictedYPixel = yPx;
        //                            }));
        //                        }
        //                        catch (Exception ex)
        //                        {
        //                            System.Diagnostics.Debug.WriteLine("Inference error: " + ex);
        //                        }
        //                        finally
        //                        {
        //                            matForInference.Dispose();
        //                        }
        //                    });
        //                }
        //                else
        //                {
        //                    PredictedXNorm = null;
        //                    PredictedYNorm = null;
        //                    PredictedXPixel = null;
        //                    PredictedYPixel = null;
        //                }
        //            }

        //        }
        //        else
        //        {
        //            // fallback: show full frame 
        //            using var frameMat = frameData.Image.Clone();

        //            var bmp = BitmapSourceConverter.ToBitmapSource(frameMat);
        //            bmp.Freeze();
        //            CurrentFrameImage = bmp;
        //        }
        //    }
        //    catch (Exception ex)
        //    {
        //        System.Diagnostics.Debug.WriteLine("LoadFrame error: " + ex);
        //    }
        //    finally
        //    {
        //        if (frameData != null)
        //        {
        //            frameData.Image?.Dispose();
        //            frameData.AuxImage?.Dispose();
        //        }
        //    }
        //}

        // Add this field somewhere in your ViewModel (prevents overlapping loads when scrubbing / playing)
        private readonly SemaphoreSlim _loadFrameGate = new(1, 1);
        private bool prevHasStylusData = false;
        private async void LoadFrame(int index)
        {
            await _loadFrameGate.WaitAsync().ConfigureAwait(true);
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

                    // clear predictions
                    PredictedXNorm = null;
                    PredictedYNorm = null;
                    PredictedXPixel = null;
                    PredictedYPixel = null;
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

                if (frameData.HasStylusData)
                {                    
                    using var paddedMat = AddHorizontalPadding(frameData.Image, HorizontalPadding);

                    // Draw surface mask / background replacement first
                    DrawSurfaceMask(paddedMat, HorizontalPadding, frameData);

                    // If we have corners, prepare homography and optionally run inference BEFORE rendering bitmap
                    if (frameData.TopLeft is Point tl &&
                        frameData.TopRight is Point tr &&
                        frameData.BottomRight is Point br &&
                        frameData.BottomLeft is Point bl)
                    {
                        // Prefer the actual model dims
                        var onnxForDims = GetOrCreateOnnx();
                        int modelW = 480;
                        int modelH = 320;

                        var dst = new[]
                        {
                            new Point2f(0, 0),
                            new Point2f(modelW - 1, 0),
                            new Point2f(modelW - 1, modelH - 1),
                            new Point2f(0, modelH - 1)
                        };

                        var quad = new[]
                        {
                            new Point2f((float)(tl.X + HorizontalPadding), (float)tl.Y),
                            new Point2f((float)(tr.X + HorizontalPadding), (float)tr.Y),
                            new Point2f((float)(br.X + HorizontalPadding), (float)br.Y),
                            new Point2f((float)(bl.X + HorizontalPadding), (float)bl.Y),
                        };

                        var Hinv = Cv2.GetPerspectiveTransform(dst, quad);

                        // Draw actual stylus location (red X)
                        if (frameData.StylusX.HasValue && frameData.StylusY.HasValue)
                        {
                            float nx = (float)(frameData.StylusX.Value / STYLUS_MAX_X);
                            float ny = (float)(frameData.StylusY.Value / STYLUS_MAX_Y);
                            var dstPt = new Point2f(nx * (modelW - 1), ny * (modelH - 1));
                            var imgPt = WarpPointByHomography(dstPt, Hinv);
                            DrawCross(paddedMat, imgPt, new Scalar(0, 0, 255)); // red (B,G,R)
                        }

                        float? xNormPred = null;
                        float? yNormPred = null;

                        // Run inference and await it (so the result corresponds to THIS rendered frame)
                        if (Inference)
                        {
                            using var matForInference = paddedMat.Clone();

                            try
                            {
                                // Capture which frame we intended (in case user scrubbed while awaiting)
                                int requestedIndex = index;

                                var (x, y) = await Task.Run(() =>
                                {
                                    var onnx = GetOrCreateOnnx();
                                    return onnx.Predict(matForInference, quad);
                                }).ConfigureAwait(true); // resume on UI thread

                                // If user moved on while we were running, drop the result
                                if (CurrentFrameIndex == requestedIndex)
                                {
                                    xNormPred = x;
                                    yNormPred = y;

                                    PredictedXNorm = x;
                                    PredictedYNorm = y;
                                    PredictedXPixel = (int)Math.Round(x * (STYLUS_MAX_X - 1));
                                    PredictedYPixel = (int)Math.Round(y * (STYLUS_MAX_Y - 1));
                                }
                                else
                                {
                                    // stale result
                                    xNormPred = null;
                                    yNormPred = null;
                                }
                            }
                            catch (Exception ex)
                            {
                                System.Diagnostics.Debug.WriteLine("Inference error: " + ex);
                                PredictedXNorm = null;
                                PredictedYNorm = null;
                                PredictedXPixel = null;
                                PredictedYPixel = null;
                            }
                        }
                        else
                        {
                            PredictedXNorm = null;
                            PredictedYNorm = null;
                            PredictedXPixel = null;
                            PredictedYPixel = null;
                            prevHasStylusData = frameData.HasStylusData;
                        }

                        // Draw inferred position (blue X) ONLY if we got a prediction for this same frame
                        if (xNormPred.HasValue && yNormPred.HasValue)
                        {
                            var dstPt = new Point2f(xNormPred.Value * (modelW - 1), yNormPred.Value * (modelH - 1));
                            var imgPt = WarpPointByHomography(dstPt, Hinv);
                            DrawCross(paddedMat, imgPt, new Scalar(255, 0, 0)); // blue
                        }
                    }
                    else
                    {
                        // no corners => clear predictions
                        PredictedXNorm = null;
                        PredictedYNorm = null;
                        PredictedXPixel = null;
                        PredictedYPixel = null;
                    }

                    // Now render bitmap (after prediction has been applied)
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

                    // clear predictions (since we're not in the stylus-data view)
                    PredictedXNorm = null;
                    PredictedYNorm = null;
                    PredictedXPixel = null;
                    PredictedYPixel = null;
                    prevHasStylusData = frameData.HasStylusData;
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

                _loadFrameGate.Release();
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

        // Warp a point (in model/image coordinates) through a 3x3 homography matrix to image coords
        private static Point2f WarpPointByHomographyPrecise(Point2f pt, Mat H)
        {
            // H is 3x3 CV_64F or CV_32F
            double a = H.At<double>(0, 0);
            double b = H.At<double>(0, 1);
            double c = H.At<double>(0, 2);
            double d = H.At<double>(1, 0);
            double e = H.At<double>(1, 1);
            double f = H.At<double>(1, 2);
            double g = H.At<double>(2, 0);
            double h = H.At<double>(2, 1);
            double i = H.At<double>(2, 2);

            double wx = a * pt.X + b * pt.Y + c;
            double wy = d * pt.X + e * pt.Y + f;
            double w = g * pt.X + h * pt.Y + i;

            if (Math.Abs(w) < double.Epsilon) w = double.Epsilon;
            return new Point2f((float)(wx / w), (float)(wy / w));
        }

        private static Point WarpPointByHomography(Point2f pt, Mat H)
        {
            var warped = WarpPointByHomographyPrecise(pt, H);
            return new Point((int)Math.Round(warped.X), (int)Math.Round(warped.Y));
        }

        private static void DrawCross(Mat img, Point p, Scalar color, int size = 8, int thickness = 2)
        {
            Cv2.Line(img, new Point(p.X - size, p.Y - size), new Point(p.X + size, p.Y + size), color, thickness);
            Cv2.Line(img, new Point(p.X - size, p.Y + size), new Point(p.X + size, p.Y - size), color, thickness);
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

            // 5) Dispose inference model
            lock (_onnxLock)
            {
                _onnx?.Dispose();
                _onnx = null;
            }

            GC.SuppressFinalize(this);
        }
    }
}
