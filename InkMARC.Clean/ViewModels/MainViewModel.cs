using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using InkMARC.Clean.Model;
using InkMARC.Clean.Services;
using InkMARC.Clean.Services.Interfaces;
using Microsoft.Win32;
using OpenCvSharp;
using OpenCvSharp.WpfExtensions;
using System;
using System.ComponentModel;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Threading;
using System.Threading.Tasks;
using System.Windows.Input;
using System.Windows.Media.Animation;
using System.Windows.Media.Imaging;

namespace InkMARC.Clean.ViewModels
{
    public partial class MainViewModel : ObservableObject, IDisposable
    {
        private readonly IVideoService _videoService;
        private CancellationTokenSource? _playCts;

        private IFrameSource? _frameSource;

        private bool _disposed;

        // Reused mats to avoid allocations every frame
        private Mat? _surfaceMask;   // 8UC1, 0 outside polygon, 255 inside
        private Mat? _hsv;           // 8UC3 for HSV conversion
        private Mat? _colorMask;     // 8UC1 for green/blue detection
        private Mat? _tmpMask;       // 8UC1 temp for OR / AND operations
        private Point[]? _lastSurfacePoly;

        private Mat? _ycrcb;           // for skin detection
        private Mat? _skinMask;        // foreground (hand/arm)
        private Mat? _notSkinMask;     // ~skin, used to subtract from colorMask
        private Mat? _pureGreenMask;

        private Mat? _lab;            // CV_8UC3
        private Mat? _labA;           // CV_8UC1
        private Mat? _labB;           // CV_8UC1
        private Mat? _penMaskLab;     // CV_8UC1  (pen pixels)
        private Mat? _bGeAMask;       // CV_8UC1  (b >= a)

        private Mat? _bgBgr;       // CV_8UC3
        private Mat? _bgNoise;     // CV_8SC3 (signed noise)
        private Mat? _bgGrad16;   // CV_16SC3

        private Mat? _bg16;        // CV_16SC3
        private Mat? _noise8s;     // CV_8SC3
        private Mat? _noise16s;    // CV_16SC3
        private Mat? _col16;       // CV_16SC1 (H x 1)
        private Mat? _grad1_16;    // CV_16SC1 (H x W)

        // Cached base+gradient (16S)
        private Mat? _bgBaseGrad16;
        private Scalar _bgBaseCached = new Scalar(double.NaN, double.NaN, double.NaN);
        private int _bgRowsCached = -1, _bgColsCached = -1, _bgGradAmpCached = int.MinValue;

        // Noise bank (16S)
        private Mat[]? _noise16Bank;
        private int _noiseBankRows = -1, _noiseBankCols = -1, _noiseBankAmp = int.MinValue;
        private const int NoiseBankSize = 32;

        public MainViewModel(IVideoService videoService)
        {
            _videoService = videoService ?? throw new ArgumentNullException(nameof(videoService));
            _videoService.FrameCountChanged += VideoService_FrameCountChanged;

            OpenVideoCommand = new RelayCommand(OpenVideo);
            MoveFrameCommand = new RelayCommand<int>(MoveFrame);
            TogglePlayCommand = new RelayCommand(TogglePlay);
            ProcessAllCommand = new RelayCommand(ProcessAll);
            ExportDatasetCommand = new RelayCommand(ExportDataset);
            ExportMultipleDatasetsCommand = new RelayCommand(ExportMultipleDatasets);
            MarkInvalidCommand = new RelayCommand(MarkInvalid);
            ReprocessFrameCommand = new RelayCommand(ReprocessFrame);
            CycleResolutionCommand = new RelayCommand(CycleSizes);
            CycleColorsCommand = new RelayCommand(CycleColors);
            OpenPictureCommand = new RelayCommand(OpenPicture);
            OpenDatasetCommand = new RelayCommand(OpenDataset);
        }

        #region progress reporting

        [ObservableProperty] private bool _isExporting;
        [ObservableProperty] private double _exportProgress01;          // 0..1 for a ProgressBar
        [ObservableProperty] private string _exportStatusText = string.Empty;

        [ObservableProperty] private int _exportProgressFrames;         // optional: show "X / Y"
        [ObservableProperty] private int _exportProgressTotalFrames;    // optional

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

        #endregion
        
        private void VideoService_FrameCountChanged(object? sender, int e)
        {
            FrameCount = e;
        }

        private int _frameCount;
        public int FrameCount
        {
            get => _frameCount;
            set { if (_frameCount != value) { _frameCount = value; OnPropertyChanged(nameof(FrameCount)); } }
        }

        private int _currentFrameIndex;
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
        private double viewW = 1080.0; // 1220
        [ObservableProperty]
        private double viewH = 2161.0; // 2550

        private const int HorizontalPadding = 300;

        [ObservableProperty]
        private string backgroundColor = "White";
        [ObservableProperty]
        private string foregroundColor = "Black";

        [ObservableProperty]
        private bool hasYellowBackground = false;

        private Scalar _backgroundScalar = new Scalar(255, 255, 255);

        public bool ShowSurfaceMask { get; set; }        
        public string? CurrentRawOcr { get; set; }

        private bool _hasTextBar;
        public bool HasTextBar
        {
            get => _hasTextBar;
            private set { if (_hasTextBar != value) { _hasTextBar = value; OnPropertyChanged(nameof(HasTextBar)); } }
        }

        // Commands
        public ICommand OpenVideoCommand { get; }
        public ICommand ProcessAllCommand { get; }
        public ICommand ExportDatasetCommand { get; }
        public ICommand ExportMultipleDatasetsCommand { get; }
        public ICommand MoveFrameCommand { get; }
        public ICommand TogglePlayCommand { get; }
        public ICommand MarkInvalidCommand { get; }
        public ICommand ReprocessFrameCommand { get; }
        public ICommand CycleResolutionCommand { get; }
        public ICommand CycleColorsCommand { get; }
        public ICommand OpenPictureCommand { get; }
        public ICommand OpenDatasetCommand { get; }

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

                // Set index and force load of first frame so UI updates immediately
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
            // Uses WinForms dialog without adding a global using.
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
            BackgroundColor = (object)BackgroundColor switch
            {
                "White" => "Black",
                "Black" => "Gray",
                "Gray" => "SaddleBrown",
                "SaddleBrown" => "DarkGreen",
                "DarkGreen" => "Tan",
                "Tan" => "White",
                _ => "White",
            };
            ForegroundColor = (object)BackgroundColor switch
            {                 
                "White" => "Black",
                "Black" => "White",
                "Gray" => "White",
                "SaddleBrown" => "White",
                "DarkGreen" => "White",
                "Tan" => "Black",
                _ => "Black",
            };
            _backgroundScalar = BackgroundColor switch
            {
                "White" => new Scalar(255, 255, 255),
                "Black" => new Scalar(0, 0, 0),
                "Gray" => new Scalar(128, 128, 128),
                "SaddleBrown" => new Scalar(139, 69, 19),
                "DarkGreen" => new Scalar(0, 100, 0),
                "Tan" => new Scalar(210, 180, 140),
                _ => new Scalar(255, 255, 255),
            };
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
                catch {  /* ignore */ }
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
                        var delayMs = Math.Max(1, 1000.0 / Math.Max(1.0, _videoService.FramesPerSecond));
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
                catch {  /* ignore */ }
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
                                       System.Action<double, string>? progress = null)
        {
            const int attrCount = 5;
            string[] colours = { "White", "Black", "Gray", "SaddleBrown", "DarkGreen", "Tan" }; // same as your export :contentReference[oaicite:5]{index=5}

            var dispatcher = System.Windows.Application.Current.Dispatcher;

            // Find first usable frame to establish dimensions (same logic as your current export) :contentReference[oaicite:6]{index=6}
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

            int rgbLen = exportH * exportW * 3;
            var rgbBuffer = new byte[rgbLen];
            var cornersBuffer = new float[8];
            var labelsBuffer = new float[attrCount];
            var labelMaskBuffer = new byte[attrCount];

            var zeroRgb = new byte[rgbLen];
            var zeroCorners = new float[8];
            var zeroLabels = new float[attrCount];
            var zeroMask = new byte[attrCount];

            using var rgbMat = new Mat(exportH, exportW, MatType.CV_8UC3);

            int totalPasses = colours.Length;
            long totalWork = (long)frameCount * totalPasses;

            const int reportEveryNFrames = 200;

            for (int pass = 0; pass < totalPasses; pass++)
            {
                ct.ThrowIfCancellationRequested();

                string colourName = colours[pass];

                // Set scalar used by DrawSurfaceMask (same as your current mapping) :contentReference[oaicite:7]{index=7}
                _backgroundScalar = colourName switch
                {
                    "White" => new Scalar(255, 255, 255),
                    "Black" => new Scalar(0, 0, 0),
                    "Gray" => new Scalar(128, 128, 128),
                    "SaddleBrown" => new Scalar(139, 69, 19),
                    "DarkGreen" => new Scalar(0, 100, 0),
                    "Tan" => new Scalar(210, 180, 140),
                    _ => new Scalar(255, 255, 255),
                };

                dispatcher.BeginInvoke(new Action(() =>
                {
                    BackgroundColor = colourName;
                    ForegroundColor = colourName switch
                    {
                        "White" => "Black",
                        "Tan" => "Black",
                        _ => "White",
                    };
                }));

                string outPath = Path.Combine(baseDir, $"{baseName}_{colourName}.h5"); // same naming scheme :contentReference[oaicite:8]{index=8}

                using var session = SessionManager.CreateNew(
                    path: outPath,
                    frameCount: (ulong)frameCount,
                    height: exportH,
                    width: exportW,
                    attrCount: attrCount,
                    chunkFrames: 16);

                for (int frameIndex = 0; frameIndex < frameCount; frameIndex++)
                {
                    ct.ThrowIfCancellationRequested();

                    FrameData? frameData = null;
                    try
                    {
                        frameData = frameSource.GetFrameForExport(frameIndex);

                        if (frameData?.Image == null || frameData.Image.Empty())
                        {
                            session.WriteFrame((ulong)frameIndex, zeroRgb, zeroCorners, zeroLabels, zeroMask);
                            continue;
                        }

                        int yOffset = frameData.AuxImage?.Height ?? 0;

                        using var cropped = GetBelowDatabarRoi(frameData.Image, yOffset);

                        // Draw mask in cropped coordinates (no horizontal pad during export) :contentReference[oaicite:9]{index=9}
                        DrawSurfaceMask(cropped, 0, frameData);

                        Cv2.CvtColor(cropped, rgbMat, ColorConversionCodes.BGR2RGB);
                        Marshal.Copy(rgbMat.Data, rgbBuffer, 0, rgbLen);

                        // corners TL,TR,BR,BL (same as current) :contentReference[oaicite:10]{index=10}
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

                        // labels + mask (same as current) :contentReference[oaicite:11]{index=11}
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

                        session.WriteFrame((ulong)frameIndex, rgbBuffer, cornersBuffer, labelsBuffer, labelMaskBuffer);
                    }
                    finally
                    {
                        frameData?.Image?.Dispose();
                        frameData?.AuxImage?.Dispose();
                    }

                    if (frameIndex == 0 || frameIndex == frameCount - 1 || (frameIndex % reportEveryNFrames == 0))
                    {
                        long done = (long)pass * frameCount + (frameIndex + 1);
                        double pct = (totalWork <= 0) ? 1.0 : (done / (double)totalWork); // 0..1 for caller
                        string status = $"Exporting {baseName}: {colourName} ({pass + 1}/{totalPasses}) - frame {frameIndex + 1}/{frameCount}";

                        progress?.Invoke(pct, status);
                    }
                }
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


        //private void ExportDataset()
        //{
        //    if (_frameSource is null || FrameCount <= 0)
        //    {
        //        System.Windows.MessageBox.Show("No video is open.");
        //        return;
        //    }

        //    // Stop any existing play/process/export loop that uses _playCts
        //    if (_playCts != null)
        //    {
        //        var cts = _playCts;
        //        _playCts = null;

        //        try { cts.Cancel(); }
        //        catch { /* ignore */ }
        //        finally { cts.Dispose(); }
        //    }

        //    var dlg = new SaveFileDialog
        //    {
        //        Title = "Export dataset (base filename)",
        //        Filter = "HDF5 files|*.h5;*.hdf5|All files|*.*",
        //        FileName = Path.GetFileNameWithoutExtension(CurrentFilePath) + "_dataset.h5",
        //        InitialDirectory = string.IsNullOrWhiteSpace(CurrentFilePath) ? null : Path.GetDirectoryName(CurrentFilePath),
        //        AddExtension = true,
        //        DefaultExt = ".h5",
        //        OverwritePrompt = true
        //    };

        //    if (dlg.ShowDialog() != true)
        //        return;

        //    var basePath = dlg.FileName;
        //    var baseDir = Path.GetDirectoryName(basePath) ?? Environment.CurrentDirectory;
        //    var baseName = Path.GetFileNameWithoutExtension(basePath);

        //    string[] colours = { "White", "Black", "Gray", "SaddleBrown", "DarkGreen", "Tan" };

        //    bool prevShowSurfaceMask = ShowSurfaceMask;
        //    ShowSurfaceMask = true;

        //    // Save & restore background scalar so export doesn’t permanently change view state
        //    var prevBackgroundScalar = _backgroundScalar;
        //    var prevBackgroundColor = BackgroundColor;
        //    var prevForegroundColor = ForegroundColor;

        //    var newCts = new CancellationTokenSource();
        //    _playCts = newCts;
        //    var ct = newCts.Token;

        //    IsExporting = true;
        //    ExportProgress = 0;
        //    ExportStatus = "Starting export...";


        //    Task.Run(() =>
        //    {
        //        try
        //        {
        //            const int attrCount = 5;

        //            var dispatcher = System.Windows.Application.Current.Dispatcher;

        //            // Find first usable frame to establish dimensions
        //            FrameData? firstFrame = null;
        //            for (int i = 0; i < FrameCount; i++)
        //            {
        //                ct.ThrowIfCancellationRequested();
        //                firstFrame = _frameSource.GetFrameForExport(i);
        //                if (firstFrame?.Image != null && !firstFrame.Image.Empty())
        //                    break;

        //                firstFrame?.Image?.Dispose();
        //                firstFrame?.AuxImage?.Dispose();
        //                firstFrame = null;
        //            }

        //            if (firstFrame?.Image == null)
        //                throw new InvalidOperationException("Could not read any frames to export.");

        //            int yCrop = firstFrame.AuxImage?.Height ?? 0;
        //            int exportH = firstFrame.Image.Height - yCrop;
        //            int exportW = firstFrame.Image.Width;

        //            firstFrame.Image.Dispose();
        //            firstFrame.AuxImage?.Dispose();

        //            int rgbLen = exportH * exportW * 3;
        //            var rgbBuffer = new byte[rgbLen];
        //            var cornersBuffer = new float[8];
        //            var labelsBuffer = new float[attrCount];
        //            var labelMaskBuffer = new byte[attrCount];

        //            // Prebuilt “empty” buffers (avoid huge Array.Clear on missing frames)
        //            var zeroRgb = new byte[rgbLen];
        //            var zeroCorners = new float[8];
        //            var zeroLabels = new float[attrCount];
        //            var zeroMask = new byte[attrCount];

        //            using var rgbMat = new Mat(exportH, exportW, MatType.CV_8UC3);

        //            int totalPasses = colours.Length;
        //            long totalWork = (long)FrameCount * totalPasses;

        //            const int reportEveryNFrames = 200; // increase = faster, less UI churn

        //            for (int pass = 0; pass < totalPasses; pass++)
        //            {
        //                ct.ThrowIfCancellationRequested();

        //                string colourName = colours[pass];

        //                // Set ONLY the scalar used by DrawSurfaceMask (no UI cycling required)
        //                _backgroundScalar = colourName switch
        //                {
        //                    "White" => new Scalar(255, 255, 255),
        //                    "Black" => new Scalar(0, 0, 0),
        //                    "Gray" => new Scalar(128, 128, 128),
        //                    "SaddleBrown" => new Scalar(139, 69, 19),
        //                    "DarkGreen" => new Scalar(0, 100, 0),
        //                    "Tan" => new Scalar(210, 180, 140),
        //                    _ => new Scalar(255, 255, 255),
        //                };

        //                // Optional: update UI colours once per pass for clarity (non-blocking)
        //                dispatcher.BeginInvoke(new Action(() =>
        //                {
        //                    BackgroundColor = colourName;
        //                    ForegroundColor = colourName switch
        //                    {
        //                        "White" => "Black",
        //                        "Tan" => "Black",
        //                        _ => "White",
        //                    };
        //                }));

        //                string outPath = Path.Combine(baseDir, $"{baseName}_{colourName}.h5");

        //                using var session = SessionManager.CreateNew(
        //                    path: outPath,
        //                    frameCount: (ulong)FrameCount,
        //                    height: exportH,
        //                    width: exportW,
        //                    attrCount: attrCount,
        //                    chunkFrames: 16);

        //                for (int frameIndex = 0; frameIndex < FrameCount; frameIndex++)
        //                {
        //                    ct.ThrowIfCancellationRequested();

        //                    FrameData? frameData = null;

        //                    try
        //                    {
        //                        frameData = _frameSource.GetFrameForExport(frameIndex);

        //                        if (frameData?.Image == null || frameData.Image.Empty())
        //                        {
        //                            session.WriteFrame((ulong)frameIndex, zeroRgb, zeroCorners, zeroLabels, zeroMask);
        //                            continue;
        //                        }

        //                        int yOffset = frameData.AuxImage?.Height ?? 0;

        //                        // Crop out the text strip (top yOffset pixels) but keep full width.
        //                        using var cropped = GetBelowDatabarRoi(frameData.Image, yOffset);

        //                        // Draw mask in cropped coordinates (no horizontal pad during export)
        //                        DrawSurfaceMask(cropped, 0, frameData);

        //                        // Convert for writing
        //                        Cv2.CvtColor(cropped, rgbMat, ColorConversionCodes.BGR2RGB);
        //                        Marshal.Copy(rgbMat.Data, rgbBuffer, 0, rgbLen);

        //                        // Corners (TL,TR,BR,BL) in padded coordinates
        //                        var tl = frameData.TopLeft;
        //                        var tr = frameData.TopRight;
        //                        var br = frameData.BottomRight;
        //                        var bl = frameData.BottomLeft;

        //                        cornersBuffer[0] = (float)(tl?.X ?? 0);
        //                        cornersBuffer[1] = (float)(tl?.Y ?? 0);
        //                        cornersBuffer[2] = (float)(tr?.X ?? 0);
        //                        cornersBuffer[3] = (float)(tr?.Y ?? 0);
        //                        cornersBuffer[4] = (float)(br?.X ?? 0);
        //                        cornersBuffer[5] = (float)(br?.Y ?? 0);
        //                        cornersBuffer[6] = (float)(bl?.X ?? 0);
        //                        cornersBuffer[7] = (float)(bl?.Y ?? 0);

        //                        // Labels + mask (overwrite fully; no clears)
        //                        labelMaskBuffer[0] = frameData.StylusX.HasValue ? (byte)1 : (byte)0;
        //                        labelsBuffer[0] = frameData.StylusX.GetValueOrDefault();

        //                        labelMaskBuffer[1] = frameData.StylusY.HasValue ? (byte)1 : (byte)0;
        //                        labelsBuffer[1] = frameData.StylusY.GetValueOrDefault();

        //                        labelMaskBuffer[2] = frameData.StylusPressure.HasValue ? (byte)1 : (byte)0;
        //                        labelsBuffer[2] = frameData.StylusPressure.GetValueOrDefault();

        //                        labelMaskBuffer[3] = frameData.StylusTiltX.HasValue ? (byte)1 : (byte)0;
        //                        labelsBuffer[3] = frameData.StylusTiltX.GetValueOrDefault();

        //                        labelMaskBuffer[4] = frameData.StylusTiltY.HasValue ? (byte)1 : (byte)0;
        //                        labelsBuffer[4] = frameData.StylusTiltY.GetValueOrDefault();

        //                        session.WriteFrame((ulong)frameIndex, rgbBuffer, cornersBuffer, labelsBuffer, labelMaskBuffer);
        //                    }
        //                    finally
        //                    {
        //                        if (frameData != null)
        //                        {
        //                            frameData.Image?.Dispose();
        //                            frameData.AuxImage?.Dispose();
        //                        }
        //                    }

        //                    // Progress update (throttled + non-blocking)
        //                    if (frameIndex == 0 || frameIndex == FrameCount - 1 || (frameIndex % reportEveryNFrames == 0))
        //                    {
        //                        long done = (long)pass * FrameCount + (frameIndex + 1);
        //                        double pct = (totalWork <= 0) ? 100.0 : (done * 100.0 / totalWork);
        //                        var status = $"Exporting {colourName} ({pass + 1}/{totalPasses}) - frame {frameIndex + 1}/{FrameCount}";

        //                        dispatcher.BeginInvoke(new Action(() =>
        //                        {
        //                            ExportProgress = pct;
        //                            ExportStatus = status;
        //                        }));
        //                    }
        //                }
        //            }

        //            System.Windows.Application.Current.Dispatcher.BeginInvoke(new Action(() =>
        //            {
        //                ExportProgress = 100;
        //                ExportStatus = "Export complete.";
        //            }));
        //        }
        //        catch (OperationCanceledException)
        //        {
        //            System.Windows.Application.Current.Dispatcher.BeginInvoke(new Action(() =>
        //            {
        //                ExportStatus = "Export cancelled.";
        //            }));
        //        }
        //        catch (Exception ex)
        //        {
        //            System.Windows.Application.Current.Dispatcher.BeginInvoke(new Action(() =>
        //            {
        //                ExportStatus = "Export failed: " + ex.Message;
        //                System.Windows.MessageBox.Show("Export failed: " + ex);
        //            }));
        //        }
        //        finally
        //        {
        //            System.Windows.Application.Current.Dispatcher.BeginInvoke(new Action(() =>
        //            {
        //                // restore state
        //                ShowSurfaceMask = prevShowSurfaceMask;
        //                _backgroundScalar = prevBackgroundScalar;
        //                BackgroundColor = prevBackgroundColor;
        //                ForegroundColor = prevForegroundColor;
        //                IsExporting = false;
        //            }));

        //            if (ReferenceEquals(_playCts, newCts))
        //                _playCts = null;

        //            newCts.Dispose();
        //        }
        //    }, ct);
        //}

        private static Mat GetBelowDatabarRoi(Mat src, int yOffset)
        {
            if (src == null) throw new ArgumentNullException(nameof(src));
            if (src.Empty()) throw new ArgumentException("src is empty", nameof(src));

            if (yOffset <= 0) return src; // ROI is the whole image
            yOffset = Math.Clamp(yOffset, 0, src.Rows - 1);

            var rect = new Rect(0, yOffset, src.Cols, src.Rows - yOffset);
            return new Mat(src, rect); // NO CLONE: view into src data
        }

        [ThreadStatic]
        private static Mat? s_paddedReuse;

        private void MarkInvalid()
        {
            // stub
        }

        private void ReprocessFrame()
        {
            // stub
            LoadFrame(CurrentFrameIndex);
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
        private Mat AddHorizontalPadding(Mat src, int padX)
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

        private static void MakeYellowPenMaskLab(Mat bgr,
                                                 Mat lab,     // CV_8UC3 scratch
                                                 Mat labA,    // CV_8UC1 scratch
                                                 Mat labB,    // CV_8UC1 scratch
                                                 Mat bGeAMask,// CV_8UC1 scratch
                                                 Mat penMask) // CV_8UC1 output)
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

        private int _perfMaskCounter;
        private void DrawSurfaceMask(Mat target, int padX, FrameData frameData)
        {            
            if (!ShowSurfaceMask)
                return;

            if (frameData.TopLeft is null || frameData.TopRight is null || frameData.BottomLeft is null || frameData.BottomRight is null)
                return;           

            // --- allocate / resize scratch Mats once ---
            void EnsureSize(ref Mat? m, MatType type)
            {
                if (m == null || m.Width != target.Width || m.Height != target.Height || m.Type() != type)
                {
                    m?.Dispose();
                    m = new Mat(target.Rows, target.Cols, type);
                }
            }

            EnsureSize(ref _hsv, MatType.CV_8UC3);
            EnsureSize(ref _ycrcb, MatType.CV_8UC3);

            EnsureSize(ref _surfaceMask, MatType.CV_8UC1);
            EnsureSize(ref _colorMask, MatType.CV_8UC1);
            EnsureSize(ref _tmpMask, MatType.CV_8UC1);
            EnsureSize(ref _skinMask, MatType.CV_8UC1);
            EnsureSize(ref _notSkinMask, MatType.CV_8UC1);
            EnsureSize(ref _pureGreenMask, MatType.CV_8UC1);

            // HSV channels + intermediate masks
            EnsureSize(ref _h, MatType.CV_8UC1);
            EnsureSize(ref _s, MatType.CV_8UC1);
            EnsureSize(ref _v, MatType.CV_8UC1);
            EnsureSize(ref _hueMask, MatType.CV_8UC1);
            EnsureSize(ref _brightMask, MatType.CV_8UC1);
            EnsureSize(ref _darkMask, MatType.CV_8UC1);
            EnsureSize(ref _highSatMask, MatType.CV_8UC1);
            EnsureSize(ref _darkHighSatMask, MatType.CV_8UC1);
            EnsureSize(ref _valueCondMask, MatType.CV_8UC1);

            EnsureSize(ref _tubeMask, MatType.CV_8UC1);
            EnsureSize(ref _tipCandMask, MatType.CV_8UC1);
            EnsureSize(ref _ccLabels, MatType.CV_32SC1);
            EnsureSize(ref _ccStats, MatType.CV_32SC1);
            EnsureSize(ref _ccCentroids, MatType.CV_64FC1);

            EnsureSize(ref _lab, MatType.CV_8UC3);
            EnsureSize(ref _labA, MatType.CV_8UC1);
            EnsureSize(ref _labB, MatType.CV_8UC1);
            EnsureSize(ref _penMaskLab, MatType.CV_8UC1);
            EnsureSize(ref _bGeAMask, MatType.CV_8UC1);

            EnsureSize(ref _bgBgr, MatType.CV_8UC3);
            EnsureSize(ref _bg16, MatType.CV_8UC3);
            EnsureSize(ref _bgGrad16, MatType.CV_16SC3);
            EnsureSize(ref _noise8s, MatType.CV_8SC3);
            EnsureSize(ref _noise16s, MatType.CV_8SC3);
            EnsureSize(ref _col16, MatType.CV_16SC1);
            EnsureSize(ref _grad1_16, MatType.CV_16SC1);                        

            // --- polygon + clip ---
            var bl = Point2f.FromPoint(frameData.BottomLeft.Value) + new Point2f(padX, 0);
            var br = Point2f.FromPoint(frameData.BottomRight.Value) + new Point2f(padX, 0);
            var tr = Point2f.FromPoint(frameData.TopRight.Value) + new Point2f(padX, 0);
            var tl = Point2f.FromPoint(frameData.TopLeft.Value) + new Point2f(padX, 0);

            var poly = new List<Point2f>(4) { bl, br, tr, tl };
            var clipRect = new Rect2f(0, 0, target.Width, target.Height);
            var clipped = PolygonClipper.ClipToRect(poly, clipRect);
            if (clipped.Count < 3)
                return;

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
            Cv2.InRange(_h!, new Scalar(39), new Scalar(108), _hueMask!);

            // brightMask = V > 40
            Cv2.Threshold(_v!, _brightMask!, 40, 255, ThresholdTypes.Binary);

            // darkMask = V <= 40
            Cv2.Threshold(_v!, _darkMask!, 40, 255, ThresholdTypes.BinaryInv);

            // highSatMask = S > 27
            Cv2.Threshold(_s!, _highSatMask!, 27, 255, ThresholdTypes.Binary);

            // darkHighSatMask = darkMask & highSatMask
            Cv2.BitwiseAnd(_darkMask!, _highSatMask!, _darkHighSatMask!);

            // valueCondMask = brightMask | darkHighSatMask
            Cv2.BitwiseOr(_brightMask!, _darkHighSatMask!, _valueCondMask!);

            // colorMask = hueMask & valueCondMask
            Cv2.BitwiseAnd(_hueMask!, _valueCondMask!, _colorMask!);

            // --- Pen highlight (yellow-vs-skin OR legacy red) ---
            if (HasYellowBackground)
            {                
                // Yellow pen (Lab) mask: pen pixels not on skin
                MakeYellowPenMaskLab(
                    target,        // ideally your original frame (pre-green), but target is OK here since it hasn't been modified yet
                    _lab!,
                    _labA!,
                    _labB!,
                    _bGeAMask!,
                    _penMaskLab!);

                // Restrict to surface
                Cv2.BitwiseAnd(_penMaskLab!, _surfaceMask!, _penMaskLab!);

                // Optional: light cleanup
                Cv2.MedianBlur(_penMaskLab!, _penMaskLab!, 3);

                // Visualise pen pixels (red overlay)
                target.SetTo(new Scalar(0, 0, 255), _penMaskLab!);                
            }
            
            // restrict to polygon
            Cv2.BitwiseAnd(_colorMask!, _surfaceMask!, _colorMask!);

            // apply green replacement
            EnsureSize(ref _bgBgr, MatType.CV_8UC3);
            if (_bgNoise == null || _bgNoise.Width != target.Width || _bgNoise.Height != target.Height || _bgNoise.Type() != MatType.CV_8SC3)
            {
                _bgNoise?.Dispose();
                _bgNoise = new Mat(target.Rows, target.Cols, MatType.CV_8SC3);
            }          

            BuildBackgroundFast(_bgBgr!, _backgroundScalar, noiseAmp: 2, gradAmp: 6, frameIndex: frameData.FrameIndex);            
            
            _bgBgr!.CopyTo(target, _colorMask!);            
            
            // --- kill remaining near-green anywhere ---
            Cv2.InRange(target, new Scalar(0, 250, 0), new Scalar(15, 255, 25), _pureGreenMask!);
            Cv2.Dilate(_pureGreenMask!, _pureGreenMask!, KernelGreen, iterations: 1);
            target.SetTo(new Scalar(40, 40, 40), _pureGreenMask!);
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

            // Start from constant base in 16S (note: SetTo works fine on 16S mats)
            _bgBaseGrad16.SetTo(baseBgr);

            // Ensure gradient is up to date for this size/gradAmp
            EnsureGradient(rows, cols, gradAmp); // should set _bgGrad16 as CV_16SC3

            // Cache base+gradient
            Cv2.Add(_bgBaseGrad16, _bgGrad16!, _bgBaseGrad16);

            _bgRowsCached = rows;
            _bgColsCached = cols;
            _bgGradAmpCached = gradAmp;
            _bgBaseCached = baseBgr;
        }


        // NOTE: added frameIndex parameter for deterministic noise selection
        private void BuildBackgroundFast(Mat bgBgr8u, Scalar baseBgr, int noiseAmp, int gradAmp, int frameIndex)
        {
            int rows = bgBgr8u.Rows;
            int cols = bgBgr8u.Cols;

            // Ensure scratch output (16S working buffer)
            if (_bg16 == null || _bg16.Rows != rows || _bg16.Cols != cols || _bg16.Type() != MatType.CV_16SC3)
            {
                _bg16?.Dispose();
                _bg16 = new Mat(rows, cols, MatType.CV_16SC3);
            }
            
            // Cache base+gradient (built only when size/base/gradAmp changes)
            EnsureBaseGrad(rows, cols, baseBgr, gradAmp);            

            // Start from cached base+grad
            _bgBaseGrad16!.CopyTo(_bg16);
            
            // Cache noise bank (built only when size/noiseAmp changes)
            EnsureNoiseBank(rows, cols, noiseAmp);            
          
            // Add deterministic noise for this frame
            var noise16 = _noise16Bank![frameIndex % NoiseBankSize];
            Cv2.Add(_bg16!, noise16, _bg16!);            
            
            // Convert back with saturation to 8U
            _bg16!.ConvertTo(bgBgr8u, MatType.CV_8UC3);            
        }


        // Cache key for gradient        
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

            // Reuse a cached Hx1 column and repeated HxW gradient
            if (_col16 == null || _col16.Rows != rows || _col16.Cols != 1 || _col16.Type() != MatType.CV_16SC1)
            {
                _col16?.Dispose();
                _col16 = new Mat(rows, 1, MatType.CV_16SC1);
            }

            // Fill column offsets
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

            // Repeat column into full image (HxW)
            Cv2.Repeat(_col16, 1, cols, _grad1_16);

            // Merge into 3 channels WITHOUT Clone() allocations
            Cv2.Merge(new[] { _grad1_16, _grad1_16, _grad1_16 }, _bgGrad16);

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
                catch { /* ignore */ }
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
