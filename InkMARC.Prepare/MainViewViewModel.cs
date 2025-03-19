using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Diagnostics.Metrics;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Reflection.Metadata;
using System.Text;
using System.Text.Json;
using System.Text.RegularExpressions;
using System.Threading.Tasks;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using CommunityToolkit.Mvvm;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using InkMARC.Models.Primatives;
using Microsoft.Win32;
using OpenCvSharp;
using OpenCvSharp.Detail;
using OpenCvSharp.Extensions;

namespace InkMARC.Prepare
{
    internal partial class MainViewViewModel : ObservableObject
    {
        private VideoCapture? videoCapture;
        private int frameCount = 0;
        private int pointIndex = 0;
        private int millisecondOffsets = 0;
        private string recordName = string.Empty;
        private const string IDPattern = @"_\d+_(\w+)_\d+\.json";

        private JsonSerializerOptions jsonSerializerOptions = new JsonSerializerOptions
        {
            WriteIndented = true
        };

        public int FrameCount => frameCount;
        public int PointCount => Points.Count;

        public int MillisecondOffsets
        {
            get
            {
                return millisecondOffsets;
            }
            set
            {
                SetProperty(ref millisecondOffsets, value);
            }
        }

        public string PointData
        {
            get
            {
                if (Points is null || Points.Count == 0)
                {
                    return string.Empty;
                }
                return JsonSerializer.Serialize(Points[pointIndex], jsonSerializerOptions);
            }
        }

        public int MaxProgress => Points.Count * 3;

        public float PredictedPressure => predictedPressure;       
        private float predictedPressure = 0.0f;

        public bool ShowProgressBar
        {
            get => showProgressBar;
            private set
            {
                SetProperty(ref showProgressBar, value);
                OnPropertyChanged(nameof(ShowCurrentIndex));
            }
        }

        public bool ShowCurrentIndex
        {
            get => !showProgressBar;
        }

        public List<InkMARCPoint> Points { get; private set; } = new List<InkMARCPoint>();

        public int PointIndex { get => pointIndex; set => pointIndex = value; }

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

        private void UpdateImageForTimestamp(long timestampInMicroseconds)
        {
            if (videoCapture is null)
            {
                return;
            }

            // Calculate timestamp in milliseconds for OpenCV
            double timestampInMilliseconds = timestampInMicroseconds / 1000.0;

            // Set the position of the video capture
            videoCapture.Set(VideoCaptureProperties.PosMsec, timestampInMilliseconds);

            // Retrieve the frame
            Mat frame = new Mat();
            if (videoCapture.Read(frame))
            {
                // Convert the frame to ImageSource
                CurrentImage = ConvertMatToImageSource(frame);
                predictedPressure = predictedPressure > 0.85 ? 1 : 0;
                OnPropertyChanged(nameof(predictedPressure));
            }
            else
            {
                // If no frame is retrieved, clear the image
                CurrentImage = null;
            }
        }

        private Bitmap? GetImage(long timestampInMicroseconds)
        {
            if (videoCapture is null)
                return null;

            // Calculate timestamp in milliseconds for OpenCV
            double timestampInMilliseconds = timestampInMicroseconds / 1000.0;

            // Set the position of the video capture
            videoCapture.Set(VideoCaptureProperties.PosMsec, timestampInMilliseconds);

            // Retrieve the frame
            Mat frame = new();
            if (videoCapture.Read(frame))
            {
                // Convert the frame to ImageSource
                return OpenCvSharp.Extensions.BitmapConverter.ToBitmap(frame);
            }
            return null;
        }

        // Helper method to convert OpenCV Mat to WPF ImageSource
        private ImageSource ConvertMatToImageSource(Mat frame)
        {
            Bitmap bitmap = OpenCvSharp.Extensions.BitmapConverter.ToBitmap(frame);

            using (var memoryStream = new MemoryStream())
            {
                bitmap.Save(memoryStream, System.Drawing.Imaging.ImageFormat.Png);
                memoryStream.Seek(0, SeekOrigin.Begin);

                BitmapImage bitmapImage = new BitmapImage();
                bitmapImage.BeginInit();
                bitmapImage.CacheOption = BitmapCacheOption.OnLoad;
                bitmapImage.StreamSource = memoryStream;
                bitmapImage.EndInit();

                return bitmapImage;
            }
        }

        public void ExtractFrames(string videoPath, string outputFolder)
        {
            // Ensure output folder exists
            Directory.CreateDirectory(outputFolder);

            using var capture = new VideoCapture(videoPath);

            for (int frameIndex = 0; frameIndex < frameCount; frameIndex++)
            {
                using var frame = new Mat();
                bool success = capture.Read(frame);

                if (!success || frame.Empty())
                    break;

                // Get the current position in milliseconds
                double currentPosMsec = capture.Get(VideoCaptureProperties.PosMsec);

                // Convert to microseconds
                long currentPosUsec = (long)(currentPosMsec * 1000);

                // Build file name as the microseconds timestamp
                string fileName = $"frame_{currentPosUsec}.png";

                // Combine with output folder
                string framePath = Path.Combine(outputFolder, fileName);

                // Save the frame
                Cv2.ImWrite(framePath, frame);

                Console.WriteLine($"Extracted frame {frameIndex} at {currentPosMsec} ms ({currentPosUsec} µs) to {fileName}");
            }
        }

        [RelayCommand]
        public void MovePoint(string parameter)
        {
            if (int.TryParse(parameter, out int adjust))
            {
                if (adjust > 0)
                {
                    if (pointIndex < Points.Count)
                    {
                        pointIndex += adjust;
                        OnPropertyChanged(nameof(PointIndex));
                        OnPropertyChanged(nameof(PointData));
                        UpdateImageForTimestamp((long)Points[pointIndex].Timestamp + (long)millisecondOffsets * 1000);
                    }
                }
                else
                {
                    if (pointIndex + adjust >= 0)
                    {
                        pointIndex += adjust;
                        OnPropertyChanged(nameof(PointIndex));
                        OnPropertyChanged(nameof(PointData));
                        UpdateImageForTimestamp((long)Points[pointIndex].Timestamp + (long)millisecondOffsets * 1000);
                    }
                }
            }
        }

        [RelayCommand]
        public void MoveOffset(string parameter)
        {
            if (int.TryParse(parameter, out int adjust))
            {
                if (pointIndex >= 0 && pointIndex < Points.Count)
                {
                    millisecondOffsets += adjust;
                    OnPropertyChanged(nameof(MillisecondOffsets));
                    UpdateImageForTimestamp((long)Points[pointIndex].Timestamp + (long)millisecondOffsets * 1000);
                }
            }
        }

        [RelayCommand]
        public async Task SyncData(object parameter)
        {
            // Open a file dialog to select a video file
            SaveFileDialog saveFileDialog = new SaveFileDialog
            {
                Filter = "Hdf5 |*.h5",
                DefaultExt = "h5",
                Title = "Save data to HDF5 file"
            };
            bool? result = saveFileDialog.ShowDialog();
            if (result == true)
            {
                ShowProgressBar = true;
                DataSave.CreateFile(saveFileDialog.FileName);

                var progress = new Progress<int>(value =>
                {
                    PointIndex = value;
                    OnPropertyChanged(nameof(PointIndex));
                });

                int progCounter = 0;

                // Buffer for processed data
                var processedData = new ConcurrentQueue<(string, float[,,], float[])>();

                // Parallel processing of images
                await Task.Run(() =>
                {
                    const int batchSize = 50;
                    var enumerator = Points.GetEnumerator();
                    DataSave.StartRecordBatch();
                    while (true)
                    {
                        var Frames = new List<(Bitmap Frame, InkMARCPoint Point)>();
                        for (int i = 0; i < batchSize && enumerator.MoveNext(); i++)
                        {
                            var point = enumerator.Current;
                            var image = GetImage((long)point.Timestamp + (long)millisecondOffsets * 1000);
                            if (image is not null)
                            {
                                Frames.Add((image, point));
                            }                            
                            ((IProgress<int>)progress).Report(++progCounter);     
                        }

                        if (Frames.Count == 0)
                        {
                            break;
                        }

                        Parallel.ForEach(Frames, frame =>
                        {
                            try
                            {
                                string imageRecordName = $"{recordName}_{pointIndex}";
                                float[,,] imageData = DataSave.BitmapToFloatArray(frame.Frame);
                                float[] pointData = DataSave.InkMARCPointNormalized(frame.Point);
                                processedData.Enqueue((imageRecordName, imageData, pointData));
                                frame.Frame.Dispose();
                                ((IProgress<int>)progress).Report(++progCounter);
                            }
                            catch (Exception ex)
                            {
                                Console.WriteLine($"Error saving record: {ex.Message}");
                            }
                        });

                        while (processedData.TryDequeue(out var data))
                        {
                            DataSave.AddIndividualRecordBatch(data.Item1, data.Item2, data.Item3);
                            ((IProgress<int>)progress).Report(++progCounter);
                        }
                    }
                });
                DataSave.EndRecordBatch();
                DataSave.CloseFile();
                ShowProgressBar = false;
            }
        }

        //public async Task SyncData(object parameter)
        //{
        //    // Open a file dialog to select a video file
        //    SaveFileDialog saveFileDialog = new SaveFileDialog
        //    {
        //        Filter = "Hdf5 |*.h5",
        //        DefaultExt = "h5",
        //        Title = "Save data to HDF5 file"
        //    };
        //    bool? result = saveFileDialog.ShowDialog();
        //    if (result == true)
        //    {
        //        ShowProgressBar = true;
        //        DataSave.CreateFile(saveFileDialog.FileName);

        //        var progress = new Progress<int>(value =>
        //        {
        //            PointIndex = value;
        //            OnPropertyChanged(nameof(PointIndex));
        //        });

        //        // Run the loop on a separate thread
        //        await Task.Run(() =>
        //        {
        //            for (int i = 0; i < Points.Count; i++)
        //            {
        //                try
        //                {
        //                    var image = GetImage((long)Points[i].Timestamp + (long)millisecondOffsets * 1000);
        //                    if (image is not null)
        //                    {
        //                        DataSave.AddRecord(recordName, image, Points[i]);
        //                        image.Dispose();
        //                    }

        //                    ((IProgress<int>)progress).Report(i);
        //                }
        //                catch (Exception ex)
        //                {
        //                    Console.WriteLine($"Error saving record: {ex.Message}");
        //                }
        //            }
        //        });
        //        DataSave.CloseFile();
        //        ShowProgressBar = false;
        //    }
        //}

        [RelayCommand]
        public void LoadPoints(object parameter)
        {
            // Use FolderBrowserDialog to select a directory
            OpenFolderDialog openFolderDialog = new OpenFolderDialog()
            {
                Title = "Select a folder"
            };
            bool? result = openFolderDialog.ShowDialog();
            if (result == true)
            {
                string directoryPath = openFolderDialog.FolderName;

                // List to hold all points
                List<InkMARCPoint> allPoints = new List<InkMARCPoint>();

                List<string> fileNames = [];

                // Iterate through all JSON files in the directory
                foreach (string filePath in Directory.GetFiles(directoryPath, "*.json"))
                {
                    string fName = Path.GetFileName(filePath);
                    Match match = MyRegex().Match(fName);
                    if (match.Success)
                        fileNames.Add(match.Groups[1].Value);

                    string jsonContent = File.ReadAllText(filePath);
                    var drawingData = JsonSerializer.Deserialize<InkMARCData>(jsonContent);

                    if (drawingData?.DrawingLines != null)
                    {
                        foreach (var line in drawingData.DrawingLines)
                        {
                            if (line.Points != null)
                            {
                                allPoints.AddRange(line.Points);
                            }
                        }
                    }
                }

                // Find the most common identifier
                recordName = fileNames
                    .GroupBy(id => id)                // Group by each identifier
                    .OrderByDescending(g => g.Count()) // Order groups by their count in descending order
                    .FirstOrDefault()?.Key ?? string.Empty;           // Get the key (identifier) of the first group

                // Sort points by Timestamp
                var uniquePoints = new HashSet<ulong>();
                Points = [.. allPoints
                    .Where(p => uniquePoints.Add(p.Timestamp)) // Add returns false if Timestamp already exists
                    .OrderBy(p => p.Timestamp)];

                OnPropertyChanged(nameof(PointIndex));
                OnPropertyChanged(nameof(PointData));
                OnPropertyChanged(nameof(PointCount));
                OnPropertyChanged(nameof(MaxProgress));
            }
            // Open a file dialog to select a video file
            OpenFileDialog openFileDialog = new OpenFileDialog
            {
                Filter = "Video Files|*.mp4;*.avi;*.mov"
            };
            result = openFileDialog.ShowDialog();
            if (result == true)
            {
                string videoPath = openFileDialog.FileName;
                if (!string.IsNullOrEmpty(videoPath))
                {
                    if (videoCapture != null)
                    {
                        videoCapture.Dispose();
                    }
                    videoCapture = new VideoCapture(videoPath);
                    frameCount = videoCapture is null ? 0 : (int)videoCapture.Get(VideoCaptureProperties.FrameCount);
                    OnPropertyChanged(nameof(FrameCount));
                }
            }
        }

        [GeneratedRegex(IDPattern)]
        private static partial Regex MyRegex();
    }

    public class InkMARCData
    {
        public List<InkMARCDrawingLine> DrawingLines { get; set; }
    }
}