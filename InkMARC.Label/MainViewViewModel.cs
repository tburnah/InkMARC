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

namespace InkMARC.Label
{
    internal partial class MainViewViewModel : ObservableObject
    {
        private VideoCapture? videoCapture;
        private int frameCount = 0;
        private int frameIndex = 0;
        private string recordName = string.Empty;

        private JsonSerializerOptions jsonSerializerOptions = new JsonSerializerOptions
        {
            WriteIndented = true
        };

        public int FrameIndex
        {
            get => frameIndex;
            set => frameIndex = value;
        }
        
        public int FrameCount => frameCount;
        public int PointCount => Points.Count;

        public int StartFrame { get; set; }

        public int StopFrame { get; set; }

        SortedDictionary<int, bool> stateChanges = new SortedDictionary<int, bool>();

        [RelayCommand]
        private void ClearData()
        {
            stateChanges.Clear();
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

                _parametersInitialized = true;

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
            ShowProgressBar = false;
        }

        private async Task ExportData2Async()
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
                // Set starting frame index.
                frameIndex = StartFrame;

                // Get the first frame to initialize the reusable image resources.
                Bitmap? firstImage = GetImage2();
                if (firstImage is not null)
                {
                    // Use the first image to initialize your datasets.
                    DataSave.InitializeChunkedDatasets(firstImage, GetStateAtFrame(frameIndex));
                    firstImage.Dispose();

                    progCounter++;
                    ((IProgress<int>)progress).Report(progCounter);
                    frameIndex++; // advance frame index after processing the first frame.
                }
                else
                {
                    Console.WriteLine("Failed to retrieve first frame.");
                    return;
                }

                // Process the remaining frames.
                for (; frameIndex < StopFrame; frameIndex++)
                {
                    Bitmap? image = GetImage2();
                    if (image is not null)
                    {
                        DataSave.WriteFrame(image, GetStateAtFrame(frameIndex));
                        image.Dispose();
                        progCounter++;

                        // Only update the UI every 10 frames (or whatever frequency you choose)
                        if (progCounter % 10 == 0)
                        {
                            ((IProgress<int>)progress).Report(progCounter);
                        }
                    }
                    else
                    {
                        Console.WriteLine($"Skipping frame {frameIndex} because image retrieval failed.");
                    }
                }
            });

            // Finalize the datasets and clean up.
            DataSave.FinalizeDatasets();
            ShowProgressBar = false;
        }

        private bool _parametersInitialized = false;
        private int _frameWidth;
        private int _frameHeight;
        private int _squareSize;
        private Mat _squareFrame = null!;
        private Mat _rotatedFrame = null!;
        private Rect _roi;
        private Mat _rotationMatrix = null!;

        private Bitmap? GetImage2()
        {
            if (videoCapture is null)
                return null;

            // Create a new Mat to hold the raw frame.
            Mat frame = new();

            // For the very first frame, explicitly position the capture.
            if (!_parametersInitialized)
            {
                if (!videoCapture.Set(VideoCaptureProperties.PosFrames, frameIndex))
                {
                    Console.WriteLine($"Failed to set frame index {frameIndex}");
                    return null;
                }
            }

            // For subsequent frames, assume the capture reads sequentially.
            bool success = videoCapture.Read(frame);
            if (!success || frame.Empty())
            {
                Console.WriteLine($"Failed to read frame at index {frameIndex}");
                return null;
            }

            // On the first successful read, precompute and allocate reusable Mats.
            if (!_parametersInitialized)
            {
                _frameWidth = frame.Width;
                _frameHeight = frame.Height;
                _squareSize = Math.Max(_frameWidth, _frameHeight);

                // Allocate a square frame (filled with black) and a destination Mat for rotation.
                _squareFrame = new Mat(new OpenCvSharp.Size(_squareSize, _squareSize), frame.Type(), Scalar.Black);
                _rotatedFrame = new Mat(new OpenCvSharp.Size(_squareSize, _squareSize), frame.Type());

                // Calculate ROI to center the original frame into the square.
                int xOffset = (_squareSize - _frameWidth) / 2;
                int yOffset = (_squareSize - _frameHeight) / 2;
                _roi = new Rect(xOffset, yOffset, _frameWidth, _frameHeight);

                // Compute the center of the square and the rotation matrix.
                Point2f center = new Point2f(_squareSize / 2f, _squareSize / 2f);
                _rotationMatrix = Cv2.GetRotationMatrix2D(center, rotation, 1.0);

                _parametersInitialized = true;
            }

            // Reuse the preallocated square frame:
            _squareFrame.SetTo(Scalar.Black);
            // Copy the current frame into the center of the square.
            using (Mat roiMat = new Mat(_squareFrame, _roi))
            {
                frame.CopyTo(roiMat);
            }

            // Rotate the square frame using the precomputed rotation matrix.
            Cv2.WarpAffine(_squareFrame, _rotatedFrame, _rotationMatrix, _squareFrame.Size());

            // Convert the final rotated Mat to a Bitmap.
            Bitmap bitmap = BitmapConverter.ToBitmap(_rotatedFrame);
            return bitmap;
        }

        [RelayCommand]
        private async Task ExportDataAsync()
        {
            OnPropertyChanged(nameof(MaxProgress));
            ShowProgressBar = true;

            // Create the file.
            DataSave.CreateFile(recordName);

            var progress = new Progress<int>(value =>
            {
                FrameIndex = value;
                OnPropertyChanged(nameof(FrameIndex));
            });
            int progCounter = 0;

            // Run the inner data loop on a background thread.
            await Task.Run(() =>
            {
                frameIndex = StartFrame;
                Bitmap? image = GetImage();
                if (image is not null)
                {
                    DataSave.InitializeChunkedDatasets(image, GetStateAtFrame(frameIndex));
                    frameIndex++;
                }

                for (frameIndex = StartFrame; frameIndex < StopFrame; frameIndex++)
                {
                    // If GetImage or GetStateAtFrame access UI elements,
                    // consider moving that logic outside of this loop or
                    // dispatching those calls back to the UI thread.
                    image = GetImage();
                    if (image is not null)
                    {
                        DataSave.WriteFrame(image, GetStateAtFrame(frameIndex));
                        image.Dispose();
                        ((IProgress<int>)progress).Report(++progCounter);
                    }
                }
            });

            // Finalize (dispose) the chunked datasets and close the file.
            DataSave.FinalizeDatasets();
            ShowProgressBar = false;
        }

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
            stateChanges.Add(frameIndex, isTouched);
            UpdateCurrentState();
        }

        public bool GetStateAtFrame(int frame)
        {
            // If there are no recorded state changes, decide on a default:
            if (stateChanges.Count == 0)
                return false; // or however you wish to default

            // Iterate in reverse order of keys.
            foreach (var kvp in stateChanges.Reverse())
            {
                if (kvp.Key <= frame)
                    return kvp.Value;
            }
            // If no state change has occurred before this frame, return default.
            return false;
        }

        public bool currentState = false;
        public bool CurrentState => currentState;

        private void UpdateCurrentState()
        {
            if (SetProperty(ref currentState, GetStateAtFrame(frameIndex)))
                OnPropertyChanged(nameof(CurrentState));
        }

        public List<InkMARCPoint> Points { get; private set; } = new List<InkMARCPoint>();

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
                }
            }
        }

        [RelayCommand]
        public void LoadVideo(object parameter)
        {           
            // Open a file dialog to select a video file
            OpenFileDialog openFileDialog = new OpenFileDialog
            {
                Filter = "Video Files|*.mp4;*.avi;*.mov"
            };
            var result = openFileDialog.ShowDialog();
            if (result == true)
            {
                string videoPath = openFileDialog.FileName;
                // Save the video file name without the extension into recordName
                recordName = Path.ChangeExtension(videoPath, ".h5");

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