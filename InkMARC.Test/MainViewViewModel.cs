using System.IO;
using System.Text.RegularExpressions;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using OpenCvSharp;
using Microsoft.WindowsAPICodePack.Dialogs;
using System.Collections.ObjectModel;
using OpenCvSharp.WpfExtensions;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Diagnostics;
using System.Drawing;
using System.Security.Cryptography.X509Certificates;

namespace InkMARC.Test
{
    public enum PlaybackState : int
    {
        Unknown = -1,
        Video = 0,
        Images = 1,
        Live = 2
    }

    internal partial class MainViewViewModel : ObservableObject
    {
        #region Private Data

        private VideoCapture? videoCapture;
        private int frameCount = 0;
        private int frameIndex = 0;
        private double framesPerSecond = 0;
        private bool isTouched = false;
        private int rotation;
        private bool currentState = false;
        private ImageSource? _currentImage;
        private int lastFrameIndex = -1;
        private string[] imageFiles;
        private int currentIndex = 0;
        private Dictionary<int, BitmapImage> imageCache = new Dictionary<int, BitmapImage>();
        private const int CacheSize = 9;

        // Instantiate your TensorFlow predictor (assumes you have implemented TensorFlowImagePredictor)
        private readonly OnnxImagePredictor _imagePredictor = new OnnxImagePredictor();

        #endregion

        #region Public Properties

        /// <summary>
        /// Gets or sets the path to the video file.
        /// </summary>
        public string VideoPath { get; set; }

        public int FrameIndex
        {
            get => frameIndex;
            set => frameIndex = value;
        }

        public int FrameCount => frameCount;

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

        public ObservableCollection<ImageSource> ImageSources { get; private set; } = new ObservableCollection<ImageSource>();

        public int Rotation => rotation;

        public PlaybackState PlaybackState { get; set; } = PlaybackState.Unknown;

        #endregion

        #region Relay Commands

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
        public void LoadImageFolder(object parameter)
        {
            CommonOpenFileDialog fileDialog = new CommonOpenFileDialog
            {
                Title = "Select Image Folder",
                IsFolderPicker = true
            };

            if (fileDialog.ShowDialog() != CommonFileDialogResult.Ok || string.IsNullOrEmpty(fileDialog.FileName))
            {
                return;
            }

            string directory = fileDialog.FileName;

            if (!string.IsNullOrEmpty(directory))
            {
                PlaybackState = PlaybackState.Images;
                imageFiles = Directory.GetFiles(directory, "*.*", SearchOption.AllDirectories)
                    .Where(file => Regex.IsMatch(file, @"(.*\.png|.*\.jpg|.*\.jpeg|.*\.bmp|.*\.gif)"))
                    .OrderBy(file => file)
                    .ToArray();

                if (imageFiles.Length == 0)
                {
                    return;
                }

                currentIndex = 0;
                LoadImage(currentIndex);
                PreloadNearbyImages();
                frameIndex = 0;
                frameCount = imageFiles.Length;
                OnPropertyChanged(nameof(FrameCount));
                OnPropertyChanged(nameof(FrameIndex));

            }
        }

        [RelayCommand]
        public void LoadVideo(object parameter)
        {
            // Open a folder selection dialog
            CommonOpenFileDialog fileDialog = new CommonOpenFileDialog
            {
                Title = "Select Video File",
                IsFolderPicker = false,
                DefaultExtension = ".mp4",
                Filters = { new CommonFileDialogFilter("Video Files", "*.mp4;*.avi;*.mov") }
            };

            if (fileDialog.ShowDialog() != CommonFileDialogResult.Ok || string.IsNullOrEmpty(fileDialog.FileName))
            {
                return; // User canceled
            }

            string videoPath = fileDialog.FileName;

            // Load the video
            if (!string.IsNullOrEmpty(videoPath))
            {
                PlaybackState = PlaybackState.Video;
                if (videoCapture != null)
                {
                    videoCapture.Dispose();
                }
                videoCapture = new VideoCapture(videoPath);
                frameIndex = 0;
                frameCount = videoCapture is null ? 0 : (int)videoCapture.Get(VideoCaptureProperties.FrameCount);
                OnPropertyChanged(nameof(FrameCount));
                OnPropertyChanged(nameof(FrameIndex));                
                UpdateImage();
            }
        }

        [RelayCommand]
        public void MoveOffset(string parameter)
        {
            if (int.TryParse(parameter, out int adjust))
            {
                switch (PlaybackState)
                {
                    case PlaybackState.Video:
                        MoveVideoOffset(adjust);
                        break;
                    case PlaybackState.Images:
                        MoveImageOffset(adjust);
                        break;
                    default:
                        break;
                }

                UpdateImage();
                // After updating the image, run the prediction.
                PredictCurrentImagePressure();
            }
        }

        [RelayCommand]
        public void TestFolder(string parameter)
        {
            if (Boolean.TryParse(parameter, out bool expected))
            {
                int correct = 0;
                List<double> accuracies = new List<double>();

                CommonOpenFileDialog fileDialog = new CommonOpenFileDialog
                {
                    Title = "Select Image Folder",
                    IsFolderPicker = true
                };

                if (fileDialog.ShowDialog() != CommonFileDialogResult.Ok || string.IsNullOrEmpty(fileDialog.FileName))
                {
                    return;
                }

                string directory = fileDialog.FileName;

                if (!string.IsNullOrEmpty(directory))
                {
                    PlaybackState = PlaybackState.Images;
                    imageFiles = Directory.GetFiles(directory, "*.*", SearchOption.AllDirectories)
                        .Where(file => Regex.IsMatch(file, @"(.*\.png|.*\.jpg|.*\.jpeg|.*\.bmp|.*\.gif)"))
                        .OrderBy(file => file)
                        .ToArray();

                    if (imageFiles.Length == 0)
                    {
                        return;
                    }

                    Debug.WriteLine("Started");
                    foreach (string imageFile in imageFiles)
                    {
                        BitmapImage image = LoadBitmap(imageFile);
                        float pressure = _imagePredictor.PredictPressure(BitmapFromSource(image));
                        accuracies.Add(pressure);
                        if (pressure > 0.00175 && expected)
                        {
                            correct++;
                        }
                        else if (pressure <= 0.00175 && !expected)
                        {
                            correct++;
                        }
                    }
                    Debug.WriteLine("Finished");

                    double average = accuracies.Average();
                    double max = accuracies.Max();
                    double min = accuracies.Min();
                    Debug.WriteLine($"Average: {average}, Max: {max}, Min: {min}, Accuracy: {(double)correct / (double)(accuracies.Count)}");
                }
            }
        }

        #endregion

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

        private void MoveVideoOffset(int adjust)
        {
            if ((frameIndex + adjust) >= 0 && (frameIndex + adjust) < FrameCount)
            {
                frameIndex += adjust;
                OnPropertyChanged(nameof(FrameIndex));
            }
        }

        private void MoveImageOffset(int adjust)
        {
            if ((FrameIndex + adjust) >= 0 && (frameIndex + adjust) < FrameCount)
            {
                if (adjust > 0)
                {
                    NextImage();
                    adjust = 1;
                }
                else if (adjust < 0)
                {
                    PreviousImage();
                    adjust = -1;
                }
                frameIndex += adjust;
                OnPropertyChanged(nameof(FrameIndex));
            }
        }

        private void UpdateImage()
        {
            switch (PlaybackState)
            {
                case PlaybackState.Video:
                    CurrentImage = GetImage();
                    break;
                case PlaybackState.Images:
                    break;
                default:
                    break;
            }
            OnPropertyChanged(nameof(CurrentImage));
        }

        private void PredictCurrentImagePressure()
        {
            // Ensure we have an image
            if (CurrentImage is BitmapSource bitmapSource)
            {
                using (Bitmap bmp = BitmapFromSource(bitmapSource))
                {
                    // Call the predictor to get the pressure value.
                    float pressure = _imagePredictor.PredictPressure(bmp);
                    Debug.WriteLine("Predicted Pressure: " + pressure);

                    currentState = pressure > 0.00175;
                    OnPropertyChanged(nameof(CurrentState));

                    // You can now use the 'pressure' value (for example, update a property or adjust the UI).
                    // For instance, you might have a property:
                    // PredictedPressure = pressure;
                    // OnPropertyChanged(nameof(PredictedPressure));
                }
            }
        }

        // Helper method to convert a BitmapSource to a System.Drawing.Bitmap.
        private static Bitmap BitmapFromSource(BitmapSource bitmapsource)
        {
            if (bitmapsource.Format != PixelFormats.Bgra32)
            {
                bitmapsource = new FormatConvertedBitmap(bitmapsource, PixelFormats.Bgra32, null, 0);
            }
            using (MemoryStream ms = new MemoryStream())
            {
                BitmapEncoder encoder = new BmpBitmapEncoder();
                encoder.Frames.Add(BitmapFrame.Create(bitmapsource));
                encoder.Save(ms);
                // Create a new Bitmap from the memory stream.
                return new Bitmap(ms);
            }
        }

        private void LoadImage(int index)
        {
            if (imageFiles == null || imageFiles.Length == 0)
                return;

            if (imageCache.TryGetValue(index, out BitmapImage cachedImage))
            {
                CurrentImage = cachedImage;
            }
            else
            {
                CurrentImage = LoadBitmap(imageFiles[index]);
                imageCache[index] = (BitmapImage)CurrentImage;
            }
        }

        public void NextImage()
        {
            if (imageFiles == null || imageFiles.Length == 0)
                return;

            currentIndex = (currentIndex + 1) % imageFiles.Length;
            LoadImage(currentIndex);
            PreloadNearbyImages();
        }

        public void PreviousImage()
        {
            if (imageFiles == null || imageFiles.Length == 0)
                return;

            currentIndex = (currentIndex - 1 + imageFiles.Length) % imageFiles.Length;
            LoadImage(currentIndex);
            PreloadNearbyImages();
        }

        private void PreloadNearbyImages()
        {
            Task.Run(() =>
            {
                var preloadIndexes = new int[]
                {
                (currentIndex - 1 + imageFiles.Length) % imageFiles.Length, // Previous
                (currentIndex + 1) % imageFiles.Length  // Next
                };

                foreach (int index in preloadIndexes)
                {
                    if (!imageCache.ContainsKey(index))
                    {
                        BitmapImage image = LoadBitmap(imageFiles[index]);
                        imageCache[index] = image;
                    }
                }

                CleanupCache();
            });
        }

        private BitmapImage LoadBitmap(string path)
        {
            BitmapImage bitmap = new BitmapImage();
            bitmap.BeginInit();
            bitmap.UriSource = new Uri(path);
            bitmap.CacheOption = BitmapCacheOption.OnLoad;  // Avoid memory buildup
            bitmap.EndInit();
            bitmap.Freeze(); // Makes it thread-safe
            return bitmap;
        }

        private void CleanupCache()
        {
            if (imageCache.Count > CacheSize)
            {
                var keysToRemove = imageCache.Keys
                    .Where(index => index != currentIndex && index != (currentIndex - 1 + imageFiles.Length) % imageFiles.Length && index != (currentIndex + 1) % imageFiles.Length)
                    .ToList();

                foreach (var key in keysToRemove)
                {
                    imageCache.Remove(key);
                }
            }
        }
    }
}