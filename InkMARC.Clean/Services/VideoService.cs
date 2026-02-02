using OpenCvSharp;
using System;
using System.Diagnostics;
using System.IO;
using InkMARC.Clean.Services.Interfaces;

namespace InkMARC.Clean.Services
{
    /// <summary>
    /// Provides video file access and frame extraction using OpenCV.
    /// Also provides a simple video writer for MJPEG AVI export.
    /// </summary>
    public class VideoService : IVideoService, IDisposable
    {
        private VideoCapture? _videoCapture;
        private VideoWriter? _videoWriter;
        private int _lastFrameIndex = -1;
        private readonly Mat _frameBuffer = new();

        /// <summary>
        /// Raised when the frame count changes after opening a new video.
        /// </summary>
        public event EventHandler<int>? FrameCountChanged;

        private int _frameCount = 0;
        /// <summary>
        /// Gets the total number of frames in the current video.
        /// </summary>
        public int FrameCount
        {
            get => _frameCount;
            private set
            {
                if (_frameCount != value)
                {
                    _frameCount = value;
                    FrameCountChanged?.Invoke(this, _frameCount);
                }
            }
        }

        /// <summary>
        /// Gets the frames per second of the current video.
        /// </summary>
        public double FramesPerSecond { get; private set; } = 0;

        /// <summary>
        /// Gets whether a video is currently open for reading.
        /// </summary>
        public bool IsOpen => _videoCapture?.IsOpened() ?? false;

        /// <summary>
        /// Whether a video writer is currently open for writing.
        /// </summary>
        public bool IsWriterOpen => _videoWriter?.IsOpened() ?? false;

        /// <summary>
        /// Gets the threshold in microseconds for frame timing.
        /// </summary>
        public double ThresholdMicroseconds => (1000.0 / FramesPerSecond) * 500.0;

        /// <summary>
        /// Opens a video file for reading.
        /// </summary>
        /// <param name="videoPath">The path to the video file.</param>
        public void Open(string videoPath)
        {
            _videoCapture?.Dispose();
            _videoCapture = null;
            _lastFrameIndex = -1;

            CloseWriter();

            _videoCapture = new VideoCapture(videoPath);

            if (!_videoCapture.IsOpened())
            {
                Console.WriteLine($"Failed to open video file: {videoPath}");

                // Try converting the file to a compatible format
                string? convertedPath = ConvertToMp4(videoPath);
                if (convertedPath != null)
                {
                    _videoCapture = new VideoCapture(convertedPath);
                }

                if (!_videoCapture.IsOpened())
                    throw new InvalidOperationException($"Unable to open video: {videoPath}");
            }

            FrameCount = (int)_videoCapture.Get(VideoCaptureProperties.FrameCount);
            FramesPerSecond = _videoCapture.Get(VideoCaptureProperties.Fps);

            if (FramesPerSecond <= 0)
                throw new InvalidOperationException("Invalid FPS detected in video.");
        }

        /// <summary>
        /// Opens a video writer for writing MJPEG AVI.
        /// </summary>
        public void OpenWriter(string aviPath, int fps, int width, int height)
        {
            // MJPEG fourcc for AVI
            int fourcc = VideoWriter.FourCC('M', 'J', 'P', 'G');

            _videoWriter?.Release();
            _videoWriter?.Dispose();

            if (File.Exists(aviPath))
                File.Delete(aviPath);

            _videoWriter = new VideoWriter(aviPath, fourcc, fps, new OpenCvSharp.Size(width, height), true);

            if (!_videoWriter.IsOpened())
                throw new Exception($"Failed to open VideoWriter: {aviPath}");

            _lastFrameIndex = -1;
        }

        /// <summary>
        /// Writes a BGR image to the open writer. Writer expects sequential writes but will not enforce index.
        /// </summary>
        public void WriteFrame(Mat imageBgr)
        {
            if (_videoWriter == null || !_videoWriter.IsOpened())
                throw new InvalidOperationException("Video writer not open.");

            if (imageBgr.Empty())
                throw new ArgumentException("imageBgr is empty.", nameof(imageBgr));

            if (!imageBgr.IsContinuous())
            {
                using var tmp = imageBgr.Clone();
                _videoWriter.Write(tmp);
            }
            else
            {
                _videoWriter.Write(imageBgr);
            }

            _lastFrameIndex++;
        }

        /// <summary>
        /// Closes the writer (if any).
        /// </summary>
        public void CloseWriter()
        {
            _videoWriter?.Release();
            _videoWriter?.Dispose();
            _videoWriter = null;
        }

        /// <summary>
        /// Converts the video to MP4 using ffmpeg if needed.
        /// </summary>
        /// <param name="originalPath">Original video path.</param>
        /// <returns>Path to converted file, or null if conversion failed.</returns>
        private static string? ConvertToMp4(string originalPath)
        {
            string tempMp4Path = Path.Combine(Path.GetTempPath(), Path.GetFileNameWithoutExtension(originalPath) + $".{Guid.NewGuid()}.converted.mp4");
            var ffmpegArgs = $"-i \"{originalPath}\" -c:v libx264 -preset fast -crf 23 \"{tempMp4Path}\"";

            var process = new Process
            {
                StartInfo = new ProcessStartInfo
                {
                    FileName = "ffmpeg",
                    Arguments = ffmpegArgs,
                    RedirectStandardOutput = true,
                    RedirectStandardError = true,
                    UseShellExecute = false,
                    CreateNoWindow = true
                }
            };

            process.Start();
            process.WaitForExit();

            if (File.Exists(tempMp4Path))
            {
                return tempMp4Path;
            }
            return null;
        }

        /// <summary>
        /// Gets the frame at the specified index.
        /// </summary>
        /// <param name="frameIndex">Frame index.</param>
        /// <returns>Frame as Mat, or null if not available.</returns>
        public Mat? GetFrameAt(int frameIndex)
        {
            if (_videoCapture == null || !_videoCapture.IsOpened())
                return null;

            if (frameIndex == _lastFrameIndex + 1)
            {
                if (!_videoCapture.Read(_frameBuffer) || _frameBuffer.Empty())
                    return null;
            }
            else
            {
                if (!_videoCapture.Set(VideoCaptureProperties.PosFrames, frameIndex))
                    return null;

                if (!_videoCapture.Read(_frameBuffer) || _frameBuffer.Empty())
                    return null;
            }

            _lastFrameIndex = frameIndex;
            return _frameBuffer.Clone(); // avoid side effects on the internal buffer
        }

        /// <summary>
        /// Gets the next frame in sequence.
        /// </summary>
        /// <returns>Frame as Mat, or null if not available.</returns>
        public Mat? GetNextFrame()
        {
            if (_videoCapture == null || !_videoCapture.IsOpened())
                return null;

            if (!_videoCapture.Read(_frameBuffer) || _frameBuffer.Empty())
                return null;

            _lastFrameIndex++;
            return _frameBuffer.Clone();
        }

        /// <summary>
        /// Seeks to the specified frame index.
        /// </summary>
        /// <param name="frameIndex">Frame index.</param>
        /// <returns>True if seek succeeded, false otherwise.</returns>
        public bool Seek(int frameIndex)
        {
            if (_videoCapture == null || !_videoCapture.IsOpened())
                return false;

            _lastFrameIndex = -1; // reset sequential optimization
            return _videoCapture.Set(VideoCaptureProperties.PosFrames, frameIndex);
        }

        /// <summary>
        /// Releases all resources used by the VideoService.
        /// </summary>
        public void Dispose()
        {
            _videoCapture?.Dispose();
            _videoCapture = null;
            _lastFrameIndex = -1;

            CloseWriter();
            _frameBuffer.Dispose();
        }


        /// <summary>
        /// Checks if the file extension indicates a video file.
        /// </summary>
        public static bool IsVideoFile(string file)
        {
            string ext = Path.GetExtension(file).ToLowerInvariant();
            return ext == ".mp4" || ext == ".avi" || ext == ".mov";
        }
    }
}
