using OpenCvSharp;

namespace InkMARC.Label.Services.Interfaces
{
    public interface IVideoService : IDisposable
    {
        /// <summary>
        /// Raised whenever the total number of frames changes (e.g., after opening a video).
        /// </summary>
        event EventHandler<int>? FrameCountChanged;

        double ThresholdMicroseconds { get; }

        /// <summary>
        /// Total number of frames in the currently opened video.
        /// </summary>
        int FrameCount { get; }

        /// <summary>
        /// Frames per second of the currently opened video.
        /// </summary>
        double FramesPerSecond { get; }

        /// <summary>
        /// Whether a video is currently open.
        /// </summary>
        bool IsOpen { get; }

        /// <summary>
        /// Opens the video file at the given path.
        /// </summary>
        void Open(string path);

        /// <summary>
        /// Gets the frame at the specified index (0-based). Returns null if not available.
        /// Returned Mat should be a clone, safe for the caller to dispose.
        /// </summary>
        Mat? GetFrameAt(int index);

        /// <summary>
        /// Gets the next sequential frame. Returns null if end of stream.
        /// </summary>
        Mat? GetNextFrame();

        /// <summary>
        /// Seeks to the specified frame index. Returns true if successful.
        /// </summary>
        bool Seek(int index);
    }
}
