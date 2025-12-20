
using InkMARC.Clean.Model;

namespace InkMARC.Clean.Services.Interfaces
{
    /// <summary>
    /// Abstract frame source: can be backed by a video file, an HDF5 session,
    /// or anything else that can supply frames + metadata by index.
    /// </summary>
    public interface IFrameSource : IDisposable
    {
        /// <summary>
        /// Fired whenever the total frame count changes (e.g., after Open()).
        /// </summary>
        event EventHandler<int>? FrameCountChanged;

        bool SupportsPlay { get; }

        bool FileSeek { get; }

        string FileFilter { get;  }

        /// <summary>
        /// Total number of frames available in this source.
        /// </summary>
        int FrameCount { get; }

        /// <summary>
        /// Nominal frames per second for this source (for timestamps / playback).
        /// </summary>
        double FramesPerSecond { get; }

        public int ViewW { get; set; }
        public int ViewH { get; set; }

        /// <summary>
        /// <summary>
        /// Open an underlying resource (video file, HDF5 file, etc.).
        /// </summary>
        /// <param name="path">Path to the resource.</param>
        void Open(string path);

        /// <summary>
        /// Get the frame and its metadata at the given index.
        /// Returns null if the index is out of range or frame cannot be read.
        ///
        /// Caller is responsible for disposing frame.Image when finished.
        /// </summary>
        /// <param name="index">Zero-based frame index.</param>
        /// <returns>FrameData or null if not available.</returns>
        FrameData? GetFrame(int index);

        FrameData? GetFrameForExport(int index);
    }
}
