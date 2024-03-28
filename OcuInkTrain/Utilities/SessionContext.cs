using Camera.MAUI;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace OcuInkTrain.Utilities
{
    /// <summary>
    /// Represents the session context for the application.
    /// </summary>
    public static class SessionContext
    {
        /// <summary>
        /// Gets or sets the session ID.
        /// </summary>
        public static string? SessionID { get; set; }

        /// <summary>
        /// Gets the file path session ID by removing spaces from the session ID.
        /// </summary>
        public static string FilePathSessionID => SessionID?.Replace(" ", "") ?? string.Empty;

        /// <summary>
        /// Gets the list of image paths.
        /// </summary>
        public static List<string> ImagePaths { get; } = [];

        /// <summary>
        /// Gets or sets the camera window.
        /// </summary>
        public static Window? CameraWin { get; set; }
    }
}
