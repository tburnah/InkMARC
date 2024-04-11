namespace OcuInk.Models.Primatives
{
    /// <summary>
    /// Represents information about a camera.
    /// </summary>
    public class CameraInfo
    {
        /// <summary>
        /// Gets or sets the name of the camera.
        /// </summary>
        public string Name { get; set; }

        /// <summary>
        /// Gets or sets the device ID of the camera.
        /// </summary>
        public string DeviceId { get; set; }

        /// <summary>
        /// Gets or sets the position of the camera.
        /// </summary>
        public CameraPosition Position { get; set; }

        /// <summary>
        /// Gets or sets the selected resolution of the camera.
        /// </summary>
        public OcuInkSize SelectedResolution { get; set; } = new OcuInkSize(640, 480);

        /// <summary>
        /// Gets or sets the encoding quality of the camera.
        /// </summary>
        public string EncodingQuality { get; set; } = "VGA";

        /// <summary>
        /// Returns the name of the camera.
        /// </summary>
        /// <returns>The name of the camera.</returns>
        public override string ToString()
        {
            return Name;
        }
    }
}
