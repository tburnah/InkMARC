namespace InkMARC.Models.Primatives
{
    /// <summary>
    /// Represents exercise data.
    /// </summary>
    public class ExerciseData
    {
        /// <summary>
        /// Gets or sets the camera information.
        /// </summary>
        public CameraInfo? CameraInfo { get; set; }

        /// <summary>
        /// Gets or sets the list of drawing lines.
        /// </summary>
        public List<InkMARCDrawingLine>? DrawingLines { get; set; }
    }
}
