namespace OcuInkTrain.Primatives
{
    /// <summary>
    /// Represents a point in an ink stroke.
    /// </summary>
    public struct OcuInkPoint
    {
        /// <summary>
        /// Gets or sets the position of the ink point.
        /// </summary>
        public PointF Position { get; set; }

        /// <summary>
        /// Gets or sets the pressure of the ink point.
        /// </summary>
        public float Pressure { get; set; }

        /// <summary>
        /// Gets or sets the tilt along the X-axis of the ink point.
        /// </summary>
        public float TiltX { get; set; }

        /// <summary>
        /// Gets or sets the tilt along the Y-axis of the ink point.
        /// </summary>
        public float TiltY { get; set; }

        /// <summary>
        /// Gets or sets the timestamp of the ink point.
        /// </summary>
        public ulong Timestamp { get; set; }

        /// <summary>
        /// Initializes a new instance of the <see cref="OcuInkPoint"/> struct.
        /// </summary>
        public OcuInkPoint()
        {
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="OcuInkPoint"/> struct with the specified parameters.
        /// </summary>
        /// <param name="position">The position of the ink point.</param>
        /// <param name="pressure">The pressure of the ink point.</param>
        /// <param name="tiltX">The tilt along the X-axis of the ink point.</param>
        /// <param name="tiltY">The tilt along the Y-axis of the ink point.</param>
        /// <param name="timestamp">The timestamp of the ink point.</param>
        public OcuInkPoint(PointF position, float pressure, float tiltX, float tiltY, ulong timestamp)
        {
            Position = position;
            Pressure = pressure;
            TiltX = tiltX;
            TiltY = tiltY;
            Timestamp = timestamp;
        }
    }
}
