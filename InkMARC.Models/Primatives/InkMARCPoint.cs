namespace InkMARC.Models.Primatives
{
    /// <summary>
    /// Represents a point in an ink stroke.
    /// </summary>
    public struct InkMARCPoint
    {
        /// <summary>
        /// Gets or sets the X position of the ink point.
        /// </summary>
        public float X { get; set; }

        /// <summary>
        /// Gets or sets the Y position of the ink point.
        /// </summary>
        public float Y { get; set; }

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
        /// Gets or sets the timestamp of the ink point. Microseconds since boot.
        /// </summary>
        public ulong Timestamp { get; set; }

        /// <summary>
        /// Initializes a new instance of the <see cref="InkMARCPoint"/> struct.
        /// </summary>
        public InkMARCPoint()
        {
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="InkMARCPoint"/> struct with the specified parameters.
        /// </summary>
        /// <param name="position">The position of the ink point.</param>
        /// <param name="pressure">The pressure of the ink point.</param>
        /// <param name="tiltX">The tilt along the X-axis of the ink point.</param>
        /// <param name="tiltY">The tilt along the Y-axis of the ink point.</param>
        /// <param name="timestamp">The timestamp of the ink point.</param>
        public InkMARCPoint(float x, float y, float pressure, float tiltX, float tiltY, ulong timestamp)
        {
            X = x;
            Y = y;
            Pressure = pressure;
            TiltX = tiltX;
            TiltY = tiltY;
            Timestamp = timestamp;
        }
    }
}
