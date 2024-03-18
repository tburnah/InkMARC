namespace Scryv.Utilities
{
    /// <summary>
    /// A utility class for calculating the tiltX and tiltY values based on azimuth and altitude angles in degrees.
    /// </summary>
    public static class StylusTiltCalculator
    {
        /// <summary>
        /// Converts degrees to radians.
        /// </summary>
        /// <param name="degrees">The angle in degrees.</param>
        /// <returns>The angle in radians.</returns>
        private static double DegreesToRadians(double degrees)
        {
            return (Math.PI / 180) * degrees;
        }

        /// <summary>
        /// Calculates the tiltX and tiltY based on azimuth and altitude angles in degrees.
        /// </summary>
        /// <param name="azimuthDegrees">The azimuth angle in degrees.</param>
        /// <param name="altitudeDegrees">The altitude angle in degrees.</param>
        /// <returns>A tuple containing the tiltX and tiltY values.</returns>
        public static (double tiltX, double tiltY) CalculateTilt(double azimuthDegrees, double altitudeDegrees)
        {
            // Convert angles from degrees to radians
            double thetaRadians = DegreesToRadians(azimuthDegrees);
            double phiRadians = DegreesToRadians(altitudeDegrees);

            // Calculate tiltX and tiltY
            double tiltX = Math.Cos(thetaRadians) * Math.Sin(phiRadians);
            double tiltY = Math.Sin(thetaRadians) * Math.Sin(phiRadians);

            return (tiltX, tiltY);
        }
    }
}
