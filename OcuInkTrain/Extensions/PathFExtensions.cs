using OcuInkTrain.Primatives;

namespace OcuInkTrain.Extensions
{
    /// <summary>
    /// Provides extension methods for the PathF class.
    /// </summary>
    public static class PathFExtensions
    {
        /// <summary>
        /// Adds a line segment to the path from the current point to the specified position.
        /// </summary>
        /// <param name="path">The PathF object.</param>
        /// <param name="point">The OcuInkPoint object representing the position to draw the line to.</param>
        public static void LineTo(this PathF path, OcuInkPoint point)
        {
            path.LineTo(point.Position.X, point.Position.Y);
        }
    }
}
