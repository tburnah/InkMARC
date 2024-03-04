using Scryv.Interfaces;
using Scryv.Primatives;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Scryv.Extensions
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
        /// <param name="point">The ScryvInkPoint object representing the position to draw the line to.</param>
        public static void LineTo(this PathF path, ScryvInkPoint point)
        {
            path.LineTo(point.Position.X, point.Position.Y);
        }
    }
}
