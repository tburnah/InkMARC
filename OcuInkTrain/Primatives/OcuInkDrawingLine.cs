using OcuInkTrain.Interfaces;
using OcuInkTrain.Services;
using System.Collections.ObjectModel;

namespace OcuInkTrain.Primatives
{
    /// <summary>
    /// Represents a drawing line in the OcuInkTrain application.
    /// </summary>
    public class OcuInkDrawingLine : IAdvancedDrawingLine
    {
        /// <summary>
        /// Gets or sets the granularity of the drawing line.
        /// </summary>
        public int Granularity { get; set; }

        /// <summary>
        /// Gets or sets the color of the drawing line.
        /// </summary>
        public Color? LineColor { get; set; }

        /// <summary>
        /// Gets or sets the width of the drawing line.
        /// </summary>
        public float LineWidth { get; set; }

        /// <summary>
        /// Gets or sets the collection of ink points that make up the drawing line.
        /// </summary>
        public ObservableCollection<OcuInkPoint>? Points { get; set; }

        /// <summary>
        /// Gets or sets a value indicating whether the path should be smoothed when drawn.
        /// </summary>
        public bool ShouldSmoothPathWhenDrawn { get; set; }

        /// <summary>
        /// Retrieves a stream containing an image of the collection of ink points provided as a parameter.
        /// </summary>
        /// <param name="points">A collection of ink points that the image is generated from.</param>
        /// <param name="imageSize">The desired dimensions of the generated image.</param>
        /// <param name="lineWidth">The desired line width to be used in the generated image.</param>
        /// <param name="strokeColor">The desired color of the line to be used in the generated image.</param>
        /// <param name="background">The background of the generated image.</param>
        /// <param name="token">A cancellation token.</param>
        /// <returns>A value task containing a stream with the data of the requested image.</returns>
        public static ValueTask<Stream> GetImageStream(IEnumerable<OcuInkPoint> points,
                                            Size imageSize,
                                            float lineWidth,
                                            Color strokeColor,
                                            Paint background,
                                            CancellationToken token = default)
        {
            return OcuInkDrawingViewService.GetImageStream(points.ToList(), imageSize, lineWidth, strokeColor, background, token);
        }

        /// <summary>
        /// Retrieves a stream containing an image of the drawing line with the specified image size, background, and cancellation token.
        /// </summary>
        /// <param name="imageSizeWidth">The width of the desired image size.</param>
        /// <param name="imageSizeHeight">The height of the desired image size.</param>
        /// <param name="background">The background of the generated image.</param>
        /// <param name="token">A cancellation token.</param>
        /// <returns>A value task containing a stream with the data of the requested image.</returns>
        public ValueTask<Stream> GetImageStream(double imageSizeWidth, double imageSizeHeight, Paint background, CancellationToken token = default)
        {
            throw new NotImplementedException();
        }
    }
}
