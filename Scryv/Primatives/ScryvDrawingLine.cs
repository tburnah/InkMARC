using CommunityToolkit.Maui.Core.Views;
using Scryv.Interfaces;
using Scryv.Services;
using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Scryv.Primatives
{
    public class ScryvDrawingLine : IAdvancedDrawingLine
    {
        public int Granularity { get; set; }
        public Color LineColor { get; set; }
        public float LineWidth { get; set; }
        public ObservableCollection<ScryvInkPoint> Points { get; set; }
        public bool ShouldSmoothPathWhenDrawn { get; set; }

        /// <summary>
        /// Retrieves a <see cref="Stream"/> containing an image of the collection of <see cref="Point"/> that is provided as a parameter.
        /// </summary>
        /// <param name="points">A collection of <see cref="Point"/> that a image is generated from.</param>
        /// <param name="imageSize">The desired dimensions of the generated image.</param>
        /// <param name="lineWidth">The desired line width to be used in the generated image.</param>
        /// <param name="strokeColor">The desired color of the line to be used in the generated image.</param>
        /// <param name="background">Background of the generated image.</param>
        /// <param name="token"><see cref="CancellationToken"/> </param>
        /// <returns><see cref="ValueTask{Stream}"/> containing the data of the requested image with data that's provided through the <paramref name="points"/> parameter.</returns>
        public static ValueTask<Stream> GetImageStream(IEnumerable<ScryvInkPoint> points,
                                            Size imageSize,
                                            float lineWidth,
                                            Color strokeColor,
                                            Paint background,
                                            CancellationToken token = default)
        {
            return ScryvDrawingViewService.GetImageStream(points.ToList(), imageSize, lineWidth, strokeColor, background, token);
        }

        public ValueTask<Stream> GetImageStream(double imageSizeWidth, double imageSizeHeight, Paint background, CancellationToken token = default)
        {
            throw new NotImplementedException();
        }
    }
}
