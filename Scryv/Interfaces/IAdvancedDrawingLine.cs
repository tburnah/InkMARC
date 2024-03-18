using Scryv.Primatives;
using System.Collections.ObjectModel;

namespace Scryv.Interfaces
{
    /// <summary>
    /// Represents an advanced drawing line.
    /// </summary>
    public interface IAdvancedDrawingLine
    {
        /// <summary>
        /// Gets or sets the granularity of this line. Min value is 5. The higher the value, the smoother the line, the slower the program.
        /// </summary>
        int Granularity { get; set; }

        /// <summary>
        /// Gets or sets the Microsoft.Maui.Graphics.Color that is used to draw this line on the CommunityToolkit.Maui.Core.IDrawingView.
        /// </summary>
        Color LineColor { get; set; }

        /// <summary>
        /// Gets or sets the width that is used to draw this line on the CommunityToolkit.Maui.Core.IDrawingView.
        /// </summary>
        float LineWidth { get; set; }

        /// <summary>
        /// Gets or sets the collection of Microsoft.Maui.Graphics.PointF that makes up this line on the CommunityToolkit.Maui.Core.IDrawingView.
        /// </summary>
        ObservableCollection<ScryvInkPoint> Points { get; set; }

        /// <summary>
        /// Gets or sets a value indicating whether this line is smoothed (anti-aliased) when drawn.
        /// </summary>
        bool ShouldSmoothPathWhenDrawn { get; set; }

        /// <summary>
        /// Retrieves a System.IO.Stream containing an image of this line, based on the CommunityToolkit.Maui.Core.IDrawingLine.Points data.
        /// </summary>
        /// <param name="imageSizeWidth">Desired width of the image that is returned.</param>
        /// <param name="imageSizeHeight">Desired height of the image that is returned.</param>
        /// <param name="background">Background of the generated image.</param>
        /// <param name="token">System.Threading.CancellationToken.</param>
        /// <returns>System.Threading.Tasks.ValueTask&lt;Stream&gt; containing the data of the requested image with data that's currently on the CommunityToolkit.Maui.Core.IDrawingLine.</returns>
        ValueTask<Stream> GetImageStream(double imageSizeWidth, double imageSizeHeight, Paint background, CancellationToken token = default(CancellationToken));
    }
}
