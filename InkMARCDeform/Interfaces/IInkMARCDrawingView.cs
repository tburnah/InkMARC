using InkMARC.Models.Interfaces;
using InkMARC.Models.Primatives;
using System.Collections.ObjectModel;

namespace InkMARCDeform.Interfaces
{
    /// <summary>
    /// Represents a InkMARC drawing view.
    /// </summary>
    public interface IInkMARCDrawingView : IView, IElement, ITransform
    {
        /// <summary>
        /// The Microsoft.Maui.Graphics.Color that is used by default to draw a line on the CommunityToolkit.Maui.Core.IDrawingView.
        /// </summary>
        Color LineColor { get; }

        /// <summary>
        /// The width that is used by default to draw a line on the CommunityToolkit.Maui.Core.IDrawingView.
        /// </summary>
        float LineWidth { get; }

        /// <summary>
        /// The collection of lines that are currently on the CommunityToolkit.Maui.Core.IDrawingView.
        /// </summary>
        ObservableCollection<IAdvancedDrawingLine> Lines { get; }

        /// <summary>
        /// Toggles multi-line mode. When true, multiple lines can be drawn on the CommunityToolkit.Maui.Core.IDrawingView
        /// while the tap/click is released in-between lines. Note: when CommunityToolkit.Maui.Core.IDrawingView.ShouldClearOnFinish
        /// is also enabled, the lines are cleared after the tap/click is released. Additionally,
        /// CommunityToolkit.Maui.Core.IDrawingView.OnDrawingLineCompleted(CommunityToolkit.Maui.Core.IDrawingLine)
        /// will be fired after each line that is drawn.
        /// </summary>
        bool IsMultiLineModeEnabled { get; }

        /// <summary>
        /// Indicates whether the CommunityToolkit.Maui.Core.IDrawingView is cleared after
        /// releasing the tap/click and a line is drawn. Note: when CommunityToolkit.Maui.Core.IDrawingView.IsMultiLineModeEnabled
        /// is also enabled, this might cause unexpected behavior.
        /// </summary>
        bool ShouldClearOnFinish { get; }

        /// <summary>
        /// Allows to draw on the CommunityToolkit.Maui.Core.IDrawingView.
        /// </summary>
        Action<ICanvas, RectF>? DrawAction { get; }

        /// <summary>
        /// Retrieves a System.IO.Stream containing an image of the CommunityToolkit.Maui.Core.IDrawingView.Lines
        /// that are currently drawn on the CommunityToolkit.Maui.Core.IDrawingView.
        /// </summary>
        /// <param name="imageSizeWidth">Desired width of the image that is returned. The image will be resized proportionally.</param>
        /// <param name="imageSizeHeight">Desired height of the image that is returned. The image will be resized proportionally.</param>
        /// <param name="token">System.Threading.CancellationToken.</param>
        /// <returns>System.Threading.Tasks.Task`1 containing the data of the requested image with
        /// data that's currently on the CommunityToolkit.Maui.Core.IDrawingView.</returns>
        ValueTask<Stream> GetImageStream(double imageSizeWidth, double imageSizeHeight, CancellationToken token = default(CancellationToken));

        /// <summary>
        /// Clears the CommunityToolkit.Maui.Core.IDrawingView.Lines that are currently drawn
        /// on the CommunityToolkit.Maui.Core.IDrawingView.
        /// </summary>
        void Clear();

        /// <summary>
        /// Event occurred when drawing line started
        /// </summary>
        /// <param name="point">Last drawing point</param>
        void OnDrawingLineStarted(InkMARCPoint point);

        /// <summary>
        /// Event occurred when drawing line cancelled
        /// </summary>
        void OnDrawingLineCancelled();

        /// <summary>
        /// Event occurred when point drawn
        /// </summary>
        /// <param name="point">Last drawing point</param>
        void OnPointDrawn(InkMARCPoint point);

        /// <summary>
        /// Event occurred when drawing line completed
        /// </summary>
        /// <param name="lastDrawingLine">Last drawing line</param>
        void OnDrawingLineCompleted(IAdvancedDrawingLine lastDrawingLine);
    }
}
