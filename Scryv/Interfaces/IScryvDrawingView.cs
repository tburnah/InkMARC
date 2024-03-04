using CommunityToolkit.Maui.Core;
using Scryv.Primatives;
using Scryv.Views.AdvanceDrawingView;
using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Scryv.Interfaces
{
    /// <summary>
    /// Represents a Scryv drawing view.
    /// </summary>
    public interface IScryvDrawingView : IView, IElement, ITransform
    {
        //
        // Summary:
        //     The Microsoft.Maui.Graphics.Color that is used by default to draw a line on the
        //     CommunityToolkit.Maui.Core.IDrawingView.
        Color LineColor { get; }

        //
        // Summary:
        //     The width that is used by default to draw a line on the CommunityToolkit.Maui.Core.IDrawingView.
        float LineWidth { get; }

        //
        // Summary:
        //     The collection of lines that are currently on the CommunityToolkit.Maui.Core.IDrawingView.
        ObservableCollection<IAdvancedDrawingLine> Lines { get; }

        //
        // Summary:
        //     Toggles multi-line mode. When true, multiple lines can be drawn on the CommunityToolkit.Maui.Core.IDrawingView
        //     while the tap/click is released in-between lines. Note: when CommunityToolkit.Maui.Core.IDrawingView.ShouldClearOnFinish
        //     is also enabled, the lines are cleared after the tap/click is released. Additionally,
        //     CommunityToolkit.Maui.Core.IDrawingView.OnDrawingLineCompleted(CommunityToolkit.Maui.Core.IDrawingLine)
        //     will be fired after each line that is drawn.
        bool IsMultiLineModeEnabled { get; }

        //
        // Summary:
        //     Indicates whether the CommunityToolkit.Maui.Core.IDrawingView is cleared after
        //     releasing the tap/click and a line is drawn. Note: when CommunityToolkit.Maui.Core.IDrawingView.IsMultiLineModeEnabled
        //     is also enabled, this might cause unexpected behavior.
        bool ShouldClearOnFinish { get; }

        //
        // Summary:
        //     Allows to draw on the CommunityToolkit.Maui.Core.IDrawingView.
        Action<ICanvas, RectF>? DrawAction { get; }

        //
        // Summary:
        //     Retrieves a System.IO.Stream containing an image of the CommunityToolkit.Maui.Core.IDrawingView.Lines
        //     that are currently drawn on the CommunityToolkit.Maui.Core.IDrawingView.
        //
        // Parameters:
        //   imageSizeWidth:
        //     Desired width of the image that is returned. The image will be resized proportionally.
        //
        //
        //   imageSizeHeight:
        //     Desired height of the image that is returned. The image will be resized proportionally.
        //
        //
        //   token:
        //     System.Threading.CancellationToken.
        //
        // Returns:
        //     System.Threading.Tasks.Task`1 containing the data of the requested image with
        //     data that's currently on the CommunityToolkit.Maui.Core.IDrawingView.
        ValueTask<Stream> GetImageStream(double imageSizeWidth, double imageSizeHeight, CancellationToken token = default(CancellationToken));

        //
        // Summary:
        //     Clears the CommunityToolkit.Maui.Core.IDrawingView.Lines that are currently drawn
        //     on the CommunityToolkit.Maui.Core.IDrawingView.
        void Clear();

        //
        // Summary:
        //     Event occurred when drawing line started
        //
        // Parameters:
        //   point:
        //     Last drawing point
        void OnDrawingLineStarted(ScryvInkPoint point);

        //
        // Summary:
        //     Event occurred when drawing line cancelled
        void OnDrawingLineCancelled();

        //
        // Summary:
        //     Event occurred when point drawn
        //
        // Parameters:
        //   point:
        //     Last drawing point
        void OnPointDrawn(ScryvInkPoint point);

        //
        // Summary:
        //     Event occurred when drawing line completed
        //
        // Parameters:
        //   lastDrawingLine:
        //     Last drawing line
        void OnDrawingLineCompleted(IAdvancedDrawingLine lastDrawingLine);
    }
}
