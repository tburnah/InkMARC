using CommunityToolkit.Maui.Core;
using Scryv.Primatives;
using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Scryv.Interfaces
{
    public interface IAdvancedDrawingLine
    {
        //
        // Summary:
        //     The granularity of this line. Min value is 5. The higher the value, the smoother
        //     the line, the slower the program.
        int Granularity { get; set; }

        //
        // Summary:
        //     The Microsoft.Maui.Graphics.Color that is used to draw this line on the CommunityToolkit.Maui.Core.IDrawingView.
        Color LineColor { get; set; }

        //
        // Summary:
        //     The width that is used to draw this line on the CommunityToolkit.Maui.Core.IDrawingView.
        float LineWidth { get; set; }

        //
        // Summary:
        //     The collection of Microsoft.Maui.Graphics.PointF that makes up this line on the
        //     CommunityToolkit.Maui.Core.IDrawingView.
        ObservableCollection<ScryvInkPoint> Points { get; set; }

        //
        // Summary:
        //     Enables or disables if this line is smoothed (anti-aliased) when drawn.
        bool ShouldSmoothPathWhenDrawn { get; set; }

        //
        // Summary:
        //     Retrieves a System.IO.Stream containing an image of this line, based on the CommunityToolkit.Maui.Core.IDrawingLine.Points
        //     data.
        //
        // Parameters:
        //   imageSizeWidth:
        //     Desired width of the image that is returned.
        //
        //   imageSizeHeight:
        //     Desired height of the image that is returned.
        //
        //   background:
        //     Background of the generated image.
        //
        //   token:
        //     System.Threading.CancellationToken.
        //
        // Returns:
        //     System.Threading.Tasks.ValueTask`1 containing the data of the requested image
        //     with data that's currently on the CommunityToolkit.Maui.Core.IDrawingLine.
        ValueTask<Stream> GetImageStream(double imageSizeWidth, double imageSizeHeight, Paint background, CancellationToken token = default(CancellationToken));
    }
}
