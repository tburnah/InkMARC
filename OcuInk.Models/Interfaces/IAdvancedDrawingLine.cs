using OcuInk.Models.Primatives;
using System.Collections.ObjectModel;

namespace OcuInk.Models.Interfaces
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
        /// Gets or sets the Microsoft.Maui.Graphics.InkColor that is used to draw this line on the CommunityToolkit.Maui.Core.IDrawingView.
        /// </summary>
        InkColor LineColor { get; set; }

        /// <summary>
        /// Gets or sets the width that is used to draw this line on the CommunityToolkit.Maui.Core.IDrawingView.
        /// </summary>
        float LineWidth { get; set; }

        /// <summary>
        /// Gets or sets the collection of Microsoft.Maui.Graphics.InkPoint that makes up this line on the CommunityToolkit.Maui.Core.IDrawingView.
        /// </summary>
        ObservableCollection<OcuInkPoint>? Points { get; set; }

        /// <summary>
        /// Gets or sets a value indicating whether this line is smoothed (anti-aliased) when drawn.
        /// </summary>
        bool ShouldSmoothPathWhenDrawn { get; set; }
    }
}
