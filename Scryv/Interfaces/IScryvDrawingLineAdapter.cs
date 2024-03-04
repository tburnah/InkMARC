using CommunityToolkit.Maui.Core.Views;
using CommunityToolkit.Maui.Core;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Scryv.Primatives;

namespace Scryv.Interfaces
{
    /// <summary>
    /// Represents an adapter for converting ScryvDrawingLine to IAdvancedDrawingLine.
    /// </summary>
    public interface IScryvDrawingLineAdapter
    {
        /// <summary>
        /// Converts a ScryvDrawingLine to an IAdvancedDrawingLine.
        /// </summary>
        /// <param name="scryvDrawingLine">The ScryvDrawingLine to convert.</param>
        /// <returns>The converted IAdvancedDrawingLine.</returns>
        IAdvancedDrawingLine ConvertMauiDrawingLine(ScryvDrawingLine scryvDrawingLine);
    }
}
