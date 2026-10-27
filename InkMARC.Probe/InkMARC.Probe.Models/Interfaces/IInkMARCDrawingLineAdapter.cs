using InkMARC.Probe.Models.Primatives;

namespace InkMARC.Probe.Models.Interfaces
{
    /// <summary>
    /// Represents an adapter for converting InkMARCDrawingLine to IAdvancedDrawingLine.
    /// </summary>
    public interface IInkMARCDrawingLineAdapter
    {
        /// <summary>
        /// Converts a InkMARCDrawingLine to an IAdvancedDrawingLine.
        /// </summary>
        /// <param name="inkMarcDrawingLine">The InkMARCDrawingLine to convert.</param>
        /// <returns>The converted IAdvancedDrawingLine.</returns>
        IAdvancedDrawingLine ConvertMauiDrawingLine(InkMARCDrawingLine inkMarcDrawingLine);
    }
}
