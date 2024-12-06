using InkMARC.Models.Primatives;

namespace InkMARC.Models.Interfaces
{
    /// <summary>
    /// Represents an adapter for converting InkMARCDrawingLine to IAdvancedDrawingLine.
    /// </summary>
    public interface IInkMARCDrawingLineAdapter
    {
        /// <summary>
        /// Converts a InkMARCDrawingLine to an IAdvancedDrawingLine.
        /// </summary>
        /// <param name="inkMARCDrawingLine">The InkMARCDrawingLine to convert.</param>
        /// <returns>The converted IAdvancedDrawingLine.</returns>
        IAdvancedDrawingLine ConvertMauiDrawingLine(InkMARCDrawingLine inkMARCDrawingLine);
    }
}
