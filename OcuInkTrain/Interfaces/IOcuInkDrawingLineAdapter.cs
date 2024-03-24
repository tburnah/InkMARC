using OcuInkTrain.Primatives;

namespace OcuInkTrain.Interfaces
{
    /// <summary>
    /// Represents an adapter for converting OcuInkDrawingLine to IAdvancedDrawingLine.
    /// </summary>
    public interface IOcuInkDrawingLineAdapter
    {
        /// <summary>
        /// Converts a OcuInkDrawingLine to an IAdvancedDrawingLine.
        /// </summary>
        /// <param name="ocuInkDrawingLine">The OcuInkDrawingLine to convert.</param>
        /// <returns>The converted IAdvancedDrawingLine.</returns>
        IAdvancedDrawingLine ConvertMauiDrawingLine(OcuInkDrawingLine ocuInkDrawingLine);
    }
}
