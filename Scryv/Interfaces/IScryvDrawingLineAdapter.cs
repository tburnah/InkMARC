using OcuInkTrain.Primatives;

namespace OcuInkTrain.Interfaces
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
