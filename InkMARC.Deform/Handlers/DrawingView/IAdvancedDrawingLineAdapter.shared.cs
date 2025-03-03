using InkMARC.Models.Interfaces;
using InkMARC.Models.Primatives;

namespace InkMARCDeform.Handlers;

/// <summary>
/// DrawingLine Adapter
/// </summary>
public interface IAdvancedDrawingLineAdapter
{
	/// <summary>
	/// Convert <see cref="InkMARCDrawingLine"/> to <see cref="IAdvancedDrawingLine"/>.
	/// </summary>
	/// <returns><see cref="IAdvancedDrawingLine"/></returns>
	IAdvancedDrawingLine ConvertInkMARCDrawingLine(InkMARCDrawingLine inkMARCDrawingLine);
}