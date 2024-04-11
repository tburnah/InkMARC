using OcuInk.Models.Interfaces;
using OcuInk.Models.Primatives;

namespace OcuInkTrain.Handlers;

/// <summary>
/// DrawingLine Adapter
/// </summary>
public interface IAdvancedDrawingLineAdapter
{
	/// <summary>
	/// Convert <see cref="OcuInkDrawingLine"/> to <see cref="IAdvancedDrawingLine"/>.
	/// </summary>
	/// <returns><see cref="IAdvancedDrawingLine"/></returns>
	IAdvancedDrawingLine ConvertOcuInkDrawingLine(OcuInkDrawingLine ocuInkDrawingLine);
}