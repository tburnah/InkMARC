using OcuInkTrain.Interfaces;
using OcuInkTrain.Primatives;

namespace OcuInkTrain.Handlers;

/// <summary>
/// DrawingLine Adapter
/// </summary>
public interface IAdvancedDrawingLineAdapter
{
	/// <summary>
	/// Convert <see cref="ScryvDrawingLine"/> to <see cref="IAdvancedDrawingLine"/>.
	/// </summary>
	/// <returns><see cref="IAdvancedDrawingLine"/></returns>
	IAdvancedDrawingLine ConvertScryvDrawingLine(ScryvDrawingLine scryvDrawingLine);
}