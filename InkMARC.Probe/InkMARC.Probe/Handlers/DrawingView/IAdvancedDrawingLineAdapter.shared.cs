using InkMARC.Probe.Models.Interfaces;
using InkMARC.Probe.Models.Primatives;

namespace InkMARC.Probe.Handlers;

/// <summary>
/// DrawingLine Adapter
/// </summary>
public interface IAdvancedDrawingLineAdapter
{
	/// <summary>
	/// Convert <see cref="InkMARCDrawingLine"/> to <see cref="IAdvancedDrawingLine"/>.
	/// </summary>
	/// <returns><see cref="IAdvancedDrawingLine"/></returns>
	IAdvancedDrawingLine ConvertInkMARCDrawingLine(InkMARCDrawingLine inkMarcDrawingLine);
}