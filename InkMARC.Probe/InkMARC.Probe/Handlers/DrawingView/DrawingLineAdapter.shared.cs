using CommunityToolkit.Maui.Core.Extensions;
using InkMARC.Probe.Models.Interfaces;
using InkMARC.Probe.Models.Primatives;
using InkMARC.Probe.Interfaces;
using InkMARC.Probe.Primatives;

namespace InkMARC.Probe.Handlers;

/// <summary>
/// DrawingLine Adapter
/// </summary>
public sealed class AdvancedDrawingLineAdapter : IAdvancedDrawingLineAdapter
{
    /// <summary>
    /// Convert <see cref="InkMARCDrawingLine"/> to <see cref="IAdvancedDrawingLine"/>.
    /// </summary>
    /// <returns><see cref="IAdvancedDrawingLine"/></returns>
    public IAdvancedDrawingLine ConvertInkMARCDrawingLine(InkMARCDrawingLine inkMarcDrawingLine)
	{
		if (inkMarcDrawingLine is null)
			return new InkMARCDrawingLine();

		return new InkMARCDrawingLine
		{
			LineColor = inkMarcDrawingLine is null ? new InkColor() : inkMarcDrawingLine.LineColor,
			ShouldSmoothPathWhenDrawn = inkMarcDrawingLine?.ShouldSmoothPathWhenDrawn ?? true,
			Granularity = inkMarcDrawingLine?.Granularity ?? 5,
			LineWidth = inkMarcDrawingLine?.LineWidth ?? 1,
			Points = inkMarcDrawingLine?.Points?.ToObservableCollection() ?? new System.Collections.ObjectModel.ObservableCollection<InkMARCPoint>()
		};
	}
}