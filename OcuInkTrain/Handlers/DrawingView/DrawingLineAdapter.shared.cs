using CommunityToolkit.Maui.Core.Extensions;
using OcuInk.Models.Interfaces;
using OcuInk.Models.Primatives;
using OcuInkTrain.Interfaces;
using OcuInkTrain.Primatives;

namespace OcuInkTrain.Handlers;

/// <summary>
/// DrawingLine Adapter
/// </summary>
public sealed class AdvancedDrawingLineAdapter : IAdvancedDrawingLineAdapter
{
    /// <summary>
    /// Convert <see cref="OcuInkDrawingLine"/> to <see cref="IAdvancedDrawingLine"/>.
    /// </summary>
    /// <returns><see cref="IAdvancedDrawingLine"/></returns>
    public IAdvancedDrawingLine ConvertOcuInkDrawingLine(OcuInkDrawingLine ocuInkDrawingLine)
	{
		if (ocuInkDrawingLine is null)
			return new OcuInkDrawingLine();

		return new OcuInkDrawingLine
		{
			LineColor = ocuInkDrawingLine is null ? new InkColor() : ocuInkDrawingLine.LineColor,
			ShouldSmoothPathWhenDrawn = ocuInkDrawingLine?.ShouldSmoothPathWhenDrawn ?? true,
			Granularity = ocuInkDrawingLine?.Granularity ?? 5,
			LineWidth = ocuInkDrawingLine?.LineWidth ?? 1,
			Points = ocuInkDrawingLine?.Points?.ToObservableCollection() ?? new System.Collections.ObjectModel.ObservableCollection<OcuInkPoint>()
		};
	}
}