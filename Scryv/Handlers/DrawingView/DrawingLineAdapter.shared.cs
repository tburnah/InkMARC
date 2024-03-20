using CommunityToolkit.Maui.Core.Extensions;
using OcuInkTrain.Interfaces;
using OcuInkTrain.Primatives;

namespace OcuInkTrain.Handlers;

/// <summary>
/// DrawingLine Adapter
/// </summary>
public sealed class AdvancedDrawingLineAdapter : IAdvancedDrawingLineAdapter
{
    /// <summary>
    /// Convert <see cref="ScryvDrawingLine"/> to <see cref="IAdvancedDrawingLine"/>.
    /// </summary>
    /// <returns><see cref="IAdvancedDrawingLine"/></returns>
    public IAdvancedDrawingLine ConvertScryvDrawingLine(ScryvDrawingLine scryvDrawingLine)
	{
		if (scryvDrawingLine is null)
			return new ScryvDrawingLine();

		return new ScryvDrawingLine
		{
			LineColor = scryvDrawingLine?.LineColor ?? Colors.Black,
			ShouldSmoothPathWhenDrawn = scryvDrawingLine?.ShouldSmoothPathWhenDrawn ?? true,
			Granularity = scryvDrawingLine?.Granularity ?? 5,
			LineWidth = scryvDrawingLine?.LineWidth ?? 1,
			Points = scryvDrawingLine?.Points?.ToObservableCollection() ?? new System.Collections.ObjectModel.ObservableCollection<ScryvInkPoint>()
		};
	}
}