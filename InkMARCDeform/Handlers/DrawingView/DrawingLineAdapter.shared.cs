using CommunityToolkit.Maui.Core.Extensions;
using InkMARC.Models.Interfaces;
using InkMARC.Models.Primatives;
using InkMARCDeform.Interfaces;
using InkMARCDeform.Primatives;

namespace InkMARCDeform.Handlers;

/// <summary>
/// DrawingLine Adapter
/// </summary>
public sealed class AdvancedDrawingLineAdapter : IAdvancedDrawingLineAdapter
{
    /// <summary>
    /// Convert <see cref="InkMARCDrawingLine"/> to <see cref="IAdvancedDrawingLine"/>.
    /// </summary>
    /// <returns><see cref="IAdvancedDrawingLine"/></returns>
    public IAdvancedDrawingLine ConvertInkMARCDrawingLine(InkMARCDrawingLine inkMARCDrawingLine)
	{
		if (inkMARCDrawingLine is null)
			return new InkMARCDrawingLine();

		return new InkMARCDrawingLine
		{
			LineColor = inkMARCDrawingLine is null ? new InkColor() : inkMARCDrawingLine.LineColor,
			ShouldSmoothPathWhenDrawn = inkMARCDrawingLine?.ShouldSmoothPathWhenDrawn ?? true,
			Granularity = inkMARCDrawingLine?.Granularity ?? 5,
			LineWidth = inkMARCDrawingLine?.LineWidth ?? 1,
			Points = inkMARCDrawingLine?.Points?.ToObservableCollection() ?? new System.Collections.ObjectModel.ObservableCollection<InkMARCPoint>()
		};
	}
}