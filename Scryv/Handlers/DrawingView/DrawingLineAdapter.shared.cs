using CommunityToolkit.Maui.Core.Extensions;
using CommunityToolkit.Maui.Core.Views;
using Scryv.Interfaces;
using Scryv.Primatives;

namespace Scryv.Handlers;

/// <summary>
/// DrawingLine Adapter
/// </summary>
public sealed class AdvancedDrawingLineAdapter : IAdvancedDrawingLineAdapter
{
    /// <summary>
    /// Convert <see cref="ScryvDrawingLine"/> to <see cref="IAdvancedDrawingLine"/>.
    /// </summary>
    /// <returns><see cref="IScryvDrawingLine"/></returns>
    public IAdvancedDrawingLine ConvertScryvDrawingLine(ScryvDrawingLine scryvDrawingLine)
	{
		return new ScryvDrawingLine
		{
			LineColor = scryvDrawingLine.LineColor,
			ShouldSmoothPathWhenDrawn = scryvDrawingLine.ShouldSmoothPathWhenDrawn,
			Granularity = scryvDrawingLine.Granularity,
			LineWidth = scryvDrawingLine.LineWidth,
			Points = scryvDrawingLine.Points.ToObservableCollection()
		};
	}
}