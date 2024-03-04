using CommunityToolkit.Maui.Core.Views;
using Scryv.Interfaces;
using Scryv.Primatives;

namespace Scryv.Handlers;

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