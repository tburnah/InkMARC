using Scryv.Interfaces;

namespace Scryv.Handlers;

/// <summary>
/// <see cref="IScryvDrawingView"/> handler.
/// </summary>
public interface IAdvancedDrawingViewHandler
{
	/// <summary>
	/// Set <see cref="IAdvancedDrawingLineAdapter"/>.
	/// </summary>
	/// <param name="drawingLineAdapter"><see cref="IAdvancedDrawingLineAdapter"/></param>
	void SetDrawingLineAdapter(IAdvancedDrawingLineAdapter drawingLineAdapter);
}