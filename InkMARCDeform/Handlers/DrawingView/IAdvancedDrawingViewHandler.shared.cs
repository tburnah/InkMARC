using InkMARCDeform.Interfaces;

namespace InkMARCDeform.Handlers;

/// <summary>
/// <see cref="IInkMARCDrawingView"/> handler.
/// </summary>
public interface IAdvancedDrawingViewHandler
{
	/// <summary>
	/// Set <see cref="IAdvancedDrawingLineAdapter"/>.
	/// </summary>
	/// <param name="drawingLineAdapter"><see cref="IAdvancedDrawingLineAdapter"/></param>
	void SetDrawingLineAdapter(IAdvancedDrawingLineAdapter drawingLineAdapter);
}