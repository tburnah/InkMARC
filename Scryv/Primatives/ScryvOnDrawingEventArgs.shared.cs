namespace Scryv.Primatives;

/// <summary>
/// Contains last drawing point
/// </summary>
public class ScryvOnDrawingEventArgs : EventArgs
{
	/// <summary>
	/// Initializes last drawing point
	/// </summary>
	/// <param name="point">Last drawing point</param>
	public ScryvOnDrawingEventArgs(ScryvInkPoint point) => Point = point;

	/// <summary>
	/// Last drawing point
	/// </summary>
	public ScryvInkPoint Point { get; }
}