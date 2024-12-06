using InkMARC.Models.Primatives;

namespace InkMARCDeform.Primatives;

/// <summary>
/// Contains last drawing point
/// </summary>
public class InkMARCDrawingStartedEventArgs : EventArgs
{
	/// <summary>
	/// Initializes last drawing point
	/// </summary>
	/// <param name="point">Last drawing point</param>
	public InkMARCDrawingStartedEventArgs(InkMARCPoint point) => Point = point;

	/// <summary>
	/// Last drawing point
	/// </summary>
	public InkMARCPoint Point { get; }
}