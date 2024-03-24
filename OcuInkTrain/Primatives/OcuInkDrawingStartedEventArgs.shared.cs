namespace OcuInkTrain.Primatives;

/// <summary>
/// Contains last drawing point
/// </summary>
public class OcuInkDrawingStartedEventArgs : EventArgs
{
	/// <summary>
	/// Initializes last drawing point
	/// </summary>
	/// <param name="point">Last drawing point</param>
	public OcuInkDrawingStartedEventArgs(OcuInkPoint point) => Point = point;

	/// <summary>
	/// Last drawing point
	/// </summary>
	public OcuInkPoint Point { get; }
}