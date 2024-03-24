namespace OcuInkTrain.Primatives;

/// <summary>
/// Contains last drawing point
/// </summary>
public class OcuInkOnDrawingEventArgs : EventArgs
{
	/// <summary>
	/// Initializes last drawing point
	/// </summary>
	/// <param name="point">Last drawing point</param>
	public OcuInkOnDrawingEventArgs(OcuInkPoint point) => Point = point;

	/// <summary>
	/// Last drawing point
	/// </summary>
	public OcuInkPoint Point { get; }
}