namespace OcuInkTrain.Primatives;

/// <summary>
/// Contains last drawing line
/// </summary>
public class OcuInkDrawingLineCompletedEventArgs : EventArgs
{
	/// <summary>
	/// Initializes last drawing line
	/// </summary>
	/// <param name="line">Last drawing line</param>
	public OcuInkDrawingLineCompletedEventArgs(OcuInkDrawingLine line) => Line = line;

	/// <summary>
	/// Last drawing line
	/// </summary>
	public OcuInkDrawingLine Line { get; }
}