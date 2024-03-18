namespace Scryv.Primatives;

/// <summary>
/// Contains last drawing line
/// </summary>
public class ScryvDrawingLineCompletedEventArgs : EventArgs
{
	/// <summary>
	/// Initializes last drawing line
	/// </summary>
	/// <param name="line">Last drawing line</param>
	public ScryvDrawingLineCompletedEventArgs(ScryvDrawingLine line) => Line = line;

	/// <summary>
	/// Last drawing line
	/// </summary>
	public ScryvDrawingLine Line { get; }
}