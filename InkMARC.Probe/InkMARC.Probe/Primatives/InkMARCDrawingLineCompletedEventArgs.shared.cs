using InkMARC.Probe.Models.Primatives;

namespace InkMARC.Probe.Primatives;

/// <summary>
/// Contains last drawing line
/// </summary>
public class InkMARCDrawingLineCompletedEventArgs : EventArgs
{
	/// <summary>
	/// Initializes last drawing line
	/// </summary>
	/// <param name="line">Last drawing line</param>
	public InkMARCDrawingLineCompletedEventArgs(InkMARCDrawingLine line) => Line = line;

	/// <summary>
	/// Last drawing line
	/// </summary>
	public InkMARCDrawingLine Line { get; }
}