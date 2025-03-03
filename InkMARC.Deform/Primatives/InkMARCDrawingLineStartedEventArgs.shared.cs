using InkMARC.Models.Primatives;

namespace InkMARCDeform.Primatives;

/// <summary>
/// Contains last drawing point
/// </summary>
public class InkMARCDrawingLineStartedEventArgs : EventArgs
{
    /// <summary>
    /// Initialize a new instance of <see cref="InkMARCDrawingLineStartedEventArgs"/>
    /// </summary>
    /// <param name="point"></param>
    public InkMARCDrawingLineStartedEventArgs(InkMARCPoint point)
	{
		Point = point;
	}

	/// <summary>
	/// Last drawing point
	/// </summary>
	public InkMARCPoint Point { get; }
}