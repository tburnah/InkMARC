
using InkMARC.Models.Primatives;

namespace InkMARCDeform.Primatives;

/// <summary>
/// Contains last InkMARC drawing point
/// </summary>
public class InkMARCPointDrawnEventArgs : EventArgs
{
    /// <summary>
    /// Initialize a new instance of <see cref="InkMARCPointDrawnEventArgs"/>
    /// </summary>
    /// <param name="point"></param>
    public InkMARCPointDrawnEventArgs(InkMARCPoint point)
	{
		Point = point;
	}

	/// <summary>
	/// Last drawing point
	/// </summary>
	public InkMARCPoint Point { get; }
}