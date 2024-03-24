
namespace OcuInkTrain.Primatives;

/// <summary>
/// Contains last OcuInk drawing point
/// </summary>
public class OcuInkPointDrawnEventArgs : EventArgs
{
    /// <summary>
    /// Initialize a new instance of <see cref="OcuInkPointDrawnEventArgs"/>
    /// </summary>
    /// <param name="point"></param>
    public OcuInkPointDrawnEventArgs(OcuInkPoint point)
	{
		Point = point;
	}

	/// <summary>
	/// Last drawing point
	/// </summary>
	public OcuInkPoint Point { get; }
}