
namespace OcuInkTrain.Primatives;

/// <summary>
/// Contains last scryv drawing point
/// </summary>
public class ScryvPointDrawnEventArgs : EventArgs
{
    /// <summary>
    /// Initialize a new instance of <see cref="ScryvPointDrawnEventArgs"/>
    /// </summary>
    /// <param name="point"></param>
    public ScryvPointDrawnEventArgs(ScryvInkPoint point)
	{
		Point = point;
	}

	/// <summary>
	/// Last Scryv drawing point
	/// </summary>
	public ScryvInkPoint Point { get; }
}