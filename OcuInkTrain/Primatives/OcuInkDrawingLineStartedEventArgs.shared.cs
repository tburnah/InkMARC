using OcuInk.Models.Primatives;

namespace OcuInkTrain.Primatives;

/// <summary>
/// Contains last drawing point
/// </summary>
public class OcuInkDrawingLineStartedEventArgs : EventArgs
{
    /// <summary>
    /// Initialize a new instance of <see cref="OcuInkDrawingLineStartedEventArgs"/>
    /// </summary>
    /// <param name="point"></param>
    public OcuInkDrawingLineStartedEventArgs(OcuInkPoint point)
	{
		Point = point;
	}

	/// <summary>
	/// Last drawing point
	/// </summary>
	public OcuInkPoint Point { get; }
}