namespace OcuInkTrain.Primatives;

/// <summary>
/// Contains last drawing point
/// </summary>
public class ScryvDrawingLineStartedEventArgs : EventArgs
{
    /// <summary>
    /// Initialize a new instance of <see cref="ScryvDrawingLineStartedEventArgs"/>
    /// </summary>
    /// <param name="point"></param>
    public ScryvDrawingLineStartedEventArgs(ScryvInkPoint point)
	{
		Point = point;
	}

	/// <summary>
	/// Last drawing point
	/// </summary>
	public ScryvInkPoint Point { get; }
}