using System.Collections.Immutable;
using System.Collections.ObjectModel;
using CommunityToolkit.Maui.Core;
using CommunityToolkit.Maui.Core.Extensions;
using InkMARC.Models.Primatives;
using InkMARCDeform.Handlers;
using InkMARCDeform.Interfaces;
using InkMARCDeform.Primatives;
using InkMARCDeform.Views;

namespace InkMARCDeform.Extensions;

/// <summary>
/// Extension methods to support <see cref="IDrawingView"/>
/// </summary>
public static class InkMARCDrawingViewExtensions
{

	/// <summary>
	/// Get smoothed path.
	/// </summary>
	public static ObservableCollection<InkMARCPoint> CreateSmoothedPathWithGranularity(this IEnumerable<InkMARCPoint> currentPoints, int granularity)
	{
		var currentPointsList = new List<InkMARCPoint>(currentPoints);

		// not enough points to smooth effectively, so return the original path and points.
		if (currentPointsList.Count < granularity + 2)
		{
			return currentPointsList.ToObservableCollection();
		}

		var smoothedPointsList = new List<InkMARCPoint>();

		// duplicate the first and last points as control points.
		currentPointsList.Insert(0, currentPointsList[0]);
		currentPointsList.Add(currentPointsList[^1]);

		// add the first point
		smoothedPointsList.Add(currentPointsList[0]);

		var currentPointsCount = currentPointsList.Count;
		for (var index = 1; index < currentPointsCount - 2; index++)
		{
			var p0 = currentPointsList[index - 1];
			var p1 = currentPointsList[index];
			var p2 = currentPointsList[index + 1];
			var p3 = currentPointsList[index + 2];

			// add n points starting at p1 + dx/dy up until p2 using Catmull-Rom splines
			for (var i = 1; i < granularity; i++)
			{
				var t = i * (1f / granularity);
				var tt = t * t;
				var ttt = tt * t;

				// intermediate point
				var mid = GetIntermediatePoint(p0, p1, p2, p3, t, tt, ttt);
				smoothedPointsList.Add(mid);
			}

			// add p2
			smoothedPointsList.Add(p2);
		}

		// add the last point
		var last = currentPointsList[^1];
		smoothedPointsList.Add(last);

		return smoothedPointsList.ToObservableCollection();
	}

    /// <summary>
    /// Set MultiLine mode
    /// </summary>
    /// <param name="inkMARCDrawingView"><see cref="InkMARCDrawingView"/></param>
    /// <param name="multiLineMode">value</param>
    public static void SetIsMultiLineModeEnabled(this InkMARCDrawingView inkMARCDrawingView, bool multiLineMode)
	{
        inkMARCDrawingView.IsMultiLineModeEnabled = multiLineMode;
	}

    /// <summary>
    /// Set DrawAction action
    /// </summary>
    /// <param name="inkMARCDrawingView"><see cref="InkMARCDrawingView"/></param>
    /// <param name="draw">value</param>
    public static void SetDrawAction(this InkMARCDrawingView inkMARCDrawingView, Action<ICanvas, RectF>? draw)
	{
        inkMARCDrawingView.DrawAction = draw;
	}

    /// <summary>
    /// Set ClearOnFinish
    /// </summary>
    /// <param name="inkMARCDrawingView"><see cref="InkMARCDrawingView"/></param>
    /// <param name="clearOnFinish">value</param>
    public static void SetShouldClearOnFinish(this InkMARCDrawingView inkMARCDrawingView, bool clearOnFinish)
	{
        inkMARCDrawingView.ShouldClearOnFinish = clearOnFinish;
	}

    /// <summary>
    /// Set LineColor
    /// </summary>
    /// <param name="inkMARCDrawingView"><see cref="InkMARCDrawingView"/></param>
    /// <param name="lineColor">line color</param>
    public static void SetLineColor(this InkMARCDrawingView inkMARCDrawingView, Color lineColor)
	{
        inkMARCDrawingView.LineColor = lineColor;
	}

    /// <summary>
    /// Set LineWidth
    /// </summary>
    /// <param name="inkMARCDrawingView"><see cref="InkMARCDrawingView"/></param>
    /// <param name="lineWidth">line width</param>
    public static void SetLineWidth(this InkMARCDrawingView inkMARCDrawingView, float lineWidth)
	{
        inkMARCDrawingView.LineWidth = lineWidth;
	}

    /// <summary>
    /// Set Paint
    /// </summary>
    /// <param name="inkMARCDrawingView"><see cref="InkMARCDrawingView"/></param>
    /// <param name="background">background</param>
    public static void SetPaint(this InkMARCDrawingView inkMARCDrawingView, Paint background)
	{
        inkMARCDrawingView.Paint = background;
	}

    /// <summary>
    /// Set Lines
    /// </summary>
    /// <param name="inkMARCDrawingView"><see cref="InkMARCDrawingView"/></param>
    /// <param name="drawingView"><see cref="IDrawingView"/></param>
    public static void SetLines(this InkMARCDrawingView inkMARCDrawingView, IInkMARCDrawingView drawingView)
	{
		var lines = drawingView.Lines.ToImmutableList();
		if (!drawingView.IsMultiLineModeEnabled && lines.Count > 1)
		{
			lines = lines.TakeLast(1).ToImmutableList();
		}

        inkMARCDrawingView.Lines.Clear();

		foreach (var line in lines)
		{
            inkMARCDrawingView.Lines.Add(new InkMARCDrawingLine
			{
				LineColor = line.LineColor,
				ShouldSmoothPathWhenDrawn = line.ShouldSmoothPathWhenDrawn,
				Granularity = line.Granularity ,
				LineWidth = line.LineWidth,
				Points = line.Points?.ToObservableCollection()
			});
		}
	}

    /// <summary>
    /// Set Lines
    /// </summary>
    /// <param name="inkMARCDrawingView"><see cref="InkMARCDrawingView"/></param>
    /// <param name="drawingView"><see cref="IInkMARCDrawingView"/></param>
    /// <param name="adapter"><see cref="IInkMARCDrawingLineAdapter"/></param>
    public static void SetLines(this IInkMARCDrawingView drawingView, InkMARCDrawingView inkMARCDrawingView, IAdvancedDrawingLineAdapter adapter)
	{
		var lines = inkMARCDrawingView.Lines.ToImmutableList();
		if (!inkMARCDrawingView.IsMultiLineModeEnabled && lines.Count > 1)
		{
			lines = lines.TakeLast(1).ToImmutableList();
		}

		drawingView.Lines.Clear();

		foreach (var line in lines)
		{
			drawingView.Lines.Add(adapter.ConvertInkMARCDrawingLine(line));
        }
    }

    /// <summary>
    /// Calculates the intermediate point between four given points using Catmull-Rom splines.
    /// </summary>
    /// <param name="p0">The first point.</param>
    /// <param name="p1">The second point.</param>
    /// <param name="p2">The third point.</param>
    /// <param name="p3">The fourth point.</param>
    /// <param name="t">The interpolation parameter.</param>
    /// <param name="tt">The squared interpolation parameter.</param>
    /// <param name="ttt">The cubed interpolation parameter.</param>
    /// <returns>The intermediate point.</returns>
    public static InkMARCPoint GetIntermediatePoint(InkMARCPoint p0, InkMARCPoint p1, InkMARCPoint p2, InkMARCPoint p3, float t, float tt, float ttt)
    {
        // Abstracted method to calculate cubic Hermite spline for a single dimension
        static float ComputeDimension(float v0, float v1, float v2, float v3, float t, float tt, float ttt)
        {
            return 0.5f * (
                2f * v1 +
                (v2 - v0) * t +
                (2f * v0 - 5f * v1 + 4f * v2 - v3) * tt +
                (3f * v1 - v0 - 3f * v2 + v3) * ttt);
        }

        var positionX = ComputeDimension(p0.X, p1.X, p2.X, p3.X, t, tt, ttt);
        var positionY = ComputeDimension(p0.Y, p1.Y, p2.Y, p3.Y, t, tt, ttt);
        var pressure = ComputeDimension(p0.Pressure, p1.Pressure, p2.Pressure, p3.Pressure, t, tt, ttt);
        var tiltX = ComputeDimension(p0.TiltX, p1.TiltX, p2.TiltX, p3.TiltX, t, tt, ttt);
        var tiltY = ComputeDimension(p0.TiltY, p1.TiltY, p2.TiltY, p3.TiltY, t, tt, ttt);
        var timestamp = (ulong)ComputeDimension(p0.Timestamp, p1.Timestamp, p2.Timestamp, p3.Timestamp, t, tt, ttt);

        return new InkMARCPoint(positionX, positionY, pressure, tiltX, tiltY, timestamp);
    }   
}