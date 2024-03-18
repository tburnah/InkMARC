using System.Collections.Immutable;
using System.Collections.ObjectModel;
using CommunityToolkit.Maui.Core;
using CommunityToolkit.Maui.Core.Extensions;
using Scryv.Handlers;
using Scryv.Interfaces;
using Scryv.Primatives;
using Scryv.Views;

namespace Scryv.Extensions;

/// <summary>
/// Extension methods to support <see cref="IDrawingView"/>
/// </summary>
public static class ScryvDrawingViewExtensions
{

	/// <summary>
	/// Get smoothed path.
	/// </summary>
	public static ObservableCollection<ScryvInkPoint> CreateSmoothedPathWithGranularity(this IEnumerable<ScryvInkPoint> currentPoints, int granularity)
	{
		var currentPointsList = new List<ScryvInkPoint>(currentPoints);

		// not enough points to smooth effectively, so return the original path and points.
		if (currentPointsList.Count < granularity + 2)
		{
			return currentPointsList.ToObservableCollection();
		}

		var smoothedPointsList = new List<ScryvInkPoint>();

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
    /// <param name="scryvDrawingView"><see cref="ScryvDrawingView"/></param>
    /// <param name="multiLineMode">value</param>
    public static void SetIsMultiLineModeEnabled(this ScryvDrawingView scryvDrawingView, bool multiLineMode)
	{
        scryvDrawingView.IsMultiLineModeEnabled = multiLineMode;
	}

    /// <summary>
    /// Set DrawAction action
    /// </summary>
    /// <param name="scryvDrawingView"><see cref="ScryvDrawingView"/></param>
    /// <param name="draw">value</param>
    public static void SetDrawAction(this ScryvDrawingView scryvDrawingView, Action<ICanvas, RectF>? draw)
	{
        scryvDrawingView.DrawAction = draw;
	}

    /// <summary>
    /// Set ClearOnFinish
    /// </summary>
    /// <param name="scryvDrawingView"><see cref="ScryvDrawingView"/></param>
    /// <param name="clearOnFinish">value</param>
    public static void SetShouldClearOnFinish(this ScryvDrawingView scryvDrawingView, bool clearOnFinish)
	{
        scryvDrawingView.ShouldClearOnFinish = clearOnFinish;
	}

    /// <summary>
    /// Set LineColor
    /// </summary>
    /// <param name="scryvDrawingView"><see cref="ScryvDrawingView"/></param>
    /// <param name="lineColor">line color</param>
    public static void SetLineColor(this ScryvDrawingView scryvDrawingView, Color lineColor)
	{
        scryvDrawingView.LineColor = lineColor;
	}

    /// <summary>
    /// Set LineWidth
    /// </summary>
    /// <param name="scryvDrawingView"><see cref="ScryvDrawingView"/></param>
    /// <param name="lineWidth">line width</param>
    public static void SetLineWidth(this ScryvDrawingView scryvDrawingView, float lineWidth)
	{
        scryvDrawingView.LineWidth = lineWidth;
	}

    /// <summary>
    /// Set Paint
    /// </summary>
    /// <param name="scryvDrawingView"><see cref="ScryvDrawingView"/></param>
    /// <param name="background">background</param>
    public static void SetPaint(this ScryvDrawingView scryvDrawingView, Paint background)
	{
        scryvDrawingView.Paint = background;
	}

    /// <summary>
    /// Set Lines
    /// </summary>
    /// <param name="scryvDrawingView"><see cref="ScryvDrawingView"/></param>
    /// <param name="drawingView"><see cref="IDrawingView"/></param>
    public static void SetLines(this ScryvDrawingView scryvDrawingView, IScryvDrawingView drawingView)
	{
		var lines = drawingView.Lines.ToImmutableList();
		if (!drawingView.IsMultiLineModeEnabled && lines.Count > 1)
		{
			lines = lines.TakeLast(1).ToImmutableList();
		}

        scryvDrawingView.Lines.Clear();

		foreach (var line in lines)
		{
            scryvDrawingView.Lines.Add(new ScryvDrawingLine
			{
				LineColor = line.LineColor,
				ShouldSmoothPathWhenDrawn = line.ShouldSmoothPathWhenDrawn,
				Granularity = line.Granularity,
				LineWidth = line.LineWidth,
				Points = line.Points.ToObservableCollection()
			});
		}
	}

    /// <summary>
    /// Set Lines
    /// </summary>
    /// <param name="scryvDrawingView"><see cref="ScryvDrawingView"/></param>
    /// <param name="drawingView"><see cref="IScryvDrawingView"/></param>
    /// <param name="adapter"><see cref="IScryvDrawingLineAdapter"/></param>
    public static void SetLines(this IScryvDrawingView drawingView, ScryvDrawingView scryvDrawingView, IAdvancedDrawingLineAdapter adapter)
	{
		var lines = scryvDrawingView.Lines.ToImmutableList();
		if (!scryvDrawingView.IsMultiLineModeEnabled && lines.Count > 1)
		{
			lines = lines.TakeLast(1).ToImmutableList();
		}

		drawingView.Lines.Clear();

		foreach (var line in lines)
		{
			drawingView.Lines.Add(adapter.ConvertScryvDrawingLine(line));
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
    static ScryvInkPoint GetIntermediatePoint(ScryvInkPoint p0, ScryvInkPoint p1, ScryvInkPoint p2, ScryvInkPoint p3, in float t, in float tt, in float ttt) => new(
        new()
        {
            X = 0.5f * (2f * p1.Position.X +
                    (p2.Position.X - p0.Position.X) * t +
                    (2f * p0.Position.X - 5f * p1.Position.X + 4f * p2.Position.X - p3.Position.X) * tt +
                    (3f * p1.Position.X - p0.Position.X - 3f * p2.Position.X + p3.Position.X) * ttt),
            Y = 0.5f * (2 * p1.Position.Y +
                    (p2.Position.Y - p0.Position.Y) * t +
                    (2 * p0.Position.Y - 5 * p1.Position.Y + 4 * p2.Position.Y - p3.Position.Y) * tt +
                    (3 * p1.Position.Y - p0.Position.Y - 3 * p2.Position.Y + p3.Position.Y) * ttt)
        },
        0.5f * (2 * p1.Pressure + (p2.Pressure - p0.Pressure) * t +
                   (2 * p0.Pressure - 5 * p1.Pressure + 4 * p2.Pressure - p3.Pressure) * tt +
                   (3 * p1.Pressure - p0.Pressure - 3 * p2.Pressure + p3.Pressure) * ttt),
        0.5f * (2 * p1.TiltX + (p2.TiltX - p0.TiltX) * t +
    (2 * p0.TiltX - 5 * p1.TiltX + 4 * p2.TiltX - p3.TiltX) * tt +
                (3 * p1.TiltX - p0.TiltX - 3 * p2.TiltX + p3.TiltX) * ttt),
        0.5f * (2 * p1.TiltY + (p2.TiltY - p0.TiltY) * t +
    (2 * p0.TiltY - 5 * p1.TiltY + 4 * p2.TiltY - p3.TiltY) * tt +
                (3 * p1.TiltY - p0.TiltY - 3 * p2.TiltY + p3.TiltY) * ttt),
        (ulong)(0.5f * (2 * p1.Timestamp + (p2.Timestamp - p0.Timestamp) * t +
                    (2 * p0.Timestamp - 5 * p1.Timestamp + 4 * p2.Timestamp - p3.Timestamp) * tt +
                    (3 * p1.Timestamp - p0.Timestamp - 3 * p2.Timestamp + p3.Timestamp) * ttt))
    );
}