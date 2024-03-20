using System.Collections.ObjectModel;
using CommunityToolkit.Maui.Core.Extensions;
using Scryv.Primatives;
using Scryv.Extensions;

namespace Scryv.Views;

/// <summary>
/// DrawingView Platform Control
/// </summary>
public partial class ScryvDrawingView
{
	readonly WeakEventManager weakEventManager = new();

	bool isDrawing;
	ScryvInkPoint previousPoint;
	PathF currentPath = new();
	ScryvDrawingLine? currentLine;
	Paint paint = new SolidPaint(CommunityToolkit.Maui.Core.DrawingViewDefaults.BackgroundColor);

	/// <summary>
	/// Event raised when drawing line completed 
	/// </summary>
	public event EventHandler<ScryvDrawingLineCompletedEventArgs> DrawingLineCompleted
	{
		add => weakEventManager.AddEventHandler(value);
		remove => weakEventManager.RemoveEventHandler(value);
	}

	/// <summary>
	/// Event raised when drawing started 
	/// </summary>
	public event EventHandler<ScryvDrawingStartedEventArgs> DrawingStarted
	{
		add => weakEventManager.AddEventHandler(value);
		remove => weakEventManager.RemoveEventHandler(value);
	}

	/// <summary>
	/// Event raised when drawing cancelled 
	/// </summary>
	public event EventHandler<ScryvDrawingStartedEventArgs> DrawingCancelled
	{
		add => weakEventManager.AddEventHandler(value);
		remove => weakEventManager.RemoveEventHandler(value);
	}

	/// <summary>
	/// Event raised when drawing 
	/// </summary>
	public event EventHandler<ScryvOnDrawingEventArgs> Drawing
	{
		add => weakEventManager.AddEventHandler(value);
		remove => weakEventManager.RemoveEventHandler(value);
	}

	/// <summary>
	/// Drawing Lines
	/// </summary>
	public ObservableCollection<ScryvDrawingLine> Lines { get; } = new();

	/// <summary>
	/// Enable or disable multiline mode
	/// </summary>
	public bool IsMultiLineModeEnabled { get; set; } = CommunityToolkit.Maui.Core.DrawingViewDefaults.IsMultiLineModeEnabled;

	/// <summary>
	/// Clear drawing on finish
	/// </summary>
	public bool ShouldClearOnFinish { get; set; } = CommunityToolkit.Maui.Core.DrawingViewDefaults.ShouldClearOnFinish;

	/// <summary>
	/// Line color
	/// </summary>
	public Color LineColor { get; set; } = CommunityToolkit.Maui.Core.DrawingViewDefaults.LineColor;

	/// <summary>
	/// Line width
	/// </summary>
	public float LineWidth { get; set; } = CommunityToolkit.Maui.Core.DrawingViewDefaults.LineWidth;

	/// <summary>
	/// Used to draw any shape on the canvas
	/// </summary>
	public Action<ICanvas, RectF>? DrawAction { get; set; }

	/// <summary>
	/// Drawable background
	/// </summary>
	public Paint Paint
	{
		get => paint;
		set
		{
			paint = value;
			Redraw();
		}
	}

	/// <summary>
	/// Clean up resources
	/// </summary>
	public void CleanUp()
	{
		currentPath.Dispose();
		Lines.CollectionChanged -= OnLinesCollectionChanged;
	}

	void OnStart(ScryvInkPoint point)
	{
		isDrawing = true;

		Lines.CollectionChanged -= OnLinesCollectionChanged;

		if (!IsMultiLineModeEnabled)
		{
			Lines.Clear();
			ClearPath();
		}

		previousPoint = point;
		currentPath.MoveTo(previousPoint.Position.X, previousPoint.Position.Y);
		currentLine = new ScryvDrawingLine
		{
			Points = new ObservableCollection<ScryvInkPoint>
			{
                new(
                    previousPoint.Position,
                    previousPoint.Pressure,
                    previousPoint.TiltX,
                    previousPoint.TiltY,
                    previousPoint.Timestamp
                )
            },
			LineColor = LineColor,
			LineWidth = LineWidth
		};

		Redraw();

		Lines.CollectionChanged += OnLinesCollectionChanged;
		OnDrawingStarted(point);
	}

	void OnMoving(ScryvInkPoint currentPoint)
	{
		if (!isDrawing)
		{
			return;
		}

#if !ANDROID
		AddPointToPath(currentPoint);
#endif

		Redraw();
		currentLine?.Points?.Add(currentPoint);
		OnDrawing(currentPoint);
	}

	void OnFinish()
	{
		if (currentLine is not null)
		{
			Lines.Add(currentLine);
			OnDrawingLineCompleted(currentLine);
		}

		if (ShouldClearOnFinish)
		{
			Lines.Clear();
			ClearPath();
		}

		currentLine = null;
		isDrawing = false;
	}

	void OnCancel()
	{
		currentLine = null;
		ClearPath();
		Redraw();
		isDrawing = false;
		OnDrawingCancelled();
	}

	void OnDrawingLineCompleted(ScryvDrawingLine lastDrawingLine) =>
		weakEventManager.HandleEvent(this, new ScryvDrawingLineCompletedEventArgs(lastDrawingLine), nameof(DrawingLineCompleted));

	void OnDrawing(ScryvInkPoint point) =>
		weakEventManager.HandleEvent(this, new ScryvOnDrawingEventArgs(point), nameof(Drawing));

	void OnDrawingStarted(ScryvInkPoint point) =>
		weakEventManager.HandleEvent(this, new ScryvDrawingStartedEventArgs(point), nameof(DrawingStarted));

	void OnDrawingCancelled() =>
		weakEventManager.HandleEvent(this, EventArgs.Empty, nameof(DrawingCancelled));

	void OnLinesCollectionChanged(object? sender, System.Collections.Specialized.NotifyCollectionChangedEventArgs e) => LoadLines();

	void AddPointToPath(ScryvInkPoint currentPoint) => currentPath.LineTo(currentPoint);

	void LoadLines()
	{
		ClearPath();
		Redraw();
	}

	void ClearPath()
	{
		currentPath = new PathF();
	}

	class DrawingViewDrawable : IDrawable
	{
		readonly ScryvDrawingView drawingView;

		public DrawingViewDrawable(ScryvDrawingView drawingView)
		{
			this.drawingView = drawingView;
		}

		public void Draw(ICanvas canvas, RectF dirtyRect)
		{
			canvas.SetFillPaint(drawingView.Paint, dirtyRect);
			canvas.FillRectangle(dirtyRect);

			drawingView.DrawAction?.Invoke(canvas, dirtyRect);

			DrawCurrentLines(canvas, drawingView);

			SetStroke(canvas, drawingView.LineWidth, drawingView.LineColor);
			canvas.DrawPath(drawingView.currentPath);
		}

		static void SetStroke(in ICanvas canvas, in float lineWidth, in Color lineColor)
		{
			canvas.StrokeColor = lineColor;
			canvas.StrokeSize = lineWidth;
			canvas.StrokeDashOffset = 0;
			canvas.StrokeLineCap = LineCap.Round;
			canvas.StrokeLineJoin = LineJoin.Round;
			canvas.StrokeDashPattern = Array.Empty<float>();
		}

		static void DrawCurrentLines(in ICanvas canvas, in ScryvDrawingView drawingView)
		{
			foreach (var line in drawingView.Lines)
			{
				var path = new PathF();
				var points = line.ShouldSmoothPathWhenDrawn
					? line.Points?.CreateSmoothedPathWithGranularity(line.Granularity)
					: line.Points;
#if ANDROID
#pragma warning disable CS8604 // Possible null reference argument.
				points = CreateCollectionWithNormalizedPoints(points, drawingView.Width, drawingView.Height, canvas.DisplayScale);
#pragma warning restore CS8604 // Possible null reference argument.
#endif
				if (points is not null && points.Count > 0)
				{
					path.MoveTo(points[0].Position.X, points[0].Position.Y);
					foreach (var point in points)
					{
						path.LineTo(point);
					}

					SetStroke(canvas, line.LineWidth, line.LineColor ?? Colors.Black);
					canvas.DrawPath(path);
				}
			}
		}
	}
}