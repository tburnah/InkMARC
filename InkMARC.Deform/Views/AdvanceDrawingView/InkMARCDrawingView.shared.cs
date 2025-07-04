﻿using System.Collections.ObjectModel;
using CommunityToolkit.Maui.Core.Extensions;
using InkMARCDeform.Primatives;
using InkMARCDeform.Extensions;
using System.Diagnostics;
using InkMARC.Models.Primatives;
using InkMARC.Extensions;

namespace InkMARCDeform.Views;

/// <summary>
/// DrawingView Platform Control
/// </summary>
public partial class InkMARCDrawingView
{
    readonly WeakEventManager weakEventManager = new();

    bool isDrawing;
    InkMARCPoint previousPoint;
    PathF currentPath = new();
    InkMARCDrawingLine? currentLine;
    Paint paint = new SolidPaint(CommunityToolkit.Maui.Core.DrawingViewDefaults.BackgroundColor);

    InkMARCPoint? currentCursorPoint;

    /// <summary>
    /// Event raised when drawing line completed 
    /// </summary>
    public event EventHandler<InkMARCDrawingLineCompletedEventArgs> DrawingLineCompleted
    {
        add => weakEventManager.AddEventHandler(value);
        remove => weakEventManager.RemoveEventHandler(value);
    }

    /// <summary>
    /// Event raised when drawing started 
    /// </summary>
    public event EventHandler<InkMARCDrawingStartedEventArgs> DrawingStarted
    {
        add => weakEventManager.AddEventHandler(value);
        remove => weakEventManager.RemoveEventHandler(value);
    }

    /// <summary>
    /// Event raised when drawing cancelled 
    /// </summary>
    public event EventHandler<InkMARCDrawingStartedEventArgs> DrawingCancelled
    {
        add => weakEventManager.AddEventHandler(value);
        remove => weakEventManager.RemoveEventHandler(value);
    }

    /// <summary>
    /// Event raised when drawing 
    /// </summary>
    public event EventHandler<InkMARCOnDrawingEventArgs> Drawing
    {
        add => weakEventManager.AddEventHandler(value);
        remove => weakEventManager.RemoveEventHandler(value);
    }

    /// <summary>
    /// Event raised when pressure changed
    /// </summary>
    public event EventHandler<float> PressureChanged
    {
        add => weakEventManager.AddEventHandler(value);
        remove => weakEventManager.RemoveEventHandler(value);
    }

    /// <summary>
    /// Drawing Lines
    /// </summary>
    public ObservableCollection<InkMARCDrawingLine> Lines { get; } = new();

    /// <summary>
    /// Returns the pressure of the input device
    /// </summary>
    public float Pressure => _pressure;
    private float _pressure = 0f;

    /// <summary>
    /// Enable or disable multiline mode
    /// </summary>
    public bool IsMultiLineModeEnabled { get; set; } = CommunityToolkit.Maui.Core.DrawingViewDefaults.IsMultiLineModeEnabled;

    /// <summary>
    /// Allow floating lines
    /// </summary>
    public bool AllowFloatingLines { get; set; } = false;

    /// <summary>
    /// Clear drawing on finish
    /// </summary>
    public bool ShouldClearOnFinish { get; set; } = CommunityToolkit.Maui.Core.DrawingViewDefaults.ShouldClearOnFinish;

    /// <summary>
    /// Line color
    /// </summary>
    public Color LineColor { get; set; } = CommunityToolkit.Maui.Core.DrawingViewDefaults.LineColor;

    /// <summary>
    /// Cursor Color
    /// </summary>
    public Color CursorColor { get; set; } = CommunityToolkit.Maui.Core.DrawingViewDefaults.LineColor;

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

    void OnStart(InkMARCPoint point)
    {
        isDrawing = true;
        currentCount = 0;
        Lines.CollectionChanged -= OnLinesCollectionChanged;

        SetPressure(point.Pressure);

        if (!IsMultiLineModeEnabled)
        {
            Lines.Clear();
            ClearPath();
        }

        previousPoint = point;
        currentPath.MoveTo(previousPoint.X, previousPoint.Y);
        currentLine = new InkMARCDrawingLine
        {
            Points = new ObservableCollection<InkMARCPoint>
            {
                point
            },
            LineColor = LineColor.ToInkColor(),
            LineWidth = LineWidth
        };
        Redraw();

        Lines.CollectionChanged += OnLinesCollectionChanged;
        OnDrawingStarted(point);
    }

    private int currentCount;
    const int MaxPointsInLine = 100;
    void OnMoving(InkMARCPoint currentPoint)
    {
        if (!isDrawing)
        {
            return;
        }

        currentCursorPoint = currentPoint; // Track the current point for the circle

        SetPressure(currentPoint.Pressure);

        currentCount++;
#if !ANDROID
        AddPointToPath(currentPoint);
#endif
        Redraw();
        currentLine?.Points?.Add(currentPoint);
        OnDrawing(currentPoint);
        if (currentCount > MaxPointsInLine)
        {
            AddLine(new InkMARCDrawingLine(currentLine));
            previousPoint = currentPoint;
            //currentPath.MoveTo(previousPoint.Position.X, previousPoint.Position.Y);
            currentCount = 0;
            currentLine = new InkMARCDrawingLine
            {
                Points = new ObservableCollection<InkMARCPoint>
                {
                    currentPoint
                },
                LineColor = LineColor.ToInkColor(),
                LineWidth = LineWidth
            };
        }
    }

    private InkMARCDrawingLine? savedLine = null;
    private void AddLine(InkMARCDrawingLine line)
    {
        if (savedLine is not null)
        {
            Lines.Add(savedLine);
            savedLine = null;
        }
        savedLine = line;
        Lines.Add(line);
        savedLine = null;
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
        SetPressure(0.0f);
        currentLine = null;
        currentCursorPoint = null; // Clear the current point
        isDrawing = false;
    }

    void OnCancel()
    {
        Debug.WriteLine("OnCancel");
        SetPressure(0.0f);
        currentLine = null;
        ClearPath();
        Redraw();
        isDrawing = false;
        currentCursorPoint = null; // Clear the current point
        OnDrawingCancelled();
    }

    private void SetPressure(float pressure)
    {
        float newPressure = MathF.Round(pressure, 1);
        if (_pressure != newPressure)
        {
            _pressure = newPressure;
            OnPressureChanged(_pressure);
        }
    }

    void OnDrawingLineCompleted(InkMARCDrawingLine lastDrawingLine) =>
        weakEventManager.HandleEvent(this, new InkMARCDrawingLineCompletedEventArgs(lastDrawingLine), nameof(DrawingLineCompleted));

    void OnDrawing(InkMARCPoint point) =>
        weakEventManager.HandleEvent(this, new InkMARCOnDrawingEventArgs(point), nameof(Drawing));

    void OnDrawingStarted(InkMARCPoint point) =>
        weakEventManager.HandleEvent(this, new InkMARCDrawingStartedEventArgs(point), nameof(DrawingStarted));

    void OnDrawingCancelled() =>
        weakEventManager.HandleEvent(this, EventArgs.Empty, nameof(DrawingCancelled));

    void OnLinesCollectionChanged(object? sender, System.Collections.Specialized.NotifyCollectionChangedEventArgs e) => LoadLines();

    void AddPointToPath(InkMARCPoint currentPoint) => currentPath.LineTo(currentPoint);

    void OnPressureChanged(float pressure) =>
        weakEventManager.HandleEvent(this, pressure, nameof(PressureChanged));

    void LoadLines()
    {
        ClearPath();
        Redraw();
    }

    void ClearPath()
    {
        currentPath = new PathF();
    }

    internal void SetAllowFloatingLines(bool allowFloatingLines)
    {
        AllowFloatingLines = allowFloatingLines;
    }

    class DrawingViewDrawable : IDrawable
    {
        readonly InkMARCDrawingView drawingView;

        public DrawingViewDrawable(InkMARCDrawingView drawingView)
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

            // Draw the circle at the current cursor point
            if (drawingView.currentCursorPoint is not null)
            {
                var cursorPoint = drawingView.currentCursorPoint;
                float circleRadius = 10.0f; // Adjust the size of the circle as needed
                canvas.StrokeColor = drawingView.CursorColor; // Set the color of the circle
                canvas.StrokeSize = 2; // Set the thickness of the circle outline
                canvas.DrawCircle(cursorPoint?.X ?? 0, cursorPoint?.Y ?? 0, circleRadius);
            }
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

        static void DrawCurrentLines(in ICanvas canvas, in InkMARCDrawingView drawingView)
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
                    path.MoveTo(points[0].X, points[0].Y);
                    foreach (var point in points)
                    {
                        path.LineTo(point);
                    }

                    SetStroke(canvas, line.LineWidth, line.LineColor.ToMauiColor());
                    canvas.DrawPath(path);
                }
            }
        }
    }
}