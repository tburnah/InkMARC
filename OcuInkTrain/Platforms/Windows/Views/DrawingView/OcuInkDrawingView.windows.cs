using Microsoft.Maui.Platform;
using Microsoft.UI.Xaml.Input;
using OcuInk.Models.Primatives;
using OcuInkTrain.Primatives;
using System.Diagnostics;
using WBrush = Microsoft.UI.Xaml.Media.Brush;
using WColor = Microsoft.UI.Colors;
using WSolidColorBrush = Microsoft.UI.Xaml.Media.SolidColorBrush;

namespace OcuInkTrain.Views;

/// <summary>
/// DrawingView Native Control
/// </summary>
public partial class OcuInkDrawingView : PlatformTouchGraphicsView, IDisposable
{
    bool isDisposed;

    /// <inheritdoc />
    ~OcuInkDrawingView() => Dispose(false);

    /// <summary>
    /// Initialize resources
    /// </summary>
    public void Initialize()
    {
        if (OperatingSystem.IsWindowsVersionAtLeast(10, 0, 18362))
        {
            ((Microsoft.Maui.Graphics.Win2D.W2DGraphicsView)Content).Drawable = new DrawingViewDrawable(this);            
        }
        else
        {
            System.Diagnostics.Trace.WriteLine("DrawingView requires Windows 10.0.18362 or higher.");
        }
        redrawTimer.AutoReset = false;
        redrawTimer.Elapsed += (s, e) => Redraw();
        
        Lines.CollectionChanged += OnLinesCollectionChanged;
    }

    /// <inheritdoc />
    public void Dispose()
    {
        // Do not change this code. Put cleanup code in 'Dispose(bool disposing)' method
        Dispose(disposing: true);
        GC.SuppressFinalize(this);
    }

    /// <inheritdoc />
    protected virtual void Dispose(bool disposing)
    {
        if (!isDisposed)
        {
            if (disposing)
            {
                currentPath.Dispose();
            }

            isDisposed = true;
        }
    }

    /// <inheritdoc />
    protected override void OnPointerPressed(PointerRoutedEventArgs e)
    {
        base.OnPointerPressed(e);

        var currentPoint = e.GetCurrentPoint(this);
        var wPoint = currentPoint.Position;

        var ocuInkPoint = new OcuInkPoint(
            wPoint._x, wPoint._y,
            currentPoint.Properties.Pressure,
            currentPoint.Properties.XTilt,
            currentPoint.Properties.YTilt,
            currentPoint.Timestamp
            );
        OnStart(ocuInkPoint);
    }

    /// <inheritdoc />
    protected override void OnPointerMoved(PointerRoutedEventArgs e)
    {
        base.OnPointerMoved(e);
        var currentPoint = e.GetCurrentPoint(this);
        var wPoint = currentPoint.Position;

        var ocuInkPoint = new OcuInkPoint(
            wPoint._x, wPoint._y,
            currentPoint.Properties.Pressure,
            currentPoint.Properties.XTilt,
            currentPoint.Properties.YTilt,
            currentPoint.Timestamp
            );
        OnMoving(ocuInkPoint);
    }

    /// <inheritdoc />
    protected override void OnPointerReleased(PointerRoutedEventArgs e)
    {
        base.OnPointerReleased(e);
        OnFinish();
    }

    /// <inheritdoc />
    protected override void OnPointerCanceled(PointerRoutedEventArgs e)
    {
        base.OnPointerCanceled(e);
        OnCancel();
    }

    void Redraw()
    {
        Invalidate();
    }
}