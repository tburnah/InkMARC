using Microsoft.Maui.Platform;
using Microsoft.UI.Xaml.Input;
using InkMARC.Models.Primatives;
using InkMARCDeform.Primatives;
using System.Diagnostics;
using WBrush = Microsoft.UI.Xaml.Media.Brush;
using WColor = Microsoft.UI.Colors;
using WSolidColorBrush = Microsoft.UI.Xaml.Media.SolidColorBrush;

namespace InkMARCDeform.Views;

/// <summary>
/// DrawingView Native Control
/// </summary>
public partial class InkMARCDrawingView : PlatformTouchGraphicsView, IDisposable
{
    bool isDisposed;

    /// <inheritdoc />
    ~InkMARCDrawingView() => Dispose(false);

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

        var inkMARCPoint = new InkMARCPoint(
            wPoint._x, wPoint._y,
            currentPoint.Properties.Pressure,
            currentPoint.Properties.XTilt,
            currentPoint.Properties.YTilt,
            currentPoint.Timestamp
            );
        OnStart(inkMARCPoint);
    }

    protected override void OnPointerExited(PointerRoutedEventArgs e)
    {
        base.OnPointerExited(e);
        OnFinish();
    }

    protected override void OnPointerEntered(PointerRoutedEventArgs e)
    {
        base.OnPointerEntered(e);
        if (e.Pointer.IsInContact || (AllowFloatingLines && e.Pointer.PointerDeviceType == Microsoft.UI.Input.PointerDeviceType.Pen))
        {
            var currentPoint = e.GetCurrentPoint(this);
            var wPoint = currentPoint.Position;

            var inkMARCPoint = new InkMARCPoint(
                wPoint._x, wPoint._y,
                currentPoint.Properties.Pressure,
                currentPoint.Properties.XTilt,
                currentPoint.Properties.YTilt,
                currentPoint.Timestamp
                );
            OnStart(inkMARCPoint);
        }
    }

    protected override void OnPointerCaptureLost(PointerRoutedEventArgs e)
    {
        base.OnPointerCaptureLost(e);
        OnCancel();
    }

    /// <inheritdoc />
    protected override void OnPointerMoved(PointerRoutedEventArgs e)
    {
        base.OnPointerMoved(e);
        var currentPoint = e.GetCurrentPoint(this);
        var wPoint = currentPoint.Position;

        var inkMARCPoint = new InkMARCPoint(
            wPoint._x, wPoint._y,
            currentPoint.Properties.Pressure,
            currentPoint.Properties.XTilt,
            currentPoint.Properties.YTilt,
            currentPoint.Timestamp
            );
        OnMoving(inkMARCPoint);
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