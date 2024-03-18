using Microsoft.Maui.Platform;
using Microsoft.UI.Xaml.Input;
using Scryv.Primatives;
using WBrush = Microsoft.UI.Xaml.Media.Brush;
using WColor = Microsoft.UI.Colors;
using WSolidColorBrush = Microsoft.UI.Xaml.Media.SolidColorBrush;

namespace Scryv.Views;

/// <summary>
/// DrawingView Native Control
/// </summary>
public partial class ScryvDrawingView : PlatformTouchGraphicsView, IDisposable
{
	bool isDisposed;

	/// <inheritdoc />
	~ScryvDrawingView() => Dispose(false);

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

		var scryvPoint = new ScryvInkPoint(
			new(wPoint._x, wPoint._y),
			currentPoint.Properties.Pressure,
			currentPoint.Properties.XTilt,
			currentPoint.Properties.YTilt,
			e.GetCurrentPoint(this).Timestamp);
        
		OnStart(scryvPoint);
	}

	/// <inheritdoc />
	protected override void OnPointerMoved(PointerRoutedEventArgs e)
	{
		base.OnPointerMoved(e);
        var currentPoint = e.GetCurrentPoint(this);
        var wPoint = currentPoint.Position;

        var scryvPoint = new ScryvInkPoint(        
            new(wPoint._x, wPoint._y),
            currentPoint.Properties.Pressure,
            currentPoint.Properties.XTilt,
            currentPoint.Properties.YTilt,
            e.GetCurrentPoint(this).Timestamp
        );
        OnMoving(scryvPoint);
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