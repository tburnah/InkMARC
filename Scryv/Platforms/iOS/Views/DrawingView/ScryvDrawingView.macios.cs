using Foundation;
using Microsoft.Maui.Graphics.Platform;
using Microsoft.Maui.Platform;
using OcuInkTrain.Primatives;
using OcuInkTrain.Utilities;
using UIKit;

namespace OcuInkTrain.Views;

public partial class ScryvDrawingView : PlatformTouchGraphicsView
{
	readonly List<UIScrollView> scrollViewParents = new();

    private ScryvInkPoint ScryvInkPointFromTouch(NSSet touches)
    {
        var touch = (UITouch)touches.AnyObject;

        // Accessing basic touch properties
        var location = touch.PreviousLocationInView(this).AsPointF(); // Location of touch
        var pressure = touch.Force; // Pressure of touch
        var timestamp = touch.Timestamp; // Timestamp of touch event

        // Accessing stylus-specific properties (if available)
        var azimuthAngle = touch.GetAzimuthAngle(this); // Direction of the stylus
        var altitudeAngle = touch.AltitudeAngle; // Elevation angle of the stylus        

        var (tiltX, tiltY) = StylusTiltCalculator.CalculateTilt(azimuthAngle, altitudeAngle);

        var point = new ScryvInkPoint(location, (float)pressure, (float)tiltX, (float)tiltY, (ulong)timestamp);
        return point;
    }
    
	/// <inheritdoc />
    public override void TouchesBegan(NSSet touches, UIEvent? evt)
    {
        base.TouchesBegan(touches, evt);
        DetectScrollViews();
        SetParentTouches(false);

        ScryvInkPoint point = ScryvInkPointFromTouch(touches);
        OnStart(point);
    }

    /// <inheritdoc />
    public override void TouchesMoved(NSSet touches, UIEvent? evt)
	{
		base.TouchesMoved(touches, evt);
        ScryvInkPoint point = ScryvInkPointFromTouch(touches);

        OnMoving(point);
	}

	/// <inheritdoc />
	public override void TouchesEnded(NSSet touches, UIEvent? evt)
	{
		base.TouchesEnded(touches, evt);
		OnFinish();
		SetParentTouches(true);
	}

	/// <inheritdoc />
	public override void TouchesCancelled(NSSet touches, UIEvent? evt)
	{
		base.TouchesCancelled(touches, evt);
		OnCancel();
		SetParentTouches(true);
	}

	/// <summary>
	/// Initialize resources
	/// </summary>
	public void Initialize()
	{
		Drawable = new DrawingViewDrawable(this);
		Lines.CollectionChanged += OnLinesCollectionChanged;
	}

	/// <inheritdoc/>
	protected override void Dispose(bool disposing)
	{
		if (disposing)
		{
			currentPath.Dispose();
		}

		base.Dispose(disposing);
	}

	void DetectScrollViews()
	{
		if (scrollViewParents.Count > 0)
		{
			return;
		}

		var parent = Superview;

		while (parent is not null)
		{
			if (parent is UIScrollView scrollView)
			{
				scrollViewParents.Add(scrollView);
			}

			parent = parent.Superview;
		}
	}

	void SetParentTouches(bool enabled)
	{
		foreach (var scrollViewParent in scrollViewParents)
		{
			scrollViewParent.ScrollEnabled = enabled;
		}
	}

	void Invalidate()
	{
		SetNeedsDisplay();
	}

	void Redraw()
	{
		Invalidate();
	}
}