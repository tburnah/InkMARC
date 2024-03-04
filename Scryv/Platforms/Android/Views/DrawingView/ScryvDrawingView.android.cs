using System.Collections.ObjectModel;
using Android.Content;
using Android.Views;
using CommunityToolkit.Maui.Core.Extensions;
using Microsoft.Maui.Platform;
using Scryv.Primatives;
using Scryv.Utilities;
using AColor = Android.Graphics.Color;
using APaint = Android.Graphics.Paint;
using APath = Android.Graphics.Path;
using AView = Android.Views.View;

namespace Scryv.Views;

public partial class ScryvDrawingView : PlatformTouchGraphicsView
{
	/// <summary>
	/// Initialize a new instance of <see cref="ScryvDrawingView" />.
	/// </summary>
	public ScryvDrawingView(Context context) : base(context)
	{
		previousPoint = new();
	}

	/// <inheritdoc />
	protected override void Dispose(bool disposing)
	{
		if (disposing)
		{
			currentPath.Dispose();
		}

		base.Dispose(disposing);
	}

	/// <inheritdoc />
	public override bool OnTouchEvent(MotionEvent? e)
	{
		base.OnTouchEvent(e);
		ArgumentNullException.ThrowIfNull(e);

		var touchX = e.GetX();
		var touchY = e.GetY();

		// For a stylus, reports the tilt angle of the stylus in radians where 0 radians indicates that the stylus is being held perpendicular 
		// to the surface, and PI / 2 radians indicates that the stylus is being held flat against the surface.
        var tilt = e.GetAxisValue(Axis.Tilt);

        // For a stylus, the orientation indicates the direction in which the stylus is pointing in relation to the vertical axis
		// of the current orientation of the screen. The range is from -PI radians to PI radians, where 0 is pointing up, -PI/2 radians 
		// is pointing left, -PI or PI radians is pointing down, and PI/2 radians is pointing right.
        var orientation = e.GetAxisValue(Axis.Orientation);

        var (tiltX, tiltY) = StylusTiltCalculator.CalculateTilt(orientation, tilt);

        var point = new ScryvInkPoint()
		{
			Position = new(touchX / (float)DeviceDisplay.MainDisplayInfo.Density, touchY / (float)DeviceDisplay.MainDisplayInfo.Density),
			Pressure = e.Pressure,
            TiltX = (float)tiltX,
            TiltY = (float)tiltY,
            Timestamp = (ulong)e.EventTime
		};
		
		switch (e.Action)
		{
			case MotionEventActions.Down:
				Parent?.RequestDisallowInterceptTouchEvent(true);
				OnStart(point);
				break;

			case MotionEventActions.Move:
				if (touchX > 0 && touchY > 0 && touchX < Width && touchY < Height)
				{
					AddPointToPath(point);
				}

				OnMoving(point);
				break;

			case MotionEventActions.Up:
				Parent?.RequestDisallowInterceptTouchEvent(false);
				OnFinish();
				break;
			case MotionEventActions.Cancel:
				Parent?.RequestDisallowInterceptTouchEvent(false);
				OnCancel();
				break;

			default:
				return false;
		}

		Redraw();

		return true;
	}

	/// <summary>
	/// Initialize resources
	/// </summary>
	public void Initialize()
	{
		Drawable = new DrawingViewDrawable(this);
		Lines.CollectionChanged += OnLinesCollectionChanged;
	}

	void Redraw()
	{
		Invalidate();
	}

	static ObservableCollection<ScryvInkPoint> CreateCollectionWithNormalizedPoints(in ObservableCollection<ScryvInkPoint> points, in int drawingViewWidth, in int drawingViewHeight, in float canvasScale)
	{
		var newPoints = new List<ScryvInkPoint>();
		foreach (var point in points)
		{
			var pointX = Math.Clamp(point.Position.X, 0, drawingViewWidth / canvasScale);
			var pointY = Math.Clamp(point.Position.Y, 0, drawingViewHeight / canvasScale);
            newPoints.Add(new ScryvInkPoint() { Position = new PointF(pointX, pointY), Pressure = point.Pressure, TiltX = point.TiltX, TiltY = point.TiltY, Timestamp = point.Timestamp });
        }

		return newPoints.ToObservableCollection();
	}
}