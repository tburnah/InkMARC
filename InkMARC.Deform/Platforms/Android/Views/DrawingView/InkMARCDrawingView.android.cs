using System.Collections.ObjectModel;
using Android.Content;
using Android.Views;
using CommunityToolkit.Maui.Core.Extensions;
using Microsoft.Maui.Platform;
using InkMARC.Models.Primatives;
using InkMARCDeform.Primatives;
using InkMARCDeform.Utilities;
using AColor = Android.Graphics.Color;
using APaint = Android.Graphics.Paint;
using APath = Android.Graphics.Path;
using AView = Android.Views.View;

namespace InkMARCDeform.Views;

public partial class InkMARCDrawingView : PlatformTouchGraphicsView
{
	/// <summary>
	/// Initialize a new instance of <see cref="InkMARCDrawingView" />.
	/// </summary>
	public InkMARCDrawingView(Context context) : base(context)
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

		var point = new InkMARCPoint(touchX / (float)DeviceDisplay.MainDisplayInfo.Density, touchY / (float)DeviceDisplay.MainDisplayInfo.Density,
			e.Pressure,
			(float)tiltX,
			(float)tiltY,
			(ulong)e.EventTime);		
		
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

	static ObservableCollection<InkMARCPoint> CreateCollectionWithNormalizedPoints(in ObservableCollection<InkMARCPoint> points, in int drawingViewWidth, in int drawingViewHeight, in float canvasScale)
	{
		var newPoints = new List<InkMARCPoint>();
		foreach (var point in points)
		{
			newPoints.Add(new InkMARCPoint(Math.Clamp(point.X, 0, drawingViewWidth / canvasScale), Math.Clamp(point.Y, 0, drawingViewHeight / canvasScale), point.Pressure, point.TiltX, point.TiltY, point.Timestamp));
        }

		return newPoints.ToObservableCollection();
	}
}