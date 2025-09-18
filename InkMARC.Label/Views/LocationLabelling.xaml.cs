using InkMARC.Label.Services;
using MaterialDesignColors.Recommended;
using OpenCvSharp;
using System.ComponentModel;
using System.Diagnostics;
using System.Text;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Controls.Primitives;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Media.Media3D;
using System.Windows.Navigation;
using System.Windows.Shapes;

namespace InkMARC.Label
{
    public partial class LocationLabelling : UserControl
    {
        private LocationLabellingViewModel? viewModel;
        private readonly List<Path> overlayPluses = [];
        private readonly List<Path> inferredPluses = [];
        private readonly List<Rectangle> cornerRects = [];
        private readonly Ellipse drawingPoint;

        // Shared brushes and pens
        private static readonly Brush OverlayCircleStroke = Brushes.Blue.Clone();
        private static readonly Brush OverlayCircleStrokeInactive = Brushes.Transparent.Clone();
        private static readonly Pen OverlayCirclePen = new(OverlayCircleStroke, 1);
        private static readonly Brush CornerFill = Brushes.Transparent;
        private static readonly Pen GreenCornerPen = new(Brushes.Green, 1);
        private const double OverlayPlusRadius = 4;
        private static readonly Geometry OverlayPlusGeometry;

        private System.Windows.Point _pTL, _pTR, _pBR, _pBL;
        private System.Windows.Point _dragStartCanvas;

        // Drag state
        private Thumb? _activeThumb;
        private bool _overlayInitDone;

        static LocationLabelling()
        {
            if (OverlayCircleStroke.CanFreeze) OverlayCircleStroke.Freeze();
            if (OverlayCirclePen.CanFreeze) OverlayCirclePen.Freeze();
            if (GreenCornerPen.CanFreeze) GreenCornerPen.Freeze();
            if (OverlayCircleStrokeInactive.CanFreeze) OverlayCircleStrokeInactive.Freeze();

            var g = new GeometryGroup();
            g.Children.Add(new LineGeometry(new System.Windows.Point(-OverlayPlusRadius, 0),
                                            new System.Windows.Point(OverlayPlusRadius, 0)));
            g.Children.Add(new LineGeometry(new System.Windows.Point(0, -OverlayPlusRadius),
                                            new System.Windows.Point(0, OverlayPlusRadius)));
            g.Freeze();
            OverlayPlusGeometry = g;
        }

        public LocationLabelling()
        {
            InitializeComponent();

            drawingPoint = new Ellipse
            {
                Width = 1,
                Height = 1,
                Fill = CornerFill,
                Stroke = Brushes.White,
                StrokeThickness = 0.5,
                SnapsToDevicePixels = true
            };
            OverlayCanvas.Children.Add(drawingPoint);

            DataContextChanged += OnDataContextChanged;
            Unloaded += LocationLabelling_Unloaded;
            Loaded += OnLoaded;
            SizeChanged += OnSizeChanged;  // keep camera in sync with overlay size
            LayoutUpdated += OnLayoutUpdated; // <-- added
        }

        private void LocationLabelling_Unloaded(object sender, RoutedEventArgs e)
        {
            if (viewModel != null)
                viewModel.PropertyChanged -= ViewModel_PropertyChanged;
        }

        private void OnDataContextChanged(object sender, DependencyPropertyChangedEventArgs e)
        {
            if (e.OldValue is LocationLabellingViewModel oldVm)
                oldVm.PropertyChanged -= ViewModel_PropertyChanged;

            viewModel = e.NewValue as LocationLabellingViewModel;

            if (viewModel != null)
                viewModel.PropertyChanged += ViewModel_PropertyChanged;

            // If the overlay depends on VM data and the control is already loaded, refresh
            if (IsLoaded)
            {
                _overlayInitDone = false;
                TryInitOverlay();
                // any immediate redraws that need VM data:
                UpdateInferredPluses();
                UpdateOverlayCircles(movePrev: false);
                UpdateCornerRects();
            }
        }

        private void OnLoaded(object? sender, RoutedEventArgs e)
        {
            _overlayInitDone = false;   // allow first-time init
            TryInitOverlay();           // will run now if sizes are ready; otherwise LayoutUpdated will handle it
        }

        private void OnLayoutUpdated(object? sender, EventArgs e)
        {
            // Run once, after Viewbox has produced ActualWidth/Height for VideoImage/OverlayCanvas.
            if (!_overlayInitDone) TryInitOverlay();
        }

        private void TryInitOverlay()
        {
            // We can drag anywhere in the container:
            if (ImageContainer.ActualWidth <= 0 || ImageContainer.ActualHeight <= 0) return;

            // Find where the VideoImage is *actually drawn* inside ImageContainer
            // (accounts for Viewbox scaling + centering)
            var imgRect = new System.Windows.Rect(0, 0, VideoImage.ActualWidth, VideoImage.ActualHeight);
            var t = VideoImage.TransformToAncestor(ImageContainer);
            var r = t.TransformBounds(imgRect);   // displayed rect in container coordinates

            // Start the quad covering the displayed image rect
            _pTL = new System.Windows.Point(r.Left, r.Top);
            _pTR = new System.Windows.Point(r.Right, r.Top);
            _pBR = new System.Windows.Point(r.Right, r.Bottom);
            _pBL = new System.Windows.Point(r.Left, r.Bottom);

            // Size/move the thumbs and mesh
            PlaceHandle(TL, _pTL);
            PlaceHandle(TR, _pTR);
            PlaceHandle(BR, _pBR);
            PlaceHandle(BL, _pBL);

            FitCameraToOverlay();
            UpdateMeshFromHandles();

            _overlayInitDone = true;
        }

        private void OnSizeChanged(object? sender, SizeChangedEventArgs e)
        {
            FitCameraToOverlay();
            if (_overlayInitDone) UpdateMeshFromHandles();
        }

        private void FitCameraToOverlay()
        {
            if (Camera == null) return;

            // If you kept the canvas name as OverlayCanvas, this matches your code.
            // If you renamed it to "Overlay" in XAML, change the reference accordingly.
            var w = ImageContainer.ActualWidth;
            if (w > 0) Camera.Width = w;
        }

        private static void PlaceHandle(FrameworkElement thumb, System.Windows.Point p)
        {
            double r = thumb.Width / 2.0;
            Canvas.SetLeft(thumb, p.X - r);
            Canvas.SetTop(thumb, p.Y - r);
        }

        private static void SetThumbCenter(FrameworkElement thumb, System.Windows.Point center)
        {
            Canvas.SetLeft(thumb, center.X - thumb.Width / 2.0);
            Canvas.SetTop(thumb, center.Y - thumb.Height / 2.0);
        }

        private void Corner_DragStarted(object sender, DragStartedEventArgs e)
        {
            _activeThumb = (Thumb)sender;
            _dragStartCanvas = new System.Windows.Point(
                Canvas.GetLeft(_activeThumb) + _activeThumb.Width / 2.0,
                Canvas.GetTop(_activeThumb) + _activeThumb.Height / 2.0);
        }

        private void Corner_DragDelta(object sender, DragDeltaEventArgs e)
        {
            if (_activeThumb == null) return;

            // Overlay is in container space already; deltas are 1:1
            double dx = e.HorizontalChange;
            double dy = e.VerticalChange;

            var newCenter = new System.Windows.Point(_dragStartCanvas.X + dx, _dragStartCanvas.Y + dy);

            // Allow dragging beyond the old canvas: either clamp to container,
            // or remove clamping entirely.
            // newCenter = ClampToContainer(newCenter);   // optional
            SetThumbCenter(_activeThumb, newCenter);
            _dragStartCanvas = newCenter;

            if (_activeThumb == TL) _pTL = newCenter;
            else if (_activeThumb == TR) _pTR = newCenter;
            else if (_activeThumb == BR) _pBR = newCenter;
            else if (_activeThumb == BL) _pBL = newCenter;

            UpdateMeshFromHandles();
        }

        private (double sx, double sy) GetOverlayScale()
        {
            // Transform from the OverlayCanvas up to the Viewbox (screen space-ish).
            // If your names differ, adjust "ImageViewbox".
            var t = OverlayCanvas2.TransformToAncestor(ImageViewbox) as MatrixTransform;
            var m = t?.Matrix ?? Matrix.Identity;

            // m maps OverlayCanvas units -> Viewbox units; the diagonal is the scale.
            // Defensive: handle negative or zero (mirrors etc.).
            var sx = m.M11 != 0 ? Math.Abs(m.M11) : 1.0;
            var sy = m.M22 != 0 ? Math.Abs(m.M22) : 1.0;
            return (sx, sy);
        }

        private Point3D OverlayToWorld(System.Windows.Point p)
        {
            double cx = OverlayCanvas2.ActualWidth / 2.0;
            double cy = OverlayCanvas2.ActualHeight / 2.0;

            double x = p.X - cx;       // center X
            double y = cy - p.Y;       // flip Y once (screen down -> world up)

            return new Point3D(x, y, 0);
        }

        private void UpdateMeshFromHandles()
        {
            if (QuadMesh == null) return;

            var tl = OverlayToWorld(_pTL);
            var tr = OverlayToWorld(_pTR);
            var br = OverlayToWorld(_pBR);
            var bl = OverlayToWorld(_pBL);

            QuadMesh.Positions = new Point3DCollection
            {
                new Point3D(tl.X, tl.Y, 0),  // 0
                new Point3D(tr.X, tr.Y, 0),  // 1
                new Point3D(br.X, br.Y, 0),  // 2
                new Point3D(bl.X, bl.Y, 0),  // 3
            };
        }


        private void Corner_DragCompleted(object sender, DragCompletedEventArgs e)
        {
            _activeThumb = null;
        }

        private void ViewModel_PropertyChanged(object? sender, PropertyChangedEventArgs e)
        {
            if (e.PropertyName == nameof(LocationLabellingViewModel.XOffset) ||
                e.PropertyName == nameof(LocationLabellingViewModel.YOffset) ||
                e.PropertyName == nameof(LocationLabellingViewModel.XOffsets) ||
                e.PropertyName == nameof(LocationLabellingViewModel.YOffsets))
            {
                Application.Current.Dispatcher.Invoke(() =>
                {
                    UpdateInferredPluses();
                    UpdateOverlayCircles(false);
                    UpdateCornerRects();
                });
            }
            else if (e.PropertyName == nameof(LocationLabellingViewModel.ScaledPoints) ||
                e.PropertyName == nameof(LocationLabellingViewModel.InferredCorners))
            {
                Application.Current.Dispatcher.Invoke(() =>
                {
                    UpdateInferredPluses();
                    UpdateOverlayCircles(true);
                    UpdateCornerRects();
                    UpdateLocation();
                });
            }

            if (e.PropertyName == nameof(LocationLabellingViewModel.IsSelectingPoints))
            {
                Application.Current.Dispatcher.Invoke(() =>
                {
                    this.Cursor = viewModel.IsSelectingPoints ? Cursors.Cross : Cursors.Arrow;
                });
            }
        }

        private void UpdateLocation()
        {
            // 113 from top, 15 from each side
            if (viewModel?.RotatedPoints is null || viewModel.ClosestPoint is null)
                return;

            Point2f[] ordered = new Point2f[4];
            Point2f[] scaledPoints = viewModel.ScaledPoints.ToArray();
            for (int i = 0; i < ordered.Length; i++)
            {
                var pt = scaledPoints[i];
                var x = pt.X + viewModel.XOffset + viewModel.XOffsets[i + 1];
                var y = pt.Y + viewModel.YOffset + viewModel.YOffsets[i + 1];
                scaledPoints[i] = new Point2f(x, y);
            }
            BoundsUtilities.OrderClockwise(scaledPoints, ordered);
            var tl = ordered[0];
            var tr = ordered[1];
            var br = ordered[2];
            var bl = ordered[3];
            var point = WorldToPixel((float)(viewModel?.ClosestPoint?.X + 16), (float)(viewModel?.ClosestPoint?.Y + 76), viewModel.CanvasWidth, viewModel.CanvasHeight, tl, tr, br, bl);
            MoveElement(drawingPoint, point.X, point.Y);

        }

        public static Point2d WorldToPixel(
            float x, float y,             // world point inside the rectangle
            float W, float H,             // rectangle width/height in world units
            Point2f tlPx, Point2f trPx, Point2f brPx, Point2f blPx) // image quad
        {
            // 1) source (world) corners
            var src = new[]
            {
                new Point2f(0f, 0f),         // TL
                new Point2f((float)W, 0f),   // TR
                new Point2f((float)W, (float)H), // BR
                new Point2f(0f, (float)H)    // BL
            };

            float xr = x - 16;
            float yr = y - 76;

            // 2) destination (pixel) corners in matching order
            var dst = new[]
            {
                new Point2f((float)tlPx.X, (float)tlPx.Y),
                new Point2f((float)trPx.X, (float)trPx.Y),
                new Point2f((float)brPx.X, (float)brPx.Y),
                new Point2f((float)blPx.X, (float)blPx.Y)
            };

            // 3) homography: world -> pixel
            using var homography = Cv2.GetPerspectiveTransform(src, dst);

            // 4) apply H to [x, y, 1]^T
            var X = homography.Get<double>(0, 0) * xr + homography.Get<double>(0, 1) * yr + homography.Get<double>(0, 2);
            var Y = homography.Get<double>(1, 0) * xr + homography.Get<double>(1, 1) * yr + homography.Get<double>(1, 2);
            var Wp = homography.Get<double>(2, 0) * xr + homography.Get<double>(2, 1) * yr + homography.Get<double>(2, 2);

            return new Point2d(X / Wp, Y / Wp);
        }

        private void UpdateInferredPluses()
        {
            int needed = viewModel.InferredCorners.Length;

            while (inferredPluses.Count < needed)
            {
                var plus = CreateOverlayPlus();
                inferredPluses.Add(plus);
                OverlayCanvas.Children.Add(plus);
            }

            while (inferredPluses.Count > needed)
            {
                var last = inferredPluses[^1];
                OverlayCanvas.Children.Remove(last);
                inferredPluses.RemoveAt(inferredPluses.Count - 1);
            }

            for (int i = 0; i < needed; i++)
            {
                var pt = viewModel.InferredCorners[i];
                var x = pt.X;
                var y = pt.Y;
                inferredPluses[i].Stroke = Brushes.Red;
                MoveElement(inferredPluses[i], x, y);
            }
        }

        private void UpdateOverlayCircles(bool movePrev = false)
        {
            int needed = viewModel.RotatedPoints.Count;

            // Add new plus if needed
            while (overlayPluses.Count < needed)
            {
                var plus = CreateOverlayPlus();
                overlayPluses.Add(plus);
                OverlayCanvas.Children.Add(plus);
            }

            //while (prevPoints.Count < needed)
            //{
            //    var rect = CreateCornerRect(Brushes.Yellow);
            //    prevPoints.Add(rect);
            //    OverlayCanvas.Children.Add(rect);
            //}

            // Remove extra plusses if needed
            while (overlayPluses.Count > needed)
            {
                var last = overlayPluses[^1];
                OverlayCanvas.Children.Remove(last);
                overlayPluses.RemoveAt(overlayPluses.Count - 1);
            }

            //while (prevPoints.Count > needed)
            //{
            //    var rect = prevPoints[^1];
            //    OverlayCanvas.Children.Remove(rect);
            //    prevPoints.RemoveAt(prevPoints.Count - 1);
            //}

            // Update positions
            for (int i = 0; i < needed; i++)
            {
                //var (oldX, oldY) = GetTranslate(overlayPluses[i]);

                var pt = viewModel.ScaledPoints[i];
                var x = pt.X + viewModel.XOffset + viewModel.XOffsets[i + 1];
                var y = pt.Y + viewModel.YOffset + viewModel.YOffsets[i + 1];

                //if (movePrev) MoveCornerRect(prevPoints[i], oldX, oldY);

                overlayPluses[i].Stroke = viewModel.CurrentState ? OverlayCircleStroke : OverlayCircleStrokeInactive;
                MoveElement(overlayPluses[i], x, y);
            }
        }

        private void UpdateCornerRects()
        {
            int needed = viewModel?.CenterPoints is not null ? viewModel.CenterPoints.Length : 0;

            while (cornerRects.Count < needed)
            {
                var rect = CreateCornerRect(Brushes.Green);
                cornerRects.Add(rect);
                OverlayCanvas.Children.Add(rect);
            }

            while (cornerRects.Count > needed)
            {
                var rect = cornerRects[^1];
                OverlayCanvas.Children.Remove(rect);
                cornerRects.RemoveAt(cornerRects.Count - 1);
            }

            if (viewModel?.CenterPoints is not null)
            {
                for (int i = 0; i < cornerRects.Count; i++)
                {
                    MoveCornerRect(cornerRects[i], viewModel.CenterPoints[i].X, viewModel.CenterPoints[i].Y);
                }
            }
        }

        private Path CreateOverlayPlus()
        {
            var path = new Path
            {
                Data = OverlayPlusGeometry,
                Stroke = OverlayCircleStroke,   // reusing your existing brush
                StrokeThickness = 1,
                SnapsToDevicePixels = true
            };
            RenderOptions.SetEdgeMode(path, EdgeMode.Aliased);
            return path;
        }

        private static (double x, double y) GetTranslate(UIElement el)
        {
            if (el.RenderTransform is TranslateTransform tt)
                return (tt.X, tt.Y);

            // Fallback if something else set positions
            double left = Canvas.GetLeft(el);
            double top = Canvas.GetTop(el);
            return (double.IsNaN(left) ? 0 : left, double.IsNaN(top) ? 0 : top);
        }

        private static void MoveElement(UIElement el, double x, double y)
        {
            if (el.RenderTransform is TranslateTransform tt)
            {
                tt.X = x;
                tt.Y = y;
            }
            else
            {
                el.RenderTransform = new TranslateTransform(x, y);
            }
        }

        private Rectangle CreateCornerRect(Brush color)
        {
            const double size = 1;
            var rectangle = new Rectangle
            {
                Width = size,
                Height = size,
                Fill = CornerFill,
                Stroke = color,
                StrokeThickness = 0.5,
                SnapsToDevicePixels = true
            };
            RenderOptions.SetEdgeMode(rectangle, EdgeMode.Aliased);
            return rectangle;
        }

        private void MoveCornerRect(Rectangle rect, double x, double y)
        {
            const double size = 2;
            Canvas.SetLeft(rect, x - size / 2);
            Canvas.SetTop(rect, y - size / 2);
        }

        private void TextBox_PreviewTextInput(object sender, System.Windows.Input.TextCompositionEventArgs e)
        {
            e.Handled = !float.TryParse(((TextBox)sender).Text + e.Text, out _);
        }

        public void ExternalKeyPressPreview(object sender, KeyEventArgs e)
        {
            if (viewModel == null)
                return;

            switch (e.Key)
            {
                case Key.Space:
                    viewModel.ToggleTouchedCommand?.Execute(null);
                    e.Handled = true;
                    break;
                case Key.Left:
                    if (Keyboard.Modifiers == ModifierKeys.Shift)
                        viewModel.MoveOffsetCommand.Execute("-15");
                    else if (Keyboard.Modifiers == ModifierKeys.Control)
                        viewModel.MoveOffsetCommand.Execute("-60");
                    else
                        viewModel.MoveOffsetCommand.Execute("-1");
                    break;
                case Key.Right:
                    if (Keyboard.Modifiers == ModifierKeys.Shift)
                        viewModel.MoveOffsetCommand.Execute("15");
                    else if (Keyboard.Modifiers == ModifierKeys.Control)
                        viewModel.MoveOffsetCommand.Execute("60");
                    else
                        viewModel.MoveOffsetCommand.Execute("1");
                    break;
                case Key.A:
                    viewModel.DecrementXOffsetCommand.Execute(null);
                    break;
                case Key.D:
                    viewModel.IncrementXOffsetCommand.Execute(null);
                    break;
                case Key.W:
                    viewModel.DecrementYOffsetCommand.Execute(null);
                    break;
                case Key.S:
                    viewModel.IncrementYOffsetCommand.Execute(null);
                    break;
                case Key.Q:
                    viewModel.RotateCounterclockwiseCommand.Execute(null);
                    break;
                case Key.E:
                    viewModel.RotateClockwiseCommand.Execute(null);
                    break;
                case Key.Z:
                    viewModel.IncreaseScaleCommand.Execute(null);
                    break;
                case Key.C:
                    viewModel.DecreaseScaleCommand.Execute(null);
                    break;
                case Key.G:
                    viewModel.PullToTemplateMatchCommand.Execute(null);
                    break;
                case Key.D1:
                    viewModel.SelectCorner("1");
                    break;
                case Key.D2:
                    viewModel.SelectCorner("2");
                    break;
                case Key.D3:
                    viewModel.SelectCorner("3");
                    break;
                case Key.D4:
                    viewModel.SelectCorner("4");
                    break;
                case Key.D0:
                    viewModel.SelectCorner("0");
                    break;
            }
        }

        private void Image_MouseLeftButtonDown(object sender, MouseButtonEventArgs e)
        {
            if (viewModel is null) return;

            if (viewModel.IsSelectingPoints)
            {
                var image = (Image)sender;
                var pos = e.GetPosition(image);

                if (image.Source is BitmapSource bitmap)
                {
                    // Save in bitmap pixel coordinates
                    viewModel.SelectedPoints.Add(new System.Windows.Point(pos.X, pos.Y));
                    // Draw on overlay canvas (same scale and origin as image now)
                    DrawOverlayPlus(pos.X, pos.Y); // Optional: Keep this for immediate feedback
                }

                if (viewModel.SelectedPoints.Count == 4)
                {
                    viewModel.IsSelectingPoints = false;
                    viewModel.RunPythonTrackingFromSelectedPoints();
                }

            }
            else if (viewModel.IsAutoModeInProgress)
            {
                viewModel.ToggleAutoModePlaying();
            }
        }

        // Optionally keep for ad-hoc drawing (e.g. point selection visual feedback)
        private void DrawOverlayPlus(double x, double y)
        {
            var plus = CreateOverlayPlus();
            MoveElement(plus, x, y);
            OverlayCanvas.Children.Add(plus);
        }

    }
}