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
        private LocationLabellingViewModel viewModel;
        private readonly List<Ellipse> overlayCircles = new();
        private readonly List<Rectangle> cornerRects = new();

        // Shared brushes and pens
        private static readonly Brush OverlayCircleFill = Brushes.Transparent;
        private static readonly Brush OverlayCircleStroke = Brushes.Blue.Clone();
        private static readonly Brush OverlayCircleStrokeInactive = Brushes.Blue.Clone();
        private static readonly Brush OverlayCircleStrokeSelected = Brushes.Yellow.Clone();
        private static readonly Brush OverlayCircleStrokeSelectedInactive = Brushes.Yellow.Clone();
        private static readonly Pen OverlayCirclePen = new Pen(OverlayCircleStroke, 1);
        private static readonly Brush CornerFill = Brushes.Transparent;
        private static readonly Pen GreenCornerPen = new Pen(Brushes.Green, 1);

        private System.Windows.Point _pTL, _pTR, _pBR, _pBL;
        private System.Windows.Point _dragStartCanvas;

        // Drag state
        private Thumb? _activeThumb;
        private bool _overlayInitDone;

        // Initial “image rect” we’ll place in the overlay (pixels)
        private double _imgW, _imgH;

        static LocationLabelling()
        {
            if (OverlayCircleStroke.CanFreeze) OverlayCircleStroke.Freeze();
            if (OverlayCircleStrokeSelected.CanFreeze) OverlayCircleStrokeSelected.Freeze();
            if (OverlayCirclePen.CanFreeze) OverlayCirclePen.Freeze();
            if (GreenCornerPen.CanFreeze) GreenCornerPen.Freeze();
            OverlayCircleStrokeInactive.Opacity = 0.2;
            if (OverlayCircleStrokeInactive.CanFreeze) OverlayCircleStrokeInactive.Freeze();
            OverlayCircleStrokeSelectedInactive.Opacity = 0.2;
            if (OverlayCircleStrokeSelectedInactive.CanFreeze) OverlayCircleStrokeSelectedInactive.Freeze();
        }

        public LocationLabelling()
        {
            InitializeComponent();
            viewModel = DataContext as LocationLabellingViewModel;

            if (viewModel != null)
            {
                viewModel.PropertyChanged += ViewModel_PropertyChanged;
            }

            Loaded += OnLoaded;
            SizeChanged += OnSizeChanged;  // keep camera in sync with overlay size
            LayoutUpdated += OnLayoutUpdated; // <-- added
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
            if (e.PropertyName == nameof(LocationLabellingViewModel.ScaledPoints) ||
                e.PropertyName == nameof(LocationLabellingViewModel.XOffset) ||
                e.PropertyName == nameof(LocationLabellingViewModel.YOffset) ||
                e.PropertyName == nameof(LocationLabellingViewModel.XOffsets) ||
                e.PropertyName == nameof(LocationLabellingViewModel.YOffsets))
            {
                Application.Current.Dispatcher.Invoke(() =>
                {
                    UpdateOverlayCircles();
                    UpdateCornerRects();
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

        private void UpdateOverlayCircles()
        {
            int needed = viewModel.RotatedPoints.Count;

            // Add new ellipses if needed
            while (overlayCircles.Count < needed)
            {
                var ellipse = CreateOverlayCircle();
                overlayCircles.Add(ellipse);
                OverlayCanvas.Children.Add(ellipse);
            }

            // Remove extra ellipses if needed
            while (overlayCircles.Count > needed)
            {
                var ellipse = overlayCircles[^1];
                OverlayCanvas.Children.Remove(ellipse);
                overlayCircles.RemoveAt(overlayCircles.Count - 1);
            }

            // Update positions
            for (int i = 0; i < needed; i++)
            {
                //if (i != 2)
                //{
                var pt = viewModel.ScaledPoints[i];
                overlayCircles[i].Stroke = viewModel.CurrentState ? OverlayCircleStroke : OverlayCircleStrokeInactive;
                MoveOverlayCircle(overlayCircles[i], pt.X + viewModel.XOffset + viewModel.XOffsets[i + 1], pt.Y + viewModel.YOffset + viewModel.YOffsets[i + 1]);
                //}
            }
            //if (viewModel.ScaledPoints.Count == 4)
            //{
            //    var tl = new OpenCvSharp.Point(this.viewModel.ScaledPoints[0].X + this.viewModel.XOffset + this.viewModel.XOffsets[1], this.viewModel.ScaledPoints[0].Y + this.viewModel.YOffset + this.viewModel.YOffsets[1]);
            //    var tr = new OpenCvSharp.Point(this.viewModel.ScaledPoints[1].X + this.viewModel.XOffset + this.viewModel.XOffsets[2], this.viewModel.ScaledPoints[1].Y + this.viewModel.YOffset + this.viewModel.YOffsets[2]);
            //    var bl = new OpenCvSharp.Point(this.viewModel.ScaledPoints[3].X + this.viewModel.XOffset + this.viewModel.XOffsets[4], this.viewModel.ScaledPoints[3].Y + this.viewModel.YOffset + this.viewModel.YOffsets[4]);
            //    var br = new OpenCvSharp.Point(tr.X - tl.X + bl.X, tr.Y-tl.Y+bl.Y);
            //    overlayCircles[2].Stroke = viewModel.CurrentState ? OverlayCircleStroke : OverlayCircleStrokeInactive;
            //    MoveOverlayCircle(overlayCircles[2], br.X, br.Y);
            //}
        }

        private void UpdateCornerRects()
        {
            int needed = viewModel.CenterPoints is not null ? 3 : 0;

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

            if (viewModel.CenterPoints is not null)
            {
                for (int i = 0; i < 3; i++)
                {
                    MoveCornerRect(cornerRects[i], viewModel.CenterPoints[i].X, viewModel.CenterPoints[i].Y);
                }
            }
        }

        private Ellipse CreateOverlayCircle()
        {
            const double radius = 4;
            var ellipse = new Ellipse
            {
                Width = radius * 2,
                Height = radius * 2,
                Fill = OverlayCircleFill,
                Stroke = OverlayCircleStroke,
                StrokeThickness = 1,
                SnapsToDevicePixels = true
            };
            RenderOptions.SetEdgeMode(ellipse, EdgeMode.Aliased);
            return ellipse;
        }

        private void MoveOverlayCircle(Ellipse ellipse, double x, double y)
        {
            const double radius = 4;
            Canvas.SetLeft(ellipse, x - radius);
            Canvas.SetTop(ellipse, y - radius);
        }

        private Rectangle CreateCornerRect(Brush color)
        {
            const double size = 2;
            var rectangle = new Rectangle
            {
                Width = size,
                Height = size,
                Fill = CornerFill,
                Stroke = color,
                StrokeThickness = 1,
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
                    DrawOverlayCircle(pos.X, pos.Y); // Optional: Keep this for immediate feedback
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
        private void DrawOverlayCircle(double x, double y)
        {
            const double radius = 4;
            var ellipse = new Ellipse
            {
                Width = radius * 2,
                Height = radius * 2,
                Fill = OverlayCircleFill,
                Stroke = OverlayCircleStroke,
                StrokeThickness = 1,
                SnapsToDevicePixels = true
            };
            RenderOptions.SetEdgeMode(ellipse, EdgeMode.Aliased);
            Canvas.SetLeft(ellipse, x - radius);
            Canvas.SetTop(ellipse, y - radius);
            OverlayCanvas.Children.Add(ellipse);
        }

    }
}