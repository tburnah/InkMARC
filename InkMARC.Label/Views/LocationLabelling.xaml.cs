using OpenCvSharp;
using System.ComponentModel;
using System.Diagnostics;
using System.Text;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
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
                var pt = viewModel.ScaledPoints[i];                
                overlayCircles[i].Stroke = viewModel.CurrentState ? OverlayCircleStroke : OverlayCircleStrokeInactive;
                MoveOverlayCircle(overlayCircles[i], pt.X + viewModel.XOffset + viewModel.XOffsets[i + 1], pt.Y + viewModel.YOffset + viewModel.YOffsets[i + 1]);
            }
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
            }
        }

        private void Image_MouseLeftButtonDown(object sender, MouseButtonEventArgs e)
        {
            if (viewModel is null || !viewModel.IsSelectingPoints) return;

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