using System;
using System.Collections.Generic;
using System.Linq;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Shapes;

namespace InkMARC.Label
{
    public partial class TimelineControl : UserControl
    {
        public static readonly DependencyProperty FrameCountProperty =
            DependencyProperty.Register(nameof(FrameCount), typeof(int), typeof(TimelineControl),
                new FrameworkPropertyMetadata(0, FrameworkPropertyMetadataOptions.AffectsRender, OnAnyStateChanged));

        public static readonly DependencyProperty CurrentFrameProperty =
            DependencyProperty.Register(nameof(CurrentFrame), typeof(int), typeof(TimelineControl),
                new FrameworkPropertyMetadata(0, FrameworkPropertyMetadataOptions.BindsTwoWayByDefault, OnCurrentFrameChanged));

        public static readonly DependencyProperty HighlightedRangesProperty =
            DependencyProperty.Register(nameof(HighlightedRanges), typeof(List<(int Start, int End)>), typeof(TimelineControl),
                new FrameworkPropertyMetadata(new List<(int, int)>(), FrameworkPropertyMetadataOptions.AffectsRender, OnHighlightChanged));

        public static readonly DependencyProperty StateChangesProperty =
            DependencyProperty.Register(nameof(StateChanges), typeof(SortedList<int, bool>), typeof(TimelineControl),
                new FrameworkPropertyMetadata(new SortedList<int, bool>(), FrameworkPropertyMetadataOptions.AffectsRender, OnStateChanged));

        public static readonly DependencyProperty DataStateChangesProperty =
            DependencyProperty.Register(nameof(DataStateChanges), typeof(SortedList<int, bool>), typeof(TimelineControl),
                new FrameworkPropertyMetadata(new SortedList<int, bool>(), FrameworkPropertyMetadataOptions.AffectsRender, OnDataStateChanged));

        public static readonly DependencyProperty IgnoredStateChangesProperty =
            DependencyProperty.Register(nameof(IgnoredStateChanges), typeof(SortedList<int, bool>), typeof(TimelineControl),
                new FrameworkPropertyMetadata(new SortedList<int, bool>(), FrameworkPropertyMetadataOptions.AffectsRender, OnIgnoredStateChanged));

        public static readonly DependencyProperty TouchPredictionsProperty =
            DependencyProperty.Register(nameof(TouchPredictions), typeof(List<float>), typeof(TimelineControl),
                new FrameworkPropertyMetadata(new List<float>(), FrameworkPropertyMetadataOptions.AffectsRender, OnPredictionChanged));

        public static readonly DependencyProperty TouchThresholdProperty =
            DependencyProperty.Register(nameof(TouchThreshold), typeof(float), typeof(TimelineControl),
                new FrameworkPropertyMetadata(0.5f, FrameworkPropertyMetadataOptions.AffectsRender, OnPredictionChanged));

        public static readonly DependencyProperty IgnoredVersionProperty =
            DependencyProperty.Register(nameof(IgnoredVersion), typeof(int), typeof(TimelineControl),
                new FrameworkPropertyMetadata(0, FrameworkPropertyMetadataOptions.AffectsRender, OnIgnoredVersionChanged));

        public static readonly DependencyProperty StateChangeNotifierProperty =
            DependencyProperty.Register(nameof(StateChangeNotifier), typeof(int), typeof(TimelineControl),
                new FrameworkPropertyMetadata(0, FrameworkPropertyMetadataOptions.AffectsRender, OnStateChangeNotifierChanged));

        public int StateChangeNotifier
        {
            get => (int)GetValue(StateChangeNotifierProperty); 
            set => SetValue(StateChangeNotifierProperty, value);
        }

        public List<float> TouchPredictions
        {
            get => (List<float>)GetValue(TouchPredictionsProperty);
            set => SetValue(TouchPredictionsProperty, value);
        }

        public float TouchThreshold
        {
            get => (float)GetValue(TouchThresholdProperty);
            set => SetValue(TouchThresholdProperty, value);
        }

        public int IgnoredVersion
        {
            get => (int)GetValue(IgnoredVersionProperty);
            set => SetValue(IgnoredVersionProperty, value);
        }

        public SortedList<int, bool> DataStateChanges
        {
            get => (SortedList<int, bool>)GetValue(DataStateChangesProperty);
            set => SetValue(DataStateChangesProperty, value);
        }

        public SortedList<int, bool> IgnoredStateChanges
        {
            get => (SortedList<int, bool>)GetValue(IgnoredStateChangesProperty);
            set => SetValue(IgnoredStateChangesProperty, value);
        }

        public SortedList<int, bool> StateChanges
        {
            get => (SortedList<int, bool>)GetValue(StateChangesProperty);
            set => SetValue(StateChangesProperty, value);
        }

        public int FrameCount
        {
            get => (int)GetValue(FrameCountProperty);
            set => SetValue(FrameCountProperty, value);
        }

        public int CurrentFrame
        {
            get => (int)GetValue(CurrentFrameProperty);
            set => SetValue(CurrentFrameProperty, value);
        }

        public List<(int Start, int End)> HighlightedRanges
        {
            get => (List<(int, int)>)GetValue(HighlightedRangesProperty);
            set => SetValue(HighlightedRangesProperty, value);
        }

        private Line? _frameMarker;
        private readonly List<UIElement> _stateRects = [];
        private readonly List<UIElement> _dataRects = [];
        private readonly List<UIElement> _ignoredRects = [];
        private readonly List<UIElement> _predictionRects = [];
        private readonly List<UIElement> _highlightRects = [];

        private bool _dirtyState = true;
        private bool _dirtyData = true;
        private bool _dirtyIgnored = true;
        private bool _dirtyPrediction = true;
        private bool _dirtyHighlight = true;
        private bool _dirtyFrame = true;

        private static readonly Brush IgnoredBrush;

        static TimelineControl()
        {
            var brush = new SolidColorBrush(Color.FromArgb(128, 255, 0, 0));
            brush.Freeze();  // makes it thread-safe and faster to render
            IgnoredBrush = brush;
        }

        public TimelineControl()
        {
            InitializeComponent();
            Loaded += (s, e) => DrawTimeline();
            SizeChanged += (s, e) => { MarkAllDirty(); DrawTimeline(); };
        }

        private static void OnAnyStateChanged(DependencyObject d, DependencyPropertyChangedEventArgs e)
        {
            if (d is TimelineControl control)
            {
                control.MarkAllDirty();
                control.DrawTimeline();
            }
        }

        private static void OnStateChanged(DependencyObject d, DependencyPropertyChangedEventArgs e)
        {
            if (d is TimelineControl control)
            {
                control._dirtyState = true;
                control.DrawTimeline();
            }
        }

        private static void OnDataStateChanged(DependencyObject d, DependencyPropertyChangedEventArgs e)
        {
            if (d is TimelineControl control)
            {
                control._dirtyData = true;
                control.DrawTimeline();
            }
        }

        private static void OnIgnoredVersionChanged(DependencyObject d, DependencyPropertyChangedEventArgs e)
        {
            if (d is TimelineControl control)
            {
                control._dirtyIgnored = true; 
                control.DrawTimeline();
            }
        }

        private static void OnStateChangeNotifierChanged(DependencyObject d, DependencyPropertyChangedEventArgs e)
        {
            if (d is TimelineControl control)
            {
                control._dirtyState = true;
                control.DrawTimeline();
            }
        }

        private static void OnIgnoredStateChanged(DependencyObject d, DependencyPropertyChangedEventArgs e)
        {
            if (d is TimelineControl control)
            {
                control._dirtyIgnored = true;
                control.DrawTimeline();
            }
        }

        private static void OnPredictionChanged(DependencyObject d, DependencyPropertyChangedEventArgs e)
        {
            if (d is TimelineControl control)
            {
                control._dirtyPrediction = true;
                control.DrawTimeline();
            }
        }

        private static void OnHighlightChanged(DependencyObject d, DependencyPropertyChangedEventArgs e)
        {
            if (d is TimelineControl control)
            {
                control._dirtyHighlight = true;
                control.DrawTimeline();
            }
        }

        private static void OnCurrentFrameChanged(DependencyObject d, DependencyPropertyChangedEventArgs e)
        {
            if (d is TimelineControl control)
            {
                control._dirtyFrame = true;
                control.DrawTimeline();
            }
        }

        private void MarkAllDirty()
        {
            _dirtyState = true;
            _dirtyData = true;
            _dirtyPrediction = true;
            _dirtyHighlight = true;
            _dirtyFrame = true;
        }

        private void Canvas_MouseDown(object sender, MouseButtonEventArgs e)
        {
            var pos = e.GetPosition(TimelineCanvas);
            int clickedFrame = (int)((pos.X / TimelineCanvas.ActualWidth) * FrameCount);
            CurrentFrame = Math.Max(0, Math.Min(FrameCount - 1, clickedFrame));
        }

        private void DrawTimeline()
        {
            if (FrameCount <= 0 || TimelineCanvas.ActualWidth <= 0) return;

            double width = TimelineCanvas.ActualWidth;
            double height = TimelineCanvas.ActualHeight;
            double rowHeight = height / 3.0;

            if (_dirtyState)
            {
                ClearVisuals(_stateRects);
                DrawStateRects(StateChanges, 0, rowHeight, Brushes.LightGreen, Brushes.DarkGray, _stateRects);
                _dirtyState = false;
            }

            if (_dirtyData)
            {
                ClearVisuals(_dataRects);
                DrawStateRects(DataStateChanges, rowHeight, rowHeight, Brushes.SkyBlue, Brushes.DimGray, _dataRects);
                _dirtyData = false;
            }

            if (_dirtyIgnored)
            {
                ClearVisuals(_ignoredRects);
                DrawStateRects(IgnoredStateChanges, 0, rowHeight * 3, IgnoredBrush, Brushes.Transparent, _ignoredRects);
                _dirtyIgnored = false;
            }

            if (_dirtyPrediction)
            {
                ClearVisuals(_predictionRects);
                DrawPredictionRects(height - rowHeight, rowHeight);
                _dirtyPrediction = false;
            }

            if (_dirtyHighlight)
            {
                ClearVisuals(_highlightRects);
                foreach (var (start, end) in HighlightedRanges)
                {
                    double startX = (start / (double)FrameCount) * width;
                    double endX = (end / (double)FrameCount) * width;

                    var rect = new Rectangle
                    {
                        Fill = Brushes.OrangeRed,
                        Width = Math.Max(endX - startX, 1),
                        Height = height,
                        Opacity = 0.5
                    };
                    Canvas.SetLeft(rect, startX);
                    Canvas.SetTop(rect, 0);
                    _highlightRects.Add(rect);
                    TimelineCanvas.Children.Add(rect);
                    Panel.SetZIndex(rect, 40);
                }
                _dirtyHighlight = false;
            }

            if (_dirtyFrame)
            {
                if (_frameMarker == null)
                {
                    _frameMarker = new Line
                    {
                        Stroke = Brushes.Black,
                        StrokeThickness = 2,
                        Y1 = 0,
                        Y2 = height
                    };
                    TimelineCanvas.Children.Add(_frameMarker);
                }

                double x = (CurrentFrame / (double)FrameCount) * width;
                _frameMarker.X1 = _frameMarker.X2 = x;
                _frameMarker.Y2 = height;
                _dirtyFrame = false;
            }
            // Always keep the marker on top
            if (_frameMarker != null) Panel.SetZIndex(_frameMarker, 1000);
        }

        private void DrawStateRects(SortedList<int, bool> states, double top, double height, Brush trueBrush, Brush falseBrush, List<UIElement> visualCache)
        {
            if (states == null || states.Count == 0) return;

            bool current = false;
            int prevFrame = 0;

            foreach (var kvp in states.OrderBy(k => k.Key))
            {
                int currFrame = kvp.Key;
                DrawStateSegment(prevFrame, currFrame, current, top, height, trueBrush, falseBrush, visualCache);
                current = kvp.Value;
                prevFrame = currFrame;
            }

            DrawStateSegment(prevFrame, FrameCount, current, top, height, trueBrush, falseBrush, visualCache);
        }

        private void DrawStateSegment(int start, int end, bool state, double top, double height, Brush trueBrush, Brush falseBrush, List<UIElement> visualCache)
        {
            if (end <= start) return;

            double startX = (start / (double)FrameCount) * TimelineCanvas.ActualWidth;
            double endX = (end / (double)FrameCount) * TimelineCanvas.ActualWidth;

            var rect = new Rectangle
            {
                Fill = state ? trueBrush : falseBrush,
                Width = Math.Max(endX - startX, 1),
                Height = height
            };

            Canvas.SetLeft(rect, startX);
            Canvas.SetTop(rect, top);

            visualCache.Add(rect);
            TimelineCanvas.Children.Add(rect);
        }

        private void DrawPredictionRects(double top, double height)
        {
            if (TouchPredictions == null || TouchPredictions.Count == 0) return;

            for (int i = 0; i < FrameCount; i++)
            {
                float val = i < TouchPredictions.Count ? TouchPredictions[i] : 0f;
                bool isTouched = val >= TouchThreshold;

                double startX = (i / (double)FrameCount) * TimelineCanvas.ActualWidth;
                double endX = ((i + 1) / (double)FrameCount) * TimelineCanvas.ActualWidth;
                double segmentWidth = Math.Max(endX - startX, 1);

                var rect = new Rectangle
                {
                    Fill = isTouched ? Brushes.Gold : Brushes.Transparent,
                    Width = segmentWidth,
                    Height = height
                };

                Canvas.SetLeft(rect, startX);
                Canvas.SetTop(rect, top);
                _predictionRects.Add(rect);
                TimelineCanvas.Children.Add(rect);
                Panel.SetZIndex(rect, 30);
            }
        }

        private void ClearVisuals(List<UIElement> visuals)
        {
            foreach (var element in visuals)
                TimelineCanvas.Children.Remove(element);
            visuals.Clear();
        }
    }
}
