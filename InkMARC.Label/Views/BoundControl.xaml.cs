using MaterialDesignThemes.Wpf;
using System;
using System.Collections.Generic;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Input;

namespace InkMARC.Label.Views
{
    public partial class BoundControl : UserControl
    {
        public BoundControl()
        {
            InitializeComponent();
        }

        // Define a record to hold info about the control events
        private record RoutedEventInfo(string Name, RoutedEvent Event);

        // Static dictionary of RoutedEvents
        private static readonly Dictionary<string, RoutedEventInfo> RoutedEvents = new();

        // Helper to register and expose routed events
        private static RoutedEvent RegisterRoutedEvent(string name, Type ownerType)
        {
            var routedEvent = EventManager.RegisterRoutedEvent(name, RoutingStrategy.Bubble, typeof(RoutedEventHandler), ownerType);
            RoutedEvents[name] = new RoutedEventInfo(name, routedEvent);
            return routedEvent;
        }

        // Event declarations and registrations
        public static readonly RoutedEvent UpPressedEvent = RegisterRoutedEvent(nameof(UpPressed), typeof(BoundControl));
        public static readonly RoutedEvent DownPressedEvent = RegisterRoutedEvent(nameof(DownPressed), typeof(BoundControl));
        public static readonly RoutedEvent LeftPressedEvent = RegisterRoutedEvent(nameof(LeftPressed), typeof(BoundControl));
        public static readonly RoutedEvent RightPressedEvent = RegisterRoutedEvent(nameof(RightPressed), typeof(BoundControl));
        public static readonly RoutedEvent RotateLeftPressedEvent = RegisterRoutedEvent(nameof(RotateLeftPressed), typeof(BoundControl));
        public static readonly RoutedEvent RotateRightPressedEvent = RegisterRoutedEvent(nameof(RotateRightPressed), typeof(BoundControl));
        public static readonly RoutedEvent ExpandPressedEvent = RegisterRoutedEvent(nameof(ExpandPressed), typeof(BoundControl));
        public static readonly RoutedEvent CollapsePressedEvent = RegisterRoutedEvent(nameof(CollapsePressed), typeof(BoundControl));
        public static readonly RoutedEvent ClearPressedEvent = RegisterRoutedEvent(nameof(ClearPressed), typeof(BoundControl));

        // Event accessors (C# event keyword can't be dynamic, so these must still be declared)
        public event RoutedEventHandler UpPressed { add => AddHandler(UpPressedEvent, value); remove => RemoveHandler(UpPressedEvent, value); }
        public event RoutedEventHandler DownPressed { add => AddHandler(DownPressedEvent, value); remove => RemoveHandler(DownPressedEvent, value); }
        public event RoutedEventHandler LeftPressed { add => AddHandler(LeftPressedEvent, value); remove => RemoveHandler(LeftPressedEvent, value); }
        public event RoutedEventHandler RightPressed { add => AddHandler(RightPressedEvent, value); remove => RemoveHandler(RightPressedEvent, value); }
        public event RoutedEventHandler RotateLeftPressed { add => AddHandler(RotateLeftPressedEvent, value); remove => RemoveHandler(RotateLeftPressedEvent, value); }
        public event RoutedEventHandler RotateRightPressed { add => AddHandler(RotateRightPressedEvent, value); remove => RemoveHandler(RotateRightPressedEvent, value); }
        public event RoutedEventHandler ExpandPressed { add => AddHandler(ExpandPressedEvent, value); remove => RemoveHandler(ExpandPressedEvent, value); }
        public event RoutedEventHandler CollapsePressed { add => AddHandler(CollapsePressedEvent, value); remove => RemoveHandler(CollapsePressedEvent, value); }
        public event RoutedEventHandler ClearPressed { add => AddHandler(ClearPressedEvent, value); remove => RemoveHandler(ClearPressedEvent, value); }

        public static readonly DependencyProperty ModeProperty = DependencyProperty.Register(nameof(Mode),
            typeof(BoundControlMode),
            typeof(BoundControl),
            new PropertyMetadata(BoundControlMode.Move));

        public BoundControlMode Mode
        {
            get => GetValue(ModeProperty) as BoundControlMode? ?? BoundControlMode.Move;
            set => SetValue(ModeProperty, value);
        }

        public static readonly DependencyProperty IconKindProperty = DependencyProperty.Register(nameof(IconKind), typeof(PackIconKind), typeof(BoundControl), new PropertyMetadata(PackIconKind.ClearCircle));

        public PackIconKind IconKind
        {
            get => (PackIconKind)GetValue(IconKindProperty);
            set => SetValue(IconKindProperty, value);
        }

        public static readonly DependencyProperty UpCommandProperty = DependencyProperty.Register(nameof(UpCommand), typeof(ICommand), typeof(BoundControl));
        public ICommand UpCommand
        {
            get => (ICommand)GetValue(UpCommandProperty);
            set => SetValue(UpCommandProperty, value);
        }

        public static readonly DependencyProperty DownCommandProperty = DependencyProperty.Register(nameof(DownCommand), typeof(ICommand), typeof(BoundControl));
        public ICommand DownCommand
        {
            get => (ICommand)GetValue(DownCommandProperty);
            set => SetValue(DownCommandProperty, value);
        }

        public static readonly DependencyProperty LeftCommandProperty = DependencyProperty.Register(nameof(LeftCommand), typeof(ICommand), typeof(BoundControl));
        public ICommand LeftCommand
        {
            get => (ICommand)GetValue(LeftCommandProperty);
            set => SetValue(LeftCommandProperty, value);
        }

        public static readonly DependencyProperty RightCommandProperty = DependencyProperty.Register(nameof(RightCommand), typeof(ICommand), typeof(BoundControl));
        public ICommand RightCommand
        {
            get => (ICommand)GetValue(RightCommandProperty);
            set => SetValue(RightCommandProperty, value);
        }

        public static readonly DependencyProperty RotateLeftCommandProperty = DependencyProperty.Register(nameof(RotateLeftCommand), typeof(ICommand), typeof(BoundControl));
        public ICommand RotateLeftCommand
        {
            get => (ICommand)GetValue(RotateLeftCommandProperty);
            set => SetValue(RotateLeftCommandProperty, value);
        }

        public static readonly DependencyProperty RotateRightCommandProperty = DependencyProperty.Register(nameof(RotateRightCommand), typeof(ICommand), typeof(BoundControl));
        public ICommand RotateRightCommand
        {
            get => (ICommand)GetValue(RotateRightCommandProperty);
            set => SetValue(RotateRightCommandProperty, value);
        }

        public static readonly DependencyProperty ExpandCommandProperty = DependencyProperty.Register(nameof(ExpandCommand), typeof(ICommand), typeof(BoundControl));
        public ICommand ExpandCommand
        {
            get => (ICommand)GetValue(ExpandCommandProperty);
            set => SetValue(ExpandCommandProperty, value);
        }

        public static readonly DependencyProperty CollapseCommandProperty = DependencyProperty.Register(nameof(CollapseCommand), typeof(ICommand), typeof(BoundControl));
        public ICommand CollapseCommand
        {
            get => (ICommand)GetValue(CollapseCommandProperty);
            set => SetValue(CollapseCommandProperty, value);
        }

        public static readonly DependencyProperty ClearCommandProperty = DependencyProperty.Register(nameof(ClearCommand), typeof(ICommand), typeof(BoundControl));
        public ICommand ClearCommand
        {
            get => (ICommand)GetValue(ClearCommandProperty);
            set => SetValue(ClearCommandProperty, value);
        }

        private void ModeButton_Click(object sender, RoutedEventArgs e)
        {
            if (Mode == BoundControlMode.Move)
            {
                Mode = BoundControlMode.Perspective;
                IconKind = PackIconKind.PerspectiveLess;
            }
            else
            {
                Mode = BoundControlMode.Move;
                IconKind = PackIconKind.CursorMove;
            }
        }

        private void UserControl_Loaded(object sender, RoutedEventArgs e)
        {
            Mode = BoundControlMode.Move; // Default mode
            IconKind = PackIconKind.CursorMove; // Default icon
        }

        // Shared click handler
        private void OnButtonClick(object sender, RoutedEventArgs e)
        {
            if (sender is Button button && button.Name is string name && RoutedEvents.TryGetValue(name.Replace("Button", "Pressed"), out var info))
            {
                RaiseEvent(new RoutedEventArgs(info.Event));
            }
        }
    }
}
