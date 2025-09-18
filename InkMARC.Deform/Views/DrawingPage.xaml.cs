using InkMARCDeform.ViewModel;

namespace InkMARCDeform.Views;

/// <summary>
/// Represents the DrawingPage class.
/// </summary>
public partial class DrawingPage : ContentPage
{
    private readonly DrawingPageViewModel? viewModel = null;
    public Window? CurrentWindow => this.Window;
    // Read-only properties for binding
    public double WindowWidth => Window?.Width ?? 0;
    public double WindowHeight => Window?.Height ?? 0;

    public double ImageWidth => MyFrame?.Width ?? 0;

    public double ImageHeight => MyFrame?.Height ?? 0;

    /// <summary>
    /// Initializes a new instance of the DrawingPage class.
    /// </summary>
    public DrawingPage()
    {
        InitializeComponent();
        if (BindingContext is DrawingPageViewModel drawingViewModel)
        {
            viewModel = drawingViewModel;
            viewModel.Navigation = Navigation;
            viewModel.InkMARCDrawingView = MyDrawingView;
        }
    }

    protected override void OnAppearing()
    {
        base.OnAppearing();

        // Now the page is attached to a Window
        OnPropertyChanged(nameof(WindowWidth));
        OnPropertyChanged(nameof(WindowHeight));

        OnPropertyChanged(nameof(ImageWidth));
        OnPropertyChanged(nameof(ImageHeight));

        if (Window is not null)
        {
            Window.SizeChanged -= OnWindowSizeChanged; // avoid double-subscribe
            Window.SizeChanged += OnWindowSizeChanged;
        }
    }

    void OnWindowSizeChanged(object? sender, EventArgs e)
    {
        OnPropertyChanged(nameof(WindowWidth));
        OnPropertyChanged(nameof(WindowHeight));
        OnPropertyChanged(nameof(ImageWidth));
        OnPropertyChanged(nameof(ImageHeight));
    }
}
