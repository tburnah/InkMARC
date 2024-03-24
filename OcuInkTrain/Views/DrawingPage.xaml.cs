using OcuInkTrain.ViewModel;

namespace OcuInkTrain.Views;

/// <summary>
/// Represents the DrawingPage class.
/// </summary>
public partial class DrawingPage : ContentPage
{
    private readonly DrawingPageViewModel? viewModel = null;

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
            viewModel.OcuInkDrawingView = MyDrawingView;
        }
    }
}
