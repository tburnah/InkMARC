using Scryv.ViewModel;

namespace Scryv.Views;

/// <summary>
/// Represents the DrawingPage class.
/// </summary>
public partial class DrawingPage : ContentPage
{
    private DrawingPageViewModel? viewModel = null;

    /// <summary>
    /// Initializes a new instance of the DrawingPage class.
    /// </summary>
    public DrawingPage()
    {
        InitializeComponent();
        viewModel = BindingContext as DrawingPageViewModel;
        viewModel.Navigation = Navigation;
    }
}
