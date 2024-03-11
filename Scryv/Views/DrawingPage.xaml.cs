using Scryv.ViewModel;

namespace Scryv.Views;

public partial class DrawingPage : ContentPage
{
	private DrawingPageViewModel? viewModel = null;
	public DrawingPage()
	{
		InitializeComponent();
		viewModel = BindingContext as DrawingPageViewModel;
		viewModel.Navigation = Navigation; 
	}
}