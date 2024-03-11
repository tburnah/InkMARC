using Scryv.ViewModel;
using Scryv.Utilities;

namespace Scryv.Views;

public partial class CameraSelection : ContentPage
{
    private AddCameraViewModel? viewModel = null;
	public CameraSelection()
	{
		InitializeComponent();
        viewModel = BindingContext as AddCameraViewModel;
        if (viewModel is not null)
        {
            viewModel.Current = CamView;
            viewModel.Navigation = Navigation;
        }
	}

    private void Continue_Clicked(object sender, EventArgs e)
    {
        Navigation.PushAsync(new DrawingPage());        
    }

    private void CamView1_CamerasLoaded(object sender, EventArgs e)
    {
    }

    private void Back_Clicked(object sender, EventArgs e)
    {
        Navigation.PopAsync();        
    }
}