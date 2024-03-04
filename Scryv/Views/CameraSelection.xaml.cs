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
	}

    private void Continue_Clicked(object sender, EventArgs e)
    {
        Shell.Current.GoToAsync("//DrawingPage"); 
    }

    private void CamView1_CamerasLoaded(object sender, EventArgs e)
    {
        if (sender is Camera.MAUI.CameraView cameraView)
        {
            var parent = VisualTreeHelper.FindParentOfType<Microsoft.Maui.Controls.ViewCell>(cameraView);
            if (parent is not null && parent.BindingContext is CameraSelectionViewModel cameraSelectionViewModel)
                cameraSelectionViewModel.Current = cameraView;
        }
        if (viewModel != null)
            viewModel.CameraLoaded();
    }

    private void Back_Clicked(object sender, EventArgs e)
    {
        Shell.Current.GoToAsync("//ConsentPage");
    }
}