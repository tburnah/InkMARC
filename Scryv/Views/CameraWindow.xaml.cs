using Scryv.ViewModel;

namespace Scryv.Views;

public partial class CameraWindow : ContentPage
{
	public CameraWindow()
	{
		InitializeComponent();
	}
    private void CamView_FinishedStarting(object sender, EventArgs e)
    {
        if (CameraWindowViewModel.Current is not null)
        {
            CameraWindowViewModel.Current.CamView_FinishedStarting(sender, e);
        }
    }

    private void CamView_FinishedStopping(object sender, EventArgs e)
    {
        if (CameraWindowViewModel.Current is not null)
        {
            CameraWindowViewModel.Current.CamView_FinishedStopping(sender, e);
        }
    }
}