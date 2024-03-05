namespace Scryv.Views;

public partial class UploadPage : ContentPage
{
	public UploadPage()
	{
		InitializeComponent();
	}

    private void Exit_Clicked(object sender, EventArgs e)
    {
        Application.Current.Quit();
    }
}