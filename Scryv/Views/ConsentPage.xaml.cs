using Scryv.Utilities;

namespace Scryv.Views;

public partial class ConsentPage : ContentPage
{
	public ConsentPage()
	{
		InitializeComponent();
	}

    private void Continue_Clicked(object sender, EventArgs e)
    {
        SessionContext.SessionID = SessionIDUtilities.GetUniqueSessionID();
        Shell.Current.GoToAsync("//CameraSelection");
    }

    private void Exit_Clicked(object sender, EventArgs e)
    {
        Application.Current.Quit();
    }

    private void Back_Clicked(object sender, EventArgs e)
    {
        Shell.Current.GoToAsync("//StartPage");
    }
}