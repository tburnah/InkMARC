namespace Scryv.Views;

public partial class IneligableExit : ContentPage
{
	public IneligableExit()
	{
		InitializeComponent();
	}

    private void Back_Clicked(object sender, EventArgs e)
    {
        Navigation.PopAsync();        
    }
    private void Exit_Clicked(object sender, EventArgs e)
    {
        Application.Current.Quit();
    }
}