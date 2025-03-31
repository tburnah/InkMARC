namespace InkMARC.Cue.Resources;

public partial class InfoConsentPage : ContentPage
{
	public InfoConsentPage()
	{        
        InitializeComponent();
        NavigationPage.SetHasNavigationBar(this, false);
    }

    private async void OnAgreeClicked(object sender, EventArgs e)
    {
        await Navigation.PushAsync(new InstructionPage());
    }
}