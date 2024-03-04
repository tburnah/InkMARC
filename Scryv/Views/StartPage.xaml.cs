using CommunityToolkit.Maui.Views;
using MauiIcons.Core;
using Scryv.Views.Popups;

namespace Scryv.Views;

public partial class StartPage : ContentPage
{
    private double cardWidth = 0;
    private double cardHeight = 0;
	public StartPage()
	{
		InitializeComponent();
        _ = new MauiIcon();
    }

    private void Continue_Clicked(object sender, EventArgs e)
    {
        Shell.Current.GoToAsync("//TouchscreenCheck");
    }

    private void Exit_Clicked(object sender, EventArgs e)
    {        
        Application.Current.Quit();
    }

    private void StackLayout_SizeChanged(object sender, EventArgs e)
    {
        if (sender is StackLayout stackLayout)
        {
            if (stackLayout.Width < stackLayout.Height)
            {
                stackLayout.Orientation = StackOrientation.Vertical;
                cardWidth = stackLayout.Width - 16;
                cardHeight = (stackLayout.Height - 48) / 3;
            }
            else
            {
                stackLayout.Orientation = StackOrientation.Horizontal;
                cardWidth = (stackLayout.Width - 48) / 3;
                cardHeight = stackLayout.Height - 16;
            }
            double newWidth = cardWidth < cardHeight ? cardWidth : cardHeight;
            double newHeight = cardWidth < cardHeight ? cardWidth : cardHeight;
            TabletChoice.WidthRequest = newWidth;
            TabletChoice.HeightRequest = newHeight;
            StylusChoice.WidthRequest = newWidth;                
            StylusChoice.HeightRequest = newHeight;
            CameraChoice.WidthRequest = newWidth;
            CameraChoice.HeightRequest = newHeight;
        }
    }

    private void TouchscreenHelpButtonTapped(object sender, TappedEventArgs e)
    {
        if (StartPageGrid.Width < StartPageGrid.Height)
            this.ShowPopup<TouchscreenCheckHelp>(new TouchscreenCheckHelp(StartPageGrid.Width));
        else
            this.ShowPopup<TouchscreenCheckHelp>(new TouchscreenCheckHelp(StartPageGrid.Height));
    }

    private void StylusHelpButtonTapped(object sender, TappedEventArgs e)
    {
        if (StartPageGrid.Width < StartPageGrid.Height)
            this.ShowPopup<StylusCheckHelp>(new StylusCheckHelp(StartPageGrid.Width));
        else
            this.ShowPopup<StylusCheckHelp>(new StylusCheckHelp(StartPageGrid.Height));
    }

    private void CameraHelpButtonTapped(object sender, TappedEventArgs e)
    {
        if (StartPageGrid.Width < StartPageGrid.Height)
            this.ShowPopup<CameraCheckHelp>(new CameraCheckHelp(StartPageGrid.Width));
        else
            this.ShowPopup<CameraCheckHelp>(new CameraCheckHelp(StartPageGrid.Height));
    }

    private void ContentPage_Loaded(object sender, EventArgs e)
    {
        StackLayout_SizeChanged(ChoiceStackLayout, e);
    }
}