using CommunityToolkit.Maui.Views;
using OcuInkTrain.ViewModel;
using OcuInkTrain.Views.Popups;
using System.Timers;

namespace OcuInkTrain.Views;

/// <summary>
/// Represents the start page of the application.
/// </summary>
public partial class StartPage : ContentPage
{
    private StartPageViewModel? spvm;

    private double cardWidth = 0;
    private double cardHeight = 0;

    /// <summary>
    /// Initializes a new instance of the <see cref="StartPage"/> class.
    /// </summary>
    public StartPage()
    {
        InitializeComponent();        
        spvm = BindingContext as StartPageViewModel;
        if (spvm is not null)
            spvm.Navigation = Navigation;
    }

    /// <summary>
    /// Handles the event when the exit button is clicked.
    /// </summary>
    /// <param name="sender">The sender of the event.</param>
    /// <param name="e">The event arguments.</param>
    private void Exit_Clicked(object sender, EventArgs e)
    {
        Application.Current?.Quit();
    }

    /// <summary>
    /// Handles the event when the size of the stack layout changes.
    /// </summary>
    /// <param name="sender">The sender of the event.</param>
    /// <param name="e">The event arguments.</param>
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

            if (newWidth < 10 || newHeight < 10)
                return;

            StylusChoice.WidthRequest = newWidth;
            StylusChoice.HeightRequest = newHeight;
            CameraChoice.WidthRequest = newWidth;
            CameraChoice.HeightRequest = newHeight;

            TabletChoice.WidthRequest = newWidth;
            TabletChoice.HeightRequest = newHeight;
            StylusChoice.WidthRequest = newWidth;
            StylusChoice.HeightRequest = newHeight;
            CameraChoice.WidthRequest = newWidth;
            CameraChoice.HeightRequest = newHeight;

            InvalidateMeasure();
        }
    }

    /// <summary>
    /// Handles the event when the touchscreen help button is tapped.
    /// </summary>
    /// <param name="sender">The sender of the event.</param>
    /// <param name="e">The event arguments.</param>
    private void TouchscreenHelpButtonTapped(object sender, TappedEventArgs e)
    {
        if (StartPageGrid.Width < StartPageGrid.Height)
            this.ShowPopup<TouchscreenCheckHelp>(new TouchscreenCheckHelp(StartPageGrid.Width));
        else
            this.ShowPopup<TouchscreenCheckHelp>(new TouchscreenCheckHelp(StartPageGrid.Height));
    }

    /// <summary>
    /// Handles the event when the stylus help button is tapped.
    /// </summary>
    /// <param name="sender">The sender of the event.</param>
    /// <param name="e">The event arguments.</param>
    private void StylusHelpButtonTapped(object sender, TappedEventArgs e)
    {
        if (StartPageGrid.Width < StartPageGrid.Height)
            this.ShowPopup<StylusCheckHelp>(new StylusCheckHelp(StartPageGrid.Width));
        else
            this.ShowPopup<StylusCheckHelp>(new StylusCheckHelp(StartPageGrid.Height));
    }

    /// <summary>
    /// Handles the event when the camera help button is tapped.
    /// </summary>
    /// <param name="sender">The sender of the event.</param>
    /// <param name="e">The event arguments.</param>
    private void CameraHelpButtonTapped(object sender, TappedEventArgs e)
    {
        if (StartPageGrid.Width < StartPageGrid.Height)
            this.ShowPopup<CameraCheckHelp>(new CameraCheckHelp(StartPageGrid.Width));
        else
            this.ShowPopup<CameraCheckHelp>(new CameraCheckHelp(StartPageGrid.Height));
    }

    /// <summary>
    /// Handles the event when the content page is loaded.
    /// </summary>
    /// <param name="sender">The sender of the event.</param>
    /// <param name="e">The event arguments.</param>
    private void ContentPage_Loaded(object sender, EventArgs e)
    {
        StackLayout_SizeChanged(ChoiceStackLayout, e);
    }

    /// <summary>
    /// Handles the event when the back button is clicked.
    /// </summary>
    /// <param name="sender">The sender of the event.</param>
    /// <param name="e">The event arguments.</param>
    private void Back_Clicked(object sender, EventArgs e)
    {
        Navigation.PopAsync();
    }
}
