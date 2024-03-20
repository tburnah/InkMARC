using Scryv.Utilities;

namespace Scryv.Views;

/// <summary>
/// Represents the ConsentPage.
/// </summary>
public partial class ConsentPage : ContentPage
{
    /// <summary>
    /// Initializes a new instance of the <see cref="ConsentPage"/> class.
    /// </summary>
    public ConsentPage()
    {
        InitializeComponent();
    }

    /// <summary>
    /// Event handler for the Continue button click.
    /// </summary>
    /// <param name="sender">The object that raised the event.</param>
    /// <param name="e">The event arguments.</param>
    private void Continue_Clicked(object sender, EventArgs e)
    {
        SessionContext.SessionID = SessionIDUtilities.GetUniqueSessionID();
        Navigation.PushAsync(new DrawingPage());
    }

    /// <summary>
    /// Event handler for the Exit button click.
    /// </summary>
    /// <param name="sender">The object that raised the event.</param>
    /// <param name="e">The event arguments.</param>
    private void Exit_Clicked(object sender, EventArgs e)
    {
        Application.Current?.Quit();
    }
}
