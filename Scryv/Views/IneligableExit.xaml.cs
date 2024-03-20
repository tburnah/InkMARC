namespace Scryv.Views;

/// <summary>
/// Represents the IneligableExit page.
/// </summary>
public partial class IneligableExit : ContentPage
{
    /// <summary>
    /// Initializes a new instance of the <see cref="IneligableExit"/> class.
    /// </summary>
    public IneligableExit()
    {
        InitializeComponent();
    }

    /// <summary>
    /// Handles the Back button click event.
    /// </summary>
    /// <param name="sender">The object that raised the event.</param>
    /// <param name="e">The event arguments.</param>
    private void Back_Clicked(object sender, EventArgs e)
    {
        Navigation.PopAsync();
    }

    /// <summary>
    /// Handles the Exit button click event.
    /// </summary>
    /// <param name="sender">The object that raised the event.</param>
    /// <param name="e">The event arguments.</param>
    private void Exit_Clicked(object sender, EventArgs e)
    {
        Application.Current?.Quit();
    }
}
