using CommunityToolkit.Maui.Views;

namespace Scryv.Views.Popups;

/// <summary>
/// Represents the TouchscreenCheckHelp popup.
/// </summary>
public partial class TouchscreenCheckHelp : Popup
{
    /// <summary>
    /// Initializes a new instance of the <see cref="TouchscreenCheckHelp"/> class.
    /// </summary>
    /// <param name="popupSize">The size of the popup.</param>
    public TouchscreenCheckHelp(double popupSize)
    {
        InitializeComponent();
        PopupGrid.WidthRequest = popupSize;
        PopupGrid.HeightRequest = popupSize;
    }
}
