using CommunityToolkit.Maui.Views;

namespace OcuInkTrain.Views.Popups;

/// <summary>
/// Represents the StylusCheckHelp popup.
/// </summary>
public partial class StylusCheckHelp : Popup
{
    /// <summary>
    /// Initializes a new instance of the <see cref="StylusCheckHelp"/> class.
    /// </summary>
    /// <param name="popupSize">The size of the popup.</param>
    public StylusCheckHelp(double popupSize)
    {
        InitializeComponent();
        PopupGrid.WidthRequest = popupSize;
        PopupGrid.HeightRequest = popupSize;
    }
}
