using CommunityToolkit.Maui.Views;

namespace OcuInkTrain.Views.Popups;

/// <summary>
/// Represents the CameraCheckHelp popup.
/// </summary>
public partial class CameraCheckHelp : Popup
{
    /// <summary>
    /// Initializes a new instance of the <see cref="CameraCheckHelp"/> class.
    /// </summary>
    /// <param name="popupSize">The size of the popup.</param>
    public CameraCheckHelp(double popupSize)
    {
        InitializeComponent();
        PopupGrid.WidthRequest = popupSize;
        PopupGrid.HeightRequest = popupSize;
    }
}
