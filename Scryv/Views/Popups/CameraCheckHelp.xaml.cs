using CommunityToolkit.Maui.Views;

namespace Scryv.Views.Popups;

public partial class CameraCheckHelp : Popup
{
	public CameraCheckHelp(double popupSize)
    {
        InitializeComponent();
        PopupGrid.WidthRequest = popupSize;
        PopupGrid.HeightRequest = popupSize;
    }
}