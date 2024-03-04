using CommunityToolkit.Maui.Views;

namespace Scryv.Views.Popups;

public partial class StylusCheckHelp : Popup
{
	public StylusCheckHelp(double popupSize)
    {
        InitializeComponent();
        PopupGrid.WidthRequest = popupSize;
        PopupGrid.HeightRequest = popupSize;
    }
}