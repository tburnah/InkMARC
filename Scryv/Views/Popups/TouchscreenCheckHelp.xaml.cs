using CommunityToolkit.Maui.Views;

namespace Scryv.Views.Popups;

public partial class TouchscreenCheckHelp : Popup
{
	public TouchscreenCheckHelp(double popupSize)
	{
		InitializeComponent();
		PopupGrid.WidthRequest = popupSize;
		PopupGrid.HeightRequest = popupSize;
	}
}