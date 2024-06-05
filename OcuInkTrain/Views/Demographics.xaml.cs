using OcuInk.ViewModel;

namespace OcuInk.Views;

public partial class Demographics : ContentPage
{
	private DemographicsViewModel? viewModel;
	public Demographics()
	{
		InitializeComponent();
		viewModel = BindingContext as DemographicsViewModel;
		if (viewModel is not null)
			viewModel.Navigation = Navigation;
	}
}