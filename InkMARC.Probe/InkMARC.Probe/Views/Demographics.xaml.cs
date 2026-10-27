using InkMARC.Probe.ViewModel;

namespace InkMARC.Probe.Views;

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