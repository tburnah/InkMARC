using Scryv.ViewModel;
using Scryv.Utilities;
using Microsoft.Extensions.Azure;
#if WINDOWS
using Microsoft.UI.Windowing;
using Microsoft.UI;
#endif
using Scryv.Interfaces;

namespace Scryv.Views;

public partial class CameraSelection : ContentPage
{
    private AddCameraViewModel? viewModel = null;
	public CameraSelection()
	{
		InitializeComponent();
        viewModel = BindingContext as AddCameraViewModel;
        if (viewModel is not null)
        {            
            viewModel.Navigation = Navigation;
        }
	}

    private void ContentPage_Loaded(object sender, EventArgs e)
    {
        var cameraWindowPage = new CameraWindow();
        cameraWindowPage.Loaded += CameraWindow_Loaded;        
        Window secondWindow = new Window(cameraWindowPage);

        secondWindow.Width = 300;
        secondWindow.Height = 200;
        
        Application.Current.OpenWindow(secondWindow);
#if WINDOWS
        var platformWindow = secondWindow.Handler.PlatformView;
        var windowHandle = WinRT.Interop.WindowNative.GetWindowHandle(platformWindow);
        // You can now use the window handle for your specific needs
        var windowId = Win32Interop.GetWindowIdFromWindow(windowHandle);
        var appWindow = AppWindow.GetFromWindowId(windowId);
        var presenter = appWindow.Presenter as OverlappedPresenter;

        appWindow.SetPresenter(AppWindowPresenterKind.CompactOverlay);
        presenter.IsMaximizable = false;
        presenter.IsMinimizable = false;        
        presenter.IsAlwaysOnTop = true;        
        int currentStyle = WinApi.GetWindowLong(windowHandle, WinApi.GWL_STYLE);
        WinApi.SetWindowLong(windowHandle, WinApi.GWL_STYLE, currentStyle & ~(int)WinApi.WS_SYSMENU);
#endif
    }

    private void CameraWindow_Loaded(object? sender, EventArgs e)
    {
        viewModel.CameraWindow_Loaded(sender, e);
    }
}