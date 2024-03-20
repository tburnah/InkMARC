using OcuInkTrain.ViewModel;
using OcuInkTrain.Utilities;
#if WINDOWS
using Microsoft.UI.Windowing;
using Microsoft.UI;
#endif

namespace OcuInkTrain.Views;

/// <summary>
/// Represents the CameraSelection view.
/// </summary>
public partial class CameraSelection : ContentPage
{
    private AddCameraViewModel? viewModel = null;

    /// <summary>
    /// Initializes a new instance of the <see cref="CameraSelection"/> class.
    /// </summary>
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

        SessionContext.CameraWin = secondWindow;

        if (Application.Current is not null)
        {
            Application.Current.OpenWindow(secondWindow);
#if WINDOWS
            var platformWindow = secondWindow.Handler.PlatformView;
            var windowHandle = WinRT.Interop.WindowNative.GetWindowHandle(platformWindow);
            // You can now use the window handle for your specific needs
            var windowId = Win32Interop.GetWindowIdFromWindow(windowHandle);
            var appWindow = AppWindow.GetFromWindowId(windowId);
            appWindow.TitleBar.ExtendsContentIntoTitleBar = true;
            var presenter = appWindow.Presenter as OverlappedPresenter;
            if (presenter is not null)
            {
                //appWindow.SetPresenter(AppWindowPresenterKind.CompactOverlay);
                presenter.IsMaximizable = false;
                presenter.IsMinimizable = false;
                presenter.IsAlwaysOnTop = true;
                presenter.SetBorderAndTitleBar(false, false);
                int currentStyle = WinApi.GetWindowLong(windowHandle, WinApi.GWL_STYLE);
                WinApi.SetWindowLong(windowHandle, WinApi.GWL_STYLE, currentStyle & ~(int)WinApi.WS_SYSMENU & ~(int)WinApi.WS_CAPTION);
            }
#endif
            secondWindow.Width = 160;
            secondWindow.Height = 120;
        }
    }

    private void CameraWindow_Loaded(object? sender, EventArgs e)
    {
        viewModel?.CameraWindow_Loaded(sender, e);
    }
}
