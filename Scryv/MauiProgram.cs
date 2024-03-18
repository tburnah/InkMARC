using Microsoft.Extensions.Logging;
using Camera.MAUI;
using CommunityToolkit.Maui;
using Microsoft.Maui.LifecycleEvents;
using MauiIcons.Material;
using Scryv.Views.AdvanceDrawingView;
using Scryv.Handlers;
using Microsoft.Maui.Controls.Hosting;

namespace Scryv
{
    /// <summary>
    /// The main entry point for the Maui application.
    /// </summary>
    public static class MauiProgram
    {
        /// <summary>
        /// Creates and configures the Maui application.
        /// </summary>
        /// <returns>The configured Maui application.</returns>
        public static MauiApp CreateMauiApp()
        {
            var builder = MauiApp.CreateBuilder();
            builder
                .UseMauiApp<App>()
                .UseMauiCameraView()
                .UseMauiCommunityToolkit()
                .UseMaterialMauiIcons()
                .ConfigureFonts(fonts =>
                {
                    fonts.AddFont("OpenSans-Regular.ttf", "OpenSansRegular");
                    fonts.AddFont("OpenSans-Semibold.ttf", "OpenSansSemibold");
                })
                .ConfigureLifecycleEvents(events =>
                {
#if WINDOWS
                    events.AddWindows(windowsLifecycleBuilder =>
                    {
                        windowsLifecycleBuilder.OnWindowCreated(window =>
                        {
                            window.Title = "Scryv";
                            var handle = WinRT.Interop.WindowNative.GetWindowHandle(window);
                            var id = Microsoft.UI.Win32Interop.GetWindowIdFromWindow(handle);
                            var appWindow = Microsoft.UI.Windowing.AppWindow.GetFromWindowId(id);
                            var titleBar = appWindow.TitleBar;
                            titleBar.ExtendsContentIntoTitleBar = true;
                            titleBar.BackgroundColor = Microsoft.UI.ColorHelper.FromArgb(0xFF, 0x0A, 0x22, 0x40);
                            titleBar.ButtonBackgroundColor = Microsoft.UI.ColorHelper.FromArgb(0xFF, 0x0A, 0x22, 0x40);
                            titleBar.InactiveBackgroundColor = Microsoft.UI.ColorHelper.FromArgb(0xFF, 0x0A, 0x22, 0x40);
                            titleBar.ButtonInactiveBackgroundColor = Microsoft.UI.ColorHelper.FromArgb(0xFF, 0x0A, 0x22, 0x40);
                        });
                    });
#endif
                });

#if DEBUG
            builder.Logging.AddDebug();
#endif
            builder.ConfigureMauiHandlers(handlers =>
            {
                handlers.AddHandler<AdvancedDrawingView, AdvancedDrawingViewHandler>();
            });
            return builder.Build();
        }
    }
}
