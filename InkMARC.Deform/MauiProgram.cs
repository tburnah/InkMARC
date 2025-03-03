using CommunityToolkit.Maui;
using Microsoft.Extensions.Logging;
using Microsoft.Maui.LifecycleEvents;
using InkMARCDeform.Handlers;
using InkMARCDeform.Utilities;
using InkMARCDeform.Views.AdvanceDrawingView;

namespace InkMARCDeform
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
                .UseMauiCommunityToolkit()                
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
                            window.Title = "InkMARC Deform";
                            var handle = WinRT.Interop.WindowNative.GetWindowHandle(window);
                            var id = Microsoft.UI.Win32Interop.GetWindowIdFromWindow(handle);
                            var appWindow = Microsoft.UI.Windowing.AppWindow.GetFromWindowId(id);
                            var titleBar = appWindow.TitleBar;
                            titleBar.ExtendsContentIntoTitleBar = true;
                            WinApi.ShowWindow(handle, WinApi.SW_MAXIMIZE);
                        })
                        .OnClosed((window, args) =>
                        {
                            if (window.Title == "InkMARC Deform" && InkMARCDeform.Utilities.SessionContext.CameraWin is not null)
                                Application.Current?.CloseWindow(InkMARCDeform.Utilities.SessionContext.CameraWin);
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
