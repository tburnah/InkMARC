using InkMARCDeform.Interfaces;

[assembly: Dependency(typeof(WindowHandleService))]
namespace InkMARCDeform.Interfaces
{
    /// <summary>
    /// Service for retrieving the window handle of a window.
    /// </summary>
    public class WindowHandleService : IWindowHandleService
    {
        /// <summary>
        /// Retrieves the window handle of the specified window.
        /// </summary>
        /// <param name="window">The window object.</param>
        /// <returns>The window handle.</returns>
        public IntPtr GetWindowHandle(Window window)
        {
            var platformWindow = window.Handler.PlatformView as Window;
            var windowHandle = WinRT.Interop.WindowNative.GetWindowHandle(platformWindow);
            return windowHandle;
        }
    }
}
