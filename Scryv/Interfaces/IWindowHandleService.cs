namespace Scryv.Interfaces
{
    /// <summary>
    /// Represents a service for retrieving the window handle of a specified window.
    /// </summary>
    public interface IWindowHandleService
    {
        /// <summary>
        /// Gets the window handle of the specified window.
        /// </summary>
        /// <param name="window">The window for which to retrieve the handle.</param>
        /// <returns>The window handle as an IntPtr.</returns>
        IntPtr GetWindowHandle(Window window);
    }
}
