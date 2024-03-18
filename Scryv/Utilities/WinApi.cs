using System.Runtime.InteropServices;

namespace Scryv.Utilities
{
    /// <summary>
    /// Provides access to Windows API functions and constants.
    /// </summary>
    public static class WinApi
    {
        /// <summary>
        /// Retrieves the window styles of a specified window.
        /// </summary>
        public const int GWL_STYLE = -16;

        /// <summary>
        /// Retrieves the extended window styles of a specified window.
        /// </summary>
        public const int GWL_EXSTYLE = -20;

        /// <summary>
        /// Specifies the system menu style for a window.
        /// </summary>
        public const uint WS_SYSMENU = 0x80000;

        /// <summary>
        /// Specifies the border style for a window.
        /// </summary>
        public const uint WS_BORDER = 0x00800000;

        /// <summary>
        /// Specifies the caption style for a window.
        /// </summary>
        public const uint WS_CAPTION = 0x00C00000;

        /// <summary>
        /// Retrieves the specified 32-bit value from the extra window memory.
        /// </summary>
        /// <param name="hWnd">A handle to the window.</param>
        /// <param name="nIndex">The zero-based offset to the value to be retrieved.</param>
        /// <returns>The requested 32-bit value.</returns>
        [DllImport("user32.dll", SetLastError = true)]
        public static extern int GetWindowLong(IntPtr hWnd, int nIndex);

        /// <summary>
        /// Changes the specified 32-bit value in the extra window memory.
        /// </summary>
        /// <param name="hWnd">A handle to the window.</param>
        /// <param name="nIndex">The zero-based offset to the value to be changed.</param>
        /// <param name="dwNewLong">The replacement value.</param>
        /// <returns>The previous 32-bit value.</returns>
        [DllImport("user32.dll")]
        public static extern int SetWindowLong(IntPtr hWnd, int nIndex, int dwNewLong);

        /// <summary>
        /// Closes the window.
        /// </summary>
        public const int SC_CLOSE = 0xF060;

        /// <summary>
        /// Specifies that the uIDEnableItem parameter gives the identifier of the menu item.
        /// </summary>
        public const int MF_BYCOMMAND = 0x00000000;

        /// <summary>
        /// Specifies that the menu item is grayed and cannot be selected.
        /// </summary>
        public const int MF_GRAYED = 0x1;

        /// <summary>
        /// Specifies that the menu item is disabled and cannot be selected.
        /// </summary>
        public const int MF_DISABLED = 0x2;

        /// <summary>
        /// Specifies the pop-up style for a window.
        /// </summary>
        public const int WS_POPUP = -2147483648;

        /// <summary>
        /// Specifies the child style for a window.
        /// </summary>
        public const int WS_CHILD = 1073741824;

        /// <summary>
        /// Specifies the visible style for a window.
        /// </summary>
        public const int WS_VISIBLE = 0x10000000;

        /// <summary>
        /// Specifies that the sibling windows are not clipped when they overlap the client area of this window.
        /// </summary>
        public const int WS_CLIPSIBLINGS = 0x04000000;

        /// <summary>
        /// Specifies the dialog frame style for a window.
        /// </summary>
        public const int WS_DLGFRAME = 0x00400000;

        /// <summary>
        /// Retrieves a handle to the system menu of the specified window.
        /// </summary>
        /// <param name="hWnd">A handle to the window.</param>
        /// <param name="bRevert">The action to be taken.</param>
        /// <returns>A handle to the system menu.</returns>
        [DllImport("user32.dll")]
        public static extern IntPtr GetSystemMenu(IntPtr hWnd, bool bRevert);

        /// <summary>
        /// Enables, disables, or grays the specified menu item.
        /// </summary>
        /// <param name="hMenu">A handle to the menu.</param>
        /// <param name="uIDEnableItem">The identifier of the menu item.</param>
        /// <param name="uEnable">The menu item state.</param>
        /// <returns>true if successful; otherwise, false.</returns>
        [DllImport("user32.dll")]
        public static extern bool EnableMenuItem(IntPtr hMenu, uint uIDEnableItem, uint uEnable);
    }
}
