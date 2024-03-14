using System;
using System.Runtime.InteropServices;

namespace Scryv.Utilities
{
    public static class WinApi
    {
        public const int GWL_STYLE = -16;
        public const uint WS_SYSMENU = 0x80000;        

        [DllImport("user32.dll", SetLastError = true)]
        public static extern int GetWindowLong(IntPtr hWnd, int nIndex);

        [DllImport("user32.dll")]
        public static extern int SetWindowLong(IntPtr hWnd, int nIndex, int dwNewLong);

        public const int SC_CLOSE = 0xF060;
        public const int MF_BYCOMMAND = 0x00000000;
        public const int MF_GRAYED = 0x1;
        public const int MF_DISABLED = 0x2;

        [DllImport("user32.dll")]
        public static extern IntPtr GetSystemMenu(IntPtr hWnd, bool bRevert);

        [DllImport("user32.dll")]
        public static extern bool EnableMenuItem(IntPtr hMenu, uint uIDEnableItem, uint uEnable);

    }
}
