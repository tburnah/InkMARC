using Scryv.Interfaces;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

[assembly: Dependency(typeof(WindowHandleService))]
namespace Scryv.Interfaces
{
    public class WindowHandleService : IWindowHandleService
    {
        public IntPtr GetWindowHandle(Window window)
        {            
            var platformWindow = window.Handler.PlatformView as Window;
            var windowHandle = WinRT.Interop.WindowNative.GetWindowHandle(platformWindow);
            return windowHandle;
        }
    }
}
