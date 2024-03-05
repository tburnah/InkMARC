using Camera.MAUI;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Scryv.Utilities
{
    public static class SessionContext
    {
        private static readonly List<CameraView> cameraViews = new List<CameraView>();

        public static string SessionID { get; set; }

        public static string FilePathSessionID => SessionID.Replace(" ", "");

        public static List<CameraView> CameraViews => cameraViews;
    }
}
