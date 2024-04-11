using OcuInk.Models.Primatives;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace OcuInk.Extensions
{
    public static class InkColorExtensions
    {
        public static Color ToMauiColor(this InkColor inkColor)
        {
            return Color.FromRgba(inkColor.Red, inkColor.Green, inkColor.Blue, inkColor.Alpha);
        }
        
        public static InkColor ToInkColor(this Color color)
        {
            return new InkColor(color.Red, color.Green, color.Blue, color.Alpha);
        }
    }
}
