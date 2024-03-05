using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Scryv.Primatives
{
    public struct ScryvInkPoint
    {
        public PointF Position { get; set; }
        public float Pressure { get; set; }
        public float TiltX { get; set; }
        public float TiltY { get; set; }
        public ulong Timestamp { get; set; }

        public ScryvInkPoint()
        {
        }

        public ScryvInkPoint(PointF position, float pressure, float tiltX, float tiltY, ulong timestamp)
        {
            Position = position;
            Pressure = pressure;
            TiltX = tiltX;
            TiltY = tiltY;
            Timestamp = timestamp;
        }
    }
}
