using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Scryv.Primatives
{
    public struct ScryvInkPoint
    {
        public PointF Position;
        public float Pressure;
        public float TiltX; 
        public float TiltY; 
        public ulong Timestamp;

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
