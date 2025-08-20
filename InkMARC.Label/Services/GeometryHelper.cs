using OpenCvSharp;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;

namespace InkMARC.Label.Services
{
    public static class GeometryHelper
    {
        public static void RotateAroundCentroidInPlace(List<Point2f> pts, float degrees)
        {
            if (pts == null || pts.Count == 0) return;
            if (MathF.Abs(degrees) < 1e-7f) return;

            // centroid
            double sumX = 0, sumY = 0;
            var span = CollectionsMarshal.AsSpan(pts);
            for (int i = 0; i < span.Length; i++) { sumX += span[i].X; sumY += span[i].Y; }
            float cx = (float)(sumX / span.Length);
            float cy = (float)(sumY / span.Length);

            // trig
            float theta = degrees * (MathF.PI / 180f);
#if NET7_0_OR_GREATER
            var (s, c) = MathF.SinCos(theta);
#else
    float s = MathF.Sin(theta);
    float c = MathF.Cos(theta);
#endif

            // translation
            float tx = cx * (1f - c) + s * cy;
            float ty = cy * (1f - c) - s * cx;

            // transform via ref to avoid copies
            for (int i = 0; i < span.Length; i++)
            {
                ref var p = ref span[i];
                float x = p.X, y = p.Y;
                p.X = x * c - y * s + tx;
                p.Y = x * s + y * c + ty;
            }
        }
    }
}
