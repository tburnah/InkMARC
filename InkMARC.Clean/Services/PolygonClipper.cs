using OpenCvSharp;
using System;
using System.Collections.Generic;
using System.Linq;

namespace InkMARC.Clean.Services
{
    public static class PolygonClipper
    {
        // Clips an arbitrary polygon against an axis-aligned rectangle.
        // Returns 0..N points (N can be > input count if edges intersect).
        public static List<Point2f> ClipToRect(IReadOnlyList<Point2f> polygon, Rect2f rect)
        {
            if (polygon == null) throw new ArgumentNullException(nameof(polygon));
            if (polygon.Count < 3) return new List<Point2f>();

            // Note: Rect2f uses X,Y as top-left. Width/Height positive.
            float left = rect.X;
            float right = rect.X + rect.Width;
            float top = rect.Y;
            float bottom = rect.Y + rect.Height;

            List<Point2f> output = polygon.ToList();

            // Clip against each boundary in turn: left, right, top, bottom
            output = ClipAgainstBoundary(output,
                inside: p => p.X >= left,
                intersect: (a, b) => IntersectWithVertical(a, b, x: left));

            output = ClipAgainstBoundary(output,
                inside: p => p.X <= right,
                intersect: (a, b) => IntersectWithVertical(a, b, x: right));

            output = ClipAgainstBoundary(output,
                inside: p => p.Y >= top,
                intersect: (a, b) => IntersectWithHorizontal(a, b, y: top));

            output = ClipAgainstBoundary(output,
                inside: p => p.Y <= bottom,
                intersect: (a, b) => IntersectWithHorizontal(a, b, y: bottom));

            // Remove near-duplicates that can occur when edges lie on the boundary
            output = RemoveNearDuplicateSequential(output, eps: 1e-3f);

            // Also remove duplicate start/end if present
            if (output.Count > 1 && NearlyEqual(output[0], output[^1], 1e-3f))
                output.RemoveAt(output.Count - 1);

            return output;
        }

        private static List<Point2f> ClipAgainstBoundary(
            List<Point2f> input,
            Func<Point2f, bool> inside,
            Func<Point2f, Point2f, Point2f> intersect)
        {
            var output = new List<Point2f>();
            if (input.Count == 0) return output;

            Point2f prev = input[^1];
            bool prevInside = inside(prev);

            for (int i = 0; i < input.Count; i++)
            {
                Point2f curr = input[i];
                bool currInside = inside(curr);

                if (currInside)
                {
                    if (!prevInside)
                    {
                        // Entering: add intersection
                        output.Add(intersect(prev, curr));
                    }
                    // Add current
                    output.Add(curr);
                }
                else
                {
                    if (prevInside)
                    {
                        // Leaving: add intersection
                        output.Add(intersect(prev, curr));
                    }
                    // Else: both outside -> add nothing
                }

                prev = curr;
                prevInside = currInside;
            }

            return output;
        }

        private static Point2f IntersectWithVertical(Point2f a, Point2f b, float x)
        {
            // Segment a->b with x = constant.
            // Parametric: p = a + t*(b-a), solve for p.X = x
            float dx = b.X - a.X;
            if (Math.Abs(dx) < 1e-8f)
            {
                // Nearly vertical segment; fallback to projecting a
                return new Point2f(x, a.Y);
            }
            float t = (x - a.X) / dx;
            float y = a.Y + t * (b.Y - a.Y);
            return new Point2f(x, y);
        }

        private static Point2f IntersectWithHorizontal(Point2f a, Point2f b, float y)
        {
            // Segment a->b with y = constant.
            float dy = b.Y - a.Y;
            if (Math.Abs(dy) < 1e-8f)
            {
                // Nearly horizontal segment; fallback to projecting a
                return new Point2f(a.X, y);
            }
            float t = (y - a.Y) / dy;
            float x = a.X + t * (b.X - a.X);
            return new Point2f(x, y);
        }

        private static List<Point2f> RemoveNearDuplicateSequential(List<Point2f> pts, float eps)
        {
            if (pts.Count <= 1) return pts;

            var outPts = new List<Point2f>(pts.Count);
            Point2f last = pts[0];
            outPts.Add(last);

            for (int i = 1; i < pts.Count; i++)
            {
                if (!NearlyEqual(last, pts[i], eps))
                {
                    last = pts[i];
                    outPts.Add(last);
                }
            }

            return outPts;
        }

        private static bool NearlyEqual(Point2f a, Point2f b, float eps)
            => Math.Abs(a.X - b.X) <= eps && Math.Abs(a.Y - b.Y) <= eps;
    }

}
