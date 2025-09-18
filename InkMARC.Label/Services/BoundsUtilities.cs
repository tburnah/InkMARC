using OpenCvSharp;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace InkMARC.Label.Services
{
    internal class BoundsUtilities
    {
        public static void EnsureTLTRBRBL(Point2f[] pts)
        {
            if (pts == null || pts.Length != 4) return;
            if (pts.Any(p => float.IsNaN(p.X) || float.IsNaN(p.Y))) return;

            // Top two: smallest Y; Bottom two: largest Y
            var byY = pts.OrderBy(p => p.Y).ToArray();
            var top = byY.Take(2).OrderBy(p => p.X).ToArray();   // left->right
            var bottom = byY.Skip(2).OrderBy(p => p.X).ToArray();   // left->right

            var TL = top[0];
            var TR = top[1];
            var BL = bottom[0];
            var BR = bottom[1];

            // Write back in TL,TR,BR,BL order
            pts[0] = TL;
            pts[1] = TR;
            pts[2] = BR;
            pts[3] = BL;
        }

        public static int[] GetXOffsets(ProjectInfo exercise, int index)
        {
            var general = exercise.BoundOffsets;
            var TL = exercise.CornerOffsetTL;
            var TR = exercise.CornerOffsetTR;
            var BL = exercise.CornerOffsetBL;
            var BR = exercise.CornerOffsetBR;

            var result = new int[5];
            result[0] = general.TryGetPredecessorValue(index, out var g) ? g.x : 0;
            result[1] = TL.TryGetPredecessorValue(index, out var tl) ? tl.x : 0;
            result[2] = TR.TryGetPredecessorValue(index, out var tr) ? tr.x : 0;
            result[3] = BL.TryGetPredecessorValue(index, out var bl) ? bl.x : 0;
            result[4] = BR.TryGetPredecessorValue(index, out var br) ? br.x : 0;

            return result;
        }

        public static int[] GetYOffsets(ProjectInfo exercise, int index)
        {
            var general = exercise.BoundOffsets;
            var TL = exercise.CornerOffsetTL;
            var TR = exercise.CornerOffsetTR;
            var BL = exercise.CornerOffsetBL;
            var BR = exercise.CornerOffsetBR;

            var result = new int[5];
            result[0] = general.TryGetPredecessorValue(index, out var g) ? g.y : 0;
            result[1] = TL.TryGetPredecessorValue(index, out var tl) ? tl.y : 0;
            result[2] = TR.TryGetPredecessorValue(index, out var tr) ? tr.y : 0;
            result[3] = BL.TryGetPredecessorValue(index, out var bl) ? bl.y : 0;
            result[4] = BR.TryGetPredecessorValue(index, out var br) ? br.y : 0;
            return result;
        }

        public static void SmoothPointTriplets(Dictionary<int, Point2f[]> points, float threshold = 5.0f)
        {
            if (points.Count < 5)
                return; // Not enough data to smooth

            var sortedKeys = points.Keys.OrderBy(k => k).ToList();

            for (int i = 0; i < sortedKeys.Count; i++)
            {
                var currentKey = sortedKeys[i];
                var currentValue = points[currentKey];

                // Get previous 2 keys/values if available
                var prev1 = i > 0 ? points[sortedKeys[i - 1]] : default;
                var prev2 = i > 1 ? points[sortedKeys[i - 2]] : default;

                // Get next 2 keys/values if available
                var next1 = i < sortedKeys.Count - 1 ? points[sortedKeys[i + 1]] : default;
                var next2 = i < sortedKeys.Count - 2 ? points[sortedKeys[i + 2]] : default;

                if (prev1 is null || prev2 is null || next1 is null || next2 is null)
                    continue; // Not enough data to smooth

                // Use currentValue, prev1, prev2, next1, next2 as needed
                var prevAvg = AveragePoints(prev2, prev1);
                var nextAvg = AveragePoints(next1, next2);

                if (prevAvg.Length != 4 || nextAvg.Length != 4)
                    continue; // Skip if averages are not valid

                Point2f newA = SmoothIfNeeded(points[i][0], prevAvg[0], nextAvg[0], threshold);
                Point2f newB = SmoothIfNeeded(points[i][1], prevAvg[1], nextAvg[1], threshold);
                Point2f newC = SmoothIfNeeded(points[i][2], prevAvg[2], nextAvg[2], threshold);
                Point2f newD = SmoothIfNeeded(points[i][3], prevAvg[3], nextAvg[3], threshold);

                points[i] = [newA, newB, newC];
            }
        }

        private static Point2f[] AveragePoints(Point2f[] p1, Point2f[] p2)
        {
            if (p1.Length != 4 || p2.Length != 4)
                return [];
            return [
                Average(p1[0], p2[0]),
                Average(p1[1], p2[1]),
                Average(p1[2], p2[2]),
                Average(p1[3], p2[3])
            ];
        }

        private static Point2f Average(Point2f p1, Point2f p2)
        {
            return new Point2f(
                (p1.X + p2.X) / 2.0f,
                (p1.Y + p2.Y) / 2.0f
            );
        }

        private static Point2f SmoothIfNeeded(Point2f current, Point2f prevAvg, Point2f nextAvg, float threshold)
        {
            var avgX = (prevAvg.X + nextAvg.X) / 2.0f;
            var avgY = (prevAvg.Y + nextAvg.Y) / 2.0f;
            var distance = MathF.Sqrt((current.X - avgX) * (current.X - avgX) + (current.Y - avgY) * (current.Y - avgY));

            if (distance > threshold)
                return new Point2f(avgX, avgY);

            return current;
        }

        public static void OrderClockwise(Point2f[] points, Point2f[] result)
        {
            if (points.Length != 4)
                throw new ArgumentException("Exactly 4 points required.");
            if (result.Length != 4)
                throw new ArgumentException("Result array must have length 4.");

            var sortedy = points.OrderBy(p => p.Y).ToList();
            var sortedx = points.OrderBy(p => p.X).ToList();

            Point2f topLeft = new();
            Point2f topRight = new();
            Point2f bottomLeft = new();
            Point2f bottomRight = new();                
            if (sortedy[0] == sortedx[0] || sortedy[0] == sortedx[1])
            {
                topLeft = sortedy[0];
                if (sortedy[1] != sortedx[2] && sortedy[1] != sortedx[3])
                {
                    throw new ArgumentException("Both of the top ys are in the top xs!");
                }
                topRight = sortedy[1];
                sortedy.RemoveAt(0);
                sortedy.RemoveAt(0);
            }
            else if (sortedy[1] == sortedx[0] || sortedy[1] == sortedx[1])
            {
                topLeft = sortedy[1];
                if (sortedy[0] != sortedx[2] && sortedy[0] != sortedx[3])
                {
                    throw new ArgumentException("Both of the top ys are in the top xs!");
                }
                topRight = sortedy[0];
                sortedy.RemoveAt(0);
                sortedy.RemoveAt(0);
            }            

            if (sortedy[0].X < sortedy[1].X)
            {
                bottomLeft = sortedy[0];
                bottomRight = sortedy[1];
            }
            else
            {
                bottomLeft = sortedy[1];
                bottomRight = sortedy[0];
            }

            result[0] = topLeft;
            result[1] = topRight;
            result[2] = bottomRight;
            result[3] = bottomLeft;            
        }
    }
}
