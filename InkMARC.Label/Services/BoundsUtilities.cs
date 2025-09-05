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

        public static int[] GetXOffsets(SessionInfo exercise, int index)
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

        public static int[] GetYOffsets(SessionInfo exercise, int index)
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
    }
}
