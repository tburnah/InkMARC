using System;
using System.Linq;
using OpenCvSharp;


namespace InkMARC.Label.Services
{

    public static class QuadScalerCv
    {
        /// <summary>
        /// Scale a quadrilateral about its top-right corner while preserving orientation/perspective.
        /// scale = 1.0 (no change), <1 shrink toward anchor, >1 expand away from anchor.
        /// Returns points in the original input order.
        /// </summary>
        public static Point2f[] ScaleQuadAboutTopLeft(Point2f[] pts, double scale)
        {
            if (pts == null || pts.Length != 4) throw new ArgumentException("Need exactly 4 points.");

            // 1) Identify TL, TR, BR, BL
            var order = OrderQuadTLTRBRBL(pts);
            var TL = pts[order[0]];
            var TR = pts[order[1]];
            var BR = pts[order[2]];
            var BL = pts[order[3]];

            // 2) Map original quad to unit square
            var src = new[] { TL, TR, BR, BL };
            var dst = new[] { new Point2f(0, 0), new Point2f(1, 0), new Point2f(1, 1), new Point2f(0, 1) };
            using var H = Cv2.GetPerspectiveTransform(src, dst);

            // 3) Invert to go unit square -> original
            using var Hinv = H.Inv();

            // 4) Scale about anchor A = (0, 0) — top-left corner fixed
            Point2f ScaleAboutAnchor(Point2f p)
            {
                float dx = p.X - 0f;
                float dy = p.Y - 0f;
                return new Point2f((float)(0.0 + scale * dx), (float)(0.0 + scale * dy));
            }

            var uTL = new Point2f(0, 0);                   // fixed anchor
            var uTR = ScaleAboutAnchor(new Point2f(1, 0)); // becomes (s, 0)
            var uBR = ScaleAboutAnchor(new Point2f(1, 1)); // becomes (s, s)
            var uBL = ScaleAboutAnchor(new Point2f(0, 1)); // becomes (0, s)

            var unitScaled = new[] { uTL, uTR, uBR, uBL };

            // 5) Map scaled unit-square corners back to original space
            var mapped = ApplyHomography(Hinv, unitScaled);
            var nTL = mapped[0];
            var nTR = mapped[1];
            var nBR = mapped[2];
            var nBL = mapped[3];

            // 6) Return in original order
            var result = new Point2f[4];
            result[order[0]] = nTL; // top-left anchor stays fixed
            result[order[1]] = nTR;
            result[order[2]] = nBR;
            result[order[3]] = nBL;
            return result;
        }


        /// <summary>
        /// Optional overload if you already know which input index is the top-right (0..3).
        /// Pass that index and we’ll permute so your chosen TR maps to (1,0) exactly.
        /// </summary>
        public static Point2f[] ScaleQuadAboutKnownTopRight(Point2f[] pts, int topRightIndex, double scale)
        {
            if (pts == null || pts.Length != 4) throw new ArgumentException("Need exactly 4 points.");
            if (topRightIndex < 0 || topRightIndex > 3) throw new ArgumentOutOfRangeException(nameof(topRightIndex));

            // Reorder so that input[topRightIndex] ends up as TR in the TL,TR,BR,BL ordering.
            var order = OrderQuadTLTRBRBL(pts);
            // If our detected TR isn't the user's TR, rotate the TL,TR,BR,BL labels consistently
            // We want 'order[1]' to equal 'topRightIndex'
            int currentTR = order[1];
            if (currentTR != topRightIndex)
            {
                // rotate labels so the TR position points to the user's index
                // Find where the user's TR sits among TL/TR/BR/BL
                int pos = Array.IndexOf(order, topRightIndex);
                // Rotate the TL,TR,BR,BL sequence by (1 - pos)
                int shift = (1 - pos + 4) % 4;
                order = new[] {
                order[(0 - shift + 4)%4],
                order[(1 - shift + 4)%4],
                order[(2 - shift + 4)%4],
                order[(3 - shift + 4)%4],
            };
            }

            // Now proceed exactly like the main method using this order
            var TL = pts[order[0]];
            var TR = pts[order[1]];
            var BR = pts[order[2]];
            var BL = pts[order[3]];

            var src = new[] { TL, TR, BR, BL };
            var dst = new[] { new Point2f(0, 0), new Point2f(1, 0), new Point2f(1, 1), new Point2f(0, 1) };
            using var H = Cv2.GetPerspectiveTransform(src, dst);
            using var Hinv = H.Inv();

            Point2f ScaleAboutAnchor(Point2f p)
            {
                float dx = p.X - 1f;
                float dy = p.Y - 0f;
                return new Point2f((float)(1.0 + scale * dx), (float)(0.0 + scale * dy));
            }

            var unitScaled = new[]
            {
            ScaleAboutAnchor(new Point2f(0,0)),
            new Point2f(1,0),
            ScaleAboutAnchor(new Point2f(1,1)),
            ScaleAboutAnchor(new Point2f(0,1)),
        };

            var mapped = ApplyHomography(Hinv, unitScaled);

            var result = new Point2f[4];
            result[order[0]] = mapped[0];
            result[order[1]] = mapped[1];
            result[order[2]] = mapped[2];
            result[order[3]] = mapped[3];
            return result;
        }

        // --- helpers ---

        private static int[] OrderQuadTLTRBRBL(Point2f[] p)
        {
            // Heuristic: TL has smallest (x+y); BR largest (x+y);
            // TR largest (x - y); BL is remaining.
            double[] sum = p.Select(pt => (double)pt.X + pt.Y).ToArray();
            double[] diff = p.Select(pt => (double)pt.X - pt.Y).ToArray();

            int iTL = ArgMin(sum);
            int iBR = ArgMax(sum);
            int iTR = ArgMax(diff);
            int iBL = Enumerable.Range(0, 4).Except(new[] { iTL, iBR, iTR }).First();

            return new[] { iTL, iTR, iBR, iBL };
        }

        private static int ArgMin(double[] a) { int k = 0; for (int i = 1; i < a.Length; i++) if (a[i] < a[k]) k = i; return k; }
        private static int ArgMax(double[] a) { int k = 0; for (int i = 1; i < a.Length; i++) if (a[i] > a[k]) k = i; return k; }

        private static Point2f ApplyHomography(Mat H, Point2f p)
        {
            double x = p.X, y = p.Y;
            double X = H.At<double>(0, 0) * x + H.At<double>(0, 1) * y + H.At<double>(0, 2);
            double Y = H.At<double>(1, 0) * x + H.At<double>(1, 1) * y + H.At<double>(1, 2);
            double W = H.At<double>(2, 0) * x + H.At<double>(2, 1) * y + H.At<double>(2, 2);
            if (Math.Abs(W) < 1e-12) W = 1e-12;
            return new Point2f((float)(X / W), (float)(Y / W));
        }

        private static Point2f[] ApplyHomography(Mat H, Point2f[] pts)
        {
            var outPts = new Point2f[pts.Length];
            for (int i = 0; i < pts.Length; i++)
                outPts[i] = ApplyHomography(H, pts[i]);
            return outPts;
        }
    }
}
