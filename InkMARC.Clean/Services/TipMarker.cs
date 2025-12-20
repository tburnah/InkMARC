using OpenCvSharp;
using System;

namespace InkMARC.Clean.Services
{
    public static class TipMarker
    {
        // Sentinel colour: magenta in BGR
        public static readonly Scalar DefaultSentinel = new Scalar(255, 0, 255);

        public static bool DetectAndPaintTip(
            Mat frameBgr,
            Mat surfaceMask,
            out Point2f tipCentroid,
            Scalar? sentinelBgr = null)
        {
            tipCentroid = default;

            if (frameBgr == null) throw new ArgumentNullException(nameof(frameBgr));
            if (surfaceMask == null) throw new ArgumentNullException(nameof(surfaceMask));
            if (frameBgr.Empty() || surfaceMask.Empty()) return false;
            if (frameBgr.Type() != MatType.CV_8UC3) throw new ArgumentException("frameBgr must be CV_8UC3.");
            if (surfaceMask.Type() != MatType.CV_8UC1) throw new ArgumentException("surfaceMask must be CV_8UC1.");

            var paint = sentinelBgr ?? DefaultSentinel;

            // --- Tunables ---
            const int surfaceDilateIters = 2;   // consider candidates near the surface
            const int minArea = 20;
            const int maxArea = 600;
            const int ringRadius = 16;          // pixels around centroid to sample surface adjacency
            const int ringSamples = 48;         // more samples = more robust arcs
            const double minGreenArcRatio = 0.15; // each arc must cover >= this fraction of ring
            const int minGreenArcs = 2;         // "green on two sides"
            const double maxSkinOverlap = 0.20; // reject blobs that are mostly skin

            // Candidate darkness threshold (V < this)
            const int vDarkThresh = 115;

            // --- Precompute HSV and Skin mask once ---
            using var hsv = new Mat();
            Cv2.CvtColor(frameBgr, hsv, ColorConversionCodes.BGR2HSV);

            using var v = new Mat();
            Cv2.ExtractChannel(hsv, v, 2);

            // Candidate mask: dark pixels (loose)
            using var cand = new Mat();
            Cv2.Threshold(v, cand, vDarkThresh, 255, ThresholdTypes.BinaryInv);

            // Restrict to near-surface region
            using var surfaceNear = new Mat();
            using (var k = Cv2.GetStructuringElement(MorphShapes.Ellipse, new Size(3, 3)))
            {
                Cv2.Dilate(surfaceMask, surfaceNear, k, iterations: surfaceDilateIters);
            }
            Cv2.BitwiseAnd(cand, surfaceNear, cand);

            // Clean speckle
            using (var k2 = Cv2.GetStructuringElement(MorphShapes.Ellipse, new Size(3, 3)))
            {
                Cv2.MorphologyEx(cand, cand, MorphTypes.Open, k2, iterations: 1);
                Cv2.MorphologyEx(cand, cand, MorphTypes.Close, k2, iterations: 1);
            }

            // Skin mask (YCrCb), restricted near surface (same region as candidates)
            using var ycrcb = new Mat();
            Cv2.CvtColor(frameBgr, ycrcb, ColorConversionCodes.BGR2YCrCb);

            using var skin = new Mat();
            Cv2.InRange(ycrcb, new Scalar(0, 133, 77), new Scalar(255, 173, 127), skin);
            Cv2.BitwiseAnd(skin, surfaceNear, skin);

            // Connected components
            using var labels = new Mat();
            using var stats = new Mat();
            using var centroids = new Mat();

            int n = Cv2.ConnectedComponentsWithStats(
                cand, labels, stats, centroids,
                PixelConnectivity.Connectivity8, MatType.CV_32S);

            if (n <= 1)
                return false;

            int bestLabel = -1;
            double bestScore = double.NegativeInfinity;

            for (int i = 1; i < n; i++)
            {
                int area = stats.Get<int>(i, (int)ConnectedComponentsTypes.Area);
                if (area < minArea || area > maxArea)
                    continue;

                // Build component mask: labels == i
                using var compMask = new Mat();
                Cv2.Compare(labels, i, compMask, CmpType.EQ); // 255 where component

                // Centroid (from ConnectedComponents centroids)
                double cx = centroids.Get<double>(i, 0);
                double cy = centroids.Get<double>(i, 1);

                // Reject if centroid out of bounds (paranoia)
                if (cx < 0 || cy < 0 || cx >= frameBgr.Width || cy >= frameBgr.Height)
                    continue;

                // Skin overlap ratio = (skin ∧ comp) / comp
                using var skinAnd = new Mat();
                Cv2.BitwiseAnd(skin, compMask, skinAnd);
                int skinCount = Cv2.CountNonZero(skinAnd);
                int compCount = Math.Max(1, Cv2.CountNonZero(compMask));
                double skinRatio = (double)skinCount / compCount;
                if (skinRatio > maxSkinOverlap)
                    continue;

                // Green adjacency as arcs on a ring around centroid
                int greenArcs = CountGreenArcs(surfaceMask, cx, cy, ringRadius, ringSamples, minGreenArcRatio);
                if (greenArcs < minGreenArcs)
                    continue;

                // Shape: contour + compactness + polygon vertices (optional)
                Cv2.FindContours(compMask, out Point[][] contours, out _, RetrievalModes.External, ContourApproximationModes.ApproxSimple);
                if (contours.Length != 1)
                    continue;

                var contour = contours[0];
                double contourArea = Cv2.ContourArea(contour);
                if (contourArea <= 1) continue;

                double peri = Cv2.ArcLength(contour, true);
                if (peri <= 1) continue;

                // Compactness (lower => pointier / less round)
                double compactness = 4.0 * Math.PI * contourArea / (peri * peri);

                // Approx poly (Mat-based to avoid signature issues)
                int verts = 6; // default "unknown"
                using (var approxMat = new Mat())
                {
                    double eps = 0.02 * peri;
                    Point[] approx = Cv2.ApproxPolyDP(contour, eps, true);
                    verts = approx.Length;
                }

                double triScore =
                    (verts == 3 ? 1.0 :
                     verts == 4 ? 0.6 :
                     verts == 5 ? 0.3 : 0.0);

                double pointyScore = (compactness < 0.65) ? 0.6 : 0.0;

                // Prefer less skin overlap (hard reject already done; now bonus)
                double skinBonus = (1.0 - skinRatio);

                // Final score: green-arcs dominates; shape helps; tiny blobs penalized slightly
                double score =
                    3.0 * greenArcs +
                    1.5 * triScore +
                    1.0 * pointyScore +
                    1.5 * skinBonus -
                    0.0015 * area;

                if (score > bestScore)
                {
                    bestScore = score;
                    bestLabel = i;
                    tipCentroid = new Point2f((float)cx, (float)cy);
                }
            }

            if (bestLabel < 0)
                return false;

            // Paint the best component
            using var bestMask = new Mat();
            Cv2.Compare(labels, bestLabel, bestMask, CmpType.EQ);
            frameBgr.SetTo(paint, bestMask);

            return true;
        }

        /// <summary>
        /// Samples surfaceMask on a ring around (cx,cy) and counts how many distinct "green arcs" exist.
        /// An arc is a contiguous run of samples where surfaceMask!=0 whose length exceeds minArcRatio * samples.
        /// </summary>
        private static int CountGreenArcs(Mat surfaceMask, double cx, double cy, int radius, int samples, double minArcRatio)
        {
            int w = surfaceMask.Width;
            int h = surfaceMask.Height;

            bool[] hit = new bool[samples];
            for (int i = 0; i < samples; i++)
            {
                double a = 2.0 * Math.PI * i / samples;
                int x = (int)Math.Round(cx + radius * Math.Cos(a));
                int y = (int)Math.Round(cy + radius * Math.Sin(a));
                if ((uint)x >= (uint)w || (uint)y >= (uint)h)
                {
                    hit[i] = false;
                    continue;
                }

                hit[i] = surfaceMask.At<byte>(y, x) != 0;
            }

            // Count contiguous runs, with wrap-around handled by doubling scan
            int minRun = Math.Max(1, (int)Math.Round(minArcRatio * samples));
            int arcs = 0;

            int run = 0;
            int totalSteps = samples * 2;
            int bestWrapLimitedSteps = samples; // only count arcs within one circle

            // Mark transitions; wrap-around run counted once
            bool inRun = false;
            int runStart = -1;

            for (int k = 0; k < totalSteps; k++)
            {
                bool v = hit[k % samples];

                if (v)
                {
                    if (!inRun)
                    {
                        inRun = true;
                        run = 1;
                        runStart = k;
                    }
                    else
                    {
                        run++;
                    }
                }
                else
                {
                    if (inRun)
                    {
                        // Close run
                        int runLen = run;
                        // Only accept runs that begin within first cycle to avoid double-counting
                        if (runStart < bestWrapLimitedSteps && runLen >= minRun)
                            arcs++;

                        inRun = false;
                        run = 0;
                        runStart = -1;
                    }
                }

                // Stop after one full cycle + possible wrap completion
                if (k >= samples && !inRun) break;
            }

            // If ended in run, close it
            if (inRun)
            {
                int runLen = run;
                if (runStart < bestWrapLimitedSteps && runLen >= minRun)
                    arcs++;
            }

            // Clamp: you can’t have more than samples/2 meaningful arcs
            return Math.Min(arcs, samples / 2);
        }
    }
}
