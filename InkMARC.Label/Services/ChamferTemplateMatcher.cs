using InkMARC.Label.Services.Interfaces;
using OpenCvSharp;

namespace InkMARC.Label.Services
{
    public class ChamferTemplateMatcher
    {
        private static readonly Mat?[] templates = new Mat?[4];

        static ChamferTemplateMatcher()
        {
            // Initialize templates to empty Mat objects
            for (int i = 0; i < templates.Length; i++)
            {
                templates[i] = new Mat();
            }
        }

        private static bool TryCaptureWindow(Mat image, int size, Point2f center, out Mat roi, out Point2f roiTopLeft)
        {
            roi = null!;
            roiTopLeft = default;

            int half = size / 2;
            int cx = (int)center.X;
            int cy = (int)center.Y;

            int x = Math.Max(0, cx - half);
            int y = Math.Max(0, cy - half);
            int w = Math.Min(image.Width - x, size);
            int h = Math.Min(image.Height - y, size);

            if (w <= 0 || h <= 0) return false;

            roiTopLeft = new Point2f(x, y);
            roi = new Mat(image, new OpenCvSharp.Rect(x, y, w, h)).Clone();
            return true;
        }

        private static void ExtractCornerAtCorrectedPoint(int cornerIndex, Mat image, Point2f[] centerPoints, Point2f finalCenter)
        {
            // 12×12 search window around the fully corrected center
            if (!TryCaptureWindow(image, 12, finalCenter, out Mat currentPos, out Point2f roiTopLeft))
                return;

            if (templates[cornerIndex] is null || (templates[cornerIndex]?.Empty() ?? false))
                return;

            var result = ChamferTemplateMatcher.MatchWithChamfer(currentPos, templates?[cornerIndex] ?? new Mat());
            if (result is null) { currentPos.Dispose(); return; }

            // The four matched points are in ROI-local coords; translate by actual ROI top-left:
            var c1 = new Point2f(roiTopLeft.X + result.Item1.X, roiTopLeft.Y + result.Item1.Y);
            var c2 = new Point2f(roiTopLeft.X + result.Item2.X, roiTopLeft.Y + result.Item2.Y);
            var c3 = new Point2f(roiTopLeft.X + result.Item3.X, roiTopLeft.Y + result.Item3.Y);
            var c4 = new Point2f(roiTopLeft.X + result.Item4.X, roiTopLeft.Y + result.Item4.Y);

            // Average to center; update silently (no UI invalidation here)
            centerPoints[cornerIndex] = new Point2f(
                (c1.X + c2.X + c3.X + c4.X) / 4f,
                (c1.Y + c2.Y + c3.Y + c4.Y) / 4f
            );

            currentPos.Dispose();
        }

        private static Point2f[] GetCorrectedBoundsForFrame(Dictionary<int, Point2f[]> frameData,
                                                     SessionInfo exercise,
                                                     int i)
        {
            // Default: NaNs if we have no points for this frame
            var nan = new Point2f(float.NaN, float.NaN);
            if (!frameData.TryGetValue(i, out var raw) || raw is null || raw.Length < 4)
                return [nan, nan, nan, nan];

            // Start from raw points
            var pts = raw.ToList();

            // 1) rotate about centroid (if any)
            float deg = exercise.BoundRotations.TryGetPredecessorValue(i, out var rot) ? rot : 0f;
            if (Math.Abs(deg) > float.Epsilon)
                GeometryHelper.RotateAroundCentroidInPlace(pts, deg);

            // 2) scale about TL (if any)
            var bounds = pts.ToArray(); // TL,TR,BR,BL order expected elsewhere
            float scl = exercise.BoundScales.TryGetPredecessorValue(i, out var s) ? s : 1f;
            if (Math.Abs(scl - 1f) > 1e-6f)
                bounds = QuadScalerCv.ScaleQuadAboutTopLeft(bounds, scl);

            // 3) add general + per-corner offsets
            var xOffs = BoundsUtilities.GetXOffsets(exercise, i); // [general, TL, TR, BL, BR]
            var yOffs = BoundsUtilities.GetYOffsets(exercise, i);
            for (int j = 0; j < 4; j++)
            {
                bounds[j].X += xOffs[0] + xOffs[j + 1];
                bounds[j].Y += yOffs[0] + yOffs[j + 1];
            }

            BoundsUtilities.EnsureTLTRBRBL(bounds);
            return bounds;
        }

        public static async Task RunTemplateMatchingOnAllFramesAsync(IVideoService videoService,
                                                                     SessionInfo exercise,
                                                                     Dictionary<int, Point2f[]> frameData,                                                                 
                                                                     IProgress<int>? progress = null)
        {
            if (!videoService.IsOpen || exercise is null) return;

            Point2f[]? centerPoints = GetCenterPoints(exercise);

            var startFrame = exercise.StartFrame;
            var stopFrame = exercise.StopFrame;
            var rotation = exercise.Rotation;
            var maxProgress = stopFrame - startFrame + 1;

            // ~1% progress updates to avoid UI thrash
            int reportEvery = Math.Max(1, maxProgress / 100);            

            await Task.Run(() =>
            {
                // Init templates using the corrected centers at StartFrame
                using var first = videoService.GetFrameAt(startFrame);
                using var processedFirst = FrameProcessor.ProcessToMat(first, rotation);
                if (processedFirst is null) return;

                var corr0 = GetCorrectedBoundsForFrame(frameData, exercise, startFrame);

                for (int k = 0; k < 4; ++k)
                {
#pragma warning disable CS8602 // Dereference of a possibly null reference.
                    bool ok = templates[k] != null && !templates[k].Empty();
#pragma warning restore CS8602 // Dereference of a possibly null reference.
                    if (!ok && !float.IsNaN(corr0[k].X))
                    {
                        Mat? tpl = null!;
                        // Template window smaller than search window is fine
                        ChamferTemplateMatcher.CapturePointTemplates(processedFirst, 6, corr0[k], ref tpl);
                        ok = tpl is not null && !tpl.Empty();
                        if (ok) templates[k] = tpl;
                    }
                    if (!ok) return; // bail if we can’t initialize templates
                }

                // Main loop (headless; progress only)
                for (int i = startFrame; i <= stopFrame; ++i)
                {
                    using var frame = videoService.GetFrameAt(i);
                    using var processed = FrameProcessor.ProcessToMat(frame, rotation);
                    if (processed is null || processed.Empty()) continue;

                    var corrected = GetCorrectedBoundsForFrame(frameData, exercise, i);

                    // Match only the three measured corners as before
                    for (int j = 0; j < 4; j++)
                    {
                        if (!float.IsNaN(corrected[j].X))
                            ExtractCornerAtCorrectedPoint(j, processed, centerPoints, corrected[j]);
                    }

                    // Commit without touching UI-bound image props
                    var copy = (Point2f[])centerPoints.Clone();
                    if (!exercise.CenterPoints.TryAdd(i, copy))
                        exercise.CenterPoints[i] = copy;

                    // Throttled progress
                    int done = i - startFrame + 1;
                    if (done % reportEvery == 0 || done == maxProgress)
                        progress?.Report(done);
                }

                // Persist once at the end
                exercise.SaveToFile();
            });
        }

        private static Point2f[] GetCenterPoints(SessionInfo exercise)
        {
            Point2f[]? centerPoints = null;

            if (exercise.CenterPoints.TryGetValue(0, out Point2f[]? value))
            {
                // If length < 4, create a new array of length 4 and copy the old values
                if (value.Length < 4)
                {
                    var newArr = new Point2f[4];
                    for (int i = 0; i < value.Length; i++)
                        newArr[i] = value[i];

                    // Optional: fill the remaining with NaN to mark as unused
                    for (int i = value.Length; i < 4; i++)
                        newArr[i] = new Point2f(float.NaN, float.NaN);

                    centerPoints = newArr;
                }
                else
                {
                    centerPoints = value;
                }
            }
            else
            {
                // No entry found: start fresh with a 4-element array
                centerPoints = new Point2f[4]
                {
                    new Point2f(float.NaN, float.NaN),
                    new Point2f(float.NaN, float.NaN),
                    new Point2f(float.NaN, float.NaN),
                    new Point2f(float.NaN, float.NaN)
                };
            }
            return centerPoints;
        }

        public static void CapturePointTemplates(Mat image, int size, Point2f point, ref Mat output)
        {
            int xCenter = (int)point.X;
            int yCenter = (int)point.Y;

            int halfSize = (int)(size / 2);
            int x = Math.Max(0, xCenter - halfSize);
            int y = Math.Max(0, yCenter - halfSize);
            int width = Math.Min(image.Width - x, size);
            int height = Math.Min(image.Height - y, size);

            if (width > 0 && height > 0)
            {
                OpenCvSharp.Rect roi = new(x, y, width, height);
                output = new Mat(image, roi).Clone();  // Clone to decouple from original
                Cv2.ImWrite("template.png", output); // Save for debugging
            }
        }

        private static double CalculateChamferScore(Mat sceneDist, Mat templateEdges, int offsetX, int offsetY)
        {
            // Ensure the ROI is fully within the bounds of sceneDist
            if (offsetX < 0 || offsetY < 0 ||
                offsetX + templateEdges.Cols > sceneDist.Cols ||
                offsetY + templateEdges.Rows > sceneDist.Rows)
            {
                return double.MaxValue; // Invalid region
            }

            // Extract region of interest
            OpenCvSharp.Rect roi = new(offsetX, offsetY, templateEdges.Cols, templateEdges.Rows);
            Mat distRoi = new(sceneDist, roi);

            // Sum distances where templateEdges > 0
            double sum = 0;
            double count = 0;

            for (int j = 0; j < templateEdges.Rows; ++j)
            {
                for (int i = 0; i < templateEdges.Cols; ++i)
                {
                    if (templateEdges.At<byte>(j, i) > 0) // Edge pixel
                    {
                        sum += distRoi.At<float>(j, i);
                        count++;
                    }
                }
            }

            return count > 0 ? sum / count : double.MaxValue; // Avoid division by zero
        }

        public static Tuple<Point2f, Point2f, Point2f, Point2f>? MatchWithChamfer(Mat imgScene, Mat imgTemplate)
        {
            try
            {
                // Get Edges
                Mat sceneGray = imgScene.CvtColor(ColorConversionCodes.BGR2GRAY);
                Mat templateGray = imgTemplate.CvtColor(ColorConversionCodes.BGR2GRAY);

                //Cv2.ImWrite("template_gray.png", templateGray);

                Mat sceneEdges = new();
                Mat templateEdges = new();

                Cv2.Canny(sceneGray, sceneEdges, 12, 150);
                Cv2.Canny(templateGray, templateEdges, 12, 150);

                Mat invertedSceneEdges = new();
                Cv2.BitwiseNot(sceneEdges, invertedSceneEdges);

                //Cv2.ImWrite("template_edges.png", templateEdges);

                Mat sceneDist = new();
                Cv2.DistanceTransform(invertedSceneEdges, sceneDist, DistanceTypes.L2, DistanceTransformMasks.Mask3);

                Mat distVis = new();
                Cv2.Normalize(sceneDist, distVis, 0, 255, NormTypes.MinMax);
                distVis.ConvertTo(distVis, MatType.CV_8U); // Convert to 8-bit for saving

                // Optional: Apply colormap to visualize depth more clearly
                Mat distColor = new();
                Cv2.ApplyColorMap(distVis, distColor, ColormapTypes.Jet);

                // Slide template over scene to find best match
                double bestScore = double.MaxValue;
                Point2f bestPoint = new();

                int heatmapRows = sceneDist.Rows - templateEdges.Rows + 1;
                int heatmapCols = sceneDist.Cols - templateEdges.Cols + 1;

                Mat chamferScoreMap = new(heatmapRows, heatmapCols, MatType.CV_32F, Scalar.All(0));

                for (int y = 0; y <= sceneDist.Rows - templateEdges.Rows; ++y)
                {
                    for (int x = 0; x <= sceneDist.Cols - templateEdges.Cols; ++x)
                    {
                        // Calculate Chamfer score
                        double score = CalculateChamferScore(sceneDist, templateEdges, x, y);

                        chamferScoreMap.Set(y, x, (float)score); // Record the score

                        if (score < bestScore)
                        {
                            bestScore = score;
                            bestPoint = new Point2f(x, y);
                        }
                    }
                }

                // Normalize to 0–255 for display
                Mat scoreVis = new();
                Cv2.Normalize(chamferScoreMap, scoreVis, 0, 255, NormTypes.MinMax);
                scoreVis.ConvertTo(scoreVis, MatType.CV_8U);

                // Optional: apply colormap for heatmap-style visualization
                Mat heatmapColor = new();
                Cv2.ApplyColorMap(scoreVis, heatmapColor, ColormapTypes.Jet);

                // Return matching rectangle corners
                if (bestScore < double.MaxValue)
                {
                    Point2f topLeft = new(bestPoint.X, bestPoint.Y);
                    Point2f topRight = new(bestPoint.X + imgTemplate.Cols, bestPoint.Y);
                    Point2f bottomRight = new(bestPoint.X + imgTemplate.Cols, bestPoint.Y + imgTemplate.Rows);
                    Point2f bottomLeft = new(bestPoint.X, bestPoint.Y + imgTemplate.Rows);
                    return Tuple.Create(topLeft, topRight, bottomRight, bottomLeft);
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error during Chamfer matching: {ex.Message}");
            }

            return null;
        }
    }
}
