using InkMARC.Label.Services.Interfaces;
using InkMARC.Models.Primatives;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Text.Json;
using System.Threading.Tasks;

namespace InkMARC.Label.Services
{
    public class SessionDataService
    {
        private static readonly JsonSerializerOptions IndentedOptions = new() { WriteIndented = true };

        public static async Task AssignStateChangesFromData(ProjectInfo exercise, Progress<int> progress, IVideoService videoService)
        {
            if (exercise == null)
                return;

            // Optionally clear any existing state changes
            exercise.StateChanges.Clear();

            bool sequenceActive = false; // Tracks if we are inside a sequence of frames with a datapoint
            int startFrame = exercise.StartFrame;
            int stopFrame = exercise.StopFrame;

            const int progressUpdateFrequency = 10;

            // Run the analysis on a background thread.
            await Task.Run(() =>
            {
                // Loop through each frame between start and stop
                for (int i = startFrame; i <= stopFrame; i++)
                {
                    // Check if there is a datapoint for this frame.
                    bool hasDataPoint = FindClosestDataPointOptimized(i, exercise, videoService) != null;

                    if (!sequenceActive && hasDataPoint)
                    {
                        // We just entered a sequence where frames have a datapoint.
                        exercise.StateChanges[i] = true;
                        sequenceActive = true;
                    }
                    else if (sequenceActive && !hasDataPoint)
                    {
                        // We just left a sequence: record the first frame where no datapoint is available.
                        exercise.StateChanges[i] = false;
                        sequenceActive = false;
                    }

                    // Report progress as the number of frames processed.
                    // Report progress only every 'progressUpdateFrequency' frames.
                    if ((i - startFrame) % progressUpdateFrequency == 0)
                    {
                        ((IProgress<int>)progress).Report(i - startFrame + 1);
                    }
                }
            });

            if (sequenceActive && stopFrame >= startFrame)
            {
                exercise.StateChanges[stopFrame] = false;
            }
        }

        /// <summary>
        /// Optimized version of FindClosestDataPoint. If _drawingLine is sorted by timestamp,
        /// you could further optimize this with a binary search.
        /// </summary>
        public static InkMARCPoint? FindClosestDataPointOptimized(int currentFrameIndex, ProjectInfo exercise, IVideoService videoService)
        {
            // Compute the video time for the frame.
            double frameVideoTimeMs = currentFrameIndex * 1000.0 / videoService.FramesPerSecond;
            double expectedDataTimestamp = exercise.DrawingLine[0].Timestamp + (frameVideoTimeMs - (exercise?.FirstPointOffset ?? -1)) * 1000.0;

            InkMARCPoint? closestPoint = null;
            double smallestDiff = double.MaxValue;

            if (exercise?.DrawingLine is not null)
            {
                // Linear search: Consider binary search if _drawingLine is sorted.
                foreach (var point in exercise.DrawingLine)
                {
                    double diff = Math.Abs(point.Timestamp - expectedDataTimestamp);
                    if (diff < smallestDiff)
                    {
                        smallestDiff = diff;
                        closestPoint = point;
                    }
                }
            }

            return (smallestDiff <= videoService.ThresholdMicroseconds) ? closestPoint : null;
        }

        /// <summary>
        /// Updates FormattedJson with the point that matches the current frame timestamp.
        /// </summary>
        public static string UpdateFormattedJson(ProjectInfo exercise, int frameIndex, IVideoService videoService)
        {
            if (exercise.DrawingLine is not null)
            {
                if (exercise.DrawingLine.Count == 0)
                {
                    return "No DrawingLines available.";
                }

                InkMARCPoint? closestPoint = exercise.DrawingLine[0];
                if (exercise.FirstPointOffset >= 0)
                {
                    closestPoint = FindClosestDataPointOptimized(frameIndex, exercise, videoService);
                }

                if (closestPoint != null)
                {
                    return JsonSerializer.Serialize(closestPoint, IndentedOptions);
                }                    
            }
            return "No matching point found.";
        }

        public static async Task<SortedList<int, bool>> ExtractFramesForStateChangesAsync(ProjectInfo project, IVideoService videoService, Progress<int> progress)
        {
            var map = new SortedList<int, bool>();
            if (project == null)
                return map;

            int startFrame = project.StartFrame;
            int stopFrame = project.StopFrame;

            const int progressUpdateFrequency = 10;

            await Task.Run(() =>
            {
                bool? previousState = null;

                for (int i = startFrame; i <= stopFrame; i++)
                {
                    bool hasDataPoint = SessionDataService.FindClosestDataPointOptimized(i, project, videoService) != null;

                    // Record a change only if the state differs from the previous state
                    if (previousState == null || previousState.Value != hasDataPoint)
                    {
                        map[i] = hasDataPoint;
                        previousState = hasDataPoint;
                    }

                    if ((i - startFrame) % progressUpdateFrequency == 0)
                    {
                        ((IProgress<int>)progress).Report(i - startFrame + 1);
                    }
                }
            });

            return map;
        }
    }
}
