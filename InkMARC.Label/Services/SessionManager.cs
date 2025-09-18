using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
using System.Text.Json;
using System.Text.RegularExpressions;
using System.Threading.Tasks;

namespace InkMARC.Label.Services
{
    public class SessionManager
    {

        public static TimeSpan GetFullSessionDrawingDuration(string folderPath, string sessionId)
        {
            var files = Directory.GetFiles(folderPath, $"data_*_{sessionId}_*.json");

            long? sessionStart = null;
            long? sessionEnd = null;

            foreach (var file in files)
            {
                var json = File.ReadAllText(file);
                using var doc = JsonDocument.Parse(json);

                if (!doc.RootElement.TryGetProperty("DrawingLines", out var linesArray)) continue;

                foreach (var line in linesArray.EnumerateArray())
                {
                    if (!line.TryGetProperty("Points", out var pointsArray)) continue;

                    foreach (var point in pointsArray.EnumerateArray())
                    {
                        if (point.TryGetProperty("Timestamp", out var tsProp) && tsProp.TryGetInt64(out var timestamp))
                        {
                            if (sessionStart == null || timestamp < sessionStart) sessionStart = timestamp;
                            if (sessionEnd == null || timestamp > sessionEnd) sessionEnd = timestamp;
                        }
                    }
                }
            }

            if (sessionStart == null || sessionEnd == null)
                throw new Exception("No timestamps found across session files.");

            long durationMicroseconds = sessionEnd.Value - sessionStart.Value;
            return TimeSpan.FromMilliseconds(durationMicroseconds / 1000.0);
        }

        /// <summary>
        /// Builds a dictionary from session ID to a dictionary mapping exercise number to a tuple (file path, date).
        /// Used for video and JSON files.
        /// </summary>
        public static Dictionary<string, Dictionary<int, Tuple<string, DateTime?>>> BuildSessionIdDictionary(IEnumerable<string> files)
        {
            var dict = new Dictionary<string, Dictionary<int, Tuple<string, DateTime?>>>();

            foreach (var file in files)
            {
                var parsed = ExtractSessionIDAndIndex(Path.GetFileName(file));
                if (parsed != null)
                {
                    if (!dict.TryGetValue(parsed.Item1, out Dictionary<int, Tuple<string, DateTime?>>? value))
                    {
                        value = [];
                        dict[parsed.Item1] = value;
                    }

                    value[parsed.Item2] = Tuple.Create(file, parsed.Item3);
                }
            }
            return dict;
        }


        /// <summary>
        /// Builds a dictionary for session data (.session files) mapping session ID and exercise number to the file path.
        /// </summary>
        public static Dictionary<string, Dictionary<int, string>> BuildSessionDataDictionary(IEnumerable<string> files)
        {
            var dict = new Dictionary<string, Dictionary<int, string>>();
            foreach (var file in files)
            {
                var name = Path.GetFileNameWithoutExtension(file);
                var parts = name.Split('_');
                if (parts.Length == 2 && int.TryParse(parts[1], out int exercise))
                {
                    if (!dict.TryGetValue(parts[0], out Dictionary<int, string>? value))
                    {
                        value = [];
                        dict[parts[0]] = value;
                    }

                    value[exercise] = file;
                }
            }
            return dict;
        }

        /// <summary>
        /// Builds a dictionary for H5 files mapping session ID and exercise number to the file path.
        /// </summary>
        public static Dictionary<string, Dictionary<int, string>> BuildSessionIdDictionarySimple(IEnumerable<string> files)
        {
            var dict = new Dictionary<string, Dictionary<int, string>>();
            foreach (var file in files)
            {
                var parsed = ExtractSessionIDAndIndex(Path.GetFileName(file));
                if (parsed != null)
                {
                    if (!dict.TryGetValue(parsed.Item1, out Dictionary<int, string>? value))
                    {
                        value = [];
                        dict[parsed.Item1] = value;
                    }

                    value[parsed.Item2] = file;
                }
            }
            return dict;
        }

        public static Tuple<string, int, DateTime?>? ExtractSessionIDAndIndex(string fileName)
        {
            // Regex patterns for different filename variations
            string[] patterns =
            [
                // Pattern 1: type_sessionID_timestamp_smoothed.json
                @"^(?:data|video)_(?<sessionID>[a-zA-Z0-9]+)_(?<timestamp>\d+)_smoothed\.json$",

                // Pattern 2: type_sessionID_timestamp_index.extension
                @"^(?:data|video)_(?<sessionID>[a-zA-Z0-9]+)_(?<timestamp>[0-9T:\-.Z]+)_(?<index>\d+)\.\w+$",
    
                // Pattern 3: type_filetime_sessionID_index.extension (sessionID after timestamp)
                @"^(?:data|video)_(?<timestamp>\d+)_(?<sessionID>[a-zA-Z0-9]+)_(?<index>\d+)\.\w+$",

                // Pattern 4: type_sessionID_filetime_index.extension
                @"^(?:data|video)_(?<sessionID>[a-zA-Z0-9]+)_(?<timestamp>\d+)_(?<index>\d+)\.\w+$",
    
                // Pattern 5: type_sessionID_index.extension (index is 1–2 digits only)
                @"^(?:data|video)_(?<sessionID>[a-zA-Z0-9]+)_(?<index>\d{1,2})\.\w+$",

                // Pattern 7: type_sessionID_Participant_index_AppView.extension
                @"^(?:data|video)_(?<sessionID>[a-zA-Z0-9]+)_Participant(?<index>\d+)_AppView\d+\.\w+$",

                // Pattern 7: type_sessionID_timestamp.extension (no index) 
                @"^(?:data|video)_(?<sessionID>[a-zA-Z0-9]+)_(?<timestamp>\d+)\.\w+$"
            ];

            foreach (var pattern in patterns)
            {
                var match = Regex.Match(fileName, pattern, RegexOptions.IgnoreCase);
                if (match.Success)
                {
                    string sessionID = match.Groups["sessionID"].Value;

                    if (!(match.Groups["timestamp"].Success && int.TryParse(match.Groups["index"].Value, out int index)))
                    {
                        index = 0;
                    }

                    DateTime? extractedDateTime = null;
                    if (match.Groups["timestamp"].Success)
                    {
                        string timestampStr = match.Groups["timestamp"].Value;
                        extractedDateTime = ParseTimestamp(timestampStr);
                    }

                    return Tuple.Create(sessionID, index, extractedDateTime);
                }
            }

            // Return null if no pattern matches
            return null;
        }

        /// <summary>
        /// Parses a timestamp from the filename into a DateTime object.
        /// </summary>
        private static DateTime? ParseTimestamp(string timestampStr)
        {
            if (DateTimeOffset.TryParse(timestampStr, CultureInfo.InvariantCulture, DateTimeStyles.RoundtripKind, out DateTimeOffset dto))
            {
                return dto.UtcDateTime; // Convert to UTC DateTime
            }
            else if (DateTime.FromFileTimeUtc(long.Parse(timestampStr)) is DateTime dt)
            {
                return dt;
            }

            return null; // Invalid timestamp
        }

        public static Dictionary<string, string> MatchSessionsToVideosWithinThreshold(Dictionary<string, TimeSpan> sessionDurations, Dictionary<string, TimeSpan> videoDurations, double maxAllowedDifferenceSeconds = 30.0)
        {
            var matched = new Dictionary<string, string>();
            var remainingVideos = new Dictionary<string, TimeSpan>(videoDurations); // copy so we can remove matched videos

            foreach (var session in sessionDurations.OrderBy(sd => sd.Value))
            {
                string sessionId = session.Key;
                TimeSpan sessionTime = session.Value;

                string? bestMatch = null;
                double bestDiff = double.MaxValue;

                foreach (var video in remainingVideos)
                {
                    var diff = (video.Value - sessionTime).TotalSeconds;

                    if (diff >= 0 && diff <= maxAllowedDifferenceSeconds && diff < bestDiff)
                    {
                        bestDiff = diff;
                        bestMatch = video.Key;
                    }
                }

                if (bestMatch != null)
                {
                    matched[sessionId] = bestMatch;
                    remainingVideos.Remove(bestMatch);
                }
                else
                {
                    Console.WriteLine($"No suitable video match found for session {sessionId} (duration: {sessionTime})");
                }
            }

            return matched;
        }
    }
}
