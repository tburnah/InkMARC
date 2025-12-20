using System;
using System.Collections.Concurrent;
using System.Diagnostics;
using System.Linq;


namespace InkMARC.Clean.Services
{
    internal static class Perf
    {
        private sealed class Stat
        {
            public long Count;
            public long TotalTicks;
            public long MaxTicks;
        }

        private static readonly ConcurrentDictionary<string, Stat> _stats = new();

        public static long Now() => Stopwatch.GetTimestamp();

        public static void Add(string name, long startTicks)
        {
            long dt = Stopwatch.GetTimestamp() - startTicks;
            var s = _stats.GetOrAdd(name, _ => new Stat());

            // Simple thread-safe accumulation; minor races are acceptable for profiling.
            System.Threading.Interlocked.Increment(ref s.Count);
            System.Threading.Interlocked.Add(ref s.TotalTicks, dt);

            // Max update (best-effort)
            long curMax;
            while (true)
            {
                curMax = System.Threading.Volatile.Read(ref s.MaxTicks);
                if (dt <= curMax) break;
                if (System.Threading.Interlocked.CompareExchange(ref s.MaxTicks, dt, curMax) == curMax) break;
            }
        }

        public static double TicksToMs(long ticks) => ticks * 1000.0 / Stopwatch.Frequency;

        public static void DumpTop(int topN = 25)
        {
            var rows = _stats.Select(kvp =>
            {
                var s = kvp.Value;
                long c = Math.Max(1, System.Threading.Volatile.Read(ref s.Count));
                long tot = System.Threading.Volatile.Read(ref s.TotalTicks);
                long mx = System.Threading.Volatile.Read(ref s.MaxTicks);
                return new
                {
                    Name = kvp.Key,
                    Count = c,
                    AvgMs = TicksToMs(tot / c),
                    MaxMs = TicksToMs(mx),
                    TotalMs = TicksToMs(tot),
                };
            })
            .OrderByDescending(r => r.TotalMs)
            .Take(topN)
            .ToList();

            Debug.WriteLine("---- PERF (Top) ----");
            foreach (var r in rows)
                Debug.WriteLine($"{r.Name,-40}  n={r.Count,6}  avg={r.AvgMs,8:0.000} ms  max={r.MaxMs,8:0.000} ms  total={r.TotalMs,10:0.0} ms");
        }

        // Optional: clear between runs
        public static void Reset() => _stats.Clear();
    }
}
