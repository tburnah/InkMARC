using System;
using System.Collections.Generic;

public static class SortedListExtensions
{
    /// <summary>
    /// Returns the index of the largest key <= target, or -1 if all keys are greater.
    /// </summary>
    public static int FindPredecessorIndex<TKey, TValue>(
        this SortedList<TKey, TValue> list,
        TKey target)
        where TKey : notnull
    {
        if (list == null || list.Count == 0) return -1;

        var keys = list.Keys;
        var cmp = list.Comparer ?? Comparer<TKey>.Default;

        int lo = 0, hi = keys.Count - 1;
        int best = -1;

        while (lo <= hi)
        {
            int mid = lo + ((hi - lo) >> 1);
            int rel = cmp.Compare(keys[mid], target);

            if (rel == 0) return mid;     // exact hit
            if (rel < 0) { best = mid; lo = mid + 1; }
            else { hi = mid - 1; }
        }
        return best;
    }

    /// <summary>
    /// Tries to get the value for the largest key <= target.
    /// </summary>
    public static bool TryGetPredecessorValue<TKey, TValue>(
        this SortedList<TKey, TValue> list,
        TKey target,
        out TValue value)
        where TKey : notnull
    {
        int idx = list.FindPredecessorIndex(target);
        if (idx >= 0)
        {
            value = list.Values[idx];
            return true;
        }
        value = default!;
        return false;
    }

    /// <summary>
    /// Upserts an entry at <paramref name="frameIndex"/>, taking the missing component
    /// (x or y) from the predecessor (largest key ≤ _frameIndex), defaulting to 0.
    /// Use named args: list.UpsertAt(_frameIndex, x: newX) or list.UpsertAt(_frameIndex, y: newY).
    /// </summary>
    public static void UpsertAt(this SortedList<int, (int x, int y)> list,
                                int frameIndex,
                                int? x = null,
                                int? y = null)
    {
        ArgumentNullException.ThrowIfNull(list);

        var prev = list.TryGetPredecessorValue(frameIndex, out var pt) ? pt : (x:0,y:0);
        var newX = x ?? prev.x;
        var newY = y ?? prev.y;

        list[frameIndex] = (newX, newY);
    }
}
