namespace InkMARC.Models.Extensions
{
    /// <summary>
    /// Provides extension methods for the <see cref="float"/> type.
    /// </summary>
    public static class FloatExtensions
    {
        /// <summary>
        /// Clamps a value between a minimum and maximum value.
        /// </summary>
        /// <param name="value">The value to clamp.</param>
        /// <param name="min">The minimum value.</param>
        /// <param name="max">The maximum value.</param>
        /// <returns>The clamped value.</returns>
        public static float Clamp(this float value, float min, float max)
        {
            return float.Clamp(value, min, max);
        }
    }
}
