namespace InkMARCDeform.Interfaces
{
    /// <summary>
    /// Represents an exercise.
    /// </summary>
    public interface IExercise
    {
        /// <summary>
        /// Gets the prompt for the exercise.
        /// </summary>
        string Prompt { get; }

        /// <summary>
        /// Gets the trace image for the exercise, if available.
        /// </summary>
        string? TraceImage { get; }

        /// <summary>
        /// Gets the minimum desired pressure for the exercise.
        /// </summary>
        float MinDesiredPressure { get; }

        /// <summary>
        /// Gets the maximum desired pressure for the exercise.
        /// </summary>
        float MaxDesiredPressure { get; }

        /// <summary>
        /// Gets a value indicating whether floating lines are allowed.
        /// </summary>
        bool AllowFloatingLines { get; }
    }
}
