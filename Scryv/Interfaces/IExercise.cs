namespace Scryv.Interfaces
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
    }
}
