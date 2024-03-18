using Scryv.Interfaces;

namespace Scryv.Exercises
{
    /// <summary>
    /// Represents an exercise to trace a spiral.
    /// </summary>
    public class SpiralExercise : IExercise
    {
        /// <summary>
        /// Gets the prompt for the exercise.
        /// </summary>
        public string Prompt => "Trace the spiral.";

        /// <summary>
        /// Gets the path to the trace image for the exercise.
        /// </summary>
        public string? TraceImage => "spiral.png";
    }
}
