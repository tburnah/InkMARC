using Scryv.Interfaces;

namespace Scryv.Exercises
{
    /// <summary>
    /// Represents a fruit exercise.
    /// </summary>
    public class FruitExercise : IExercise
    {
        /// <summary>
        /// Gets the prompt for the exercise.
        /// </summary>
        public string Prompt => "Trace the image shown.";

        /// <summary>
        /// Gets the path to the trace image.
        /// </summary>
        public string? TraceImage => "fruit.png";
    }
}
