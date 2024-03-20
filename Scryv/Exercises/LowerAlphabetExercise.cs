using OcuInkTrain.Interfaces;

namespace OcuInkTrain.Exercises
{
    /// <summary>
    /// Represents an exercise that requires writing as much of the lower case alphabet as possible.
    /// </summary>
    public class LowerAlphabetExercise : IExercise
    {
        /// <summary>
        /// Gets the prompt for the exercise.
        /// </summary>
        public string Prompt => "Write as much of the lower case alphabet as you can fit.";

        /// <summary>
        /// Gets the trace image for the exercise.
        /// </summary>
        public string? TraceImage => null;
    }
}
