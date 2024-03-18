using Scryv.Interfaces;

namespace Scryv.Exercises
{
    /// <summary>
    /// Represents a pet exercise.
    /// </summary>
    public class PetExercise : IExercise
    {
        /// <summary>
        /// Gets the prompt for the exercise.
        /// </summary>
        public string Prompt => "Draw a picture of a pet you would like to have.";

        /// <summary>
        /// Gets the trace image for the exercise.
        /// </summary>
        public string? TraceImage => null;
    }
}
