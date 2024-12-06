using InkMARCDeform.Interfaces;

namespace InkMARCDeform.Exercises
{
    /// <summary>
    /// Represents a landscape exercise that prompts the user to draw a picture of a place outside.
    /// </summary>
    public class LandscapeExercise : IExercise
    {
        /// <summary>
        /// Gets the prompt for the exercise.
        /// </summary>
        public string Prompt => "Draw a picture of a place outside. You can show things like trees, mountains, or a river.";

        /// <summary>
        /// Gets the trace image for the exercise.
        /// </summary>
        public string? TraceImage => null;
    }
}
