using InkMARCDeform.Interfaces;

namespace InkMARCDeform.Exercises
{
    /// <summary>
    /// Represents an exercise for tracing a given image.
    /// </summary>
    public class ShapesExercise : IExercise
    {
        /// <summary>
        /// Gets the prompt for the exercise.
        /// </summary>
        public string Prompt => "Trace the given image.";

        /// <summary>
        /// Gets the path to the image to be traced.
        /// </summary>
        public string? TraceImage => "shapes.png";
    }
}
