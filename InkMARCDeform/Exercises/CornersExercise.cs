using InkMARCDeform.Interfaces;

namespace InkMARCDeform.Exercises
{
    /// <summary>
    /// Exercise for corner touching
    /// </summary>
    public class CornersExercise : IExercise
    {
        /// <summary>
        /// Gets the prompt for the exercise.
        /// </summary>
        public string Prompt => "Touch the 4 dots at the corners.";

        /// <summary>
        /// Gets the path to the image.
        /// </summary>
        public string? TraceImage => "corners.png";
    }
}
