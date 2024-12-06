using InkMARCDeform.Interfaces;

namespace InkMARCDeform.Exercises
{
    /// <summary>
    /// Represents a food exercise.
    /// </summary>
    public class FoodExercise : IExercise
    {
        /// <summary>
        /// Gets the prompt for the exercise.
        /// </summary>
        public string Prompt => "Draw a picture of your favorite food.";

        /// <summary>
        /// Gets the trace image for the exercise.
        /// </summary>
        public string? TraceImage => null;
    }
}
