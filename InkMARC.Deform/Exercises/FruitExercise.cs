using InkMARCDeform.Interfaces;

namespace InkMARCDeform.Exercises
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

        /// <summary>
        /// Gets the minimum desired pressure for the exercise.
        /// </summary>
        public float MinDesiredPressure => 0.1f;

        /// <summary>
        /// Gets the maximum desired pressure for the exercise.
        /// </summary>
        public float MaxDesiredPressure => 0.3f;

        /// <summary>
        /// Gets a value indicating whether floating lines are allowed.
        /// </summary>
        public bool AllowFloatingLines => false;
    }
}
