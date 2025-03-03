using InkMARCDeform.Interfaces;

namespace InkMARCDeform.Exercises
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
