using InkMARCDeform.Interfaces;

namespace InkMARCDeform.Exercises
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
