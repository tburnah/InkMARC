using InkMARCDeform.Interfaces;

namespace InkMARCDeform.Exercises
{
    /// <summary>
    /// Represents a shaded exercise.
    /// </summary>
    public class ShadedExercise : IExercise
    {
        /// <summary>
        /// Gets the prompt for the exercise.
        /// </summary>
        public string Prompt => "Color the shading, or shadows and lightness of the shapes.";

        /// <summary>
        /// Gets the trace image for the exercise.
        /// </summary>
        public string? TraceImage => "shadedshapes.png";

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
