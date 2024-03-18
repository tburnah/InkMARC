using Scryv.Interfaces;

namespace Scryv.Exercises
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
    }
}
