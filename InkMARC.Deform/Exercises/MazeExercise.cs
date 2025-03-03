using InkMARCDeform.Interfaces;

namespace InkMARCDeform.Exercises
{
    /// <summary>
    /// Represents a maze exercise.
    /// </summary>
    public class MazeExercise : IExercise
    {
        /// <summary>
        /// Gets the prompt for the maze exercise.
        /// </summary>
        public string Prompt => "Draw a line through the maze.";

        /// <summary>
        /// Gets the trace image for the maze exercise.
        /// </summary>
        public string? TraceImage => "maze.png";

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
