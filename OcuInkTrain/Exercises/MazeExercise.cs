using OcuInkTrain.Interfaces;

namespace OcuInkTrain.Exercises
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
    }
}
