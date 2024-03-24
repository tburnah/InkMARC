using OcuInkTrain.Interfaces;

namespace OcuInkTrain.Exercises
{
    /// <summary>
    /// Represents a portrait exercise.
    /// </summary>
    public class PortraitExercise : IExercise
    {
        /// <summary>
        /// Gets the prompt for the exercise.
        /// </summary>
        public string Prompt => "Trace the portrait shown.";

        /// <summary>
        /// Gets the image file name to trace.
        /// </summary>
        public string? TraceImage => "girl.png";
    }
}
