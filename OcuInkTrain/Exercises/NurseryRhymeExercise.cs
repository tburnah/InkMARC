using OcuInkTrain.Interfaces;

namespace OcuInkTrain.Exercises
{
    /// <summary>
    /// Represents an exercise for writing the words to a favorite nursery rhyme or children's song.
    /// </summary>
    public class NurseryRhymeExercise : IExercise
    {
        /// <summary>
        /// Gets the prompt for the exercise.
        /// </summary>
        public string Prompt => "Write the words to a favorite nursery rhyme or children's song.";

        /// <summary>
        /// Gets the trace image for the exercise.
        /// </summary>
        public string? TraceImage => null;
    }
}
