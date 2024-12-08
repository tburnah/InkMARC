using InkMARC.Primatives;
using InkMARCDeform.Interfaces;

namespace InkMARC.Exercises
{
    /// <summary>
    /// Represents a customizable exercise.
    /// </summary>
    public class CustomizableExercise : IExercise
    {
        private string _prompt;
        private string _pressurePrompt;
        private float _minPressure;
        private float _maxPressure;
        private bool _allowFloatingLines;
        private IExercise? _exercise;

        /// <summary>
        /// Initializes a new instance of the <see cref="CustomizableExercise"/> class.
        /// </summary>
        /// <param name="prompt">The prompt for the exercise.</param>
        /// <param name="pressure">The pressure type for the exercise.</param>
        /// <exception cref="ArgumentOutOfRangeException">Thrown when the pressure type is not recognized.</exception>
        public CustomizableExercise(string prompt, PressureType pressure)
        {
            _prompt = prompt;            
            (_minPressure, _maxPressure, _allowFloatingLines, _pressurePrompt) = pressure switch
            {
                PressureType.None => (0.0f, 0.0f, true, " Hover just above the screen, not actually touching."),
                PressureType.Low => (0.1f, 0.4f, false, " Use light pressure."),
                PressureType.Medium => (0.3f, 0.7f, false, " Use medium pressure."),
                PressureType.High => (0.5f, 1.0f, false, "Use heavy pressure."),
                _ => throw new ArgumentOutOfRangeException(nameof(pressure), pressure, null)
            };
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="CustomizableExercise"/> class.
        /// </summary>
        /// <param name="exerc">The exercise to customize.</param>
        /// <param name="pressure">The pressure type for the exercise.</param>
        /// <exception cref="ArgumentOutOfRangeException">Thrown when the pressure type is not recognized.</exception>
        public CustomizableExercise(IExercise exerc, PressureType pressure)
            : this(exerc.Prompt, pressure)
        {
            _exercise = exerc;
        }

        /// <summary>
        /// Gets the prompt for the exercise.
        /// </summary>
        public string Prompt => _prompt + _pressurePrompt;

        /// <summary>
        /// Gets the path to the image.
        /// </summary>
        public string? TraceImage => _exercise?.TraceImage ?? null;

        /// <summary>
        /// Gets the minimum desired pressure for the exercise.
        /// </summary>
        public float MinDesiredPressure => _minPressure;

        /// <summary>
        /// Gets the maximum desired pressure for the exercise.
        /// </summary>
        public float MaxDesiredPressure => _maxPressure;

        /// <summary>
        /// Gets a value indicating whether floating lines are allowed.
        /// </summary>
        public bool AllowFloatingLines => _allowFloatingLines;
    }
}
