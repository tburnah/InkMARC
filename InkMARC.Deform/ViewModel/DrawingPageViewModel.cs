using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using InkMARC.Exercises;
using InkMARC.Models.Interfaces;
using InkMARC.Primatives;
using InkMARCDeform.Exercises;
using InkMARCDeform.Interfaces;
using InkMARCDeform.Utilities;
using InkMARCDeform.Views;
using InkMARCDeform.Views.AdvanceDrawingView;
using System.Collections.ObjectModel;
using System.Diagnostics;
using System.Reflection;

namespace InkMARCDeform.ViewModel
{
    /// <summary>
    /// Represents the view model for the DrawingPage.
    /// </summary>
    internal partial class DrawingPageViewModel : ObservableObject
    {
        private Color prevColor = Colors.Black;
        private int currentExerciseIndex = 0;
        private IExercise? currentExercise;
        private float pressure;
        private AdvancedDrawingView? inkMARCDrawingView;

        private void ResetExerciseState()
        {
            ClearDrawing();
            pressure = -1f;
            Pressure = 0f;
            OnPropertyChanged(nameof(Prompt));
            OnPropertyChanged(nameof(ImageSource));
            OnPropertyChanged(nameof(ShowTraceImage));
            OnPropertyChanged(nameof(CurrentExerciseNumber));
            OnPropertyChanged(nameof(FloatingLinesAllowed));
            OnPropertyChanged(nameof(PressureBackground));
        }

        /// <summary>
        /// Gets or sets the InkMARCDrawingView.
        /// </summary>
        public AdvancedDrawingView? InkMARCDrawingView
        {
            get => inkMARCDrawingView;
            set
            {
                inkMARCDrawingView = value;
                if (inkMARCDrawingView is not null)
                {
                    FlashColor(Colors.Blue);
                    long ticksSinceBoot = Stopwatch.GetTimestamp();
                    long frequency = Stopwatch.Frequency;
                    startingTimeStamp = (ulong)(ticksSinceBoot * 1_000_000 / frequency);
                }
            }
        }

        /// <summary>
        /// Gets or sets the pressure.
        /// </summary>
        public float Pressure
        {
            get => pressure;
            set
            {
                if (SetProperty(ref pressure, value))
                {
                    UpdatePressureRelatedProperties();
                }
            }
        }

        /// <summary>
        /// Updates the pressure related properties.
        /// </summary>
        private void UpdatePressureRelatedProperties()
        {
            OnPropertyChanged(nameof(PressureBackground));
            if (PressureBackground != prevColor && InkMARCDrawingView != null)
            {
                prevColor = PressureBackground;
                InkMARCDrawingView.CursorColor = prevColor;
            }
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="DrawingPageViewModel"/> class.
        /// </summary>
        public DrawingPageViewModel()
        {
            List<string> shortPrompts =
            [
                "Write 3 abbreviations of country names, separated by commas.  I.e. USA, UK, etc.",
                "Write 3 frequently used acronyms, separated by commas.  I.e. LOL, OMG, FYI, etc."
            ];

            int index = SessionContext.Marked ? 1 : 0;
            string selectedShortPrompt = shortPrompts[index];

            List<string> mediumStraightPrompts =
            [
                "Draw a city scape with straight lines.",
                "Draw a signpost with arrow signs pointing in different directions.",                
            ];
            
            string selectedMediumStraightPrompt = mediumStraightPrompts[index];

            List<string> mediumCurvedPrompts =
            [
                "Draw planets circling a star.",
                "Draw flower petals radiating outward from a central point.",                
            ];
            
            string selectedMediumCurvedPrompt = mediumCurvedPrompts[index];

            List<IExercise> longStraight =
            [
                new CustomizableExercise("Draw a large tic, tac, toe board.", PressureType.Undefined),
                new MazeExercise()
            ];
            
            IExercise selectedLongStraight = longStraight[index];

            List<IExercise> longCurved =
            [
                new CustomizableExercise("Draw a large infinity symbol.", PressureType.Undefined),
                new CustomizableExercise("Draw the waves of an ocean scene with large, rolling curves.", PressureType.Undefined),
                new SpiralExercise()
            ];
            
            IExercise selectedLongCurved = longCurved[index];

            List<IExercise> allExercises =
            [
                //new CustomizableExercise(selectedShortPrompt, PressureType.None),
                //new CustomizableExercise(selectedMediumStraightPrompt, PressureType.None),
                //new CustomizableExercise(selectedMediumCurvedPrompt, PressureType.None),
                //new CustomizableExercise(selectedLongStraight, PressureType.None),
                //new CustomizableExercise(selectedLongCurved, PressureType.None),

                new CustomizableExercise(selectedShortPrompt, PressureType.Low),
                new CustomizableExercise(selectedMediumStraightPrompt, PressureType.Low),
                new CustomizableExercise(selectedMediumCurvedPrompt, PressureType.Low),
                new CustomizableExercise(selectedLongStraight, PressureType.Low),
                new CustomizableExercise(selectedLongCurved, PressureType.Low),

                new CustomizableExercise(selectedShortPrompt, PressureType.High),                             
                new CustomizableExercise(selectedMediumStraightPrompt, PressureType.High),                                
                new CustomizableExercise(selectedMediumCurvedPrompt, PressureType.High),                
                new CustomizableExercise(selectedLongStraight, PressureType.High),                                
                new CustomizableExercise(selectedLongCurved, PressureType.High)
            ];            

            this.DrawingLines = [];
            Exercises = new ObservableCollection<IExercise>(allExercises);
            Exercises.Insert(0, new Instructions1Exercise());
            this.CurrentExercise = this.Exercises.First();

            ResetExerciseState();
        }

        #region Exercises
        /// <summary>
        /// Gets the collection of exercises.
        /// </summary>
        public ObservableCollection<IExercise> Exercises { get; private set; }

        /// <summary>
        /// Gets or sets the navigation service.
        /// </summary>
        public INavigation? Navigation { get; set; }

        /// <summary>
        /// Gets or sets the current exercise.
        /// </summary>
        public IExercise? CurrentExercise
        {
            get => currentExercise;
            set => SetProperty(ref currentExercise, value);
        }

        /// <summary>
        /// Gets the current exercise number.
        /// </summary>
        public int CurrentExerciseNumber => this.currentExerciseIndex + 1;

        /// <summary>
        /// Gets the total number of exercises.
        /// </summary>
        public int TotalExercises => this.Exercises.Count;

        #endregion

        /// <summary>
        /// Gets the collection of drawing lines.
        /// </summary>
        public ObservableCollection<IAdvancedDrawingLine> DrawingLines { get; private set; }

        /// <summary>
        /// Gets the prompt for the current exercise.
        /// </summary>
        public string Prompt => CurrentExercise?.Prompt ?? string.Empty;

        /// <summary>
        /// Gets the image source for the current exercise.
        /// </summary>
        public string ImageSource => CurrentExercise?.TraceImage ?? string.Empty;

        /// <summary>
        /// Gets a value indicating whether to show the trace image.
        /// </summary>
        public bool ShowTraceImage => !string.IsNullOrEmpty(ImageSource);

        /// <summary>
        /// Gets the minimum desired pressure for the current exercise.
        /// </summary>
        public float MinDesiredPressure => CurrentExercise?.MinDesiredPressure ?? 0.0f;

        /// <summary>
        /// Gets the maximum desired pressure for the current exercise.
        /// </summary>
        public float MaxDesiredPressure => CurrentExercise?.MaxDesiredPressure ?? 1.0f;

        /// <summary>
        /// Gets the pressure background.
        /// </summary>
        public Color PressureBackground
        {
            get
            {
                if ((pressure < MinDesiredPressure) || (pressure > MaxDesiredPressure))
                {
                    return Colors.Red;
                }
                else
                {
                    return Colors.Green;
                }
            }
        }

        /// <summary>
        /// Gets a value indicating whether floating lines are allowed.
        /// </summary>
        public bool FloatingLinesAllowed => this.CurrentExercise?.AllowFloatingLines ?? false;

        /// <summary>
        /// Clears the drawing lines.
        /// </summary>
        [RelayCommand]
        public void ClearDrawing()
        {
            this.DrawingLines.Clear();
        }

        [RelayCommand]
        public void Undo()
        {
            if (DrawingLines.Count > 0)
                DrawingLines.RemoveAt(DrawingLines.Count - 1);
        }

        /// <summary>
        /// Continues to the next exercise.
        /// </summary>
        [RelayCommand]
        public async Task Continue()
        {
            if (DrawingLines.Count > 0)
            {
                DataUtilities.SaveAdvancedDrawingLines(DrawingLines.ToList(), startingTimeStamp, DataUtilities.GetDataFileName(currentExerciseIndex));
                using var stream = await InkMARCDrawingView.GetImageStream(InkMARCDrawingView.Width, InkMARCDrawingView.Height);
                string imagePath = DataUtilities.GetImageFileName(currentExerciseIndex);
                using var fileStream = new FileStream(imagePath, FileMode.Create, FileAccess.Write);
                await stream.CopyToAsync(fileStream);
            }
            if (currentExerciseIndex < TotalExercises - 1)
            {
                ClearDrawing();
                currentExerciseIndex++;
                CurrentExercise = Exercises[currentExerciseIndex];
                ResetExerciseState();
            }
            else
            {
                if (Navigation is not null)
                    await Navigation.PushAsync(new UploadPage());
            }
        }

        [RelayCommand]
        public void ChooseColor(object color)
        {
            if (color is Color selectedColor && InkMARCDrawingView is not null)
            {
                InkMARCDrawingView.LineColor = selectedColor;
            }
        }

        private ulong startingTimeStamp;

        private void FlashColor(Color color)
        {           
            // Set the background to the new color
            inkMARCDrawingView.BackgroundColor = color;

            Task.Delay(500).ContinueWith(t => inkMARCDrawingView.BackgroundColor = Colors.Transparent);
        }
    }
}
