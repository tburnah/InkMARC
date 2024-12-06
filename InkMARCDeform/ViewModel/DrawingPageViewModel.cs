using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using InkMARC.Models.Interfaces;
using InkMARCDeform.Exercises;
using InkMARCDeform.Interfaces;
using InkMARCDeform.Utilities;
using InkMARCDeform.Views;
using InkMARCDeform.Views.AdvanceDrawingView;
using System.Collections.ObjectModel;

namespace InkMARCDeform.ViewModel
{
    /// <summary>
    /// Represents the view model for the DrawingPage.
    /// </summary>
    internal partial class DrawingPageViewModel : ObservableObject
    {
        public AdvancedDrawingView? InkMARCDrawingView { get; set; }

        /// <summary>
        /// Initializes a new instance of the <see cref="DrawingPageViewModel"/> class.
        /// </summary>
        public DrawingPageViewModel()
        {
            this.DrawingLines = new ObservableCollection<IAdvancedDrawingLine>();
            Exercises = new ObservableCollection<IExercise>
                {
                    new CornersExercise(),
                    new ShapesExercise(),
                    new LowerAlphabetExercise(),
                    new FruitExercise(),
                    new NurseryRhymeExercise(),
                    new PortraitExercise(),
                    new SignatureExercise(),
                    new SpiralExercise(),
                    new LandscapeExercise(),
                    new ShadedExercise(),
                    new FoodExercise(),
                    new MazeExercise(),
                    new PetExercise()
                };
            this.CurrentExercise = this.Exercises.First();
            OnPropertyChanged(nameof(Prompt));
            OnPropertyChanged(nameof(ImageSource));
            OnPropertyChanged(nameof(ShowTraceImage));
            OnPropertyChanged(nameof(CurrentExerciseNumber));            
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

        private int currentExerciseIndex = 0;
        private IExercise? currentExercise;

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
        public string Prompt => this.CurrentExercise?.Prompt ?? string.Empty;

        /// <summary>
        /// Gets the image source for the current exercise.
        /// </summary>
        public string ImageSource => this.CurrentExercise?.TraceImage ?? string.Empty;

        /// <summary>
        /// Gets a value indicating whether to show the trace image.
        /// </summary>
        public bool ShowTraceImage => !string.IsNullOrEmpty(this.ImageSource);

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
                DataUtilities.SaveAdvancedDrawingLines(DrawingLines.ToList(), DataUtilities.GetDataFileName(currentExerciseIndex));
                using var stream = await InkMARCDrawingView.GetImageStream(InkMARCDrawingView.Width, InkMARCDrawingView.Height);
                string imagePath = DataUtilities.GetImageFileName(currentExerciseIndex);
                using (var fileStream = new FileStream(imagePath, FileMode.Create, FileAccess.Write))
                {                    
                    await stream.CopyToAsync(fileStream);                    
                }
                SessionContext.ImagePaths.Add(imagePath);
            }
            if (currentExerciseIndex < TotalExercises - 1)
            {
                ClearDrawing();
                currentExerciseIndex++;
                CurrentExercise = Exercises[currentExerciseIndex];
                OnPropertyChanged(nameof(Prompt));
                OnPropertyChanged(nameof(ImageSource));
                OnPropertyChanged(nameof(ShowTraceImage));
                OnPropertyChanged(nameof(CurrentExerciseNumber));
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
    }
}
