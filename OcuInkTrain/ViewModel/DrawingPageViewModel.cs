using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using OcuInkTrain.Exercises;
using OcuInkTrain.Interfaces;
using OcuInkTrain.Utilities;
using OcuInkTrain.Views;
using OcuInkTrain.Views.AdvanceDrawingView;
using System.Collections.ObjectModel;

namespace OcuInkTrain.ViewModel
{
    /// <summary>
    /// Represents the view model for the DrawingPage.
    /// </summary>
    internal partial class DrawingPageViewModel : ObservableObject
    {
        public AdvancedDrawingView? OcuInkDrawingView { get; set; }

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

        #region CameraView Properties        

        private void StartRecording()
        {
            if (CameraWindowViewModel.Current != null)
            {
                CameraWindowViewModel.Current.StartRecording.Execute(null);
            }
        }

        private void StopRecording()
        {
            if (CameraWindowViewModel.Current != null)
            {
                CameraWindowViewModel.Current.StopRecording.Execute(null);
            }
        }

        #endregion

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

        /// <summary>
        /// Continues to the next exercise.
        /// </summary>
        [RelayCommand]
        public async Task Continue()
        {
            if (DrawingLines.Count > 0)
            {
                DataUtilities.SaveAdvancedDrawingLines(DrawingLines.ToList(), DataUtilities.GetDataFileName(currentExerciseIndex));
                using var stream = await OcuInkDrawingView.GetImageStream(OcuInkDrawingView.Width, OcuInkDrawingView.Height);
                using (var fileStream = new FileStream(DataUtilities.GatImageFileName(currentExerciseIndex), FileMode.Create, FileAccess.Write))
                {
                    await stream.CopyToAsync(fileStream);
                }
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
                CameraWindowViewModel.Current?.RestartCameraPreview(DataUtilities.GetVideoFileName(DateTime.Now.ToFileTime(), currentExerciseIndex));
            }
            else
            {
                StopRecording();
                if (Navigation is not null)
                    await Navigation.PushAsync(new UploadPage());
            }
        }
    }
}
