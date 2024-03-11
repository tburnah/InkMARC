using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using Scryv.Exercises;
using Scryv.Interfaces;
using Scryv.Utilities;
using Scryv.Views;
using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Scryv.ViewModel
{
    internal partial class DrawingPageViewModel : ObservableObject
    {
        public ObservableCollection<IAdvancedDrawingLine> DrawingLines { get; private set; }

        public ObservableCollection<IExercise> Exercises { get; private set; }

        public INavigation? Navigation { get; set; }

        public IExercise? CurrentExercise
        {
            get => currentExercise;
            set
            {
                currentExercise = value;

            }
        }

        private int currentExerciseIndex = 0;
        private IExercise? currentExercise;

        public int CurrentExerciseNumber => this.currentExerciseIndex + 1;
        public int TotalExercises => this.Exercises.Count;

        public DrawingPageViewModel()
        {
            this.DrawingLines = new ObservableCollection<IAdvancedDrawingLine>();
            Exercises = new ObservableCollection<IExercise>
            {
                new ShapesExercise(),
                new LowerAlphabetExercise(),
                new FruitExercise(),
                new UpperAlphabetExercise(),
                new PortraitExercise(),
                new SignatureExercise(),
                new SpiralExercise(),
                new LandscapeExercise(),
                new ShadedExercise(),
                new MazeExercise()
            };
            this.CurrentExercise = this.Exercises[this.currentExerciseIndex];
            OnPropertyChanged(nameof(Prompt));
            OnPropertyChanged(nameof(ImageSource));
            OnPropertyChanged(nameof(ShowTraceImage));
            OnPropertyChanged(nameof(CurrentExerciseNumber));
            for (int i = 0; i < SessionContext.CameraViews.Count; i++)
            {
                var cameraView = SessionContext.CameraViews[i];
                cameraView.StartRecordingAsync(DataUtilities.GetVideoFileName(CurrentExerciseNumber, i, DateTime.Now.ToFileTime()));
            }
        }

        public string Prompt => this.CurrentExercise?.Prompt ?? string.Empty;

        public string ImageSource => this.CurrentExercise?.TraceImage ?? string.Empty;

        public bool ShowTraceImage => !string.IsNullOrEmpty(this.ImageSource);

        [RelayCommand]
        public void ClearDrawing()
        {
            this.DrawingLines.Clear();
        }

        [RelayCommand]
        public void Continue()
        {
            if (DrawingLines.Count > 0)
                DataUtilities.SaveAdvancedDrawingLines(DrawingLines.ToList(), DataUtilities.GetDataFileName());
            ClearDrawing();
            if (currentExerciseIndex < TotalExercises - 1)
            {
                currentExerciseIndex++;
                CurrentExercise = Exercises[currentExerciseIndex];
                OnPropertyChanged(nameof(Prompt));
                OnPropertyChanged(nameof(ImageSource));
                OnPropertyChanged(nameof(ShowTraceImage));
                OnPropertyChanged(nameof(CurrentExerciseNumber));
            }
            else
            {
                foreach (var cameraView in SessionContext.CameraViews)
                {
                    cameraView.StopRecordingAsync();
                }
                Navigation.PushAsync(new UploadPage());
            }
        }
    }
}
