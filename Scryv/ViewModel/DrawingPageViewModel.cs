using Camera.MAUI;
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
using System.Timers;

namespace Scryv.ViewModel
{
    internal partial class DrawingPageViewModel : ObservableObject
    {
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
                new FoodExercise(),
                new MazeExercise(),
                new PetExercise()
            };            
            this.CurrentExercise = this.Exercises.First();
            OnPropertyChanged(nameof(Prompt));
            OnPropertyChanged(nameof(ImageSource));
            OnPropertyChanged(nameof(ShowTraceImage));
            OnPropertyChanged(nameof(CurrentExerciseNumber));
            //StartRecording();
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
        public ObservableCollection<IExercise> Exercises { get; private set; }

        public INavigation? Navigation { get; set; }

        public IExercise? CurrentExercise
        {
            get => currentExercise;
            set => SetProperty(ref currentExercise, value);
        }

        private int currentExerciseIndex = 0;
        private IExercise? currentExercise;

        public int CurrentExerciseNumber => this.currentExerciseIndex + 1;
        public int TotalExercises => this.Exercises.Count;

        #endregion

        public ObservableCollection<IAdvancedDrawingLine> DrawingLines { get; private set; }

        public string Prompt => this.CurrentExercise?.Prompt ?? string.Empty;

        public string ImageSource => this.CurrentExercise?.TraceImage ?? string.Empty;

        public bool ShowTraceImage => !string.IsNullOrEmpty(this.ImageSource);

        [RelayCommand]
        public void ClearDrawing()
        {
            this.DrawingLines.Clear();
        }

        [RelayCommand]
        public async void Continue()
        {
            if (DrawingLines.Count > 0)
            {
                DataUtilities.SaveAdvancedDrawingLines(DrawingLines.ToList(), DataUtilities.GetDataFileName(currentExerciseIndex));
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
                CameraWindowViewModel.Current.RestartCameraPreview(DataUtilities.GetVideoFileName(DateTime.Now.ToFileTime(), currentExerciseIndex));
            }
            else
            { 
                StopRecording();
                Navigation.PushAsync(new UploadPage());
            }
        }
    }
}
