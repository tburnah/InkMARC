using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using Scryv.Exercises;
using Scryv.Interfaces;
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

        public IExercise? CurrentExercise { get; set; }

        private int currentExerciseIndex = 0;

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
                new NumericSequenceExercise(),
                new MathematicalEquationExercise()                
            };
            this.CurrentExercise = this.Exercises[this.currentExerciseIndex];
            OnPropertyChanged(nameof(Prompt));
            OnPropertyChanged(nameof(ImageSource));
            OnPropertyChanged(nameof(ShowTraceImage));
            OnPropertyChanged(nameof(CurrentExerciseNumber));
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
        }
    }
}
