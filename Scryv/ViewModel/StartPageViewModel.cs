using CommunityToolkit.Maui.Core;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using OcuInkTrain.Utilities;
using OcuInkTrain.Views;
using System.Diagnostics;

namespace OcuInkTrain.ViewModel
{
    /// <summary>
    /// ViewModel for the StartPage.
    /// </summary>
    public partial class StartPageViewModel : ObservableObject
    {
        private bool isTabletSelected;
        private bool isStylusSelected;
        private bool isCameraSelected;

        private IPopupService popupService;

        /// <summary>
        /// Gets or sets the navigation service.
        /// </summary>
        public INavigation? Navigation { get; set; }

        /// <summary>
        /// Initializes a new instance of the <see cref="StartPageViewModel"/> class.
        /// </summary>
        public StartPageViewModel()
        {
            this.popupService = DependencyService.Get<IPopupService>();
            string loadVideoPath = DataUtilities.GetVideosFolderPath();
            Debug.WriteLine(loadVideoPath);
        }

        /// <summary>
        /// Gets or sets a value indicating whether the tablet is selected.
        /// </summary>
        public bool IsTabletSelected
        {
            get => isTabletSelected;
            set
            {
                SetProperty(ref isTabletSelected, value);
                OnPropertyChanged(nameof(IsTabletSelected));
                OnPropertyChanged(nameof(TabletChoiceImage));
                OnPropertyChanged(nameof(TabletChoiceBorderColor));
                OnPropertyChanged(nameof(TabletChoiceColor));
            }
        }

        /// <summary>
        /// Gets or sets a value indicating whether the stylus is selected.
        /// </summary>
        public bool IsStylusSelected
        {
            get => isStylusSelected;
            set
            {
                SetProperty(ref isStylusSelected, value);
                OnPropertyChanged(nameof(IsStylusSelected));
                OnPropertyChanged(nameof(StylusChoiceImage));
                OnPropertyChanged(nameof(StylusChoiceBorderColor));
                OnPropertyChanged(nameof(StylusChoiceColor));
            }
        }

        /// <summary>
        /// Gets or sets a value indicating whether the camera is selected.
        /// </summary>
        public bool IsCameraSelected
        {
            get => isCameraSelected;
            set
            {
                SetProperty(ref isCameraSelected, value);
                OnPropertyChanged(nameof(IsCameraSelected));
                OnPropertyChanged(nameof(CameraChoiceImage));
                OnPropertyChanged(nameof(CameraChoiceBorderColor));
                OnPropertyChanged(nameof(CameraChoiceColor));
            }
        }

        private Color getBackgroundColor(bool value)
        {
            if (value)
            {
                if (Application.Current is not null && Application.Current.Resources.TryGetValue("Selected", out var activeColor))
                {
                    return (Color)activeColor;
                }
                return Colors.GreenYellow;
            }
            if (Application.Current is not null && Application.Current.Resources.TryGetValue("White", out var inactiveColor))
            {
                return (Color)inactiveColor;
            }
            return Colors.White;
        }

        private Color getBackgroundBorderColor(bool value)
        {
            if (value)
            {
                if (Application.Current is not null && Application.Current.Resources.TryGetValue("SelectionBorder", out var activeColor))
                {
                    return (Color)activeColor;
                }
                return Colors.GreenYellow;
            }
            if (Application.Current is not null && Application.Current.Resources.TryGetValue("UnselectedBorder", out var inactiveColor))
            {
                return (Color)inactiveColor;
            }
            return Colors.White;
        }

        /// <summary>
        /// Gets the image for the tablet choice.
        /// </summary>
        public string TabletChoiceImage => IsTabletSelected ? "check.png" : "tablet.png";

        /// <summary>
        /// Gets the image for the stylus choice.
        /// </summary>
        public string StylusChoiceImage => IsStylusSelected ? "check.png" : "stylus.png";

        /// <summary>
        /// Gets the image for the camera choice.
        /// </summary>
        public string CameraChoiceImage => IsCameraSelected ? "check.png" : "webcam.png";

        /// <summary>
        /// Gets the color for the tablet choice.
        /// </summary>
        public Color TabletChoiceColor => getBackgroundColor(IsTabletSelected);

        /// <summary>
        /// Gets the color for the stylus choice.
        /// </summary>
        public Color StylusChoiceColor => getBackgroundColor(IsStylusSelected);

        /// <summary>
        /// Gets the color for the camera choice.
        /// </summary>
        public Color CameraChoiceColor => getBackgroundColor(IsCameraSelected);

        /// <summary>
        /// Gets the border color for the tablet choice.
        /// </summary>
        public Color TabletChoiceBorderColor => getBackgroundBorderColor(IsTabletSelected);

        /// <summary>
        /// Gets the border color for the stylus choice.
        /// </summary>
        public Color StylusChoiceBorderColor => getBackgroundBorderColor(IsStylusSelected);

        /// <summary>
        /// Gets the border color for the camera choice.
        /// </summary>
        public Color CameraChoiceBorderColor => getBackgroundBorderColor(IsCameraSelected);

        /// <summary>
        /// Command to toggle the tablet choice.
        /// </summary>
        [RelayCommand]
        public void PressTabletChoice()
        {
            IsTabletSelected = !IsTabletSelected;
        }

        /// <summary>
        /// Command to toggle the stylus choice.
        /// </summary>
        [RelayCommand]
        public void PressStylusChoice()
        {
            IsStylusSelected = !IsStylusSelected;
        }

        /// <summary>
        /// Command to toggle the camera choice.
        /// </summary>
        [RelayCommand]
        public void PressCameraChoice()
        {
            IsCameraSelected = !IsCameraSelected;
        }

        /// <summary>
        /// Command to handle the continue button press.
        /// </summary>
        [RelayCommand]
        public void PressContinue()
        {
            if (Navigation is not null)
            {
                if (isCameraSelected && IsStylusSelected && IsTabletSelected)
                    Navigation.PushAsync(new CameraSelection());
                else
                    Navigation.PushAsync(new IneligableExit());
            }
        }
    }
}
