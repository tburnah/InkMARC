using CommunityToolkit.Maui.Core;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using Scryv.Views;
using Scryv.Views.Popups;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Scryv.ViewModel
{
    public partial class StartPageViewModel : ObservableObject
    {
        private bool isTabletSelected;
        private bool isStylusSelected;
        private bool isCameraSelected;

        private IPopupService popupService;

        public INavigation? Navigation { get; set; }

        public StartPageViewModel()
        {
            this.popupService = DependencyService.Get<IPopupService>();
        }

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
                if (Application.Current.Resources.TryGetValue("Selected", out var activeColor))
                {
                    return (Color)activeColor;
                }
                return Colors.GreenYellow;
            }
            if (Application.Current.Resources.TryGetValue("White", out var inactiveColor))
            {
                return (Color)inactiveColor;
            }
            return Colors.White;
        }

        private Color getBackgroundBorderColor(bool value)
        {
            if (value)
            {
                if (Application.Current.Resources.TryGetValue("SelectionBorder", out var activeColor))
                {
                    return (Color)activeColor;
                }
                return Colors.GreenYellow;
            }
            if (Application.Current.Resources.TryGetValue("UnselectedBorder", out var inactiveColor))
            {
                return (Color)inactiveColor;
            }
            return Colors.White;
        }

        public string TabletChoiceImage => IsTabletSelected ? "check.png" : "tablet.png";

        public string StylusChoiceImage => IsStylusSelected ? "check.png" : "stylus.png";

        public string CameraChoiceImage => IsCameraSelected ? "check.png" : "webcam.png";

        public Color TabletChoiceColor => getBackgroundColor(IsTabletSelected);

        public Color StylusChoiceColor => getBackgroundColor(IsStylusSelected);
        
        public Color CameraChoiceColor => getBackgroundColor(IsCameraSelected);

        public Color TabletChoiceBorderColor => getBackgroundBorderColor(IsTabletSelected);

        public Color StylusChoiceBorderColor => getBackgroundBorderColor(IsStylusSelected);

        public Color CameraChoiceBorderColor => getBackgroundBorderColor(IsCameraSelected);

        [RelayCommand]
        public void PressTabletChoice()
        {
            IsTabletSelected = !IsTabletSelected;
        }

        [RelayCommand]
        public void PressStylusChoice()
        {
            IsStylusSelected = !IsStylusSelected;
        }

        [RelayCommand]
        public void PressCameraChoice()
        {
            IsCameraSelected = !IsCameraSelected;
        }

        [RelayCommand]
        public void PressContinue()
        {
            if (isCameraSelected && IsStylusSelected && IsTabletSelected)
                Navigation.PushAsync(new CameraSelection());
            else
                Navigation.PushAsync(new IneligableExit());                
        }
    }
}
