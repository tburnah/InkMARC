using Camera.MAUI;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Scryv.ViewModel
{
    public partial class AvailableCameraModel : ObservableObject
    {
        public event EventHandler Selected;

        public INavigation? Navigation { get; set; }

        public AvailableCameraModel(CameraInfo cameraInfo, bool isPhone)
        {
            IsPhone = isPhone;
            CameraInfo = cameraInfo;
            
            if (cameraInfo is null)
            {
                CameraName = "Phone";
                CameraImage = "phonestand.png";
            }
            else
            {
                CameraName = cameraInfo.Name;
                if (cameraInfo.Position == CameraPosition.Front)
                {
                    CameraImage = "frontcamera.png";
                }
                else if (CameraInfo.Position == CameraPosition.Back)
                {
                    CameraImage = "rearcamera.png";
                }
                else
                {
                    CameraImage = "webcam.png";
                }
            }
        }

        public bool IsPhone { get; set; }

        private string cameraName;
        public string CameraName
        {
            get => cameraName;
            set => SetProperty(ref cameraName, value);
        }

        private string cameraImage;
        public string CameraImage
        {
            get => cameraImage;
            set => SetProperty(ref cameraImage, value);
        }

        private CameraInfo cameraInfo;
        public CameraInfo CameraInfo
        {
            get => cameraInfo;
            set => SetProperty(ref cameraInfo, value);
        }

        private bool isSelected = false;
        public bool IsSelected
        {
            get => isSelected;
            set
            {
                SetProperty(ref isSelected, value);
                if (isSelected)
                {
                    Selected?.Invoke(this, EventArgs.Empty);
                }
                OnPropertyChanged(nameof(BackgroundBorderColor));
            }
        }

        public Color BackgroundBorderColor
        {
            get
            {
                if (isSelected)
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
                return Colors.Gray;
            }
        }

        [RelayCommand]
        public void Tapped()
        {
            IsSelected = !IsSelected; 
        }
    }
}
