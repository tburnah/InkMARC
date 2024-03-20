using Camera.MAUI;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;

namespace Scryv.ViewModel
{
    /// <summary>
    /// Represents the view model for an available camera.
    /// </summary>
    public partial class AvailableCameraModel : ObservableObject
    {
        /// <summary>
        /// Event that is raised when the camera is selected.
        /// </summary>
        public event EventHandler? Selected;

        /// <summary>
        /// Gets or sets the navigation service.
        /// </summary>
        public INavigation? Navigation { get; set; }

        /// <summary>
        /// Initializes a new instance of the <see cref="AvailableCameraModel"/> class.
        /// </summary>
        /// <param name="cameraInfo">The camera information.</param>
        /// <param name="isPhone">A value indicating whether the camera is a phone camera.</param>
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

        /// <summary>
        /// Gets or sets a value indicating whether the camera is a phone camera.
        /// </summary>
        public bool IsPhone { get; set; }

        private string? cameraName;
        /// <summary>
        /// Gets or sets the name of the camera.
        /// </summary>
        public string? CameraName
        {
            get => cameraName;
            set => SetProperty(ref cameraName, value);
        }

        private string? cameraImage;
        /// <summary>
        /// Gets or sets the image of the camera.
        /// </summary>
        public string? CameraImage
        {
            get => cameraImage;
            set => SetProperty(ref cameraImage, value);
        }

        private CameraInfo? cameraInfo;
        /// <summary>
        /// Gets or sets the camera information.
        /// </summary>
        public CameraInfo? CameraInfo
        {
            get => cameraInfo;
            set => SetProperty(ref cameraInfo, value);
        }

        private bool isSelected = false;
        /// <summary>
        /// Gets or sets a value indicating whether the camera is selected.
        /// </summary>
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

        /// <summary>
        /// Gets the background border color of the camera.
        /// </summary>
        public Color BackgroundBorderColor
        {
            get
            {
                if (isSelected)
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
                return Colors.Gray;
            }
        }

        /// <summary>
        /// Command that is executed when the camera is tapped.
        /// </summary>
        [RelayCommand]
        public void Tapped()
        {
            IsSelected = !IsSelected;
        }
    }
}
