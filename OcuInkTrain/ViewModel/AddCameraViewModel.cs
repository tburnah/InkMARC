using Camera.MAUI;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using OcuInkTrain.Views;
using System.Collections.ObjectModel;
using System.ComponentModel;


namespace OcuInkTrain.ViewModel
{
    /// <summary>
    /// ViewModel for adding a camera.
    /// </summary>
    internal partial class AddCameraViewModel : ObservableObject
    {
        /// <summary>
        /// Gets or sets the navigation service.
        /// </summary>
        public INavigation? Navigation { get; set; }

        /// <summary>
        /// Gets a value indicating whether the continue button is enabled.
        /// </summary>
        public bool ContinueEnabled => Selected is not null;

        #region Camera Selection

        private ObservableCollection<AvailableCameraModel> availableCameraModels = new ObservableCollection<AvailableCameraModel>();
        /// <summary>
        /// Gets the collection of available camera models.
        /// </summary>
        public ObservableCollection<AvailableCameraModel> AvailableCameraModels => availableCameraModels;

        /// <summary>
        /// Gets or sets the selected camera model.
        /// </summary>
        public AvailableCameraModel? Selected { get; set; }

        private void Model_Selected(object? sender, EventArgs e)
        {
            if (sender is AvailableCameraModel model)
            {
                Selected = model;
                foreach (var m in AvailableCameraModels)
                {
                    if (m != model)
                        m.IsSelected = false;
                }
                if (model.CameraInfo is not null)
                {
                    Camera = model.CameraInfo;
                }
                OnPropertyChanged(nameof(ContinueEnabled));
            }
        }

        #endregion

        #region CameraView Properties

        /// <summary>
        /// Gets or sets the current camera.
        /// </summary>
        public CameraInfo? Camera
        {
            get => CameraWindowViewModel.Current?.Camera ?? null;
            set
            {
                if (CameraWindowViewModel.Current is not null)
                {
                    if (CameraWindowViewModel.Current.Camera is null)
                    {
                        CameraWindowViewModel.Current.Camera = value;
                        CameraWindowViewModel.Current.AutoStartRecording = true;
                    }
                    else
                    {
                        CameraWindowViewModel.Current.Camera = value;
                        CameraWindowViewModel.Current?.RestartCameraPreview();
                    }
                }
            }
        }

        /// <summary>
        /// Gets or sets the collection of cameras.
        /// </summary>
        public ObservableCollection<CameraInfo> Cameras
        {
            get => CameraWindowViewModel.Current?.Cameras ?? new();
            set
            {
                if (CameraWindowViewModel.Current is not null)
                    CameraWindowViewModel.Current.Cameras = value;
            }
        }

        /// <summary>
        /// Sets the number of cameras.
        /// </summary>
        public int NumCameras
        {
            set
            {
                MainThread.BeginInvokeOnMainThread(() =>
                {
                    AvailableCameraModels.Clear();
                    AvailableCameraModel model;
                    if (value > 0)
                    {
                        //Camera = Cameras.First();
                        foreach (var camera in Cameras)
                        {
                            model = new AvailableCameraModel(camera, false);
                            model.Selected += Model_Selected;
                            AvailableCameraModels.Add(model);
                        }
                    }
                });
                OnPropertyChanged(nameof(availableCameraModels));
            }
        }

        #endregion

        #region Commands

        /// <summary>
        /// Command to choose continue.
        /// </summary>
        [RelayCommand]
        public async void ChooseContinue()
        {
            if (Navigation is not null)
                await Navigation.PushAsync(new ConsentPage());
        }

        /// <summary>
        /// Command to go back.
        /// </summary>
        [RelayCommand]
        public async Task Back()
        {
            if (Navigation is not null)
                await Navigation.PopAsync();
        }

        #endregion

        /// <summary>
        /// Event handler for the CameraWindow loaded event.
        /// </summary>
        public void CameraWindow_Loaded(object? sender, EventArgs e)
        {
            if (CameraWindowViewModel.Current is not null)
            {
                CameraWindowViewModel.Current.PropertyChanged += Camera_PropertyChanged;
                if (CameraWindowViewModel.Current.NumCameras > 0)
                {
                    NumCameras = CameraWindowViewModel.Current.NumCameras;
                }
            }
        }

        private void Camera_PropertyChanged(object? sender, PropertyChangedEventArgs e)
        {
            if (e.PropertyName == nameof(CameraWindowViewModel.NumCameras) && CameraWindowViewModel.Current is not null)
            {
                NumCameras = CameraWindowViewModel.Current.NumCameras;
            }
        }
    }
}
