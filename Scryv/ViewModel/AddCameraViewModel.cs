using Camera.MAUI;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using Scryv.Utilities;
using Scryv.Views;
using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.ComponentModel;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Timers;

namespace Scryv.ViewModel
{
    internal partial class AddCameraViewModel : ObservableObject
    {
        public INavigation? Navigation { get; set; }

        public bool ContinueEnabled => Selected is not null;

        #region Camera Selection

        private ObservableCollection<AvailableCameraModel> availableCameraModels = new ObservableCollection<AvailableCameraModel>();
        public ObservableCollection<AvailableCameraModel> AvailableCameraModels => availableCameraModels;

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
        
        public ObservableCollection<CameraInfo> Cameras
        {
            get => CameraWindowViewModel.Current?.Cameras ?? new();
            set
            {
                if (CameraWindowViewModel.Current is not null)
                    CameraWindowViewModel.Current.Cameras = value;                
            }
        }        

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

        [RelayCommand]
        public async void ChooseContinue()
        {
            Navigation.PushAsync(new DrawingPage());
        }

        [RelayCommand]
        public void Back()
        {
            Navigation.PopAsync();
        }

        #endregion

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
            if (e.PropertyName == nameof(CameraWindowViewModel.NumCameras))
            {
                NumCameras = CameraWindowViewModel.Current.NumCameras;
            }            
        }
    }
}
