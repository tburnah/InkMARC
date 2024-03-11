using Camera.MAUI;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using Scryv.Utilities;
using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.ComponentModel;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Scryv.ViewModel
{
    internal partial class AddCameraViewModel : ObservableObject
    {
        private ObservableCollection<AvailableCameraModel> availableCameraModels = new ObservableCollection<AvailableCameraModel>();
        public ObservableCollection<AvailableCameraModel> AvailableCameraModels => availableCameraModels;

        public INavigation? Navigation { get; set; }

        public AvailableCameraModel? Selected { get; set; }

        public bool ContinueEnabled => Selected is not null;

        public CameraView? Current
        {
            get => current;
            set
            {
                current = value;
                if (!SessionContext.CameraViews.Contains(value))
                {
                    SessionContext.CameraViews.Add(value);
                }                
            }
        }

        private CameraInfo camera = null;
        public CameraInfo Camera
        {
            get => camera;
            set
            {
                SetProperty(ref camera, value);
            }
        }

        public bool CamerasVisible
        {
            get => camerasVisible;
            set => SetProperty(ref camerasVisible, value);
        }

        private bool camerasVisible;
        private CameraView? current = null;

        public ObservableCollection<CameraInfo> Cameras
        {
            get => cameras;
            set
            {
                cameras = value;
            }
        }
        private ObservableCollection<CameraInfo> cameras = new();

        public string SessionID => SessionContext.SessionID;

        public bool TextVisible => Selected?.IsPhone ?? false;

        public bool BarCodeVisibile => TextVisible;

        public bool CameraVisible => !(Selected?.IsPhone ?? true);

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
                    model = new AvailableCameraModel(null, true);
                    model.Selected += Model_Selected;
                    AvailableCameraModels.Add(model);
                });
                OnPropertyChanged(nameof(availableCameraModels));
            }
        }

        private void Model_Selected(object? sender, EventArgs e)
        {
            if (sender is AvailableCameraModel model)
            {                
                Current.AutoStartPreview = false;
                Selected = model;
                foreach (var m in AvailableCameraModels)
                {
                    if (m != model)
                        m.IsSelected = false;
                }
                if (model.CameraInfo is not null)
                {
                    Current.AutoStartPreview = false;
                    Camera = model.CameraInfo;
                    Current.AutoStartPreview = true;
                }
                OnPropertyChanged(nameof(TextVisible));
                OnPropertyChanged(nameof(CameraVisible));
                OnPropertyChanged(nameof(BarCodeVisibile));
                OnPropertyChanged(nameof(ContinueEnabled));
            }            
        }
    }
}
