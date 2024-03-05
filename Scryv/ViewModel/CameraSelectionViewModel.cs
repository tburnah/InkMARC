using Camera.MAUI;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using Scryv.Utilities;
using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Scryv.ViewModel
{
    internal partial class CameraSelectionViewModel : ObservableObject
    {
        public event EventHandler<CameraSelectionViewModel>? RemoveMe;

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
        public ObservableCollection<string> CameraTypes { get; } = new ObservableCollection<string>
        {
            "WebCam",
            "Phone"
        };

        public string SelectedCameraType 
        { 
            get => selectedCameraType; 
            set
            {
                SetProperty(ref selectedCameraType, value);

                if (value == "Phone")
                {
                    camerasVisible = false;
                    OnPropertyChanged(nameof(CamerasVisible));
                    OnPropertyChanged(nameof(PhoneVisible));
                }
                else if (value == "WebCam")
                {
                    camerasVisible = true;
                    OnPropertyChanged(nameof(PhoneVisible));
                    OnPropertyChanged(nameof(CamerasVisible));
                }
                else
                {
                    camerasVisible = false;
                    OnPropertyChanged(nameof(CamerasVisible));
                    OnPropertyChanged(nameof(PhoneVisible));
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

        public bool PhoneVisible => !camerasVisible && selectedCameraType == "Phone";

        private ObservableCollection<CameraInfo> cameras = new();        
        private string selectedCameraType = string.Empty;
        private string buttonText = "Start";
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

        public string SessionID => SessionContext.SessionID;

        public int NumCameras
        {
            set
            {
                if (value > 0)
                    Camera = Cameras.First();
            }
        }

        public string ButtonText
        {
            get => buttonText;
            set => SetProperty(ref buttonText, value);
        }

        [RelayCommand]
        private void StartCommand(object args)
        {
            if (args is CameraView cameraView)
            {
                if (ButtonText == "Stop")
                {
                    cameraView.StopCameraAsync();
                    ButtonText = "Start";                    
                }
                else
                {
                    cameraView.StartCameraAsync();
                    ButtonText = "Stop";
                }
            }
        }

        [RelayCommand]
        private void Remove()
        {
            if (current is not null && SessionContext.CameraViews.Contains(current))
            {
                SessionContext.CameraViews.Remove(current);
            }
            RemoveMe?.Invoke(this, this);
        }
    }
}
