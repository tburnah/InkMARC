using Camera.MAUI;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
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
        public CameraView? Current { get; set; } = null;

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

        public ObservableCollection<CameraInfo> Cameras
        {
            get => cameras;
            set
            {
                cameras = value;
            }
        }

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
    }
}
