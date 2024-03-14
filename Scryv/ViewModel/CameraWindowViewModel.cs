using CommunityToolkit.Mvvm.ComponentModel;
using Camera.MAUI;
using System.Collections.ObjectModel;
using System.Timers;
using Scryv.Utilities;

namespace Scryv.ViewModel
{
    public class CameraWindowViewModel : ObservableObject
    {
        public static CameraWindowViewModel Current { get; set; }

        private ObservableCollection<AvailableCameraModel> availableCameraModels = new ObservableCollection<AvailableCameraModel>();
        public ObservableCollection<AvailableCameraModel> AvailableCameraModels => availableCameraModels;

        private System.Timers.Timer cameraTimer;
        private bool leaving = false;

        public CameraWindowViewModel()
        {
            cameraTimer = new System.Timers.Timer(500);
            cameraTimer.AutoReset = false;
            cameraTimer.Elapsed += CameraTimer_Elapsed;

            StartCamera = new Command(() =>
            {
                AutoStartPreview = true;
                OnPropertyChanged(nameof(AutoStartPreview));
            });
            StopCamera = new Command(() =>
            {
                AutoStartPreview = false;
                OnPropertyChanged(nameof(AutoStartPreview));
            });
            RecordingFile = DataUtilities.GetVideoFileName(DateTime.Now.ToFileTime(), 0);

            OnPropertyChanged(nameof(RecordingFile));

            StartRecording = new Command(() =>
            {
                AutoStartRecording = true;
                OnPropertyChanged(nameof(AutoStartRecording));
            });
            StopRecording = new Command(() =>
            {
                AutoStartRecording = false;
                OnPropertyChanged(nameof(AutoStartRecording));
            });
            OnPropertyChanged(nameof(StartCamera));
            OnPropertyChanged(nameof(StopCamera));
            OnPropertyChanged(nameof(StartRecording));
            OnPropertyChanged(nameof(StopRecording));
            Current = this;
        }

        public CameraInfo Camera
        {
            get => camera;
            set
            {
                SetProperty(ref camera, value);
            }
        }
        private CameraInfo camera = null;
        public ObservableCollection<CameraInfo> Cameras
        {
            get => cameras;
            set
            {
                cameras = value;
            }
        }
        private ObservableCollection<CameraInfo> cameras = new();

        private int numCameras = 0;
        private bool autoStartRecording = false;

        public int NumCameras
        {
            get => numCameras;
            set => SetProperty(ref numCameras, value);
        }

        public bool AutoStartPreview { get; set; } = false;
        public bool AutoStartRecording { 
            get => autoStartRecording; 
            set => SetProperty(ref autoStartRecording, value); 
        }
        private void CameraTimer_Elapsed(object? sender, ElapsedEventArgs e)
        {
            AutoStartRecording = true;
            OnPropertyChanged(nameof(AutoStartRecording));
        }

        public Command StartCamera { get; set; }
        public Command StopCamera { get; set; }        
        public string RecordingFile { get; set; }
        public Command StartRecording { get; set; }
        public Command StopRecording { get; set; }

        public void RestartCameraPreview(string recordingFile = "")
        {
            if (!string.IsNullOrEmpty(recordingFile))
            {
                RecordingFile = recordingFile;
                OnPropertyChanged(nameof(RecordingFile));
            }
            AutoStartRecording = false;
            OnPropertyChanged(nameof(AutoStartRecording));
        }

        public void CamView_FinishedStarting(object sender, EventArgs e)
        {

        }

        public void CamView_FinishedStopping(object sender, EventArgs e)
        {
            cameraTimer.Start();
        }
    }
}
