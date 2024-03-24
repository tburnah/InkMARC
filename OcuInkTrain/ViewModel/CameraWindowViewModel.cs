using Camera.MAUI;
using CommunityToolkit.Mvvm.ComponentModel;
using OcuInkTrain.Utilities;
using System.Collections.ObjectModel;
using System.Timers;

namespace OcuInkTrain.ViewModel
{
    /// <summary>
    /// Represents the view model for the camera window.
    /// </summary>
    public class CameraWindowViewModel : ObservableObject
    {
        /// <summary>
        /// Gets or sets the current instance of the <see cref="CameraWindowViewModel"/>.
        /// </summary>
        public static CameraWindowViewModel? Current { get; set; }

        private ObservableCollection<AvailableCameraModel> availableCameraModels = new ObservableCollection<AvailableCameraModel>();
        /// <summary>
        /// Gets the collection of available camera models.
        /// </summary>
        public ObservableCollection<AvailableCameraModel> AvailableCameraModels => availableCameraModels;

        private readonly System.Timers.Timer cameraTimer;

        /// <summary>
        /// Initializes a new instance of the <see cref="CameraWindowViewModel"/> class.
        /// </summary>
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

        private CameraInfo? camera = null;
        /// <summary>
        /// Gets or sets the current camera.
        /// </summary>
        public CameraInfo? Camera
        {
            get => camera;
            set
            {
                SetProperty(ref camera, value);
            }
        }

        private ObservableCollection<CameraInfo> cameras = new();
        /// <summary>
        /// Gets or sets the collection of cameras.
        /// </summary>
        public ObservableCollection<CameraInfo> Cameras
        {
            get => cameras;
            set
            {
                cameras = value;
            }
        }

        private int numCameras = 0;
        private bool autoStartRecording = false;

        /// <summary>
        /// Gets or sets the number of cameras.
        /// </summary>
        public int NumCameras
        {
            get => numCameras;
            set => SetProperty(ref numCameras, value);
        }

        /// <summary>
        /// Gets or sets a value indicating whether the camera preview should automatically start.
        /// </summary>
        public bool AutoStartPreview { get; set; } = false;

        /// <summary>
        /// Gets or sets a value indicating whether the camera recording should automatically start.
        /// </summary>
        public bool AutoStartRecording
        {
            get => autoStartRecording;
            set => SetProperty(ref autoStartRecording, value);
        }

        private void CameraTimer_Elapsed(object? sender, ElapsedEventArgs e)
        {
            AutoStartRecording = true;
            OnPropertyChanged(nameof(AutoStartRecording));
        }

        /// <summary>
        /// Gets or sets the command to start the camera.
        /// </summary>
        public Command StartCamera { get; set; }

        /// <summary>
        /// Gets or sets the command to stop the camera.
        /// </summary>
        public Command StopCamera { get; set; }

        /// <summary>
        /// Gets or sets the recording file.
        /// </summary>
        public string RecordingFile { get; set; }

        /// <summary>
        /// Gets or sets the command to start recording.
        /// </summary>
        public Command StartRecording { get; set; }

        /// <summary>
        /// Gets or sets the command to stop recording.
        /// </summary>
        public Command StopRecording { get; set; }

        /// <summary>
        /// Restarts the camera preview.
        /// </summary>
        /// <param name="recordingFile">The recording file to set.</param>
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

        /// <summary>
        /// Event handler for when the camera view finishes starting.
        /// </summary>
        /// <param name="sender">The sender of the event.</param>
        /// <param name="e">The event arguments.</param>
        public void CamView_FinishedStarting(object sender, EventArgs e)
        {

        }

        /// <summary>
        /// Event handler for when the camera view finishes stopping.
        /// </summary>
        /// <param name="sender">The sender of the event.</param>
        /// <param name="e">The event arguments.</param>
        public void CamView_FinishedStopping(object sender, EventArgs e)
        {
            cameraTimer.Start();
        }
    }
}
