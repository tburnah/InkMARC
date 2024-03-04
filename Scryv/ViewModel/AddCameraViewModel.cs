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
    internal partial class AddCameraViewModel : ObservableObject
    {
        private System.Timers.Timer timer = new System.Timers.Timer(100);

        private ObservableCollection<CameraSelectionViewModel> cameras = new ObservableCollection<CameraSelectionViewModel>();

        public ObservableCollection<CameraSelectionViewModel> Cameras { get => cameras; set => cameras = value; }

        public AddCameraViewModel()
        {
            timer.Elapsed += Timer_Elapsed;
            timer.AutoReset = false;
        }

        private void Timer_Elapsed(object? sender, System.Timers.ElapsedEventArgs e)
        {
            MainThread.InvokeOnMainThreadAsync(() =>
            {
                CameraSelectionViewModel newCamera = new();
                newCamera.RemoveMe += RemoveCamera;
                Cameras.Add(newCamera);
                OnPropertyChanged(nameof(ContinueEnabled));
            });            
        }

        [RelayCommand]
        public void AddCamera()
        {
            addCameraSpinnerVisible = true;
            OnPropertyChanged(nameof(AddCameraSpinnerVisible));
            timer.Start();
        }

        public void CameraLoaded()
        {
            addCameraSpinnerVisible = false;
            OnPropertyChanged(nameof(AddCameraSpinnerVisible));
        }

        public bool ContinueEnabled => cameras.Count > 0;

        private bool addCameraSpinnerVisible = false;
        public bool AddCameraSpinnerVisible => addCameraSpinnerVisible;

        public void RemoveCamera(object sender, CameraSelectionViewModel camera)
        {
            camera.RemoveMe -= RemoveCamera;
            Cameras.Remove(camera);
            OnPropertyChanged(nameof(ContinueEnabled));
        }
    }
}
