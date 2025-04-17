using Plugin.Maui.Audio;
using System;
using System.Threading.Tasks;

namespace InkMARC.Evaluate
{
    public partial class MainPage : ContentPage
    {
        private float threshold = 0.5f;
        private bool isCalibrated = false;

        public MainPage()
        {
            InitializeComponent();
            StartCalibrationAsync();
        }

        private async void StartCalibrationAsync()
        {
            var touchValues = new List<float>();
            var noTouchValues = new List<float>();

            // Prompt: Pen Down
            await DisplayAlert("Calibration", "Place the stylus on the surface and act like you're writing. Then tap OK.", "OK");
            touchValues = await CaptureInferenceSamples(10);

            // Prompt: Pen Up
            await DisplayAlert("Calibration", "Lift the stylus off the surface. Then tap OK.", "OK");
            noTouchValues = await CaptureInferenceSamples(10);

            // Calculate threshold
            float avgTouch = touchValues.Average();
            float avgNoTouch = noTouchValues.Average();
            threshold = (avgTouch + avgNoTouch) / 2f;
            isCalibrated = true;

            await DisplayAlert("Calibration Complete", $"Threshold set to {threshold:F2}", "OK");

            // Normal inference after calibration
            cameraPreview.OnInferenceResult = (result) =>
            {
                string status = result > threshold ? "Pen Down" : "Pen Up";
                MainThread.BeginInvokeOnMainThread(() =>
                {
                    screenPrompt.Text = $"{result}, {status}";
                });
            };
        }

        private Task<List<float>> CaptureInferenceSamples(int sampleCount)
        {
            var tcs = new TaskCompletionSource<List<float>>();
            var results = new List<float>();

            // Temporarily override inference handler to collect samples
            cameraPreview.OnInferenceResult = (result) =>
            {
                results.Add(result);
                if (results.Count >= sampleCount)
                {
                    tcs.TrySetResult(results);
                }
            };

            return tcs.Task;
        }
    }


    //public partial class MainPage : ContentPage
    //{
    //    public MainPage()
    //    {
    //        InitializeComponent();

    //        cameraPreview.OnInferenceResult = async (result) =>
    //        {
    //            screenPrompt.Text = $"Result: {result}";
    //        };
    //    }
    //}
}
