using Plugin.Maui.Audio;
using System;
using System.Threading.Tasks;

namespace InkMARC.Evaluate
{
    public partial class MainPage : ContentPage
    {
        public MainPage()
        {
            InitializeComponent();

            cameraPreview.OnInferenceResult = async (result) =>
            {
                screenPrompt.Text = $"Result: {result}";
            };
        }
    }
}
