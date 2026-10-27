
using InkMARC.Probe.Views;

namespace InkMARC.Probe
{
    /// <summary>
    /// InkMARC.Probe - Stylus capture and recording yielding video
    /// </summary>
    public partial class App : Application
    {
        /// <summary>
        /// Creates the App.
        /// </summary>
        public App()
        {
            InitializeComponent();

            MainPage = new NavigationPage(new ConsentPage());            
        }
    }
}
