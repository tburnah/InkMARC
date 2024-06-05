
using OcuInkTrain.Views;

namespace OcuInkTrain
{
    /// <summary>
    /// OcuInk Train - Stylus capture and recording yielding video
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
