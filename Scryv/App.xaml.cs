
using Scryv.Views;

namespace Scryv
{
    /// <summary>
    /// Scryv - Stylus capture and recording yielding video
    /// </summary>
    public partial class App : Application
    {
        public App()
        {
            InitializeComponent();

            MainPage = new NavigationPage(new StartPage());
        }
    }
}
