using InkMARC.Cue.Resources;

namespace InkMARC.Cue
{
    public partial class App : Application
    {
        public App()
        {
            InitializeComponent();

            MainPage = new NavigationPage(new InfoConsentPage());
        }        
    }
}