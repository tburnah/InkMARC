using InkMARC.Evaluate;

namespace InkMARC.Evaluate
{
    public partial class App : Application
    {
        public App()
        {
            InitializeComponent();

            MainPage = new NavigationPage(new MainPage());
        }        
    }
}