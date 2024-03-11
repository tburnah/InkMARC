using Scryv.Views;

namespace Scryv
{
    public partial class AppShell : Shell
    {
        public AppShell()
        {
            InitializeComponent();
            Routing.RegisterRoute(nameof(IneligableExit), typeof(IneligableExit));
        }
    }
}
