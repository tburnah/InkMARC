using Microsoft.Extensions.DependencyInjection;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Shapes;

namespace InkMARC.Label.Views
{
    /// <summary>
    /// Interaction logic for MainView.xaml
    /// </summary>
    public partial class MainView : Window
    {
        public MainView()
        {
            InitializeComponent();

            MainContent.Content = CreateLocationControl();
        }

        private LocationLabelling CreateLocationControl()
        {
            LocationLabelling control = new();
            control.DataContext = App.AppHost.Services.GetRequiredService<LocationLabellingViewModel>();
            return control;
        }

        private void NavList_SelectionChanged(object sender, SelectionChangedEventArgs e)
        {
            if (e.AddedItems.Count == 0) return;

            var selected = ((ListBoxItem)e.AddedItems[0])?.Tag as string;

            switch (selected)
            {
                case "main":
                    MainContent.Content = CreateLocationControl();
                    break;
                case "second":
                    MainContent.Content = new TouchLabelling();
                    break;
            }
        }

        private void Window_PreviewKeyDown(object sender, KeyEventArgs e)
        {
            if (MainContent.Content != null)
            {
                if (MainContent.Content is LocationLabelling locationLabelling)
                {
                    locationLabelling.ExternalKeyPressPreview(sender, e);
                    e.Handled = true;
                }                
            }
        }
    }
}
