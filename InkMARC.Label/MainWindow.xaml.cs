﻿using System.Text;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Navigation;
using System.Windows.Shapes;

namespace InkMARC.Label
{
    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    public partial class MainWindow : Window
    {
        private MainViewViewModel viewModel;
        public MainWindow()
        {
            InitializeComponent();
            viewModel = DataContext as MainViewViewModel;
        }

        private void TextBox_PreviewTextInput(object sender, System.Windows.Input.TextCompositionEventArgs e)
        {
            e.Handled = !float.TryParse(((TextBox)sender).Text + e.Text, out _);
        }

        private void Window_PreviewKeyDown(object sender, KeyEventArgs e)
        {
            if (viewModel == null)
                return;

            switch (e.Key)
            {
                case Key.Space:
                    viewModel.ToggleTouchedCommand?.Execute(null);
                    e.Handled = true;
                    break;
            }
        }
    }
}