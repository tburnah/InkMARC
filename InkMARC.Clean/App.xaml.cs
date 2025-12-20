using InkMARC.Clean.Views;
using InkMARC.Clean.ViewModels;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using System.Configuration;
using System.Data;
using System.Threading.Tasks;
using System.Windows;

namespace InkMARC.Clean
{
    /// <summary>
    /// Interaction logic for App.xaml
    /// </summary>
    public partial class App : Application
    {
#pragma warning disable CS8618 // Non-nullable field must contain a non-null value when exiting constructor. Consider adding the 'required' modifier or declaring as nullable.
        public static IHost AppHost { get; private set; }
#pragma warning restore CS8618 // Non-nullable field must contain a non-null value when exiting constructor. Consider adding the 'required' modifier or declaring as nullable.

        public App()
        {
            AppHost = Host.CreateDefaultBuilder()
                .ConfigureAppConfiguration((context, config) =>
                {
                    // Additional configuration can be set up here
                })
                .ConfigureServices((context, services) =>
                {
                    services.AddSingleton<Services.Interfaces.IVideoService, Services.VideoService>();
                    services.AddTransient<MainViewModel>();
                    services.AddTransient<MainView>(); 
                })
                .Build();
        }

        protected override async void OnExit(ExitEventArgs e)
        {
            await AppHost.StopAsync(TimeSpan.FromSeconds(5));
            AppHost.Dispose();
            base.OnExit(e);
        }

        protected override async void OnStartup(StartupEventArgs e)
        {
            await AppHost.StartAsync();            
            var main = AppHost.Services.GetRequiredService<MainView>();            
            MainWindow = main;
            main.Show();
            base.OnStartup(e);
        }
    }

}
