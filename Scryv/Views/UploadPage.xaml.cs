#if ANDROID
using Android.Content;
#endif
using Scryv.Utilities;
#if WINDOWS
using System.Diagnostics;
#endif

namespace Scryv.Views;

/// <summary>
/// Final page of the app.
/// </summary>
public partial class UploadPage : ContentPage
{
    /// <summary>
    /// Initializes a new instance of the <see cref="UploadPage"/> class.
    /// </summary>
	public UploadPage()
	{
		InitializeComponent();
	}

    private void Exit_Clicked(object sender, EventArgs e)
    {
        Application.Current?.Quit();
    }

#pragma warning disable CS1998 // Async method lacks 'await' operators and will run synchronously
    private async void ContentPage_Loaded(object sender, EventArgs e)
#pragma warning restore CS1998 // Async method lacks 'await' operators and will run synchronously
    {
        //AzureStorageService azureStorageService = new AzureStorageService();

        //string dataDirectory = "data";
        //string[] dataFiles = Directory.GetFiles(FileSystem.AppDataDirectory);

        //foreach (string file in dataFiles)
        //{
        //    try
        //    {
        //        FileInfo fileInfo = new FileInfo(file);
        //        using (FileStream fs = new FileStream(file, FileMode.Open))
        //        {
        //            await azureStorageService.UploadFileAsync("uploadedfiles", fileInfo.Name, fs);
        //        }
        //    }
        //    catch (Exception ex)
        //    {
        //        Console.WriteLine(ex.Message);
        //    }
        //}
    }

    private void Button_Clicked(object sender, EventArgs e)
    {
#if WINDOWS
        Process.Start("explorer.exe", DataUtilities.GetDataFolder());
#elif ANDROID
        Intent intent = new Intent(Intent.ActionView);
        intent.SetDataAndType(Android.Net.Uri.Parse(DataUtilities.GetDataFolder()), "resource/folder");
        intent.SetFlags(ActivityFlags.NewTask);
        Android.App.Application.Context.StartActivity(intent);
#endif
    }
}