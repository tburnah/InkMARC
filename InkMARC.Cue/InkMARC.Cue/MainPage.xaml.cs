using Plugin.Maui.Audio;
using System;
using System.Threading.Tasks;

namespace InkMARC.Cue
{
    public partial class MainPage : ContentPage
    {
        private const int InitialWait = 3000;
        private const int CountdownDuration = 1000;
        private const int MinTouchDuration = 5000;
        private const int MaxTouchDuration = 10001;
        private const int RecordingDelay = 2000;
        private const int CountdownStart = 3;

        private readonly string _selectedMusic;
        private readonly IAudioManager _audioManager;
        private IAudioPlayer _audioPlayer;
        private readonly Random _random = new();
        private int videoIndex = 1;
        private string sessionID = SessionIDUtilities.GetUniqueSessionID();

        public MainPage(string selectedMusic)
        {
            InitializeComponent();
            _selectedMusic = selectedMusic;
            _audioManager = AudioManager.Current;

            Loaded += async (s, e) => await StartMusicSequence();
        }

        private async Task StartMusicSequence()
        {
            if (string.IsNullOrWhiteSpace(_selectedMusic))
            {
                screenPrompt.Text = "No music selected.";
                return;
            }

            try
            {
                var stream = await FileSystem.OpenAppPackageFileAsync(System.IO.Path.Combine("Resources/Music", _selectedMusic));
                _audioPlayer = _audioManager.CreatePlayer(stream);

                await Task.Delay(InitialWait); // Initial wait

                for (int i = 0; i < 5; i++)
                {
                    // Countdown before music starts
                    await ShowCountdown("Pen touching in", "Pen Touching");

                    _audioPlayer.Play();

                    var touchDuration = _random.Next(MinTouchDuration, MaxTouchDuration); // 5–10 sec
                    await Task.Delay(RecordingDelay); // Wait 2s before recording
                    await StartRecording($"Touched_{i + 1}");
                    await Task.Delay(touchDuration - InitialWait); // Remaining duration - 1s before stop

                    await StopRecording();

                    await ShowCountdown("Pen not touching in", "Pen not touching");

                    _audioPlayer.Pause();

                    var noTouchDuration = _random.Next(MinTouchDuration, MaxTouchDuration); // 5–10 sec
                    await Task.Delay(RecordingDelay);
                    await StartRecording($"NoTouched_{i + 1}");
                    await Task.Delay(noTouchDuration - InitialWait);
                    await StopRecording();
                }

                screenPrompt.Text = "Done.";
            }
            catch (Exception ex)
            {
                System.Diagnostics.Debug.WriteLine($"StartMusicSequence Exception: {ex.Message}");
                screenPrompt.Text = "An error occurred.";
            }
        }

        private async Task StartRecording(string baseName)
        {
            string fileName = $"{baseName}_{sessionID}_{videoIndex++}.mp4";
            string fullPath = Path.Combine(FileSystem.AppDataDirectory, fileName);

#if ANDROID
            try
            {
                var publicMovies = Android.OS.Environment.GetExternalStoragePublicDirectory(Android.OS.Environment.DirectoryMovies);
                fullPath = Path.Combine(publicMovies.AbsolutePath, fileName);
                var handler = cameraPreview.Handler as InkMARC.Cue.Platforms.Android.CameraPreviewHandler;
                await MainThread.InvokeOnMainThreadAsync(() =>
                {
                    handler?.StartRecording(fullPath);
                });
            }
            catch (Exception ex)
            {
                System.Diagnostics.Debug.WriteLine($"StartRecording Exception: {ex.Message}");
            }
#endif

            await Task.CompletedTask;
        }

        private async Task StopRecording()
        {
#if ANDROID
            try
            {
                var handler = cameraPreview.Handler as InkMARC.Cue.Platforms.Android.CameraPreviewHandler;
                await MainThread.InvokeOnMainThreadAsync(() =>
                {
                    handler?.StopRecording();
                });
            }
            catch (Exception ex)
            {
                System.Diagnostics.Debug.WriteLine($"StopRecording Exception: {ex.Message}");
            }
#endif

            await Task.CompletedTask;
        }

        private Task ShowCountdown2(string messagePrefix, string finalMessage)
        {
            var tcs = new TaskCompletionSource();

            int counter = CountdownStart;
            Dispatcher.StartTimer(TimeSpan.FromMilliseconds(CountdownDuration), () =>
            {
                if (counter > 0)
                {
                    screenPrompt.Text = $"{messagePrefix} {counter--}";
                    return true; // Continue timer
                }
                else
                {
                    screenPrompt.Text = finalMessage;
                    tcs.SetResult();
                    return false; // Stop timer
                }
            });

            return tcs.Task;
        }

        private async Task ShowCountdown(string messagePrefix, string finalMessage)
        {
            await ShowCountdown2(messagePrefix, finalMessage);
        }
    }

}
