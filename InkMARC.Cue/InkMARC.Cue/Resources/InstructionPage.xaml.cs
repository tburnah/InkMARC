using System.Reflection;
using Plugin.Maui.Audio;
using TagLib;
using File = TagLib.File;

namespace InkMARC.Cue.Resources;

public partial class InstructionPage : ContentPage
{
    private List<(string Title, string FileName)> musicList = new();
    private IAudioManager audioManager;
    private IAudioPlayer audioPlayer;
    private Stream? _currentAudioStream; // Class-level variable to keep the stream open

    public InstructionPage()
    {
        InitializeComponent();
        NavigationPage.SetHasNavigationBar(this, false);
        audioManager = AudioManager.Current;
        LoadMusic();
    }

    private async void LoadMusic()
    {
        try
        {
            using var listStream = await FileSystem.OpenAppPackageFileAsync("Resources/Music/music_list.txt");
            using var reader = new StreamReader(listStream);

            while (!reader.EndOfStream)
            {
                var fileName = reader.ReadLine()?.Trim();
                if (string.IsNullOrWhiteSpace(fileName)) continue;

                var musicPath = Path.Combine("Resources/Music", fileName);

                try
                {
                    using var stream = await FileSystem.OpenAppPackageFileAsync(musicPath);
                    using var memReadStream = new MemoryStream();
                    await stream.CopyToAsync(memReadStream);
                    memReadStream.Position = 0;

                    // Create a separate write stream
                    using var memWriteStream = new MemoryStream();

                    using (var tagFile = TagLib.File.Create(new StreamFileAbstraction(fileName, memReadStream, memWriteStream)))
                    {
                        var title = tagFile.Tag.Title ?? Path.GetFileNameWithoutExtension(fileName);
                        musicList.Add((title, fileName));
                        Console.WriteLine($"Loaded music: {title} ({fileName})");
                    }
                }
                catch (Exception fileEx)
                {
                    Console.WriteLine($"Error loading file {fileName}: {fileEx.Message}");
                    // Optionally, display an alert or log the error
                }
            }

            musicPicker.ItemsSource = musicList.Select(m => m.Title).ToList();
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Error in LoadMusic: {ex.Message}");
            await DisplayAlert("Error", "Failed to load music list.", "OK");
        }
    }

    private async void OnPlayClicked(object sender, EventArgs e)
    {
        try
        {
            if (musicPicker.SelectedIndex < 0) return;

            if (audioPlayer != null && audioPlayer.IsPlaying)
            {
                audioPlayer?.Stop();
                audioPlayer?.Dispose();
                _currentAudioStream?.Dispose();
                _currentAudioStream = null;
                playButton.Text = "▶ Preview";
            }
            else
            {
                playButton.Text = "■ Stop";
                var (_, fileName) = musicList[musicPicker.SelectedIndex];

                // Keep the stream open by not disposing it immediately
                _currentAudioStream = await FileSystem.OpenAppPackageFileAsync(Path.Combine("Resources/Music", fileName));
                audioPlayer = audioManager.CreatePlayer(_currentAudioStream);
                audioPlayer.Play();

                // Subscribe to the PlaybackEnded event to dispose resources
                audioPlayer.PlaybackEnded += AudioPlayer_OnPlaybackEnded;
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Error in OnPlayClicked: {ex.Message}");
            await DisplayAlert("Error", "Failed to play audio.", "OK");
        }
    }

    private void AudioPlayer_OnPlaybackEnded(object sender, EventArgs e)
    {
        audioPlayer?.Dispose();
        audioPlayer = null;
        _currentAudioStream?.Dispose();
        _currentAudioStream = null;
        playButton.Text = "▶ Preview";
    }

    private void OnMusicSelected(object sender, EventArgs e)
    {
        playButton.IsEnabled = musicPicker.SelectedIndex >= 0;
    }

    private async void OnBeginClicked(object sender, EventArgs e)
    {
        try
        {
            // Optional: Save the selected music choice to pass into the main page
            var selectedMusic = musicList[musicPicker.SelectedIndex == -1 ? 0 : musicPicker.SelectedIndex].FileName;
            await Navigation.PushAsync(new MainPage(selectedMusic));
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Error in OnBeginClicked: {ex.Message}");
            await DisplayAlert("Error", "Failed to navigate to the main page.", "OK");
        }
    }
}

public class StreamFileAbstraction : TagLib.File.IFileAbstraction
{
    private readonly Stream _readStream;
    private readonly Stream _writeStream;

    public StreamFileAbstraction(string name, Stream readStream, Stream writeStream)
    {
        Name = name;
        _readStream = readStream;
        _writeStream = writeStream;
    }

    public string Name { get; }

    public Stream ReadStream => _readStream;

    public Stream WriteStream => _writeStream;

    public void CloseStream(Stream stream) => stream.Dispose();
}
