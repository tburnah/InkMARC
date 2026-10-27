namespace Camera.MAUI;

public class CameraInfo
{
    public string Name { get; internal set; }
    public string DeviceId { get; internal set; }
    public CameraPosition Position { get; internal set; }
    public bool HasFlashUnit { get; internal set; }
    public float MinZoomFactor { get; internal set; }
    public float MaxZoomFactor { get; internal set; }
    public float HorizontalViewAngle { get; internal set; }
    public float VerticalViewAngle { get; internal set; }

    public List<Size> AvailableResolutions { get; internal set; }
    public Size SelectedResolution { get; internal set; } = new Size(640, 480);
    public string EncodingQuality { get; internal set; } = "VGA";
    public override string ToString()
    {
        return Name;
    }
}
