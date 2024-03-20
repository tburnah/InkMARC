using OcuInkTrain.Utilities;
using OcuInkTrain.ViewModel;

namespace OcuInkTrain.Views;

/// <summary>
/// Represents the camera window view.
/// </summary>
public partial class CameraWindow : ContentPage
{
    /// <summary>
    /// Initializes a new instance of the <see cref="CameraWindow"/> class.
    /// </summary>
    public CameraWindow()
    {
        InitializeComponent();
    }

    /// <summary>
    /// Handles the event when the camera view finishes starting.
    /// </summary>
    /// <param name="sender">The sender object.</param>
    /// <param name="e">The event arguments.</param>
    private void CamView_FinishedStarting(object sender, EventArgs e)
    {
        if (CameraWindowViewModel.Current is not null)
        {
            CameraWindowViewModel.Current.CamView_FinishedStarting(sender, e);
        }
    }

    /// <summary>
    /// Handles the event when the camera view finishes stopping.
    /// </summary>
    /// <param name="sender">The sender object.</param>
    /// <param name="e">The event arguments.</param>
    private void CamView_FinishedStopping(object sender, EventArgs e)
    {
        if (CameraWindowViewModel.Current is not null)
        {
            CameraWindowViewModel.Current.CamView_FinishedStopping(sender, e);
        }
    }

    private double currentX, currentY;

    /// <summary>
    /// Handles the pan updated event.
    /// </summary>
    /// <param name="sender">The sender object.</param>
    /// <param name="e">The pan updated event arguments.</param>
    private void OnPanUpdated(object sender, PanUpdatedEventArgs e)
    {
        if (e.StatusType == GestureStatus.Started)
        {
            currentX = SessionContext.CameraWin?.X ?? this.X;
            currentY = SessionContext.CameraWin?.Y ?? this.Y;
        }
        if (e.StatusType == GestureStatus.Running)
        {
            if (SessionContext.CameraWin is null)
                return;

            // Calculate the new position based on the pan gesture
            double newX = currentX + e.TotalX;
            double newY = currentY + e.TotalY;

            // Update the position of the main window
            SessionContext.CameraWin.X = newX;
            SessionContext.CameraWin.Y = newY;
        }
    }
}
