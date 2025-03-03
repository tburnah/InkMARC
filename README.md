# InkMARC Label

## Prerequisites

To run this solution, ensure you have the following prerequisites installed:

1. **.NET 9 SDK**: This project targets .NET 9. You can download the .NET 9 SDK from the [official .NET website](https://dotnet.microsoft.com/download/dotnet/9.0).

2. **Visual Studio 2022**: It is recommended to use Visual Studio 2022 for development. Ensure you have the latest version installed. You can download it from the [Visual Studio website](https://visualstudio.microsoft.com/vs/).

3. **HDF5CSharp Library**: This project uses the HDF5CSharp library for handling HDF5 files. You can install it via NuGet:
   
4. **OpenCvSharp**: The project uses OpenCvSharp for image processing. Install it via NuGet:
   
5. **CommunityToolkit.Mvvm**: This project uses the CommunityToolkit.Mvvm library for MVVM support. Install it via NuGet:
   
6. **MaterialDesignInXamlToolkit**: The project uses Material Design in XAML Toolkit for UI components. Install it via NuGet:
   
7. **Microsoft.Windows.Compatibility**: This project uses some Windows-specific APIs. Install the compatibility pack via NuGet:
   
8. **Microsoft.WindowsAPICodePack-Shell**: This project uses the Windows API Code Pack for file dialogs. Install it via NuGet:
   
## Getting Started

1. **Clone the repository**:
   
2. **Open the solution**: Open the `InkMARC.Label.sln` file in Visual Studio 2022.

3. **Restore NuGet packages**: Visual Studio should automatically restore the required NuGet packages. If not, you can restore them manually:
   
4. **Build the solution**: Build the solution to ensure all dependencies are correctly installed and there are no build errors.

5. **Run the application**: Start debugging (F5) or run the application without debugging (Ctrl+F5) from Visual Studio.

## Usage

- **Load Folder**: Use the "Open" button to select a folder containing video files.
- **Export Data**: Use the "Save" button to export data to an HDF5 file.
- **Navigate Frames**: Use the arrow keys or toolbar buttons to navigate through video frames.
- **Mark Touch Points**: Use the "Begin Touch" button to mark touch points in the video.

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request with your changes.

