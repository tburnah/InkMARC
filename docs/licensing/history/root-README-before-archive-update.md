# InkMARC Suite

InkMARC is a collection of applications designed for analyzing hand deformation, processing video frames, and labeling ink data for research in stylus-based interactions. This repository includes the following solutions:

- **InkMARC.Deform**: Studies how the hand deforms when using a stylus with varying pressures.
- **InkMARC.Prepare**: Processes and analyzes video frames and ink data.
- **InkMARC.Label**: Provides tools for labeling and exporting ink-related data.

All solutions are built using .NET 9 and leverage technologies such as OpenCV, HDF5, and the CommunityToolkit.Mvvm framework.

## Features

### InkMARC Deform
- Analyze hand deformation under different stylus pressures.
- Cross-platform support with .NET MAUI.
- Customizable UI using the .NET MAUI Community Toolkit.

### InkMARC Prepare
- Extract frames from video files.
- Predict pressure from video frames.
- Sync data with HDF5 files.
- Load and process ink points from JSON files.

### InkMARC Label
- Load and annotate video files.
- Navigate frames and mark touch points.
- Export data to an HDF5 file.
- Material Design-based UI for an improved user experience.

## License
This software is licensed under the Creative Commons Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0).
You may use, share, and modify this code for non-commercial purposes only, as long as proper attribution is given.
See the [LICENSE](LICENSE) file for the full legal text, or visit the [Creative Commons site](https://creativecommons.org/licenses/by-nc/4.0/) for a human-readable summary.
[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)

## Prerequisites
To build and run the solutions, ensure you have the following installed:

- **.NET 9 SDK**: [Download .NET 9](https://dotnet.microsoft.com/download/dotnet/9.0)
- **Visual Studio 2022** (version 17.0 or later) with necessary workloads
- **FFmpeg**: Required for video frame extraction and processing in `InkMARC.Prepare` and `InkMARC.Label`.
  - [Download FFmpeg](https://ffmpeg.org/download.html) and install it.
  - Make sure the `ffmpeg` executable is accessible via your system `PATH`.
    - On **Windows**, you can do this by adding the FFmpeg `bin` folder to the system environment variable `PATH`.
    - On **macOS/Linux**, install via a package manager (`brew install ffmpeg`, `apt install ffmpeg`, etc.) or add it to your `PATH` manually.
- **Additional Dependencies (NuGet Packages):**
  - OpenCvSharp
  - CommunityToolkit.Mvvm
  - Microsoft.Maui.Graphics
  - HDF5CSharp
  - MaterialDesignInXamlToolkit (for InkMARC Label)
  - Microsoft.Windows.Compatibility (for InkMARC Label)
  - Microsoft.WindowsAPICodePack-Shell (for InkMARC Label)

## Installation

1. **Clone the repository:**
   ```sh
   git clone https://github.com/yourusername/inkmarc.git
   cd inkmarch
   ```
2. **Restore project dependencies:**
   ```sh
   dotnet restore
   ```

## Building and Running

### InkMARC Deform
1. Open the solution in Visual Studio 2022.
2. Set the startup project to `InkMARC.Deform`.
3. Select the target platform (Android, iOS, Windows, etc.).
4. Press `F5` to build and run the application.

### InkMARC Prepare
1. Open `InkMARC.sln` in Visual Studio 2022.
2. Restore NuGet packages manually if needed.
3. Build the solution (`Ctrl+Shift+B`).
4. Start debugging (`F5`).

### InkMARC Label
1. Open `InkMARC.Label.sln` in Visual Studio 2022.
2. Restore NuGet packages manually if needed.
3. Build the solution.
4. Start debugging (`F5`) or run without debugging (`Ctrl+F5`).

## Usage

### InkMARC Deform
- Use the stylus to interact with the application and observe hand deformations under different pressures.

### InkMARC Prepare
- **Extract Frames:** Use the `ExtractFrames` method in `MainViewViewModel`.
- **Predict Pressure:** Use `UpdateImageForTimestamp`.
- **Sync Data:** Use `SyncData` to sync with an HDF5 file.
- **Load Points:** Use `LoadPoints` to load ink points from JSON.

### InkMARC Label
- **Load Folder:** Open a folder containing video files.
- **Export Data:** Save data to an HDF5 file.
- **Navigate Frames:** Use arrow keys or toolbar buttons.
- **Mark Touch Points:** Use "Begin Touch" to mark touch points in the video.

## Contributing
To contribute to the project, follow these steps:

1. Fork the repository.
2. Create a new branch:
   ```sh
   git checkout -b feature-branch
   ```
3. Make your changes and commit them:
   ```sh
   git commit -m "Description of changes"
   ```
4. Push to the branch:
   ```sh
   git push origin feature-branch
   ```
5. Create a pull request.

## License

This project is licensed under the Apache 2.0 License. See the [LICENSE](LICENSE) file for details.

