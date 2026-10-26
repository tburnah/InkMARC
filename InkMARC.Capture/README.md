# InkMARC.Capture

**Copyright (c) 2024–2026 Tavish Burnah. All rights reserved in original contributions.**
See the [root licence](../LICENSE), [copyright scope](../COPYRIGHT) and
[third-party notices](../THIRD_PARTY_NOTICES.md).

InkMARC.Capture is the Android D3 collection application associated with
Chapter 5, §5.3.1 of the InkMARC thesis. It combines a CameraX preview and
recording path, an OpenGL renderer, AprilTag detection, exercise traces and
USB Wacom stylus input.

## Project layout

This is a self-contained Gradle Android project. Its main entry points are:

- `app/src/main/java/com/example/inkmarc_capture/MainActivity.kt`
- `app/src/main/java/com/example/inkmarc_capture/Renderer.kt`
- `app/src/main/java/com/example/inkmarc_capture/UsbHandler.kt`
- `app/src/main/cpp/jni-apriltag.cpp`

The AprilTag native dependency is retained as the
`app/src/main/cpp/third_party/apriltag` Git submodule. Initialise it after
cloning with `git submodule update --init --recursive`.

## Build scope

Open this directory in Android Studio, or run `gradlew.bat assembleDebug` on
Windows. The project declares compile/target SDK 36, minimum SDK 26, NDK
29.0.14206865 and Java 11 bytecode targets. A connected Android device needs
camera and USB-host support; no device build or data-collection session is
claimed as part of this archive import.

The project includes an ONNX model, shaders and exercise assets. Participant
recordings, credentials and signing material are not included. Build outputs,
IDE state and local SDK configuration remain ignored.

## Rights and archive status

This imported desktop-origin source is preserved as an Android project rather
than folded into the older Cue application. It remains subject to the root
repository's rights, third-party and release-review notices.
