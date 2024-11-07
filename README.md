## WaveNeuron Music Visualizer

WaveNeuron Music Visualizeris an interactive and dynamic visualization tool that creates mesmerizing wave patterns influenced by real-time audio input and visual sources such as webcams, images, or videos. Leveraging the power of PyTorch for GPU-accelerated computations, OpenCV for video processing, and Pygame for rendering, Wave Pond offers a visually stunning experience customizable to your preferences.

## Features

Real-Time Audio Visualization: Capture and visualize audio input from your microphone with customizable sensitivity.

Multiple Background Sources: Choose between live webcam feeds, static images, or pre-recorded videos as the background.

Dynamic Wave Patterns: Interactive wave patterns that respond to audio and visual inputs, creating a fluid and engaging display.

Customizable Settings: Adjust wave amplitude, speed, ripple size, effect visibility, color palettes, and more.

System Tray Integration: Access quick controls like settings, help toggling, pausing/resuming, and exiting via the system tray icon.

GPU Acceleration: Utilizes PyTorch with CUDA support for efficient and smooth performance on compatible NVIDIA GPUs.

Interactive Controls: Zoom in/out, drag wave nodes, toggle fullscreen, and view help sections directly within the application.

## Requirements

Before installing Wave Pond PyTorch, ensure your system meets the following requirements:

## Operating System: Windows, macOS, or Linux

(Made with Windows)

Python Version: 3.7 or higher

## Hardware:

GPU: NVIDIA GPU with CUDA support for optimal performance (e.g., NVIDIA GeForce RTX 3060 barely runs it) 

Webcam: For live video input (optional)

Microphone: For audio input (optional)

Installation

## Clone the Repository

git clone https://github.com/anttiluode/WaveNeuronMusicVisualizer.git

cd WaveNeuronMusicVisualizer

Create a Virtual Environment (Optional but Recommended)

## Install Dependencies

pip install -r requirements.txt

Or install them manually: 

pip install opencv-python numpy pygame pyaudio torch torchvision torchaudio pystray pillow

PyTorch with CUDA: To utilize GPU acceleration, install the CUDA-enabled version of PyTorch. Visit PyTorch.org to find the appropriate installation command based on your CUDA version.

Example for CUDA 11.7:

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117

PyAudio Installation:

Windows: Install the PyAudio wheel from Unofficial Binaries.

## Run 

python app.py 

## System Tray Icon

Upon launching, Wave Pond PyTorch will minimize to the system tray. You can access controls via the tray icon.

## Controls

Interact with Wave Pond PyTorch using the following keyboard and mouse controls:

## Keyboard Controls:

UP/DOWN Arrow Keys: Increase/Decrease wave amplitude (Range: 0-50)

LEFT/RIGHT Arrow Keys: Decrease/Increase wave speed (Range: 0-50)

A/Z Keys: Increase/Decrease audio sensitivity (Range: 0-200)

S/X Keys: Increase/Decrease ripple size

D/C Keys: Increase/Decrease effect visibility (Range: 0-1)

SPACE: Change color palette

F: Toggle FPS display

H: Toggle Help Section

F11: Toggle Fullscreen mode

P: Pause/Resume visualization

ESC: Quit the application

## Mouse Controls:

Mouse Wheel: Zoom in/out of the visualization

Left Click on 'Menu' Button: Open Settings window

Drag Wave Nodes: Click and drag wave nodes to reposition

## Settings

Access the Settings window to customize various aspects of the visualization:

## Open Settings:

Click the 'Menu' button within the application window.

Or right-click the system tray icon and select 'Settings'.

Menu is pretty broke. It can crash the computer.. 

## Configure Settings:

Background Source:

Webcam: Use a live webcam feed as the background.

Image: Select a static image file (JPEG, PNG) as the background.

Video: Select a video file (MP4, AVI, MOV) as the background.

## Audio Input Device:

Choose the desired microphone or audio input device from the dropdown list.

Select "Default" to use the system's default audio input.

## Visualization Settings:

(On code level) 

Number of Rings: Set the number of concentric rings for wave generation.

Neurons per Ring: Set the number of wave nodes per ring.

Resolution:

Adjust the width and height of the visualization window.

Save Settings:

After configuring, click the "Save Settings" button to apply changes.

Troubleshooting

Common Issues

License
This project is licensed under the MIT License.

