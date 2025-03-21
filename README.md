# Hand Landmark Detection & Augmented Reality Project

A modern computer vision project that detects hand landmarks in real-time and overlays interactive 3D objects using augmented reality. This project uses MediaPipe for hand detection, OpenCV for video capture and processing, and ModernGL for 3D rendering. The code is structured into two main modules: one for processing and predicting hand landmarks, and another for rendering a 3D scene based on those predictions.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [How It Works](#how-it-works)
- [Project Structure](#project-structure)
- [Installation and Setup](#installation-and-setup)
  - [Prerequisites](#prerequisites)
  - [Step-by-Step Setup using Miniconda](#step-by-step-setup-using-miniconda)
- [Usage](#usage)
- [Technical Details](#technical-details)
- [Future Enhancements](#future-enhancements)
- [License](#license)

## Overview
This project demonstrates how to combine hand landmark detection with augmented reality techniques. Using a live camera feed, the project first identifies key points on the hand (such as fingertips and joints) using MediaPipe. Then, using mathematical transformations and 3D rendering libraries like ModernGL and PyWavefront, the project overlays 3D objects (e.g., a cube and a marker) onto the video stream. The interactive experience is enhanced by detecting a "pinch" gesture (by measuring the distance between thumb and index finger) to manipulate the position of 3D objects.

## Features
- **Real-Time Hand Detection:** Leverages MediaPipe's hand landmark detector to identify and track hand gestures.
- **2D and 3D Visualization:**
  - Draws 2D landmarks over the video feed.
  - Projects and manipulates 3D objects in the scene based on hand movements.
- **Pinch Gesture Recognition:** Detects when the thumb and index finger come close together to trigger object manipulation.
- **Camera Calibration:** Approximates intrinsic camera parameters and uses perspective projection to overlay 3D graphics accurately.
- **Smooth Interaction:** Uses smoothing techniques to create seamless transitions when moving 3D objects.

## How It Works
### Hand Landmark Detection
- The `prediction.py` module uses MediaPipe's hand landmark detection to extract 2D and 3D coordinates from a live video feed.
- **Prediction Function:** Converts frames from the camera into the required format and returns detected hand landmarks.
- **Drawing Function:** Visualizes these landmarks on the image using OpenCV.
- **Camera Matrix Calculation:** Generates an intrinsic matrix based on frame dimensions, which is crucial for projecting 3D points.
- **3D Projection:** Uses OpenCV’s `solvePnP` to compute the rotation and translation vectors, transforming the hand model's 3D points to camera space.

### Augmented Reality Rendering
- The `gl.py` module creates an augmented reality window using ModernGL.
- **3D Scene Setup:** Loads 3D models (like a cube and a marker) using PyWavefront.
- **Texture Mapping:** Captures the live video feed as a texture for the background.
- **Gesture-Driven Interaction:** Checks for a pinch gesture and updates the position of the 3D object accordingly.
- **Rendering Pipeline:** Uses vertex and fragment shaders to render the 3D objects with appropriate lighting and transformations.

## Project Structure
```
├── prediction.py       # Module for hand landmark detection and processing
├── gl.py               # Module for AR rendering and integrating OpenCV with ModernGL
├── data/               # Folder containing 3D model files (e.g., crate.obj, marker.obj) and textures (e.g., crate.png)
└── README.md           # This file
```

## Installation and Setup
### Prerequisites
Before running the project, ensure you have the following installed on your system:
- Python 3.7+
- Miniconda (or Anaconda)
- A working webcam (for video capture)
- Basic libraries: MediaPipe, OpenCV, NumPy, ModernGL, Moderngl-window, PyWavefront, Pyrr

### Step-by-Step Setup using Miniconda
#### Clone the Repository
```bash
git clone https://github.com/yourusername/your-repo-name.git
cd your-repo-name
```
#### Create a Conda Environment
Create a new conda environment with Python 3.7 or later:
```bash
conda create -n ar_project python=3.8
```
#### Activate the environment
```bash
conda activate ar_project
```
#### Install Dependencies
```bash
pip install mediapipe opencv-python-headless numpy moderngl moderngl-window pywavefront pyrr
```
*Note: If you run into issues with OpenCV, try installing a different build or refer to the OpenCV documentation for guidance.*

#### Verify the Installation
```bash
python -c "import cv2; import mediapipe; import moderngl; print('All dependencies installed successfully!')"
```

#### Run the Project
```bash
python gl.py
```
The application will open a window that displays your webcam feed with overlaid 2D hand landmarks and a 3D cube that reacts to your hand gestures (pinch to manipulate).

## Usage
1. **Start the Application:** Run `python gl.py` to launch the AR window.
2. **Interact:** Use your hand in front of the camera. The application detects your hand landmarks. When you pinch (bring your thumb and index finger close together), you can interact with the 3D object.
3. **Exit:** Press the `Esc` key to close the application window.

## Technical Details
- **MediaPipe HandLandmarker:** Utilized for robust hand detection and landmark estimation. The project requires MediaPipe version 0.9.1 or higher.
- **OpenCV:** Handles real-time video capture, frame manipulation, and drawing of 2D landmarks.
- **3D Transformations:**
  - `solvePnP`: Computes rotation and translation vectors to align 3D hand landmarks with the camera's perspective.
  - **Reprojection:** Ensures the 3D points correctly map to the 2D image plane.
- **ModernGL & Shaders:**
  - Vertex and fragment shaders are used to render the 3D models with lighting effects.
  - The MVP (Model-View-Projection) matrix is computed to transform 3D objects within the scene.
- **Gesture Recognition:**
  - Detects a pinch gesture by calculating the Euclidean distance between the thumb and index finger.
  - Adjusts the position of the 3D object based on gesture input, incorporating smoothing to ensure natural movement.

## Future Enhancements
- **Model Calibration:** Improve camera calibration for more accurate 3D projections.
- **Extended Gestures:** Recognize additional hand gestures to provide more interactive AR experiences.
- **UI/UX Enhancements:** Add on-screen instructions and visual feedback to guide users during interaction.
- **Cross-Platform Support:** Enhance compatibility with different operating systems and devices.

## License
This project is licensed under the **GPL-3.0 License**.

