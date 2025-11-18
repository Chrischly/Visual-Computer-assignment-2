** Windows-only: The code uses MSMF for webcam capture, so Linux/macOS are not supported without adjustment. ** 

Real-Time Webcam Processing: CPU vs GPU Pipelines

This project compares two real-time video-processing pipelines:
A CPU pipeline built with OpenCV
A GPU pipeline using OpenGL + GLSL shaders

Both pipelines perform the same filtering and geometric transformations on a live webcam feed. The system allows switching between pipelines at runtime, enabling direct visual and performance comparison.

Build Instructions
This project uses CMake.
Requirements
- CMake ≥ 3.20
- Visual Studio 2022 (MSBuild or Ninja)
- OpenCV 4.x (via vcpkg recommended)
- GLFW
- GLAD
- glm

Dependencies (via vcpkg)
This project uses:
- opencv4
- glfw3
- glad
- glm


Features:
- Live webcam capture (default 1280×720@30FPS, adjustable by camera settings)
- Two parallel processing paths (CPU / GPU)
- Real-time filters: Pixelation & Sin City
- Real-time geometric transformations: Translation / Scaling / Rotation
- Runtime switching between CPU and GPU
- Built-in automated experiment for FPS measurement


Keyboard Controls
Key	Action
C:	CPU pipeline
G:  GPU pipeline
0/1/2:  Toggle filter (None / Pixelation / Sin City)
T:  Experiment Runner
Q/E:    Rotate
W/S/A/D: Move image
Z/X:    Zoom in/out
ESC:    Quit

Press T in the running application to execute the built-in performance test.
The experiment automatically cycles through:

- CPU vs GPU
- Filters (None, Pixelation, Sin City)
- Transform: On/Off
- Resolutions (720p, 576p, 360p)
- Results are printed directly to the console and saved to an experiments.csv file
