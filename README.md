# 🔥 Thermal Facial Recognition and Human Detection System

A real-time **night vision surveillance system** built using Python, YOLOv8 Segmentation, OpenCV, and Face Recognition libraries.

This system detects human figures through a webcam, applies a **thermal effect (Inferno colormap)** to those regions, and performs facial recognition by comparing detected faces with known images stored in a local directory. If no known person is detected, the system triggers an audible alert.

---

## 📸 Project Overview

✅ Detects humans in real-time using YOLOv8 segmentation  
✅ Applies a thermal vision effect to detected human regions  
✅ Recognizes faces by comparing them against a known faces database  
✅ Triggers an alert sound if an unknown person is detected  
✅ Supports low-light environments with dynamic brightness adjustment  
✅ Provides a live video stream in a browser using Gradio

---

## 📽️ How It Works

1. Webcam feed is captured frame by frame.
2. Each frame is processed by the YOLOv8 segmentation model to detect human masks.
3. For each detected human:
   - A thermal effect is applied only on human regions.
   - The face is cropped and checked against known face images using the `face_recognition` library.
   - If the face is recognized, their name appears above them.
   - If not, it labels them as **"Unknown Person"** and plays an alert sound.
4. The final processed video stream is displayed live using **Gradio’s** web interface.

---

## 🧩 Additional Setup (Important for Windows)

Before installing `face_recognition`, ensure you have C/C++ Build Tools installed:

### ✅ Install Visual Studio Build Tools:
- Download from [Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
- During installation, select:
  - **C++ build tools**
  - Include:
    - Windows 10 SDK
    - C++ x64/x86 build tools
- Restart your system after installation.

**📌 Reason:**  
The `face_recognition` library depends on **dlib**, which requires compilation using C++ build tools.

---

### ✅ Install CMake

```bash
pip install cmake
