🔥 Thermal Facial Recognition and Human Detection System
A real-time night vision surveillance system built using Python, YOLOv8 Segmentation, OpenCV, and Face Recognition libraries.
This system detects human figures through a webcam, applies a thermal effect (Inferno colormap) to those regions, and performs facial recognition by comparing detected faces with known images stored in a local directory. If no known person is detected, the system triggers an audible alert.

📸 Project Overview
✅ Detects humans in real-time using YOLOv8 segmentation
✅ Applies a thermal vision effect to detected human regions
✅ Recognizes faces by comparing them against a known faces database
✅ Triggers an alert sound if an unknown person is detected
✅ Supports low-light environments with dynamic brightness adjustment
✅ Provides a live video stream in a browser using Gradio

📽️ How It Works
Webcam feed is captured frame by frame.

Each frame is processed by YOLOv8 segmentation model to detect human masks.

For each detected human:

A thermal effect is applied only on human regions.

The face is cropped and checked against known face images using face_recognition library.

If the face is recognized, their name appears above them.

If not, it labels as "Unknown Person".

If any unknown human is detected, an alert beep sound is triggered.

The final processed video stream is displayed live using Gradio’s web interface.

📥 Additional Setup (Very Important for Windows)
Before installing face_recognition, make sure you have C/C++ Build Tools installed:

✅ Install Visual Studio Build Tools:
Download Visual Studio Build Tools.

During installation, select:

C++ build tools

Include:

Windows 10 SDK

C++ x64/x86 build tools

Complete the installation and restart your system.

📌 Reason:
The face_recognition library relies on dlib, which requires C++ compilation during installation.
Without C++ build tools, installing face_recognition will fail on Windows.

✅ Also Install CMake:
bash
Copy
Edit
pip install cmake
Note: If you already have CMake installed via your system PATH, you can skip this step.

✅ Updated Requirements Summary
Dependency	Reason
C++ Build Tools	Required for dlib during face_recognition install
CMake	Helps compile dlib binaries

📦 Requirements
✅ Python Version
Python 3.10.x is recommended for compatibility.

✅ Libraries & Frameworks
Package	Recommended Version
opencv-python	4.8.0.76
numpy	1.24.3
torch	2.1.0+cu118
ultralytics	8.2.0
face_recognition	1.3.0
gradio	4.27.0



📥 Installation
1️⃣ Clone the repository

bash
Copy
Edit
git clone https://github.com/yourusername/thermal-facial-recognition.git
cd thermal-facial-recognition
2️⃣ Install required libraries

bash
Copy
Edit
pip install -r requirements.txt
If you don’t have a requirements.txt, you can install them one by one:

bash
Copy
Edit
pip install opencv-python numpy torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install ultralytics
pip install face_recognition
pip install gradio
📂 Folder Structure
bash
Copy
Edit
thermal-facial-recognition/
│
├── known_faces/           # Folder containing images of known persons
│   ├── person1.jpg
│   ├── person2.jpg
│
├── yolov8m-seg.pt         # Pre-trained YOLOv8 segmentation model file
│
├── main.py                # Main application code
│
└── README.md
📌 Usage
1️⃣ Add your known face images to the known_faces/ folder.

File name without extension will be considered as the person’s name.

2️⃣ Download YOLOv8 Segmentation Model

Download yolov8m-seg.pt from Ultralytics YOLOv8 official repo or train your own.

3️⃣ Run the Application

bash
Copy
Edit
python main.py
4️⃣ The Gradio web interface will open in your browser, showing the thermal processed live video.

🔔 Features
🚶 Human detection with YOLOv8 Segmentation

🌡️ Thermal vision effect for detected humans

🧑‍🦱 Face recognition using face_recognition

📢 Alert system for unrecognized persons

🌙 Automatic brightness adjustment in low light

🌐 Live webcam feed via Gradio

📋 Notes
This project uses webcam feed (device index 0). If your system has multiple cameras, adjust cv2.VideoCapture(0) accordingly.

YOLO model path needs to be correctly set in:

python
Copy
Edit
model = YOLO(r"E:\Exsel\yolov8m-seg.pt")
Windows: Alert uses winsound for beep
Mac/Linux: Fallback system beep is used.

💡 Future Enhancements
📧 Email/SMS notification system for unknown person alerts

📁 Save snapshots of unknown persons for investigation

📊 Web dashboard for viewing incident history

🙌 Author
Nitheesh Kumar
GitHub

📜 License
This project is licensed under the MIT License.

📌 Final Note
This project is intended for educational and experimental purposes in security surveillance, smart home monitoring, and research scenarios.
Not recommended for production environments without additional safety and compliance measures.
