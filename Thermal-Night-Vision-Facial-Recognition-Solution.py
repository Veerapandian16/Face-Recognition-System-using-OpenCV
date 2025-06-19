import cv2
import numpy as np
import os
import torch
import face_recognition
import winsound
from ultralytics import YOLO
import gradio as gr

model = YOLO(r"E:\Exsel\yolov8m-seg.pt")


known_face_encodings = []
known_face_names = []

known_faces_dir = "known_faces"
for filename in os.listdir(known_faces_dir):
    img_path = os.path.join(known_faces_dir, filename)
    image = face_recognition.load_image_file(img_path)
    encoding = face_recognition.face_encodings(image)
    if encoding:
        known_face_encodings.append(encoding[0])
        known_face_names.append(os.path.splitext(filename)[0])

def adjust_brightness(image, threshold=50, brightness_factor=1.5):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    avg_brightness = np.mean(gray)
    if avg_brightness < threshold:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hsv[:, :, 2] = hsv[:, :, 2] * brightness_factor
        image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return image


def apply_thermal_to_humans(image, results):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thermal_image = cv2.applyColorMap(gray_image, cv2.COLORMAP_INFERNO)
    human_mask = np.zeros_like(gray_image, dtype=np.uint8)

    for r in results:
        if r.masks is not None:
            for mask in r.masks.xy:
                points = np.array(mask, dtype=np.int32)
                cv2.fillPoly(human_mask, [points], 255)

    human_mask_colored = cv2.merge([human_mask, human_mask, human_mask])
    blended_image = np.where(human_mask_colored > 0, thermal_image, image)
    return blended_image


def play_alert_sound():
    try:
        winsound.Beep(1000, 500)
    except:
        os.system("afplay /System/Library/Sounds/Glass.aiff" if os.name == "posix" else "play -nq -t alsa synth 0.5 sine 1000")


def video_stream():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_resized = cv2.resize(frame, (640, 480))
        frame_resized = adjust_brightness(frame_resized)

        results = model(frame_resized, conf=0.3, iou=0.3)
        human_detected = False
        recognized_faces = 0
        recognized_persons = {}

        for r in results:
            for box in r.boxes:
                box_coordinates = box.xyxy[0].tolist()
                x1, y1, x2, y2 = map(int, box_coordinates)

                if box.cls[0] != 0:
                    continue

                crop_img = frame[y1:y2, x1:x2]
                human_detected = True
                rgb_crop_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
                face_locations = face_recognition.face_locations(rgb_crop_img)
                face_encodings = face_recognition.face_encodings(rgb_crop_img, face_locations)

                for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                    matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.5)
                    face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                    best_match_index = np.argmin(face_distances) if len(face_distances) > 0 else None

                    if best_match_index is not None and matches[best_match_index]:
                        name = known_face_names[best_match_index]
                        recognized_faces += 1
                        recognized_persons[f"{x1}_{y1}_{x2}_{y2}"] = name

                if f"{x1}_{y1}_{x2}_{y2}" not in recognized_persons:
                    name = "Unknown Person"

                text_x = x1
                text_y = max(y1 - 10, 20)
                text_color = (0, 255, 255) if name != "Unknown Person" else (0, 0, 255)
                cv2.putText(frame, name, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)

        thermal_frame = apply_thermal_to_humans(frame, results)

        if human_detected and recognized_faces == 0:
            print("⚠️ No recognized faces detected! Alert triggered.")
            play_alert_sound()

        rgb_frame = cv2.cvtColor(thermal_frame, cv2.COLOR_BGR2RGB)
        yield rgb_frame 

    cap.release()


demo = gr.Interface(
    fn=video_stream,
    inputs=None,
    outputs=gr.Image(label="Thermal Facial Recognition System", type="numpy"),
    live=True
)

demo.launch()
