import streamlit as st
import cv2
import torch
from ultralytics import YOLO
import cvzone
import numpy as np
import math

model = YOLO('./yolo_weights/yolov8n.pt')
cls_names = model.names

def capture_frames():
    cap = cv2.VideoCapture(0)
    cap.set(3, 1280)
    cap.set(4, 720)
    
    while True:
        success, img = cap.read()
        if not success:
            break

        results = model(img, stream=True)
        
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                w = x2 - x1
                h = y2 - y1

                cvzone.cornerRect(img, (x1, y1, w, h))
                confidence = math.ceil(box.conf[0] * 100) / 100
                cvzone.putTextRect(img, f"{cls_names[int(box.cls[0])]} {confidence}", 
                                   (max(0, x1), max(35, y1)))

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        yield img_rgb

    cap.release()

st.title("YOLO Object Detection Live Stream")
st.text("Live object detection using YOLO model by Natan Asrat.")

# Display the video stream in Streamlit
frame_placeholder = st.empty()

# Get video frames and display them
frame_generator = capture_frames()

for frame in frame_generator:
    frame_placeholder.image(frame, channels="RGB")
