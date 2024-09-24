import streamlit as st
import torch
from ultralytics import YOLO
import cv2
import numpy as np
import math
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer
import av

# Custom Video Processor Class
class CustomVideoProcessor(VideoTransformerBase):
    def __init__(self, model):
        self.model = model

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")  # Convert to OpenCV format
        results = self.model(img, stream=True)

        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = math.ceil(box.conf[0] * 100) / 100

                # Draw the bounding box
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"{self.model.names[int(box.cls[0])]} {confidence:.2f}"
                cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# Main application logic
st.title("YOLO Object Detection Live Stream")
st.text("Live object detection using YOLO model. Please wait a bit if the stream does not appear immediately.")

# Load the YOLO model
@st.cache_resource(show_spinner=True)
def load_model():
    st.text("Loading YOLO model...")
    model = YOLO('./yolo_weights/yolov8n.pt')
    return model

model = load_model()

# Use WebRTC for live video stream
webrtc_streamer(key="example", video_processor_factory=lambda: CustomVideoProcessor(model))
