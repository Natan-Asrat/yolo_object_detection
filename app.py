import streamlit as st
import cv2
import torch
from ultralytics import YOLO
import cvzone
import numpy as np
import math
import platform
import time

# JavaScript code to request camera permissions
def camera_permission_script():
    return """
    <script>
        async function checkCameraPermission() {
            const stream = await navigator.mediaDevices.getUserMedia({ video: true });
            const videoTracks = stream.getVideoTracks();
            if (videoTracks.length > 0) {
                console.log('Camera is accessible.');
            } else {
                console.error('Camera is not accessible.');
            }
            stream.getTracks().forEach(track => track.stop());
        }
        checkCameraPermission();
    </script>
    """

# Function to get camera index based on the operating system
def get_camera_index():
    os_type = platform.system()
    if os_type in ["Windows", "Linux", "Darwin"]:
        return 0
    else:
        st.error("Unsupported operating system.")
        return None

# Function to capture frames from the camera
def capture_frames(camera_index, model):
    print("Starting video capture...")
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print("Camera could not be opened.")
        st.error("Camera could not be opened. Please check your camera connection.")
        return

    cap.set(3, 1280)
    cap.set(4, 720)
    
    while True:
        start_frame_time = time.time()
        success, img = cap.read()
        if not success:
            print("Failed to capture image.")
            break
        
        print("Image captured, processing...")
        results = model(img, stream=True)
        
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                w = x2 - x1
                h = y2 - y1

                cvzone.cornerRect(img, (x1, y1, w, h))
                confidence = math.ceil(box.conf[0] * 100) / 100
                cvzone.putTextRect(img, f"{model.names[int(box.cls[0])]} {confidence}", 
                                   (max(0, x1), max(35, y1)))

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        frame_time = time.time() - start_frame_time
        print(f"Frame processed in {frame_time:.2f} seconds.")
        yield img_rgb

    cap.release()
    print("Video capture released.")

# Main application logic
print("Starting the application...")
start_time = time.time()

# Cache the YOLO model loading
@st.cache_resource(show_spinner=True)
def load_model():
    print("Loading YOLO model...")
    model_start_time = time.time()
    model = YOLO('./yolo_weights/yolov8n.pt')
    model_end_time = time.time()
    print(f"YOLO model loaded in {model_end_time - model_start_time:.2f} seconds.")
    return model

# Load the YOLO model
model = load_model()

# Get the camera index
camera_index = get_camera_index()
if camera_index is not None:
    st.title("YOLO Object Detection Live Stream")
    st.text("Live object detection using YOLO model by Natan Asrat. Please wait a bit if the stream does not appear immediately.")

    # Inject the JavaScript to check camera permission
    st.markdown(camera_permission_script(), unsafe_allow_html=True)

    # Display the video stream in Streamlit
    frame_placeholder = st.empty()

    # Get video frames and display them
    frame_generator_start_time = time.time()
    frame_generator = capture_frames(camera_index, model)
    frame_generator_end_time = time.time()
    print(f"Frame generator initialized in {frame_generator_end_time - frame_generator_start_time:.2f} seconds.")

    for frame in frame_generator:
        frame_placeholder.image(frame, channels="RGB", use_column_width=True)

else:
    st.error("Could not initialize camera.")

end_time = time.time()
print(f"Total initialization time: {end_time - start_time:.2f} seconds.")
