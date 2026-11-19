import streamlit as st
import cv2
import numpy as np
import av
from PIL import Image
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode

st.set_page_config(page_title="Book Detector (VBBRS)", layout="wide")
st.title("📚 Vision-Based Book Detection (YOLO26n ONNX)")

# Cache the model so it loads only once across reruns and video frames
@st.cache_resource
def load_model():
    return YOLO("models/best.onnx", task="detect")

model = load_model()

# Sidebar controls
mode = st.sidebar.radio("Select Input Mode", ["📷 Live Webcam (Real-Time)", "🖼️ Upload Image"])
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.10, 0.90, 0.35, 0.05)

# --- MODE 1: REAL-TIME WEBCAM ---
if mode == "📷 Live Webcam (Real-Time)":
    st.subheader("Real-Time Camera Detection")

    class BookVideoProcessor(VideoProcessorBase):
        def __init__(self):
            self.conf = conf_threshold

        def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
            img = frame.to_ndarray(format="bgr24")

            # Run ONNX inference on the live frame
            results = model.predict(source=img, imgsz=640, conf=self.conf, verbose=False)
            annotated_frame = results[0].plot()

            # Display live book count on the top-left corner
            count = len(results[0].boxes)
            cv2.putText(
                annotated_frame,
                f"Books Detected: {count}",
                (15, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                2,
            )
            return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")

    ctx = webrtc_streamer(
        key="book-detector",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=BookVideoProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

    # Update confidence slider dynamically while camera is running
    if ctx.video_processor:
        ctx.video_processor.conf = conf_threshold

# --- MODE 2: IMAGE UPLOAD ---
else:
    st.subheader("Upload Bookshelf Image")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        results = model.predict(source=image, imgsz=640, conf=conf_threshold, verbose=False)
        
        # Plot bounding boxes (convert BGR output from .plot() to RGB for Streamlit)
        annotated_img = results[0].plot()[:, :, ::-1]
        total_books = len(results[0].boxes)

        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Original Image", use_container_width=True)
        with col2:
            st.image(annotated_img, caption=f"Detected: {total_books} Books", use_container_width=True)