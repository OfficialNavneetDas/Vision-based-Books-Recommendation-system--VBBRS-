import threading
import av
import cv2
import streamlit as st
from PIL import Image
from streamlit_webrtc import WebRtcMode, webrtc_streamer
from ultralytics import YOLO

st.set_page_config(page_title="Book Detector (VBBRS)", layout="wide")
st.title("📚 Vision-Based Book Detection (YOLO26n ONNX)")


@st.cache_resource
def load_model():
    return YOLO("models/best.onnx", task="detect")


model = load_model()

# Sidebar controls
mode = st.sidebar.radio("Select Input Mode", ["📷 Live Webcam (Real-Time)", "🖼️ Upload Image"])
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.10, 0.90, 0.35, 0.05)

# Thread-safe container for passing slider updates into the WebRTC worker thread
lock = threading.Lock()
shared_state = {"conf": conf_threshold}
with lock:
    shared_state["conf"] = conf_threshold


def get_ice_servers():
    # Base STUN servers for NAT traversal
    ice_servers = [
        {"urls": ["stun:stun.l.google.com:19302"]},
        {"urls": ["stun:stun1.l.google.com:19302"]},
    ]
    # Optional: Add TURN credentials from Streamlit Secrets for strict firewalls
    if "TURN_URL" in st.secrets:
        ice_servers.append(
            {
                "urls": [st.secrets["TURN_URL"]],
                "username": st.secrets.get("TURN_USERNAME", ""),
                "credential": st.secrets.get("TURN_CREDENTIAL", ""),
            }
        )
    return {"iceServers": ice_servers}


# --- MODE 1: REAL-TIME WEBCAM ---
if mode == "📷 Live Webcam (Real-Time)":
    st.subheader("Real-Time Camera Detection")

    def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")

        with lock:
            current_conf = shared_state["conf"]

        # Use imgsz=416 (or 640 if your server has strong CPU/GPU) for smoother FPS
        results = model.predict(source=img, imgsz=416, conf=current_conf, verbose=False)
        annotated_frame = results[0].plot()

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

    webrtc_streamer(
        key="book-detector",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=get_ice_servers(),
        video_frame_callback=video_frame_callback,
        media_stream_constraints={
            "video": {"width": {"ideal": 640}, "height": {"ideal": 480}},
            "audio": False,
        },
        async_processing=True,
    )

# --- MODE 2: IMAGE UPLOAD ---
else:
    st.subheader("Upload Bookshelf Image")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        results = model.predict(source=image, imgsz=640, conf=conf_threshold, verbose=False)

        annotated_img = results[0].plot()[:, :, ::-1]
        total_books = len(results[0].boxes)

        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Original Image", use_container_width=True)
        with col2:
            st.image(annotated_img, caption=f"Detected: {total_books} Books", use_container_width=True)