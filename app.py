from concurrent.futures import ThreadPoolExecutor
import threading
import av
import cv2
import numpy as np
import pandas as pd
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
from PIL import Image
from rapidocr_onnxruntime import RapidOCR
from streamlit_webrtc import WebRtcMode, webrtc_streamer
from ultralytics import YOLO

st.set_page_config(page_title="VBBRS - Smart Book Recommender", layout="wide")
st.title("📚 Vision-Based Book Recommendation System (VBBRS)")


@st.cache_resource
def load_models():
    detector = YOLO("models/best.onnx", task="detect")
    ocr = RapidOCR()
    return detector, ocr


model, ocr_engine = load_models()

# --- SIDEBAR: INPUT MODE & USER PREFERENCES ---
st.sidebar.header("1. Detection Settings")
mode = st.sidebar.radio("Select Input Mode", ["📷 Live Webcam (Real-Time)", "🖼️ Upload Image"])
conf_threshold = st.sidebar.slider("YOLO Confidence Threshold", 0.10, 0.90, 0.35, 0.05)

st.sidebar.markdown("---")
st.sidebar.header("2. Your Book Preferences")
st.sidebar.caption("Enter 1 or 2 favorite books to highlight similar books on the shelf.")
fav_book_1 = st.sidebar.text_input("Favorite Book 1", placeholder="e.g., Dune")
fav_book_2 = st.sidebar.text_input("Favorite Book 2 (Optional)", placeholder="e.g., The Martian")
sim_threshold = st.sidebar.slider("Similarity Highlight Threshold (%)", 5, 80, 15, 5)

# Thread-safe container for WebRTC worker thread
lock = threading.Lock()
shared_state = {"conf": conf_threshold, "latest_frame": None}
with lock:
    shared_state["conf"] = conf_threshold


def get_ice_servers():
    ice_servers = [
        {"urls": ["stun:stun.l.google.com:19302"]},
        {"urls": ["stun:stun1.l.google.com:19302"]},
    ]
    if "TURN_URL" in st.secrets:
        ice_servers.append(
            {
                "urls": [st.secrets["TURN_URL"]],
                "username": st.secrets.get("TURN_USERNAME", ""),
                "credential": st.secrets.get("TURN_CREDENTIAL", ""),
            }
        )
    return {"iceServers": ice_servers}


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_book_metadata(query: str) -> dict:
    """Queries Open Library Search API to resolve book title, author, and subjects."""
    cleaned_query = query.strip()
    if not cleaned_query or cleaned_query == "Unreadable / No text detected":
        return {"matched_title": query, "author": "Unknown", "subjects": "", "found": False}

    try:
        resp = requests.get(
            "https://openlibrary.org/search.json",
            params={
                "q": cleaned_query,
                "limit": 1,
                "fields": "title,author_name,subject",
            },
            timeout=4,
        )
        resp.raise_for_status()
        docs = resp.json().get("docs", [])
        if docs:
            doc = docs[0]
            title = doc.get("title", cleaned_query)
            authors = ", ".join(doc.get("author_name", ["Unknown"])[:2])
            subjects = ", ".join(doc.get("subject", [])[:12])
            return {
                "matched_title": title,
                "author": authors,
                "subjects": subjects,
                "found": True,
            }
    except Exception:
        pass

    return {"matched_title": cleaned_query, "author": "Unknown", "subjects": "", "found": False}


def read_spine_text(crop_bgr: np.ndarray) -> str:
    """Extracts text from a cropped book spine, auto-rotating vertical spines."""
    if crop_bgr.size == 0:
        return ""

    h, w = crop_bgr.shape[:2]
    candidates = [crop_bgr]

    if h > w * 1.2:
        candidates = [
            cv2.rotate(crop_bgr, cv2.ROTATE_90_CLOCKWISE),
            cv2.rotate(crop_bgr, cv2.ROTATE_90_COUNTERCLOCKWISE),
        ]

    best_text = ""
    best_score = 0.0

    for candidate in candidates:
        ocr_result, _ = ocr_engine(candidate)
        if ocr_result:
            texts = [line[1].strip() for line in ocr_result if float(line[2]) > 0.45]
            avg_score = float(np.mean([float(line[2]) for line in ocr_result]))
            if texts and avg_score > best_score:
                best_score = avg_score
                best_text = " ".join(texts)

    return best_text if best_text else "Unreadable / No text detected"


def compute_similarity_scores(fav_queries: list[str], shelf_metadata: list[dict]) -> list[float]:
    """Computes TF-IDF cosine similarity between user favorites and detected shelf books."""
    if not fav_queries or not shelf_metadata:
        return [0.0] * len(shelf_metadata)

    fav_profiles = []
    for fq in fav_queries:
        meta = fetch_book_metadata(fq)
        # Weight title and author alongside subject tags
        profile = f"{meta['matched_title']} {meta['matched_title']} {meta['author']} {meta['subjects']}"
        fav_profiles.append(profile.lower())

    combined_user_profile = " ".join(fav_profiles)

    shelf_profiles = []
    for item in shelf_metadata:
        profile = f"{item['matched_title']} {item['author']} {item['subjects']} {item['raw_ocr']}"
        shelf_profiles.append(profile.lower())

    corpus = [combined_user_profile] + shelf_profiles
    try:
        vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        tfidf_matrix = vectorizer.fit_transform(corpus)
        scores = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
        return [round(float(s) * 100, 1) for s in scores]
    except ValueError:
        return [0.0] * len(shelf_metadata)


def analyze_and_recommend(image_rgb: np.ndarray, conf: float, fav_books: list[str], match_cutoff: float):
    """Detects books, extracts titles, computes preference similarity, and highlights matches."""
    results = model.predict(source=image_rgb, imgsz=640, conf=conf, verbose=False)
    boxes = results[0].boxes

    annotated = image_rgb.copy()
    if boxes is None or len(boxes) == 0:
        return annotated, []

    xyxy = boxes.xyxy.cpu().numpy()
    confs = boxes.conf.cpu().numpy()
    sorted_indices = sorted(
        range(len(xyxy)),
        key=lambda i: (int(xyxy[i][1] // 100), xyxy[i][0]),
    )

    # Step 1: OCR all detected spines
    detections = []
    for book_num, idx in enumerate(sorted_indices, start=1):
        x1, y1, x2, y2 = map(int, xyxy[idx])
        crop_bgr = cv2.cvtColor(image_rgb[y1:y2, x1:x2], cv2.COLOR_RGB2BGR)
        raw_ocr = read_spine_text(crop_bgr)
        detections.append(
            {
                "book_num": book_num,
                "coords": (x1, y1, x2, y2),
                "det_conf": float(confs[idx]),
                "raw_ocr": raw_ocr,
            }
        )

    # Step 2: Enrich shelf books in parallel via Open Library
    with ThreadPoolExecutor(max_workers=6) as executor:
        enriched_list = list(
            executor.map(lambda d: fetch_book_metadata(d["raw_ocr"]), detections)
        )

    for det, meta in zip(detections, enriched_list):
        meta["raw_ocr"] = det["raw_ocr"]

    # Step 3: Score similarity against user's favorite books
    sim_scores = compute_similarity_scores(fav_books, enriched_list)

    # Step 4: Draw boxes (Highlight matches in Emerald Green, others in Slate Blue)
    records = []
    for det, meta, sim_score in zip(detections, enriched_list, sim_scores):
        x1, y1, x2, y2 = det["coords"]
        book_num = det["book_num"]
        is_match = bool(fav_books) and (sim_score >= match_cutoff)

        if is_match:
            box_color = (16, 185, 129)  # Bright Emerald Green (RGB)
            thickness = 4
            badge_text = f"★ #{book_num} MATCH ({sim_score:.0f}%)"
            # Subtle translucent green highlight overlay on matching spine
            overlay = annotated.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), box_color, -1)
            cv2.addWeighted(overlay, 0.22, annotated, 0.78, 0, annotated)
        else:
            box_color = (59, 130, 246)  # Clean Blue (RGB)
            thickness = 2
            badge_text = f"#{book_num}"

        cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, thickness)

        (tw, th), _ = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        cv2.rectangle(annotated, (x1, max(0, y1 - th - 8)), (x1 + tw + 8, y1), box_color, -1)
        cv2.putText(
            annotated,
            badge_text,
            (x1 + 4, max(th + 2, y1 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
        )

        records.append(
            {
                "Status": "⭐ Recommended Match" if is_match else "Detected",
                "Match Score (%)": sim_score,
                "Book ID": f"#{book_num}",
                "Resolved Title": meta["matched_title"],
                "Author": meta["author"],
                "Raw Spine OCR": det["raw_ocr"],
                "Genres / Subjects": meta["subjects"][:80] + ("..." if len(meta["subjects"]) > 80 else ""),
            }
        )

    # Sort table so highlighted recommendations appear first
    records.sort(key=lambda r: (r["Status"] == "⭐ Recommended Match", r["Match Score (%)"]), reverse=True)
    return annotated, records


def render_results(annotated_img: np.ndarray, records: list[dict], fav_books: list[str], original_img=None):
    """Displays annotated bookshelf image, highlighted recommendations, and full book table."""
    total_books = len(records)
    matches = [r for r in records if r["Status"] == "⭐ Recommended Match"]

    if original_img is not None:
        col1, col2 = st.columns(2)
        with col1:
            st.image(original_img, caption="Original Bookshelf", use_container_width=True)
        with col2:
            st.image(
                annotated_img,
                caption=f"Detected: {total_books} Books | Highlighted Matches: {len(matches)}",
                use_container_width=True,
            )
    else:
        st.image(
            annotated_img,
            caption=f"Detected: {total_books} Books | Highlighted Matches: {len(matches)}",
            use_container_width=True,
        )

    if fav_books:
        st.subheader("🎯 Recommended Matches for Your Taste")
        if matches:
            for m in matches:
                st.success(
                    f"**{m['Book ID']} — {m['Resolved Title']}** by *{m['Author']}* "
                    f"(Similarity: **{m['Match Score (%)']}%** | Raw OCR: `{m['Raw Spine OCR']}`)"
                )
        else:
            st.info(
                "No strong matches found above the current similarity threshold. "
                "Try lowering the **Similarity Highlight Threshold (%)** in the sidebar."
            )

    st.subheader("📖 All Detected Books")
    if records:
        df = pd.DataFrame(records)
        st.dataframe(df, use_container_width=True, hide_index=True)

        csv_data = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ Download Results (CSV)",
            data=csv_data,
            file_name="vbbrs_recommendations.csv",
            mime="text/csv",
        )
    else:
        st.warning("No books detected at the current YOLO confidence threshold.")


active_favorites = [b.strip() for b in (fav_book_1, fav_book_2) if b.strip()]

# --- MODE 1: REAL-TIME WEBCAM ---
if mode == "📷 Live Webcam (Real-Time)":
    st.subheader("Real-Time Camera Detection")

    def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")

        with lock:
            current_conf = shared_state["conf"]
            shared_state["latest_frame"] = img.copy()

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

    if st.button("📸 Capture Frame & Find Similar Books", type="primary"):
        with lock:
            captured_bgr = shared_state["latest_frame"]
        if captured_bgr is not None:
            captured_rgb = cv2.cvtColor(captured_bgr, cv2.COLOR_BGR2RGB)
            with st.spinner("Reading spines, fetching metadata, and matching preferences..."):
                annotated_cap, records = analyze_and_recommend(
                    captured_rgb, conf_threshold, active_favorites, sim_threshold
                )
            render_results(annotated_cap, records, active_favorites)
        else:
            st.warning("Start the webcam stream first before capturing a frame.")

# --- MODE 2: IMAGE UPLOAD ---
else:
    st.subheader("Upload Bookshelf Image")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        image_np = np.array(image)

        with st.spinner("Detecting books, reading titles, and matching your preferences..."):
            annotated_img, records = analyze_and_recommend(
                image_np, conf_threshold, active_favorites, sim_threshold
            )

        render_results(annotated_img, records, active_favorites, original_img=image)