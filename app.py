import tempfile
from collections import Counter
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image
from ultralytics import YOLO


st.set_page_config(page_title="YOLO Detection", layout="centered")
st.title("YOLO Multi-Object Detection")
st.caption("Upload an image and detect objects in one click.")


@st.cache_resource
def load_model() -> YOLO:
    """
    Load custom model if available, otherwise use lightweight default.
    """
    custom_model = Path("best.pt")
    model_path = str(custom_model) if custom_model.exists() else "yolov8n.pt"
    return YOLO(model_path)


def render_detection_summary(result) -> None:
    names = result.names
    class_ids = result.boxes.cls.tolist() if result.boxes is not None else []
    class_names = [names[int(class_id)] for class_id in class_ids]
    counts = Counter(class_names)

    if not counts:
        st.info("No objects detected.")
        return

    st.subheader("Detected Objects")
    for label, count in counts.items():
        st.write(f"- {label}: {count}")


def predict_image(model: YOLO, image: Image.Image, conf: float):
    image_np = np.array(image.convert("RGB"))
    results = model.predict(image_np, conf=conf, verbose=False)
    return results[0], image_np


def predict_video(model: YOLO, video_file, conf: float):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
        temp_video.write(video_file.read())
        temp_video_path = temp_video.name

    results = model.predict(source=temp_video_path, conf=conf, verbose=False, stream=False)
    return results[0]


model = load_model()

mode = st.radio("Select input type", ["Image", "Video"], horizontal=True)
confidence = st.slider("Confidence threshold", min_value=0.10, max_value=1.00, value=0.25, step=0.05)

if mode == "Image":
    uploaded_image = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])
    if uploaded_image:
        image = Image.open(uploaded_image)
        col1, col2 = st.columns(2)

        with col1:
            st.image(image, caption="Original", use_container_width=True)

        result, _ = predict_image(model, image, confidence)
        plotted = result.plot()[:, :, ::-1]  # BGR -> RGB

        with col2:
            st.image(plotted, caption="Detected", use_container_width=True)

        render_detection_summary(result)

elif mode == "Video":
    uploaded_video = st.file_uploader("Upload video", type=["mp4", "mov", "avi", "mkv"])
    if uploaded_video:
        st.video(uploaded_video)
        st.warning("Video inference may take longer on cloud deployments.")

        if st.button("Run detection on video"):
            result = predict_video(model, uploaded_video, confidence)
            st.success("Detection complete. Showing the first processed frame preview.")
            preview = result.plot()[:, :, ::-1]  # BGR -> RGB
            st.image(preview, caption="Preview frame with detections", use_container_width=True)
            render_detection_summary(result)
