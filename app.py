from collections import Counter
from pathlib import Path
import tempfile

import streamlit as st
from PIL import Image
from ultralytics import YOLO


st.set_page_config(page_title="YOLO Detection", layout="centered")
st.title("YOLO Multi-Object Detection")
st.caption("Upload one image to detect vegetables/fruits and calculate price (MYR).")


# Per detected item (each bounding box = 1 unit), in MYR.
PRICE_RM = {
    "almond": 3.50,
    "apple": 1.80,
    "asparagus": 2.00,
    "avocado": 6.00,
    "banana": 1.00,
    "beans": 3.00,
    "beet": 2.50,
    "bell pepper": 4.00,
    "blackberry": 0.35,
    "blueberry": 0.25,
    "broccoli": 5.00,
    "brussels sprouts": 0.60,
    "cabbage": 5.00,
    "carrot": 1.20,
    "cauliflower": 6.50,
    "celery": 3.50,
    "cherry": 0.45,
    "corn": 3.00,
    "cucumber": 2.50,
    "egg": 0.55,
    "eggplant": 3.50,
    "garlic": 1.50,
    "grape": 0.20,
    "green bean": 0.35,
    "green onion": 1.80,
    "hot pepper": 0.80,
    "kiwi": 2.50,
    "lemon": 1.00,
    "lettuce": 4.00,
    "lime": 0.70,
    "mandarin": 1.20,
    "mushroom": 0.50,
    "onion": 1.20,
    "orange": 1.80,
    "pattypan squash": 3.50,
    "pea": 0.15,
    "peach": 4.00,
    "pear": 3.00,
    "pineapple": 10.00,
    "potato": 2.50,
    "pumpkin": 4.50,
    "radish": 0.60,
    "raspberry": 0.40,
    "strawberry": 1.50,
    "tomato": 2.50,
    "vegetable marrow": 3.50,
    "watermelon": 6.00,
}


@st.cache_resource
def load_model():
    """
    Load custom model if available, otherwise use lightweight default.
    """
    custom_model = Path("best.pt")
    model_path = str(custom_model.resolve()) if custom_model.exists() else "yolov8n.pt"
    return YOLO(model_path), model_path


def render_detection_summary_and_price(result) -> None:
    names = result.names
    class_ids = result.boxes.cls.tolist() if result.boxes is not None else []
    class_names = [names[int(class_id)] for class_id in class_ids]
    counts = Counter(class_names)

    if not counts:
        st.info("No objects detected.")
        return

    st.subheader("Detected Objects")

    total_rm = 0.0
    rows = []
    missing_prices = []

    for cls_name in sorted(counts.keys()):
        qty = counts[cls_name]
        unit_price = PRICE_RM.get(cls_name)
        if unit_price is None:
            line_total = None
            missing_prices.append(cls_name)
        else:
            line_total = unit_price * qty
            total_rm += line_total

        rows.append(
            {
                "Item": cls_name.title(),
                "Quantity": qty,
                "Unit Price (RM)": f"{unit_price:.2f}" if unit_price is not None else "N/A",
                "Line Total (RM)": f"{line_total:.2f}" if line_total is not None else "N/A",
            }
        )

    st.dataframe(rows, use_container_width=True, hide_index=True)
    st.metric("Total Price (MYR)", f"RM {total_rm:.2f}")

    if missing_prices:
        missing = ", ".join(sorted(missing_prices))
        st.warning(f"Missing price in PRICE_RM for: {missing}")


def predict_image(model: YOLO, uploaded_file, conf: float):
    suffix = Path(uploaded_file.name).suffix or ".jpg"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        temp_file.write(uploaded_file.getvalue())
        temp_path = temp_file.name

    # Match notebook inference settings as closely as possible.
    results = model.predict(temp_path, conf=conf, iou=0.45, verbose=False)
    return results[0]


model, model_path = load_model()
confidence = st.slider("Confidence threshold", min_value=0.10, max_value=1.00, value=0.25, step=0.05)
uploaded_image = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])

if uploaded_image:
    image = Image.open(uploaded_image)
    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="Original", use_container_width=True)

    result = predict_image(model, uploaded_image, confidence)
    # Explicit channel conversion for ndarray output from OpenCV.
    plotted = result.plot()
    if hasattr(plotted, "ndim") and plotted.ndim == 3:
        plotted = plotted[:, :, ::-1]

    with col2:
        st.image(plotted, caption="Detected", use_container_width=True)

    render_detection_summary_and_price(result)
