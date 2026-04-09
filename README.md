# AIVege

A vegetable and fruit detection plus price estimation app built with YOLO and Streamlit.  
Upload an image with multiple items and the app detects each object, shows an itemized breakdown, and calculates the total price in MYR automatically.

## Features
- Detects 47 vegetable/fruit classes from a single image
- YOLO-based multi-object detection with bounding boxes and class labels
- Itemized checkout-style table (item, quantity, unit price, line total)
- Total price calculation in Malaysian Ringgit (MYR)
- Simple Streamlit interface for quick testing and demo

## Dataset
This project expects YOLO-format data configuration in:
- `data/detect_data.yaml`

You can update dataset paths and class definitions there if needed.

## Installation
```bash
# clone the repo
git clone https://github.com/jeeshen/AIVege.git

# navigate to project directory
cd AIVege

# install dependencies
pip install -r requirements.txt

# place the trained detection model at project root
# best.pt  (not included due to file size)

# run the app
streamlit run app.py
```

## Notes
- If `best.pt` is not present, the app falls back to `yolov8n.pt`.
- For Streamlit cloud deployment, keep a CPU-compatible `requirements.txt`.
