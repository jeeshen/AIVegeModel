"""
Project settings: vegetable class names, prices, folders, and training options.
"""

import os
import warnings
from pathlib import Path

# --- Paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

# Classification data: prefer data_classification/ if it has train/, else use data/.
_CLF_ALT = os.path.join(BASE_DIR, "data_classification")
if os.path.isdir(os.path.join(_CLF_ALT, "train")):
    CLF_DATA_ROOT = _CLF_ALT
else:
    CLF_DATA_ROOT = DATA_DIR

TRAIN_DIR = os.path.join(CLF_DATA_ROOT, "train")
_val_candidates = [
    os.path.join(CLF_DATA_ROOT, "valid"),
    os.path.join(CLF_DATA_ROOT, "validation"),
    os.path.join(CLF_DATA_ROOT, "val"),
]
VAL_DIR = None
for _p in _val_candidates:
    if os.path.isdir(_p):
        VAL_DIR = _p
        break
if VAL_DIR is None and os.path.isdir(TRAIN_DIR):
    warnings.warn(
        f"No 'valid', 'validation', or 'val' folder under {CLF_DATA_ROOT}. "
        "Using TRAIN_DIR for validation (metrics will look too good).",
        UserWarning,
        stacklevel=1,
    )
    VAL_DIR = TRAIN_DIR
elif VAL_DIR is None:
    VAL_DIR = _val_candidates[0]

MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# YOLO object detection (Roboflow layout under data/)
DETECT_DATA_YAML = os.path.join(DATA_DIR, "detect_data.yaml")
YOLO_DETECT_WEIGHTS = os.path.join(MODELS_DIR, "yolo_detect_vege", "weights", "best.pt")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


def get_yolo_train_device() -> int | str:
    """
    Pick device for Ultralytics training.
    Set YOLO_DEVICE to cpu, 0, 1, or 0,1 for multi-GPU.
    Default: GPU 0 if CUDA is available, else cpu.
    """
    import torch

    raw = os.environ.get("YOLO_DEVICE", "").strip()
    if raw:
        low = raw.lower()
        if low in ("cpu", "none", "-1"):
            return "cpu"
        if "," in raw:
            return raw
        try:
            return int(raw)
        except ValueError:
            return raw
    return 0 if torch.cuda.is_available() else "cpu"


def yolo_resume_requested(resume: bool | None) -> bool:
    """True if resume=True was passed, or RESUME / YOLO_RESUME env is set to a truthy value."""
    if resume is not None:
        return bool(resume)
    for key in ("RESUME", "YOLO_RESUME"):
        v = os.environ.get(key, "").strip().lower()
        if v in ("1", "true", "yes", "on"):
            return True
    return False


def get_yolo_train_device_label(device: int | str) -> str:
    """Short label for logs (GPU name or cpu)."""
    import torch

    if device == "cpu" or device == -1:
        return "cpu"
    if not torch.cuda.is_available():
        return f"{device} (CUDA not available — use torch with CUDA or YOLO_DEVICE=cpu)"
    try:
        idx = int(device) if isinstance(device, int) else int(str(device).split(",")[0].strip())
        return f"cuda:{idx} ({torch.cuda.get_device_name(idx)})"
    except Exception:
        return str(device)


def get_yolo_cls_data_yaml() -> str:
    """
    Write a small YAML for Ultralytics classification (train/val folder names).
    Ultralytics expects 'val'; this project may use 'valid' on disk.
    """
    root = Path(CLF_DATA_ROOT)
    tr = Path(TRAIN_DIR).relative_to(root).as_posix()
    va = Path(VAL_DIR).relative_to(root).as_posix()
    yml = Path(RESULTS_DIR) / "yolo_cls_dataset.yaml"
    yml.write_text(
        f"path: {root.as_posix()}\ntrain: {tr}\nval: {va}\n",
        encoding="utf-8",
    )
    return str(yml)


# --- Class names (15 vegetables) ---
CLASSES = [
    "Bean",
    "Bitter_Gourd",
    "Bottle_Gourd",
    "Brinjal",
    "Broccoli",
    "Cabbage",
    "Capsicum",
    "Carrot",
    "Cauliflower",
    "Cucumber",
    "Papaya",
    "Potato",
    "Pumpkin",
    "Radish",
    "Tomato",
]

NUM_CLASSES = len(CLASSES)
CLASS_TO_IDX = {cls: idx for idx, cls in enumerate(CLASSES)}
IDX_TO_CLASS = {idx: cls for cls, idx in CLASS_TO_IDX.items()}

# --- Prices (RM per unit or kg) ---
PRICES = {
    "Bean": 3.50,
    "Bitter_Gourd": 2.50,
    "Bottle_Gourd": 2.00,
    "Brinjal": 2.00,
    "Broccoli": 4.00,
    "Cabbage": 3.00,
    "Capsicum": 3.50,
    "Carrot": 2.50,
    "Cauliflower": 4.50,
    "Cucumber": 2.00,
    "Papaya": 5.00,
    "Potato": 2.50,
    "Pumpkin": 3.00,
    "Radish": 2.00,
    "Tomato": 2.50,
}

# --- Training hyperparameters ---
IMG_SIZE = 224
BATCH_SIZE = 32
NUM_EPOCHS = 30
YOLO_EPOCHS = 50
LEARNING_RATE = 1e-3
NUM_WORKERS = 4

# --- Saved model files ---
YOLO_MODEL_PATH = os.path.join(MODELS_DIR, "yolo_vege", "weights", "best.pt")
CNN_MODEL_PATH = os.path.join(MODELS_DIR, "cnn_vege.pth")
