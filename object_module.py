import torch
from ultralytics import YOLO
import numpy as np
from PIL import Image

# Module-level cached YOLO model
_yolo_model = None
_model_path = "yolov8n.pt"   # Fastest / lightest model


def _pil_from_input(image):
    """
    Accepts an OpenCV ndarray (BGR) or PIL.Image and returns PIL.Image in RGB.
    """
    if isinstance(image, Image.Image):
        return image.convert("RGB")

    if isinstance(image, np.ndarray):
        if image.ndim == 3 and image.shape[2] == 3:
            rgb = image[:, :, ::-1]  # BGR → RGB
            return Image.fromarray(rgb)
        return Image.fromarray(image).convert("RGB")

    raise ValueError("Input must be a PIL.Image or OpenCV ndarray.")


def _load_yolo_model(model_path=None):
    """
    Lazy-load YOLO model and store globally.
    """
    global _yolo_model, _model_path

    if model_path:
        _model_path = model_path

    if _yolo_model is None:
        print("[object_module] Loading YOLOv8 model...")
        _yolo_model = YOLO(_model_path)   # ultralytics handles device placement automatically

    return _yolo_model


def unload_yolo():
    """
    Frees YOLO model from memory (optional).
    """
    global _yolo_model
    try:
        del _yolo_model
        _yolo_model = None
        torch.cuda.empty_cache()
        print("[object_module] YOLO model unloaded.")
    except Exception:
        pass


def run_yolo(image):
    """
    Runs YOLO object detection on a PIL or OpenCV image.
    Returns a text list of detected objects with confidences.
    """

    global _yolo_model

    # Convert image to PIL
    pil_img = _pil_from_input(image)

    # Load YOLO lazily
    model = _load_yolo_model()

    # Run detection
    results = model.predict(pil_img, verbose=False)

    detections = results[0].boxes

    if detections is None or len(detections) == 0:
        return "No objects detected in the image."

    names = model.names  # class names
    output_lines = []

    for box in detections:
        cls = int(box.cls[0])
        conf = float(box.conf[0]) * 100
        label = names.get(cls, f"class_{cls}")
        output_lines.append(f"{label} ({conf:.1f}%)")

    return "Detected objects: " + ", ".join(output_lines)
