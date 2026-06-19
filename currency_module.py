import numpy as np
import tensorflow as tf
from PIL import Image
import json
import os

# >>>>>>>  FIXED PATH  <<<<<<<
MODEL_PATH = "/content/drive/MyDrive/VISUAL_ASSISTANCE/currency_model.h5"
CLASS_MAP_PATH = "/content/drive/MyDrive/VISUAL_ASSISTANCE/currency_class_map.json"

_model = None
_class_map = None
IMAGE_SIZE = (224, 224)

def _pil_from_input(image):
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if isinstance(image, np.ndarray):
        if image.ndim == 3:
            rgb = image[:, :, ::-1]
            return Image.fromarray(rgb).convert("RGB")
        return Image.fromarray(image).convert("RGB")
    raise ValueError("Invalid image type.")

def _load_model():
    global _model
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError("Currency model not found at " + MODEL_PATH)
        _model = tf.keras.models.load_model(MODEL_PATH)
    return _model

def _load_map():
    global _class_map
    if _class_map is None:
        if not os.path.exists(CLASS_MAP_PATH):
            raise FileNotFoundError("Class map not found at " + CLASS_MAP_PATH)
        with open(CLASS_MAP_PATH, "r") as f:
            _class_map = json.load(f)
    return _class_map

def identify_currency(image):
    model = _load_model()
    class_map = _load_map()

    img = _pil_from_input(image)
    img = img.resize(IMAGE_SIZE)
    arr = np.array(img).astype("float32") / 255.0
    arr = np.expand_dims(arr, 0)

    preds = model.predict(arr)[0]
    idx = int(np.argmax(preds))
    conf = float(preds[idx])

    label = class_map.get(str(idx), f"class_{idx}")

    return label, conf

def unload_currency_model():
    global _model, _class_map
    _model = None
    _class_map = None
    tf.keras.backend.clear_session()
    print("Currency model unloaded.")
