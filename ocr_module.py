# ocr_module.py
# OCR module using EasyOCR
# Provides: extract_text(image)
#           unload_ocr() to free RAM

import easyocr
import numpy as np
from PIL import Image
import torch

# Cached EasyOCR reader
_reader = None


def _pil_from_input(image):
    """
    Converts OpenCV (BGR) or PIL image into a PIL RGB image.
    """
    if isinstance(image, Image.Image):
        return image.convert("RGB")

    if isinstance(image, np.ndarray):
        if image.ndim == 3 and image.shape[2] == 3:
            rgb = image[:, :, ::-1]   # BGR → RGB
            return Image.fromarray(rgb)
        else:
            return Image.fromarray(image).convert("RGB")

    raise ValueError("Input must be a PIL.Image or OpenCV ndarray.")


def _load_reader(lang_list=['en']):
    """
    Lazy-load EasyOCR reader (loads only once).
    """
    global _reader
    if _reader is None:
        print("[ocr_module] Loading EasyOCR reader...")
        _reader = easyocr.Reader(lang_list)
    return _reader


def unload_ocr():
    """
    Clear the OCR reader from memory (optional).
    """
    global _reader
    try:
        del _reader
        _reader = None
        torch.cuda.empty_cache()
        print("[ocr_module] OCR reader unloaded.")
    except Exception:
        pass


def extract_text(image):
    """
    Runs OCR on an input image (PIL or OpenCV).
    Returns a clean string of detected text.
    """

    global _reader

    pil_img = _pil_from_input(image)
    
    # Convert back to numpy for easyocr
    img_np = np.array(pil_img)

    # Load EasyOCR lazily
    reader = _load_reader()

    # Perform OCR
    results = reader.readtext(img_np)

    if not results:
        return "No readable text found in the image."

    # Collect text segments
    extracted = []
    for bbox, text, prob in results:
        extracted.append(text)

    if len(extracted) == 0:
        return "No readable text found in the image."

    return " ".join(extracted)
