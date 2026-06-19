# vqa_module.py
# Visual Question Answering (LLaVA) module
# Provides: answer_question(image, english_question, max_new_tokens=200)
# Also: unload_model() to free GPU memory

import os
import warnings
from typing import Union

import torch
from PIL import Image
import numpy as np

# Try to import Llava class and processor; the exact import may depend on the package version.
# If your environment provides a different import path adjust accordingly.
try:
    from transformers import AutoProcessor
    from transformers import LlavaForConditionalGeneration
except Exception as e:
    # keep the import error visible but do not crash on import; the model will fail to load later with clear message
    LlavaForConditionalGeneration = None
    AutoProcessor = None
    warnings.warn(f"Could not import Llava classes from transformers here: {e}. "
                  "Make sure package versions are installed in the Colab environment.")

# Module-level cached model & processor
_model = None
_processor = None
_model_id = "llava-hf/llava-1.5-7b-hf"  # change if you want a different LLaVA model


def _pil_from_input(image: Union[np.ndarray, Image.Image]) -> Image.Image:
    """
    Accepts an OpenCV ndarray (BGR) or a PIL Image and returns a PIL Image in RGB mode.
    """
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if isinstance(image, np.ndarray):
        # OpenCV uses BGR order, convert to RGB
        if image.ndim == 3 and image.shape[2] == 3:
            img_rgb = image[:, :, ::-1]
            return Image.fromarray(img_rgb).convert("RGB")
        else:
            # grayscale or other
            return Image.fromarray(image).convert("RGB")
    raise ValueError("Input image must be a PIL.Image or an OpenCV (numpy) array.")


def _get_device():
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def _load_model(device: torch.device = None, model_id: str = None):
    """
    Lazy-load the model and processor into module-level variables.
    Uses 4-bit loading if possible to reduce memory (dependent on bitsandbytes & transformers support).
    """
    global _model, _processor, _model_id

    if model_id:
        _model_id = model_id

    if _model is not None and _processor is not None:
        return _model, _processor

    if AutoProcessor is None or LlavaForConditionalGeneration is None:
        raise ImportError(
            "Required classes (AutoProcessor, LlavaForConditionalGeneration) are not available. "
            "Ensure you installed the correct transformers/llava package in Colab."
        )

    device = device or _get_device()
    print(f"[vqa_module] Loading model {_model_id} on device {device} ... (this may take a while)")

    # Recommended: try low-mem loading settings first (4-bit + device_map='auto'), fallback to safer CPU float16 if fails
    try:
        # Preferred (if environment supports bitsandbytes + 4-bit)
        _model = LlavaForConditionalGeneration.from_pretrained(
            _model_id,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            load_in_4bit=True,
            device_map="auto",
        )
        _processor = AutoProcessor.from_pretrained(_model_id)
        print("[vqa_module] Model loaded with 4-bit + device_map='auto'.")
    except Exception as e:
        warnings.warn(f"4-bit loading failed or not supported in this environment: {e}. "
                      "Falling back to CPU/float16 load (may be slow).")
        # fallback: CPU/float16
        _model = LlavaForConditionalGeneration.from_pretrained(
            _model_id,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            device_map="auto" if torch.cuda.is_available() else None,
        )
        _processor = AutoProcessor.from_pretrained(_model_id)
        print("[vqa_module] Model loaded (fallback).")

    # Move to CPU if device is cpu, or leave device_map manage placement
    if device.type == "cpu":
        _model.to("cpu")

    return _model, _processor


def unload_model():
    """
    Unload the model and free GPU memory. Call this after you finish using the module to avoid memory hogging.
    """
    global _model, _processor
    try:
        if _model is not None:
            del _model
        if _processor is not None:
            del _processor
    except Exception:
        pass
    _model = None
    _processor = None
    # free gpu memory
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass
    print("[vqa_module] Model and processor unloaded.")


def answer_question(image: Union[np.ndarray, Image.Image],
                    english_question: str,
                    max_new_tokens: int = 200,
                    model_id: str = None) -> str:
    """
    Run LLaVA VQA on the provided image and english_question.
    - image: PIL image or OpenCV ndarray (BGR)
    - english_question: question text in English (master notebook should translate to English first)
    - returns: answer string in English

    This function lazily loads the model on first call and keeps it in memory until unload_model() is called.
    """
    global _model, _processor

    if not isinstance(english_question, str) or len(english_question.strip()) == 0:
        raise ValueError("Please provide a non-empty English question string.")

    # Convert image to PIL
    pil_img = _pil_from_input(image)

    # Load model if needed
    device = _get_device()
    if _model is None or _processor is None:
        _load_model(device=device, model_id=model_id)

    if _model is None or _processor is None:
        raise RuntimeError("Model failed to load. Check logs and package installation in the Colab environment.")

    # Build prompt (this follows your earlier format)
    prompt = f"USER: <image>\n{english_question}\nASSISTANT:"

    # Prepare inputs through processor
    # Important: put tensors to CPU first to avoid some device_map issues, model generation will handle placement
    inputs = _processor(images=pil_img, text=prompt, return_tensors="pt")
    # move inputs to model device if model expects them there (device_map auto usually handles it)
    # But to be safe, we will not force move here; many Llava variants handle CPU tensors with device_map placement.

    # Generate
    _model.eval()
    with torch.no_grad():
        try:
            output = _model.generate(**inputs, max_new_tokens=max_new_tokens)
        except Exception as e:
            # If generation fails due to device mismatch, try moving inputs to model device
            try:
                model_device = next(_model.parameters()).device
                inputs = {k: v.to(model_device) for k, v in inputs.items()}
                output = _model.generate(**inputs, max_new_tokens=max_new_tokens)
            except Exception as ee:
                raise RuntimeError(f"Generation failed: {e} / fallback failed: {ee}")

    # Decode output
    try:
        answer = _processor.decode(output[0], skip_special_tokens=True)
    except Exception:
        # If processor has no decode, try model's tokenizer or raw decode
        try:
            answer = output[0]
            if isinstance(answer, torch.Tensor):
                answer = answer.cpu().numpy().tolist()
            answer = str(answer)
        except Exception:
            answer = "Could not decode model output."

    # Post-process answer
    answer = answer.strip()
    if len(answer) == 0:
        answer = "No answer generated."

    return answer
