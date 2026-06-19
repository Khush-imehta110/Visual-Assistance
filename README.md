# Visual Assistance System for Visually Impaired 👁️‍🗨️🎙️

An end-to-end **multimodal AI-based visual assistance system** that integrates **speech, vision, and language** to help visually impaired individuals understand their surroundings in real time.  
The system accepts **voice queries**, analyzes **images**, and delivers **spoken responses**, enabling greater independence in daily tasks.

---

## 🔍 Project Overview

Millions of visually impaired individuals rely on external assistance for basic tasks such as reading text, identifying objects, or recognizing currency.  
This project proposes a **fully voice-controlled intelligent assistant** capable of understanding images and answering user queries through speech.

The system combines:
- Speech Recognition  
- Machine Translation  
- Visual Question Answering (VQA)  
- Object Detection  
- Optical Character Recognition (OCR)  
- Indian Currency Recognition  
- Text-to-Speech synthesis  

All modules are integrated into a **single unified pipeline**.

---

## 🎯 Objectives

- Enable visually impaired users to interact with images using voice
- Answer visual questions (VQA)
- Read printed and scene text (OCR)
- Detect common objects in the environment
- Identify Indian currency denominations
- Support multilingual speech queries
- Provide spoken responses in real time

---

## 🧠 System Architecture

**Pipeline Flow:**

1. **Speech-to-Text** (User Voice Input)
2. **Language Translation** → English
3. **Intent Classification**
4. **Vision Module Execution**
5. **Response Generation**
6. **Text-to-Speech Output**

---

## 🛠️ Technologies & Models Used

### 🔊 Speech Processing
- **Whisper (Medium)** – Robust multilingual speech-to-text conversion

### 🌐 Translation
- **Google Translator API** – Converts regional languages to English

### 🧠 Intent Classification
Keyword-based routing to ensure fast and accurate module selection.

| Intent | Example Keywords | Module |
|------|------------------|--------|
| OCR | read, text, words | EasyOCR |
| Object Detection | detect, find objects | YOLOv8 |
| Currency | money, note, rupees | Currency Classifier |
| VQA | describe, explain | LLaVA |

---

## 👁️ Computer Vision Modules

### 🖼️ Visual Question Answering
- **Model:** LLaVA-1.5-7B (4-bit quantized)
- Handles open-ended questions with strong reasoning capabilities

### 🚗 Object Detection
- **YOLOv8**
- Fast, lightweight, real-time detection of everyday objects

### 📝 OCR
- **EasyOCR**
- Reads printed and scene text from images

### 💵 Indian Currency Recognition
- **Base Model:** MobileNetV3-Large  
- **Classes:** ₹10, ₹20, ₹50, ₹100, ₹200, ₹500, ₹2000  
- **Dataset:** ~200 images per class  
- **Accuracy:** ~80%

---

## 📊 Datasets Used

1. **Indian Currency Dataset**
   - 7 denominations
   - Custom collected images
   - Data augmentation applied

2. **User-Captured Images**
   - Real-world images via camera input

> Note: No benchmark VQA datasets were used; LLaVA relies on pretrained knowledge.

---

## 🧪 Experiments & Results

### 📈 Performance Metrics

| Module | Metric | Performance |
|------|-------|------------|
| Currency Classification | Accuracy | ~80% |
| OCR | Character Accuracy | 85–92% |
| VQA (LLaVA) | Qualitative | Strong reasoning |

### ✅ Sample Outcomes
- Correctly described activities in images
- Accurately detected objects and currency notes
- Successfully read printed and scene text

---

## ⚠️ Error Analysis

| Module | Observed Limitations |
|------|----------------------|
| LLaVA | Struggles with blurry or low-light images |
| EasyOCR | Weak on stylized or curved fonts |
| Currency Model | Reduced accuracy for folded/damaged notes |
| Whisper | Minor errors with heavy accents |

---

## 📌 Conclusion

The Visual Assistance System demonstrates the effectiveness of **multimodal AI** in accessibility applications.  
By combining speech, vision, and language models, the system offers a powerful, flexible, and open solution to assist visually impaired individuals in understanding their environment.

---

## 🚀 Future Enhancements

- Mobile deployment using ONNX / TensorRT
- Real-time obstacle detection & navigation
- Scene narration for continuous feedback
- Improved currency model accuracy
- Reduced inference latency
- GUI for desktop and mobile platforms

---

## ✅ Project Status

### ✔️ Completed
- Speech-to-text integration
- Multilingual translation
- Intent classification
- OCR, VQA, object detection
- Currency classifier training
- Full voice-driven pipeline
---

## 📚 References

- LLaVA – Large Language and Vision Assistant  
- Whisper Speech Recognition – OpenAI  
- YOLOv8 – Ultralytics  
- EasyOCR – JaidedAI  
- MobileNetV3 – Howard et al.  
- Research on multimodal assistive AI systems
