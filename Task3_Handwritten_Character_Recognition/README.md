# ✅ Task 3: Handwritten Character & Digit Recognition System

## 📌 Project Overview
This project delivers a **Deep Convolutional Neural Network (CNN)** for recognizing handwritten characters, digits, and multi-character text sequences (MNIST / EMNIST). It integrates automated image preprocessing (thresholding, contour extraction, bounding-box centering) and vertical projection word segmentation.

---

## 🏗️ Architecture & Pipeline

1. **Image Preprocessing & Normalization**:
   - **Auto-Inversion**: Automatically detects and aligns background contrast (ensuring white strokes on black background).
   - **Thresholding**: Adaptive Otsu-style thresholding to remove illumination noise.
   - **Aspect Ratio Bounding-Box Centering**: Fits characters neatly within a 28x28 matrix with padding.
   - **Standardization**: Normalized with $\mu = 0.1307, \sigma = 0.3081$.

2. **Deep CNN Architecture**:
   - Conv2d(1 $\to$ 32, 3x3) + BatchNorm2d + ReLU
   - Conv2d(32 $\to$ 64, 3x3) + BatchNorm2d + ReLU + MaxPool2d(2x2) + Dropout(0.25)
   - Conv2d(64 $\to$ 128, 3x3) + BatchNorm2d + ReLU + MaxPool2d(2x2) + Dropout(0.25)
   - Conv2d(128 $\to$ 256, 3x3) + BatchNorm2d + ReLU + AdaptiveAvgPool2d((2, 2)) + Dropout(0.3)
   - Dense(1024 $\to$ 256) $\to$ BatchNorm1d $\to$ Dropout(0.4) $\to$ Softmax Dense Output.

3. **Multi-Character Word Segmentation**:
   - Vertical projection profile identifies spacing gaps between adjacent characters to segment full handwritten words or PIN numbers into sequential character crops.

4. **Evaluation Metrics**:
   - Top-1 and Top-3 Classification Accuracy
   - Per-character Precision, Recall, and F1-Score
   - Multi-class Confusion Matrix
   - Visual Prediction Grid with confidence scores

---

## 📂 Directory Structure

```
Task3_Handwritten_Character_Recognition/
├── data/
│   ├── dataset_loader.py        # MNIST / EMNIST dataset loader and synthetic generator
│   └── sample_images/           # Sample test images (digits & letters)
├── src/
│   ├── image_processing.py      # Thresholding, centering, and word segmentation algorithms
│   └── models.py                # PyTorch Deep CNN & CRNN sequence model definitions
├── models/
│   ├── handwritten_cnn_best.pth # Trained PyTorch CNN model weights
│   └── class_mapping.json       # Class index to character symbol mapping
├── results/
│   ├── evaluation_matrix.png    # Confusion matrix and sample prediction visualization
│   └── evaluation_metrics.json  # Comprehensive metric assessment
├── train.py                     # Training script with data augmentation
├── evaluate.py                  # Evaluation & visualization script
├── predict.py                   # Single character & multi-character word inference engine
├── app.py                       # Interactive Streamlit Web App with live image tester
└── requirements.txt             # Python dependencies
```

---

## 🚀 Quickstart & Usage

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train CNN on Character Dataset
```bash
python train.py
```

### 3. Evaluate Model & Generate Confusion Matrix
```bash
python evaluate.py
```

### 4. Run CLI Prediction on Sample Images
```bash
python predict.py
```

### 5. Launch Web Application (Flask)
```bash
python app.py
```
Then open your browser at: **http://localhost:5003**

> ⚠️ Do NOT use `streamlit run app.py` — this is a Flask app, not a Streamlit app.
