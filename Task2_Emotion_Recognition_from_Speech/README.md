# ✅ Task 2: Speech Emotion Recognition (SER) System

## 📌 Project Overview
This project implements an end-to-end **Speech Emotion Recognition (SER)** system capable of recognizing human emotional states (**Happy, Sad, Angry, Neutral, Fear, Surprise**) from acoustic speech waveforms. It combines speech signal processing (MFCCs, Spectrograms, Chroma, ZCR) with Deep Learning (**1D Convolutional Neural Networks & Bidirectional LSTMs** in PyTorch).

---

## 🏗️ Architecture & Signal Processing

1. **Acoustic Feature Extraction**:
   - **MFCCs (Mel-Frequency Cepstral Coefficients)**: 40 coefficients capturing timbral envelope and vocal tract resonance.
   - **Mel-Spectrogram**: 64-band frequency power distribution on human auditory Mel scale.
   - **Zero-Crossing Rate (ZCR) & RMS Energy**: Dynamic prosody and sound intensity envelope.
   - **Chroma & Spectral Contrast**: Pitch distribution and harmonic clarity.

2. **Deep Learning Model (1D-CNN)**:
   - **Input**: $(B, 40, T)$ temporal MFCC sequence.
   - **Architecture**:
     - Layer 1: Conv1d(40 $\to$ 64, kernel=5) + BatchNorm + MaxPool1d + Dropout(0.2)
     - Layer 2: Conv1d(64 $\to$ 128, kernel=5) + BatchNorm + MaxPool1d + Dropout(0.3)
     - Layer 3: Conv1d(128 $\to$ 256, kernel=3) + BatchNorm + Dropout(0.3)
     - Global Average Pooling $\to$ Dense(256 $\to$ 128) $\to$ BatchNorm $\to$ Linear(128 $\to$ 6 classes).
   - **Optimizer**: AdamW with Cosine Annealing learning rate schedule.

3. **Evaluation Metrics**:
   - Multi-class Accuracy & Macro/Weighted F1-Score
   - Per-emotion Precision and Recall
   - Confusion Matrix Heatmap
   - Training & Validation Loss/Accuracy Progression Curves

---

## 📂 Directory Structure

```
Task2_Emotion_Recognition_from_Speech/
├── data/
│   ├── audio_generator.py       # Audio synthesizer & dataset creator (RAVDESS/TESS format)
│   ├── dataset_manifest.csv     # Metadata index of audio files
│   └── audio_samples/           # Partitioned audio clips (angry, happy, sad, etc.)
├── src/
│   ├── feature_extraction.py    # MFCC, spectrogram, and acoustic feature extractors
│   └── models.py                # PyTorch 1D-CNN and BiLSTM model architectures
├── models/
│   ├── speech_emotion_best_model.pth # PyTorch model weights
│   └── label_encoder.joblib     # Label encoder mapping
├── results/
│   ├── emotion_evaluation.png   # Confusion matrix and learning curves
│   └── emotion_metrics.json     # Detailed evaluation metrics
├── train.py                     # Deep learning training pipeline
├── evaluate.py                  # Evaluation & visualization script
├── predict.py                   # Single audio / CLI inference engine
├── app.py                       # Interactive Streamlit Web App with audio player & waveform visualizer
└── requirements.txt             # Python dependencies
```

---

## 🚀 Quickstart & Usage

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train Model on Speech Dataset
```bash
python train.py
```

### 3. Evaluate & Generate Confusion Matrix Plots
```bash
python evaluate.py
```

### 4. Run CLI Emotion Prediction on Audio File
```bash
python predict.py
```

### 5. Launch Interactive Web App
```bash
streamlit run app.py
```
