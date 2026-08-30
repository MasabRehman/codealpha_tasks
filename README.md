# Machine Learning & Deep Learning Project Suite

This repository contains 4 machine learning and deep learning tasks organized into clean, self-contained folders with automated pipelines for dataset processing, model training, evaluation metrics, visual reporting, inference CLIs, and interactive **Flask web apps**.


---

## 📁 Repository Overview

| Task | Title | Domain / Problem | Algorithms & Architectures | Key Evaluation Metrics |
| :--- | :--- | :--- | :--- | :--- |
| **Task 1** | **Credit Scoring Model** | Creditworthiness & default risk prediction | Logistic Regression, Decision Tree, Random Forest, HistGradientBoosting | ROC-AUC, PR-AUC, KS-Statistic, F1-Score |
| **Task 2** | **Emotion Recognition from Speech** | Speech emotion classification (6 emotions) | PyTorch 1D-CNN, Bidirectional LSTM, MFCCs, Spectrograms | Multi-class Accuracy, Per-emotion F1, Confusion Matrix |
| **Task 3** | **Handwritten Character Recognition** | Character/digit & word sequence recognition | Deep PyTorch CNN (Conv2D+BatchNorm+Dropout), Contour Segmentation | Top-1 & Top-3 Accuracy, Confusion Matrix |
| **Task 4** | **Disease Prediction from Medical Data** | Multi-disease diagnostic portal (Heart, Diabetes, Cancer, Symptoms) | SVM (RBF kernel), Random Forest, XGBoost, Logistic Regression | Sensitivity (Recall), Specificity, ROC-AUC |

---

## 🚀 Quick Execution

### Run All 4 Tasks Sequentially:
```bash
python run_all.py
```

### Launch Individual Web Apps (Flask — runs on localhost directly):
```bash
# Task 1 — Credit Scoring Dashboard  →  http://localhost:5001
cd Task1_Credit_Scoring_Model && python app.py

# Task 2 — Speech Emotion Recognition  →  http://localhost:5002
cd Task2_Emotion_Recognition_from_Speech && python app.py

# Task 3 — Handwritten Character Recognition  →  http://localhost:5003
cd Task3_Handwritten_Character_Recognition && python app.py

# Task 4 — Medical Disease Diagnostic Portal  →  http://localhost:5004
cd Task4_Disease_Prediction_from_Medical_Data && python app.py
```

> ⚠️ These are **Flask** apps. Do NOT run with `streamlit run app.py`.

---

## 📑 Detailed Task Breakdown

### [Task 1: Credit Scoring Model](./Task1_Credit_Scoring_Model/README.md)
- Automated generation and preprocessing of 6,000 credit records.
- Feature engineering: Debt-to-Income (DTI), Payment-to-Income (PTI), Delinquency Severity Index, Savings Ratios.
- Continuous 300–850 FICO score conversion engine with underwriting decision logic (Approval, Conditional, Decline).

### [Task 2: Emotion Recognition from Speech](./Task2_Emotion_Recognition_from_Speech/README.md)
- Speech signal processing extracting 40 MFCC coefficients, Mel-Spectrograms, Chroma, and Zero Crossing Rates.
- Deep Learning 1D-CNN with temporal pooling and AdamW optimizer with Cosine Annealing.
- Classifies: `Angry`, `Happy`, `Sad`, `Neutral`, `Fear`, `Surprise`.

### [Task 3: Handwritten Character Recognition](./Task3_Handwritten_Character_Recognition/README.md)
- Multi-layer Deep CNN trained on character images with data augmentation (Rotations, Affine transforms).
- Automatic thresholding, image centering, polarity inversion, and vertical projection word segmentation.

### [Task 4: Disease Prediction from Medical Data](./Task4_Disease_Prediction_from_Medical_Data/README.md)
- Multi-disease diagnostic portal covering Heart Disease (UCI), Type-2 Diabetes (PIMA), Breast Cancer Cytology (Wisconsin), and Symptom-based condition matching.
- SVM with RBF kernel and probability calibration, Random Forest, and Gradient Boosting with clinical risk factor explanations.
