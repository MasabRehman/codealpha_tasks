# ✅ Task 4: Medical Disease Prediction & Clinical Decision Support System

## 📌 Project Overview
This project provides an AI-powered clinical decision support system that predicts the likelihood of multiple diseases based on structured patient data, laboratory vitals, cytology measurements, and clinical symptom profiles.

---

## 🏥 Covered Disease Modules & Datasets

1. **Heart Disease Risk (UCI Cleveland Database)**:
   - Evaluates 13 cardiovascular markers including resting BP, cholesterol, chest pain type, max heart rate, and ST depression.
2. **Type-2 Diabetes Diagnosis (PIMA Indian Diabetes)**:
   - Evaluates insulin sensitivity, plasma glucose, BMI, blood pressure, and genetic diabetes pedigree.
3. **Breast Cancer Cytology Biopsy (UCI Wisconsin Diagnostic)**:
   - Distinguishes Benign from Malignant tumors using 30 nuclear morphology characteristics.
4. **Symptom-Based Disease Matching**:
   - Matches clinical symptoms (fever, cough, chest tightness, fatigue, nausea, etc.) across differential diagnoses (COVID-19, Pneumonia, Bronchitis, Flu, Migraine, Arthritis, etc.).

---

## 🧠 Machine Learning Algorithms Evaluated

- **Support Vector Machines (SVM)** with Radial Basis Function (RBF) kernel & probability calibration.
- **Random Forest Classifier** with Gini impurity feature selection.
- **Gradient Tree Boosting / XGBoost** for structured tabular classification.
- **Logistic Regression** for clinical odds-ratio interpretability.
- **k-Nearest Neighbors (k-NN)** for non-parametric proximity estimation.

---

## 📂 Directory Structure

```
Task4_Disease_Prediction_from_Medical_Data/
├── data/
│   ├── medical_datasets.py      # Multi-disease benchmark dataset generator & loaders
│   ├── heart_disease.csv        # Cardiovascular dataset
│   ├── diabetes.csv             # Metabolic dataset
│   ├── breast_cancer.csv        # Oncology dataset
│   └── symptoms_disease.csv     # Symptom-disease relational dataset
├── src/
│   ├── preprocessing.py         # Clinical scaling, imputation, and risk stratification
│   └── models.py                # Machine learning classification models
├── models/
│   ├── heart_disease_best_model.joblib
│   ├── diabetes_best_model.joblib
│   ├── breast_cancer_best_model.joblib
│   └── symptoms_disease_best_model.joblib
├── results/
│   ├── medical_evaluation_curves.png # ROC curves and confusion matrices
│   └── medical_benchmarks.json       # Clinical benchmark metrics
├── train.py                     # Multi-disease training & benchmark pipeline
├── evaluate.py                  # Evaluation & visualization script
├── predict.py                   # Clinical diagnostic inference engine
├── app.py                       # Multi-tab Streamlit Clinical Assistant Web App
└── requirements.txt             # Python dependencies
```

---

## 🚀 Quickstart & Usage

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train All Disease Models
```bash
python train.py
```

### 3. Evaluate & Generate ROC-AUC Curves
```bash
python evaluate.py
```

### 4. Run CLI Diagnostic Predictions
```bash
python predict.py
```

### 5. Launch Clinical Web Assistant
```bash
streamlit run app.py
```
