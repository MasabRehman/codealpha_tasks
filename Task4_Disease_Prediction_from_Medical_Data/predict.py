"""
Task 4: Clinical Diagnostic Inference Engine & Risk Stratification
Supports Heart Disease, Diabetes, Breast Cancer, and Multi-Symptom Disease Prediction.
"""

import os
import sys
import joblib
import pandas as pd
import numpy as np

base_dir = os.path.dirname(os.path.abspath(__file__))
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

from src.preprocessing import stratify_medical_risk

def load_disease_model(disease_name: str):
    model_path = os.path.join(base_dir, "models", f"{disease_name}_best_model.joblib")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model for {disease_name} not found at {model_path}. Run train.py first.")
    return joblib.load(model_path)

def predict_heart_disease(patient_data: dict) -> dict:
    """
    Evaluates cardiovascular disease risk from patient vitals.
    """
    model = load_disease_model("heart_disease")
    df = pd.DataFrame([patient_data])
    prob = float(model.predict_proba(df)[0, 1])
    risk_info = stratify_medical_risk(prob)
    
    risk_factors = []
    if patient_data.get("trestbps", 0) > 135:
        risk_factors.append(f"Elevated resting blood pressure ({patient_data['trestbps']} mmHg)")
    if patient_data.get("chol", 0) > 220:
        risk_factors.append(f"Elevated total serum cholesterol ({patient_data['chol']} mg/dl)")
    if patient_data.get("exang", 0) == 1:
        risk_factors.append("Exercise-induced angina detected")
    if patient_data.get("oldpeak", 0) > 1.5:
        risk_factors.append(f"ST depression ({patient_data['oldpeak']}) indicates potential myocardial ischemia")
    if patient_data.get("ca", 0) > 0:
        risk_factors.append(f"Fluoroscopy shows {patient_data['ca']} major coronary vessels with calcification/narrowing")
        
    if not risk_factors:
        risk_factors.append("Cardiovascular biomarkers are within optimal clinical thresholds.")
        
    return {
        "condition": "Cardiovascular Disease",
        "probability": round(prob, 4),
        "risk_level": risk_info["risk_level"],
        "risk_percentage": risk_info["risk_percentage"],
        "clinical_advice": risk_info["clinical_advice"],
        "risk_factors": risk_factors
    }

def predict_diabetes(patient_data: dict) -> dict:
    """
    Evaluates Type-2 diabetes risk from metabolic and lab measurements.
    """
    model = load_disease_model("diabetes")
    df = pd.DataFrame([patient_data])
    prob = float(model.predict_proba(df)[0, 1])
    risk_info = stratify_medical_risk(prob)
    
    risk_factors = []
    if patient_data.get("Glucose", 0) >= 126:
        risk_factors.append(f"Fasting Plasma Glucose ({patient_data['Glucose']} mg/dL) exceeds diabetic threshold (126 mg/dL)")
    elif patient_data.get("Glucose", 0) >= 100:
        risk_factors.append(f"Fasting Glucose ({patient_data['Glucose']} mg/dL) indicates pre-diabetes impaired tolerance")
    if patient_data.get("BMI", 0) >= 30.0:
        risk_factors.append(f"BMI ({patient_data['BMI']}) classified in clinical obesity range")
    if patient_data.get("DiabetesPedigreeFunction", 0) > 0.6:
        risk_factors.append("Strong familial genetic predisposition (High DPF score)")
    if patient_data.get("BloodPressure", 0) > 85:
        risk_factors.append(f"Elevated diastolic blood pressure ({patient_data['BloodPressure']} mmHg)")
        
    if not risk_factors:
        risk_factors.append("Metabolic and glucose markers are within healthy reference ranges.")
        
    return {
        "condition": "Type-2 Diabetes Mellitus",
        "probability": round(prob, 4),
        "risk_level": risk_info["risk_level"],
        "risk_percentage": risk_info["risk_percentage"],
        "clinical_advice": risk_info["clinical_advice"],
        "risk_factors": risk_factors
    }

def predict_breast_cancer(biopsy_data: dict) -> dict:
    """
    Classifies fine needle aspirate (FNA) biopsy features as Benign or Malignant.
    """
    model = load_disease_model("breast_cancer")
    df = pd.DataFrame([biopsy_data])
    prob = float(model.predict_proba(df)[0, 1])
    diagnosis = "Benign" if prob > 0.5 else "Malignant"
    conf = prob if diagnosis == "Benign" else (1.0 - prob)
    
    return {
        "condition": "Breast Tissue Biopsy",
        "diagnosis": diagnosis,
        "malignancy_risk": f"{(1.0 - prob)*100:.1f}%",
        "confidence": f"{conf*100:.1f}%",
        "recommendation": "Follow-up routine mammography" if diagnosis == "Benign" else "Immediate oncology consultation & tissue histology biopsy"
    }

def predict_disease_from_symptoms(symptoms_list: list) -> dict:
    """
    Matches reported clinical symptoms against multi-condition diagnostic models.
    """
    model = load_disease_model("symptoms_disease")
    le = joblib.load(os.path.join(base_dir, "models", "symptoms_disease_label_encoder.joblib"))
    
    # Load dataset columns
    symptoms_df = pd.read_csv(os.path.join(base_dir, "data", "symptoms_disease.csv"))
    all_symptom_cols = [c for c in symptoms_df.columns if c != "Disease_Diagnosis"]
    
    input_vec = {col: (1 if col in symptoms_list else 0) for col in all_symptom_cols}
    df = pd.DataFrame([input_vec])
    
    probs = model.predict_proba(df)[0]
    top_indices = np.argsort(probs)[::-1][:3]
    
    top_predictions = [
        {"disease": le.classes_[idx], "probability": f"{probs[idx]*100:.1f}%"}
        for idx in top_indices if probs[idx] > 0.05
    ]
    
    primary_dx = top_predictions[0]["disease"] if top_predictions else "Inconclusive"
    
    return {
        "primary_suspected_disease": primary_dx,
        "matched_symptoms": symptoms_list,
        "differential_diagnoses": top_predictions
    }

if __name__ == "__main__":
    print("=" * 65)
    print("      TASK 4: MEDICAL CLINICAL DIAGNOSTIC ENGINE")
    print("=" * 65)
    
    sample_heart = {
        "age": 62, "sex": 1, "cp": 2, "trestbps": 145, "chol": 265,
        "fbs": 0, "restecg": 1, "thalach": 128, "exang": 1, "oldpeak": 2.2,
        "slope": 1, "ca": 2, "thal": 3
    }
    
    sample_diabetes = {
        "Pregnancies": 4, "Glucose": 158, "BloodPressure": 88,
        "SkinThickness": 32, "Insulin": 180, "BMI": 34.2,
        "DiabetesPedigreeFunction": 0.72, "Age": 45
    }
    
    try:
        print("\n[Assessment 1: Heart Disease Evaluation]")
        res1 = predict_heart_disease(sample_heart)
        print(f"Risk Level:  {res1['risk_level']} ({res1['risk_percentage']})")
        print(f"Advice:      {res1['clinical_advice']}")
        print("Risk Factors:" + ", ".join(res1['risk_factors']))
        
        print("\n[Assessment 2: Diabetes Risk Evaluation]")
        res2 = predict_diabetes(sample_diabetes)
        print(f"Risk Level:  {res2['risk_level']} ({res2['risk_percentage']})")
        print(f"Advice:      {res2['clinical_advice']}")
        print("Risk Factors:" + ", ".join(res2['risk_factors']))
        
        print("\n[Assessment 3: Symptom Checker]")
        res3 = predict_disease_from_symptoms(["high_fever", "chest_pain", "shortness_of_breath", "cough"])
        print(f"Primary Suspected Condition: {res3['primary_suspected_disease']}")
        print(f"Differential Diagnoses:      {res3['differential_diagnoses']}")
        print("=" * 65)
    except Exception as e:
        print(f"Note: Run train.py first to train models: {e}")
