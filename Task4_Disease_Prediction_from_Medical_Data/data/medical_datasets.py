"""
Task 4: Medical Benchmark Datasets Generator & Curation
Creates standardized clinical datasets for Heart Disease, Diabetes, Breast Cancer, and Multi-Symptom Disease Prediction.
"""

import os
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer

def create_heart_disease_dataset(n_samples: int = 1200, random_state: int = 42) -> pd.DataFrame:
    """
    Generates realistic UCI Cleveland-style Heart Disease dataset.
    Features: Age, Sex, CP, Trestbps, Chol, Fbs, Restecg, Thalach, Exang, Oldpeak, Slope, Ca, Thal.
    Target: Target (1 = Heart Disease Present, 0 = Absent)
    """
    np.random.seed(random_state)
    
    age = np.random.normal(55, 9, n_samples).clip(28, 80).astype(int)
    sex = np.random.choice([1, 0], size=n_samples, p=[0.68, 0.32]) # 1: Male, 0: Female
    cp = np.random.choice([0, 1, 2, 3], size=n_samples, p=[0.48, 0.17, 0.28, 0.07]) # Chest pain type
    trestbps = np.random.normal(132, 18, n_samples).clip(90, 200).astype(int) # Resting blood pressure
    chol = np.random.normal(246, 50, n_samples).clip(120, 560).astype(int) # Serum cholestoral
    fbs = (np.random.rand(n_samples) < 0.15).astype(int) # Fasting blood sugar > 120 mg/dl
    restecg = np.random.choice([0, 1, 2], size=n_samples, p=[0.50, 0.48, 0.02])
    thalach = (220 - age - np.random.normal(20, 15, n_samples)).clip(70, 205).astype(int) # Max heart rate
    exang = np.random.choice([0, 1], size=n_samples, p=[0.67, 0.33]) # Exercise induced angina
    oldpeak = np.random.exponential(1.0, n_samples).clip(0, 6.2).round(1) # ST depression
    slope = np.random.choice([0, 1, 2], size=n_samples, p=[0.07, 0.46, 0.47])
    ca = np.random.choice([0, 1, 2, 3], size=n_samples, p=[0.58, 0.22, 0.13, 0.07]) # Major vessels
    thal = np.random.choice([1, 2, 3], size=n_samples, p=[0.06, 0.55, 0.39]) # Thalassemia
    
    # Disease risk calculation
    z = (
        0.04 * (age - 50) +
        0.5 * sex +
        0.6 * cp +
        0.015 * (trestbps - 120) +
        0.005 * (chol - 200) -
        0.03 * (thalach - 140) +
        0.9 * exang +
        0.7 * oldpeak +
        0.8 * ca +
        0.5 * (thal == 3) - 1.2
    )
    prob = 1 / (1 + np.exp(-z))
    target = (np.random.rand(n_samples) < prob).astype(int)
    
    df = pd.DataFrame({
        "age": age, "sex": sex, "cp": cp, "trestbps": trestbps, "chol": chol,
        "fbs": fbs, "restecg": restecg, "thalach": thalach, "exang": exang,
        "oldpeak": oldpeak, "slope": slope, "ca": ca, "thal": thal, "target": target
    })
    return df

def create_diabetes_dataset(n_samples: int = 1200, random_state: int = 42) -> pd.DataFrame:
    """
    Generates PIMA Indians-style Diabetes dataset.
    Features: Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age.
    Target: Outcome (1 = Diabetic, 0 = Non-Diabetic)
    """
    np.random.seed(random_state)
    
    age = np.random.normal(33, 11, n_samples).clip(21, 80).astype(int)
    pregnancies = np.random.poisson(lam=3, size=n_samples).clip(0, 17)
    glucose = np.random.normal(120, 32, n_samples).clip(50, 200).astype(int)
    blood_pressure = np.random.normal(70, 12, n_samples).clip(40, 122).astype(int)
    skin_thickness = np.random.normal(20, 10, n_samples).clip(0, 99).astype(int)
    insulin = np.random.exponential(scale=80, size=n_samples).clip(0, 846).astype(int)
    bmi = np.random.normal(32, 7, n_samples).clip(18.0, 67.0).round(1)
    dpf = np.random.exponential(scale=0.45, size=n_samples).clip(0.08, 2.42).round(3)
    
    z = (
        0.035 * (glucose - 100) +
        0.08 * (bmi - 25) +
        0.03 * (age - 30) +
        0.7 * dpf +
        0.003 * insulin +
        0.08 * pregnancies - 2.8
    )
    prob = 1 / (1 + np.exp(-z))
    outcome = (np.random.rand(n_samples) < prob).astype(int)
    
    df = pd.DataFrame({
        "Pregnancies": pregnancies, "Glucose": glucose, "BloodPressure": blood_pressure,
        "SkinThickness": skin_thickness, "Insulin": insulin, "BMI": bmi,
        "DiabetesPedigreeFunction": dpf, "Age": age, "Outcome": outcome
    })
    return df

def create_breast_cancer_dataset() -> pd.DataFrame:
    """
    Loads standard UCI Breast Cancer Wisconsin (Diagnostic) dataset.
    """
    cancer = load_breast_cancer(as_frame=True)
    df = cancer.frame.copy()
    return df

def create_symptom_disease_dataset(n_samples: int = 1500, random_state: int = 42) -> pd.DataFrame:
    """
    Generates multi-symptom clinical diagnostic dataset for 10 common conditions.
    """
    np.random.seed(random_state)
    
    diseases = {
        "Common Cold": ["cough", "runny_nose", "sneezing", "sore_throat", "mild_fever"],
        "Influenza (Flu)": ["high_fever", "chills", "body_aches", "fatigue", "headache", "cough"],
        "COVID-19": ["high_fever", "dry_cough", "loss_of_taste_smell", "shortness_of_breath", "fatigue"],
        "Pneumonia": ["high_fever", "chest_pain", "shortness_of_breath", "cough", "fatigue"],
        "Bronchitis": ["persistent_cough", "wheezing", "chest_tightness", "mucus_production"],
        "Gastroenteritis": ["nausea", "vomiting", "diarrhea", "abdominal_pain", "dehydration"],
        "Migraine": ["severe_headache", "sensitivity_to_light", "nausea", "throbbing_pain", "vision_blur"],
        "Hypertension": ["headache", "dizziness", "chest_palpitations", "shortness_of_breath"],
        "Diabetes": ["excessive_thirst", "frequent_urination", "fatigue", "unexplained_weight_loss"],
        "Arthritis": ["joint_pain", "joint_swelling", "joint_stiffness", "reduced_mobility"]
    }
    
    all_symptoms = sorted(list(set([s for syms in diseases.values() for s in syms])))
    
    records = []
    for _ in range(n_samples):
        disease = np.random.choice(list(diseases.keys()))
        symptom_vec = {s: 0 for s in all_symptoms}
        
        # Primary symptoms with 85% probability
        for s in diseases[disease]:
            if np.random.rand() < 0.85:
                symptom_vec[s] = 1
                
        # Random occasional noise symptom with 5% probability
        noise_symptom = np.random.choice(all_symptoms)
        if np.random.rand() < 0.08:
            symptom_vec[noise_symptom] = 1
            
        symptom_vec["Disease_Diagnosis"] = disease
        records.append(symptom_vec)
        
    return pd.DataFrame(records)

def save_all_medical_datasets(data_dir: str = None):
    if data_dir is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = base_dir
        
    os.makedirs(data_dir, exist_ok=True)
    
    # 1. Heart Disease
    df_heart = create_heart_disease_dataset(n_samples=1200)
    df_heart.to_csv(os.path.join(data_dir, "heart_disease.csv"), index=False)
    
    # 2. Diabetes
    df_diabetes = create_diabetes_dataset(n_samples=1200)
    df_diabetes.to_csv(os.path.join(data_dir, "diabetes.csv"), index=False)
    
    # 3. Breast Cancer
    df_cancer = create_breast_cancer_dataset()
    df_cancer.to_csv(os.path.join(data_dir, "breast_cancer.csv"), index=False)
    
    # 4. Symptoms
    df_symptoms = create_symptom_disease_dataset(n_samples=1500)
    df_symptoms.to_csv(os.path.join(data_dir, "symptoms_disease.csv"), index=False)
    
    print(f"[Task 4] All medical datasets generated and saved to {data_dir}")

if __name__ == "__main__":
    save_all_medical_datasets()
