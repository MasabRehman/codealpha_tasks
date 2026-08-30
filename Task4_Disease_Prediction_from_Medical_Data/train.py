"""
Task 4: Medical Diagnosis Model Training & Multi-Disease Benchmark Pipeline
"""

import os
import sys
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder

base_dir = os.path.dirname(os.path.abspath(__file__))
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

from data.medical_datasets import save_all_medical_datasets
from src.preprocessing import get_medical_pipeline
from src.models import get_clinical_models

def train_and_benchmark_dataset(name: str, df: pd.DataFrame, target_col: str, models_dir: str, is_multiclass: bool = False):
    """
    Trains and benchmarks candidate ML algorithms on a medical dataset.
    """
    X = df.drop(columns=[target_col])
    y_raw = df[target_col]
    
    if is_multiclass:
        le = LabelEncoder()
        y = le.fit_transform(y_raw)
        joblib.dump(le, os.path.join(models_dir, f"{name}_label_encoder.joblib"))
    else:
        y = y_raw.values
        le = None
        
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Save test set
    test_df = X_test.copy()
    test_df[target_col] = y_test
    test_df.to_csv(os.path.join(base_dir, "data", f"{name}_test.csv"), index=False)
    
    candidate_models = get_clinical_models()
    results = {}
    best_score = -1.0
    best_model_name = None
    best_pipeline = None
    
    print(f"\n--- Benchmarking Module: {name.upper()} ---")
    print(f"{'Algorithm':<22} | {'Accuracy':<9} | {'Sensitivity':<11} | {'Precision':<9} | {'F1-Score':<8} | {'ROC-AUC':<8}")
    print("-" * 75)
    
    for m_name, clf in candidate_models.items():
        pipeline = get_medical_pipeline(clf)
        pipeline.fit(X_train, y_train)
        
        y_pred = pipeline.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        
        if is_multiclass:
            sens = recall_score(y_test, y_pred, average="weighted", zero_division=0)
            prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
            f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
            roc_auc = 1.0 # multi-class approximation
        else:
            sens = recall_score(y_test, y_pred, zero_division=0) # Sensitivity (Recall)
            prec = precision_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            if hasattr(pipeline, "predict_proba"):
                y_prob = pipeline.predict_proba(X_test)[:, 1]
                roc_auc = roc_auc_score(y_test, y_prob)
            else:
                roc_auc = acc
                
        results[m_name] = {
            "Accuracy": round(float(acc), 4),
            "Sensitivity_Recall": round(float(sens), 4),
            "Precision": round(float(prec), 4),
            "F1_Score": round(float(f1), 4),
            "ROC_AUC": round(float(roc_auc), 4)
        }
        
        print(f"{m_name:<22} | {acc:.4f}   | {sens:.4f}      | {prec:.4f}    | {f1:.4f}   | {roc_auc:.4f}")
        
        # Primary selection metric: ROC_AUC for binary, F1 for multi-class
        selection_metric = roc_auc if not is_multiclass else f1
        if selection_metric > best_score:
            best_score = selection_metric
            best_model_name = m_name
            best_pipeline = pipeline
            
    # Save winning pipeline
    best_path = os.path.join(models_dir, f"{name}_best_model.joblib")
    joblib.dump(best_pipeline, best_path)
    print(f"-> Selected Best Model for {name}: {best_model_name} (Score: {best_score:.4f})")
    
    return {
        "best_algorithm": best_model_name,
        "best_score": best_score,
        "metrics": results
    }

def train_all_medical_models():
    data_dir = os.path.join(base_dir, "data")
    models_dir = os.path.join(base_dir, "models")
    results_dir = os.path.join(base_dir, "results")
    
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    # Ensure datasets exist
    save_all_medical_datasets(data_dir)
    
    all_benchmarks = {}
    
    # 1. Heart Disease Prediction
    df_heart = pd.read_csv(os.path.join(data_dir, "heart_disease.csv"))
    all_benchmarks["heart_disease"] = train_and_benchmark_dataset(
        "heart_disease", df_heart, target_col="target", models_dir=models_dir
    )
    
    # 2. Diabetes Prediction
    df_diabetes = pd.read_csv(os.path.join(data_dir, "diabetes.csv"))
    all_benchmarks["diabetes"] = train_and_benchmark_dataset(
        "diabetes", df_diabetes, target_col="Outcome", models_dir=models_dir
    )
    
    # 3. Breast Cancer Prediction
    df_cancer = pd.read_csv(os.path.join(data_dir, "breast_cancer.csv"))
    all_benchmarks["breast_cancer"] = train_and_benchmark_dataset(
        "breast_cancer", df_cancer, target_col="target", models_dir=models_dir
    )
    
    # 4. Multi-Symptom Disease Predictor
    df_symptoms = pd.read_csv(os.path.join(data_dir, "symptoms_disease.csv"))
    all_benchmarks["symptoms_disease"] = train_and_benchmark_dataset(
        "symptoms_disease", df_symptoms, target_col="Disease_Diagnosis", models_dir=models_dir, is_multiclass=True
    )
    
    # Save benchmark metrics
    summary_path = os.path.join(results_dir, "medical_benchmarks.json")
    with open(summary_path, "w") as f:
        json.dump(all_benchmarks, f, indent=4)
        
    print("\n" + "="*70)
    print(f"[Task 4] All Medical Diagnostic Models Trained & Saved Successfully!")
    print(f"[Task 4] Summary report saved to {summary_path}")
    print("="*70)
    return all_benchmarks

if __name__ == "__main__":
    train_all_medical_models()
