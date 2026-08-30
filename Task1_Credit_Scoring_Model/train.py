"""
Task 1: Model Training and Comparative Evaluation Pipeline for Credit Scoring
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_validate, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
from sklearn.pipeline import Pipeline

from data.generate_dataset import generate_credit_data
from src.feature_engineering import get_preprocessor
from src.models import get_candidate_models

def train_credit_scoring_models():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "data")
    models_dir = os.path.join(base_dir, "models")
    results_dir = os.path.join(base_dir, "results")
    
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    dataset_file = os.path.join(data_dir, "credit_scoring_dataset.csv")
    if not os.path.exists(dataset_file):
        print("[Task 1] Generating fresh synthetic credit scoring dataset...")
        df = generate_credit_data(n_samples=6000, random_state=42)
        df.to_csv(dataset_file, index=False)
    else:
        print(f"[Task 1] Loading dataset from {dataset_file}...")
        df = pd.read_csv(dataset_file)
        
    X = df.drop(columns=["Credit_Default", "Credit_Score_Category"])
    y = df["Credit_Default"]
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Save test set for evaluation and validation
    test_df = X_test.copy()
    test_df["Credit_Default"] = y_test
    test_df.to_csv(os.path.join(data_dir, "test_data.csv"), index=False)
    print(f"[Task 1] Train samples: {len(X_train)}, Test samples: {len(X_test)}")
    
    candidate_models = get_candidate_models()
    preprocessor = get_preprocessor()
    
    benchmark_results = {}
    best_model_name = None
    best_roc_auc = -1.0
    best_pipeline = None
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    print("\n" + "="*70)
    print("           CREDIT SCORING MODEL BENCHMARK RESULTS")
    print("="*70)
    print(f"{'Model':<22} | {'Accuracy':<9} | {'Precision':<9} | {'Recall':<8} | {'F1-Score':<8} | {'ROC-AUC':<8}")
    print("-" * 70)
    
    for name, clf in candidate_models.items():
        pipeline = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("classifier", clf)
        ])
        
        # Fit on training data
        pipeline.fit(X_train, y_train)
        
        # Predict probabilities and labels on test data
        if hasattr(pipeline, "predict_proba"):
            y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
        elif hasattr(pipeline, "decision_function"):
            y_pred_proba = pipeline.decision_function(X_test)
        else:
            y_pred_proba = pipeline.predict(X_test)
            
        y_pred = pipeline.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        pr_auc = average_precision_score(y_test, y_pred_proba)
        
        benchmark_results[name] = {
            "Accuracy": round(float(acc), 4),
            "Precision": round(float(prec), 4),
            "Recall": round(float(rec), 4),
            "F1_Score": round(float(f1), 4),
            "ROC_AUC": round(float(roc_auc), 4),
            "PR_AUC": round(float(pr_auc), 4)
        }
        
        print(f"{name:<22} | {acc:.4f}   | {prec:.4f}    | {rec:.4f}   | {f1:.4f}   | {roc_auc:.4f}")
        
        # Save individual model
        joblib.dump(pipeline, os.path.join(models_dir, f"{name.lower()}_pipeline.joblib"))
        
        if roc_auc > best_roc_auc:
            best_roc_auc = roc_auc
            best_model_name = name
            best_pipeline = pipeline
            
    print("=" * 70)
    print(f"\n[Task 1] Winner: {best_model_name} with ROC-AUC = {best_roc_auc:.4f}")
    
    # Save best model
    best_model_path = os.path.join(models_dir, "credit_scoring_best_model.joblib")
    joblib.dump(best_pipeline, best_model_path)
    print(f"[Task 1] Saved best model to {best_model_path}")
    
    # Save benchmark summary
    summary_path = os.path.join(results_dir, "model_comparison.json")
    with open(summary_path, "w") as f:
        json.dump({
            "best_model": best_model_name,
            "best_roc_auc": best_roc_auc,
            "comparison": benchmark_results
        }, f, indent=4)
    print(f"[Task 1] Benchmark metrics saved to {summary_path}")
    
    return best_model_name, benchmark_results

if __name__ == "__main__":
    train_credit_scoring_models()
