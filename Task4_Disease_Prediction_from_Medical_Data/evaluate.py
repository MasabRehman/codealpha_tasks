"""
Task 4: Medical Diagnostic Evaluation & ROC-AUC Analytics
"""

import os
import sys
import json
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, classification_report

base_dir = os.path.dirname(os.path.abspath(__file__))
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

def evaluate_disease_models():
    data_dir = os.path.join(base_dir, "data")
    models_dir = os.path.join(base_dir, "models")
    results_dir = os.path.join(base_dir, "results")
    
    tasks = [
        ("heart_disease", "target", "Heart Disease Risk"),
        ("diabetes", "Outcome", "Diabetes Diagnosis"),
        ("breast_cancer", "target", "Breast Cancer Diagnosis")
    ]
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    summary_eval = {}
    
    print("\n" + "="*70)
    print("      TASK 4: MEDICAL DISEASE PREDICTION EVALUATION REPORT")
    print("="*70)
    
    for i, (name, target_col, title) in enumerate(tasks):
        test_file = os.path.join(data_dir, f"{name}_test.csv")
        model_file = os.path.join(models_dir, f"{name}_best_model.joblib")
        
        if not os.path.exists(test_file) or not os.path.exists(model_file):
            continue
            
        df_test = pd.read_csv(test_file)
        X_test = df_test.drop(columns=[target_col])
        y_test = df_test[target_col].values
        
        model = joblib.load(model_file)
        y_pred = model.predict(X_test)
        
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
        else:
            y_prob = y_pred
            
        roc_auc = roc_auc_score(y_test, y_prob)
        cm = confusion_matrix(y_test, y_pred)
        
        # Sensitivity (TPR) and Specificity (TNR)
        tn, fp, fn, tp = cm.ravel()
        sensitivity = tp / (tp + fn + 1e-6)
        specificity = tn / (tn + fp + 1e-6)
        
        print(f"\n--- {title} ---")
        print(f"ROC-AUC:     {roc_auc:.4f}")
        print(f"Sensitivity (True Positive Rate): {sensitivity*100:.2f}%")
        print(f"Specificity (True Negative Rate): {specificity*100:.2f}%")
        
        summary_eval[name] = {
            "ROC_AUC": float(roc_auc),
            "Sensitivity": float(sensitivity),
            "Specificity": float(specificity),
            "Confusion_Matrix": cm.tolist()
        }
        
        # Plot 1: ROC Curve (Top Row)
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        axes[0, i].plot(fpr, tpr, color="#2980b9", lw=2.5, label=f"ROC (AUC = {roc_auc:.3f})")
        axes[0, i].plot([0, 1], [0, 1], color="gray", linestyle="--")
        axes[0, i].set_title(f"{title} - ROC Curve", fontsize=11, fontweight="bold")
        axes[0, i].set_xlabel("False Positive Rate (1 - Specificity)")
        axes[0, i].set_ylabel("True Positive Rate (Sensitivity)")
        axes[0, i].legend(loc="lower right")
        axes[0, i].grid(True, alpha=0.3)
        
        # Plot 2: Confusion Matrix (Bottom Row)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[1, i], cbar=False)
        axes[1, i].set_title(f"{title} - Confusion Matrix", fontsize=11, fontweight="bold")
        axes[1, i].set_xlabel("Predicted")
        axes[1, i].set_ylabel("Actual")
        
    plt.tight_layout()
    plot_path = os.path.join(results_dir, "medical_evaluation_curves.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"\n[Task 4] Saved multi-disease evaluation plots to {plot_path}")
    
    with open(os.path.join(results_dir, "diagnostic_metrics.json"), "w") as f:
        json.dump(summary_eval, f, indent=4)
        
    return summary_eval

if __name__ == "__main__":
    evaluate_disease_models()
