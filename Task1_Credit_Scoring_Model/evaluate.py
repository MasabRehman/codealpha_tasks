"""
Task 1: Evaluation, Metrics Assessment & Visual Analytics for Credit Scoring
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve, roc_auc_score,
    precision_recall_curve, average_precision_score
)

def evaluate_credit_model():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "data")
    models_dir = os.path.join(base_dir, "models")
    results_dir = os.path.join(base_dir, "results")
    
    test_file = os.path.join(data_dir, "test_data.csv")
    model_file = os.path.join(models_dir, "credit_scoring_best_model.joblib")
    
    if not os.path.exists(test_file) or not os.path.exists(model_file):
        raise FileNotFoundError("Please run train.py first to produce models and test data.")
        
    df_test = pd.read_csv(test_file)
    X_test = df_test.drop(columns=["Credit_Default"])
    y_test = df_test["Credit_Default"]
    
    pipeline = joblib.load(model_file)
    y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
    y_pred = pipeline.predict(X_test)
    
    # 1. Classification Metrics
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    pr_auc = average_precision_score(y_test, y_pred_proba)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=["Good (No Default)", "High Risk (Default)"], output_dict=True)
    
    # 2. Kolmogorov-Smirnov (KS) Statistic (Standard in Credit Scoring Bureau Analysis)
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    ks_statistic = np.max(tpr - fpr)
    
    print("\n" + "="*60)
    print("       TASK 1: CREDIT SCORING MODEL EVALUATION REPORT")
    print("="*60)
    print(f"ROC-AUC Score:      {roc_auc:.4f}")
    print(f"PR-AUC Score:       {pr_auc:.4f}")
    print(f"KS Statistic:       {ks_statistic:.4f}")
    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred, target_names=["Good (No Default)", "High Risk (Default)"]))
    
    # 3. Visualization: Confusion Matrix & ROC/PR Curves
    plt.figure(figsize=(16, 5))
    
    # Plot 1: Confusion Matrix
    plt.subplot(1, 3, 1)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["Non-Default (0)", "Default (1)"],
                yticklabels=["Non-Default (0)", "Default (1)"])
    plt.title("Confusion Matrix", fontsize=12, fontweight="bold")
    plt.xlabel("Predicted Label")
    plt.ylabel("Actual True Label")
    
    # Plot 2: ROC Curve
    plt.subplot(1, 3, 2)
    plt.plot(fpr, tpr, color="#2b5c8f", lw=2.5, label=f"ROC (AUC = {roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], color="gray", linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve (KS = {ks_statistic:.3f})", fontsize=12, fontweight="bold")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Precision-Recall Curve
    plt.subplot(1, 3, 3)
    precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_pred_proba)
    plt.plot(recall_vals, precision_vals, color="#e67e22", lw=2.5, label=f"PR (AUC = {pr_auc:.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve", fontsize=12, fontweight="bold")
    plt.legend(loc="lower left")
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join(results_dir, "evaluation_curves.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"[Task 1] Saved evaluation curves to {plot_path}")
    
    # 4. Save evaluation metrics JSON
    metrics_summary = {
        "ROC_AUC": float(roc_auc),
        "PR_AUC": float(pr_auc),
        "KS_Statistic": float(ks_statistic),
        "Confusion_Matrix": cm.tolist(),
        "Classification_Report": report
    }
    with open(os.path.join(results_dir, "evaluation_metrics.json"), "w") as f:
        json.dump(metrics_summary, f, indent=4)
        
    print(f"[Task 1] Evaluation complete. Metrics stored in {results_dir}")
    return metrics_summary

if __name__ == "__main__":
    evaluate_credit_model()
