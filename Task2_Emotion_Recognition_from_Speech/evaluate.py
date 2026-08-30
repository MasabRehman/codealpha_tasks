"""
Task 2: Evaluation & Visual Performance Analytics for Speech Emotion Recognition
"""

import os
import sys
import json
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix

base_dir = os.path.dirname(os.path.abspath(__file__))
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

from src.models import Speech1DCNN

def evaluate_speech_emotion():
    data_dir = os.path.join(base_dir, "data")
    models_dir = os.path.join(base_dir, "models")
    results_dir = os.path.join(base_dir, "results")
    
    x_test_file = os.path.join(data_dir, "X_test.npy")
    y_test_file = os.path.join(data_dir, "y_test.npy")
    model_file = os.path.join(models_dir, "speech_emotion_best_model.pth")
    le_file = os.path.join(models_dir, "label_encoder.joblib")
    
    if not os.path.exists(x_test_file) or not os.path.exists(model_file):
        raise FileNotFoundError("Please run train.py first to train model and save test data.")
        
    X_test = np.load(x_test_file)
    y_test = np.load(y_test_file)
    le = joblib.load(le_file)
    class_names = list(le.classes_)
    
    model = Speech1DCNN(in_channels=40, num_classes=len(class_names))
    model.load_state_dict(torch.load(model_file, map_location=torch.device("cpu")))
    model.eval()
    
    with torch.no_grad():
        inputs = torch.tensor(X_test)
        logits = model(inputs)
        probs = F.softmax(logits, dim=1).numpy()
        preds = np.argmax(probs, axis=1)
        
    cm = confusion_matrix(y_test, preds)
    report = classification_report(y_test, preds, target_names=class_names, output_dict=True)
    
    print("\n" + "="*65)
    print("      TASK 2: SPEECH EMOTION RECOGNITION EVALUATION REPORT")
    print("="*65)
    print(classification_report(y_test, preds, target_names=class_names))
    
    # 1. Plot Confusion Matrix
    plt.figure(figsize=(14, 5))
    
    plt.subplot(1, 2, 1)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Purples",
                xticklabels=class_names, yticklabels=class_names)
    plt.title("Speech Emotion Confusion Matrix", fontsize=12, fontweight="bold")
    plt.xlabel("Predicted Emotion")
    plt.ylabel("Ground Truth Emotion")
    
    # 2. Plot Learning Curves (if available)
    history_file = os.path.join(results_dir, "training_history.json")
    if os.path.exists(history_file):
        with open(history_file) as f:
            history = json.load(f)
            
        plt.subplot(1, 2, 2)
        epochs = range(1, len(history["train_loss"]) + 1)
        plt.plot(epochs, history["train_acc"], label="Train Accuracy", color="#3498db", lw=2)
        plt.plot(epochs, history["test_acc"], label="Test Accuracy", color="#2ecc71", lw=2)
        plt.plot(epochs, history["train_loss"], label="Train Loss", color="#e74c3c", linestyle="--", lw=1.5)
        plt.plot(epochs, history["test_loss"], label="Test Loss", color="#f39c12", linestyle="--", lw=1.5)
        plt.title("Training & Validation Curves (1D-CNN)", fontsize=12, fontweight="bold")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy / Loss")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
    plt.tight_layout()
    plot_path = os.path.join(results_dir, "emotion_evaluation.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"[Task 2] Evaluation plots saved to {plot_path}")
    
    # Save metrics JSON
    metrics = {
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
        "classes": class_names
    }
    with open(os.path.join(results_dir, "emotion_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)
        
    return metrics

if __name__ == "__main__":
    evaluate_speech_emotion()
