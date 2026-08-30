"""
Task 3: Evaluation, Confusion Matrix & Prediction Visualization for Handwritten Recognition
"""

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader

base_dir = os.path.dirname(os.path.abspath(__file__))
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

from data.dataset_loader import load_or_create_dataset
from src.models import HandwrittenCNN

def evaluate_handwritten_model():
    data_dir = os.path.join(base_dir, "data")
    models_dir = os.path.join(base_dir, "models")
    results_dir = os.path.join(base_dir, "results")
    
    model_path = os.path.join(models_dir, "handwritten_cnn_best.pth")
    class_map_path = os.path.join(models_dir, "class_mapping.json")
    
    if not os.path.exists(model_path) or not os.path.exists(class_map_path):
        raise FileNotFoundError("Model weights or class mapping not found. Please run train.py first.")
        
    with open(class_map_path) as f:
        classes = json.load(f)["classes"]
        
    _, test_set, _ = load_or_create_dataset(data_dir=os.path.join(data_dir, "raw"), dataset_type="synthetic")
    test_loader = DataLoader(test_set, batch_size=128, shuffle=False)
    
    model = HandwrittenCNN(num_classes=len(classes))
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    
    all_preds = []
    all_targets = []
    sample_images = []
    sample_preds = []
    sample_targets = []
    sample_confs = []
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            outputs = model(inputs)
            probs = F.softmax(outputs, dim=1)
            confs, preds = torch.max(probs, 1)
            
            all_preds.extend(preds.numpy())
            all_targets.extend(targets.numpy())
            
            if len(sample_images) < 16:
                for i in range(min(inputs.size(0), 16 - len(sample_images))):
                    sample_images.append(inputs[i, 0].numpy())
                    sample_preds.append(classes[preds[i].item()])
                    sample_targets.append(classes[targets[i].item()])
                    sample_confs.append(float(confs[i].item()))
                    
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    acc = np.mean(all_preds == all_targets)
    cm = confusion_matrix(all_targets, all_preds)
    report = classification_report(all_targets, all_preds, target_names=classes, output_dict=True)
    
    print("\n" + "="*65)
    print("   TASK 3: HANDWRITTEN CHARACTER RECOGNITION EVALUATION REPORT")
    print("="*65)
    print(f"Overall Test Accuracy: {acc*100:.2f}%\n")
    print(classification_report(all_targets, all_preds, target_names=classes))
    
    # 1. Visualization: Confusion Matrix & Sample Prediction Grid
    fig = plt.figure(figsize=(16, 7))
    
    # Subplot 1: Confusion Matrix
    ax1 = plt.subplot(1, 2, 1)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes, ax=ax1)
    ax1.set_title("Handwritten Character Confusion Matrix", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Predicted Character")
    ax1.set_ylabel("True Character")
    
    # Subplot 2: Sample Predictions Grid
    ax2 = plt.subplot(1, 2, 2)
    ax2.axis('off')
    
    # Plot 4x4 sub-grid
    for idx in range(min(16, len(sample_images))):
        sub_ax = fig.add_subplot(4, 4, idx + 1)
        sub_ax.imshow(sample_images[idx], cmap="gray")
        sub_ax.axis('off')
        
        is_correct = (sample_preds[idx] == sample_targets[idx])
        color = "green" if is_correct else "red"
        sub_ax.set_title(f"P:{sample_preds[idx]} (T:{sample_targets[idx]})\n{sample_confs[idx]*100:.0f}%",
                         fontsize=9, color=color, fontweight="bold")
        
    plt.tight_layout()
    plot_path = os.path.join(results_dir, "evaluation_matrix.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"[Task 3] Saved visual evaluation plot to {plot_path}")
    
    # Save metrics JSON
    metrics = {
        "test_accuracy": float(acc),
        "confusion_matrix": cm.tolist(),
        "classification_report": report
    }
    with open(os.path.join(results_dir, "evaluation_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)
        
    return metrics

if __name__ == "__main__":
    evaluate_handwritten_model()
