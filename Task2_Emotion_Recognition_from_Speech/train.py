"""
Task 2: PyTorch Training Loop for Speech Emotion Recognition
Trains 1D-CNN and BiLSTM models on MFCC acoustic features.
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

# Ensure base directory is in python path
base_dir = os.path.dirname(os.path.abspath(__file__))
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

from data.audio_generator import generate_speech_dataset, EMOTIONS
from src.feature_extraction import extract_features_from_file
from src.models import Speech1DCNN, SpeechLSTM

def prepare_data(data_dir: str):
    """
    Extracts MFCC features from all audio samples in manifest.
    """
    manifest_file = os.path.join(data_dir, "dataset_manifest.csv")
    if not os.path.exists(manifest_file):
        print("[Task 2] Manifest not found. Generating emotional speech dataset...")
        manifest_df = generate_speech_dataset(samples_per_emotion=100)
    else:
        manifest_df = pd.read_csv(manifest_file)
        
    print(f"[Task 2] Extracting MFCC acoustic features for {len(manifest_df)} audio files...")
    
    X_mfcc = []
    y_labels = []
    
    for idx, row in manifest_df.iterrows():
        file_path = row["file_path"]
        if not os.path.exists(file_path):
            file_path = os.path.join(data_dir, "audio_samples", row["relative_path"])
            
        try:
            mfcc_2d, _ = extract_features_from_file(file_path, n_mfcc=40, max_len=120)
            X_mfcc.append(mfcc_2d)
            y_labels.append(row["emotion"])
        except Exception as e:
            print(f"Warning: could not process {file_path}: {e}")
            
    X = np.array(X_mfcc, dtype=np.float32)
    le = LabelEncoder()
    y = le.fit_transform(y_labels)
    
    # Save label encoder
    models_dir = os.path.join(base_dir, "models")
    os.makedirs(models_dir, exist_ok=True)
    joblib.dump(le, os.path.join(models_dir, "label_encoder.joblib"))
    
    return X, y, le

def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return total_loss / total, correct / total

def eval_epoch(model, dataloader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    f1 = f1_score(all_labels, all_preds, average="weighted")
    return total_loss / total, correct / total, f1

def run_training(epochs: int = 35, batch_size: int = 32):
    data_dir = os.path.join(base_dir, "data")
    models_dir = os.path.join(base_dir, "models")
    results_dir = os.path.join(base_dir, "results")
    
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    X, y, le = prepare_data(data_dir)
    print(f"[Task 2] Dataset shape: X = {X.shape}, y = {y.shape}, Classes = {le.classes_}")
    
    # Stratified split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Save test set numpy arrays for evaluation
    np.save(os.path.join(data_dir, "X_test.npy"), X_test)
    np.save(os.path.join(data_dir, "y_test.npy"), y_test)
    
    # PyTorch DataLoaders
    train_dataset = TensorDataset(torch.tensor(X_train), torch.tensor(y_train, dtype=torch.long))
    test_dataset = TensorDataset(torch.tensor(X_test), torch.tensor(y_test, dtype=torch.long))
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Task 2] Training on hardware device: {device}")
    
    # 1. Train 1D-CNN Model
    print("\n" + "="*60)
    print("        TRAINING 1D-CNN AUDIO EMOTION MODEL")
    print("="*60)
    
    cnn_model = Speech1DCNN(in_channels=40, num_classes=len(le.classes_)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(cnn_model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    best_test_acc = 0.0
    history = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}
    
    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = train_epoch(cnn_model, train_loader, criterion, optimizer, device)
        te_loss, te_acc, te_f1 = eval_epoch(cnn_model, test_loader, criterion, device)
        scheduler.step()
        
        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["test_loss"].append(te_loss)
        history["test_acc"].append(te_acc)
        
        if epoch % 5 == 0 or epoch == epochs:
            print(f"Epoch [{epoch:02d}/{epochs:02d}] | Train Loss: {tr_loss:.4f} | Train Acc: {tr_acc*100:.2f}% | Test Acc: {te_acc*100:.2f}% | Test F1: {te_f1:.4f}")
            
        if te_acc > best_test_acc:
            best_test_acc = te_acc
            best_model_path = os.path.join(models_dir, "speech_emotion_best_model.pth")
            torch.save(cnn_model.state_dict(), best_model_path)
            
    print(f"\n[Task 2] 1D-CNN Training Complete. Best Test Accuracy: {best_test_acc*100:.2f}%")
    print(f"[Task 2] Best model saved to {best_model_path}")
    
    # Save training history JSON
    with open(os.path.join(results_dir, "training_history.json"), "w") as f:
        json.dump(history, f, indent=4)
        
    return best_test_acc

if __name__ == "__main__":
    run_training(epochs=35)
