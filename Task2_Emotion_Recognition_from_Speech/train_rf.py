import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import json
import joblib
import os
import argparse
import time

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=15)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--lr', type=float, default=0.001)
args = parser.parse_args()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
hist_path = os.path.join(MODEL_DIR, "training_history.json")

print("Loading cached dataset for Random Forest...")
# For now, just load MELD. We can add concatenation later if active_datasets has RAVDESS
# Check active datasets
active_ds = ["meld"]
try:
    with open(os.path.join(MODEL_DIR, "active_datasets.json"), "r") as f:
        active_ds = json.load(f)
except: pass

Xs = []
ys = []
val_Xs = []
val_ys = []

try:
    if "meld" in active_ds:
        Xs.append(np.load(os.path.join(MODEL_DIR, "train_X.npy")))
        ys.append(np.load(os.path.join(MODEL_DIR, "train_y.npy")))
        val_Xs.append(np.load(os.path.join(MODEL_DIR, "val_X.npy")))
        val_ys.append(np.load(os.path.join(MODEL_DIR, "val_y.npy")))
    
    if "ravdess" in active_ds:
        r_X = np.load(os.path.join(MODEL_DIR, "ravdess_X.npy"))
        r_y = np.load(os.path.join(MODEL_DIR, "ravdess_y.npy"))
        # Split RAVDESS 80/20 for val
        split = int(0.8 * len(r_X))
        Xs.append(r_X[:split])
        ys.append(r_y[:split])
        val_Xs.append(r_X[split:])
        val_ys.append(r_y[split:])
    if "tess" in active_ds:
        try:
            t_X = np.load(os.path.join(MODEL_DIR, "tess_X.npy"))
            t_y = np.load(os.path.join(MODEL_DIR, "tess_y.npy"))
            split = int(0.8 * len(t_X))
            Xs.append(t_X[:split])
            ys.append(t_y[:split])
            val_Xs.append(t_X[split:])
            val_ys.append(t_y[split:])
        except: print("TESS cache not found, skipping...")
        
    if "emodb" in active_ds:
        try:
            e_X = np.load(os.path.join(MODEL_DIR, "emodb_X.npy"))
            e_y = np.load(os.path.join(MODEL_DIR, "emodb_y.npy"))
            split = int(0.8 * len(e_X))
            Xs.append(e_X[:split])
            ys.append(e_y[:split])
            val_Xs.append(e_X[split:])
            val_ys.append(e_y[split:])
        except: print("EMO-DB cache not found, skipping...")
        
        
    train_X = np.concatenate(Xs, axis=0)
    train_y = np.concatenate(ys, axis=0)
    val_X = np.concatenate(val_Xs, axis=0)
    val_y = np.concatenate(val_ys, axis=0)
except Exception as e:
    print(f"Error loading datasets: {e}")
    exit(1)

# Flatten the time dimension by taking the mean (N, 40, 150) -> (N, 40)
train_X_flat = np.mean(train_X, axis=2)
val_X_flat = np.mean(val_X, axis=2)

hdata = {"history": {"epochs":[], "train_loss":[], "val_loss":[], "val_acc":[]}}

# We simulate "Epochs" by iteratively training the RF with increasing number of trees!
# Max trees = 150. Step = 150 / args.epochs
step = max(1, 150 // args.epochs)

for epoch in range(1, args.epochs + 1):
    n_trees = epoch * step
    print(f"Training RF with {n_trees} trees...")
    rf = RandomForestClassifier(n_estimators=n_trees, class_weight='balanced', random_state=42, n_jobs=-1)
    rf.fit(train_X_flat, train_y)
    
    val_preds = rf.predict(val_X_flat)
    val_acc = accuracy_score(val_y, val_preds)
    
    # We simulate "Loss" just to keep the chart happy (RF doesn't use loss)
    val_loss = 2.0 - val_acc
    
    hdata["epoch"] = epoch
    hdata["total_epochs"] = args.epochs
    hdata["val_accuracy"] = round(val_acc * 100, 2)
    hdata["val_loss"] = round(val_loss, 3)
    
    hdata["history"]["epochs"].append(epoch)
    hdata["history"]["train_loss"].append(val_loss) # Mock
    hdata["history"]["val_loss"].append(val_loss)   # Mock
    hdata["history"]["val_acc"].append(val_acc * 100)
    
    with open(hist_path, "w") as f:
        json.dump(hdata, f)
        
    time.sleep(0.5) # Slight delay so the UI chart animates gracefully!

joblib.dump(rf, os.path.join(MODEL_DIR, "rf_model.pkl"))
joblib.dump(rf.classes_, os.path.join(MODEL_DIR, "rf_classes.pkl"))
print("Random Forest Training Complete and Model Saved!")
