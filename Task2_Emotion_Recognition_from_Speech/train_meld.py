"""
Train Speech Emotion Recognition model on MELD dataset.
MELD contains MP4 video clips from the TV show Friends with emotion labels.
We extract audio from the MP4s, compute MFCCs, and train the Speech1DCNN.
"""
import os
import imageio_ffmpeg
ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
ffmpeg_dir = os.path.dirname(ffmpeg_exe)
os.environ["PATH"] += os.pathsep + ffmpeg_dir
import pydub
pydub.AudioSegment.converter = ffmpeg_exe
pydub.utils.get_prober_name = lambda: ffmpeg_exe.replace('ffmpeg', 'ffprobe')

import sys
import glob
import csv
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import joblib
import json
from sklearn.preprocessing import LabelEncoder
from collections import Counter
import traceback

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from src.feature_extraction import load_audio, extract_features_from_audio
from src.models import Speech1DCNN, SpeechCNNLSTM

# --- Configuration ---
MELD_DIR = os.path.join(BASE_DIR, "data", "MELD")
MODEL_DIR = os.path.join(BASE_DIR, "models")
BATCH_SIZE = 32
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=15)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--model_type', type=str, default='cnn')
args = parser.parse_args()
EPOCHS = args.epochs
BATCH_SIZE = args.batch_size
LR = args.lr
N_MFCC = 40
MAX_LEN = 150
SR = 22050
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def find_meld_structure(base_path):
    """Find the actual MELD directory structure after extraction."""
    print(f"Scanning {base_path} for MELD structure...")
    
    # Look for CSV files
    csv_files = []
    for root, dirs, files in os.walk(base_path):
        for f in files:
            if f.endswith('.csv'):
                csv_files.append(os.path.join(root, f))
                print(f"  Found CSV: {os.path.join(root, f)}")
    
    # Look for MP4 directories
    mp4_dirs = []
    for root, dirs, files in os.walk(base_path):
        mp4_count = sum(1 for f in files if f.endswith('.mp4'))
        if mp4_count > 0:
            mp4_dirs.append((root, mp4_count))
            print(f"  Found {mp4_count} MP4s in: {root}")
    
    return csv_files, mp4_dirs


def extract_audio_from_mp4(mp4_path, wav_path):
    """Extract audio from MP4 using moviepy or subprocess."""
    if os.path.exists(wav_path):
        return True
    try:
        # Try moviepy first
        from moviepy.editor import VideoFileClip
        clip = VideoFileClip(mp4_path)
        clip.audio.write_audiofile(wav_path, fps=SR, nbits=16, codec='pcm_s16le', logger=None)
        clip.close()
        return True
    except ImportError:
        pass
    
    try:
        # Try ffmpeg subprocess
        import subprocess
        result = subprocess.run(
            [ffmpeg_exe, '-y', '-i', mp4_path, '-vn', '-acodec', 'pcm_s16le', '-ar', str(SR), '-ac', '1', wav_path],
            capture_output=True, timeout=30
        )
        return result.returncode == 0
    except Exception:
        pass
    
    try:
        # Try pydub
        from pydub import AudioSegment
        audio = AudioSegment.from_file(mp4_path, format="mp4")
        audio = audio.set_frame_rate(SR).set_channels(1)
        audio.export(wav_path, format="wav")
        return True
    except Exception:
        pass
    
    # Last resort: try librosa directly on mp4
    try:
        import librosa
        y, sr = librosa.load(mp4_path, sr=SR, mono=True)
        import soundfile as sf
        sf.write(wav_path, y, SR)
        return True
    except Exception:
        return False


def load_meld_labels(csv_path):
    """Parse MELD CSV to get dialogue_id, utterance_id, emotion mappings."""
    labels = {}
    with open(csv_path, 'r', encoding='utf-8', errors='ignore') as f:
        reader = csv.DictReader(f)
        columns = reader.fieldnames
        print(f"  CSV columns: {columns}")
        
        for row in reader:
            try:
                dia_id = row.get('Dialogue_ID', '').strip()
                utt_id = row.get('Utterance_ID', '').strip()
                emotion = row.get('Emotion', '').strip().lower()
                
                if dia_id and utt_id and emotion:
                    filename = f"dia{dia_id}_utt{utt_id}.mp4"
                    labels[filename] = emotion
            except Exception:
                continue
    
    print(f"  Loaded {len(labels)} labels from {os.path.basename(csv_path)}")
    return labels


class MELDDataset(Dataset):
    def __init__(self, features_list, labels_list, augment=False):
        self.features = features_list
        self.labels = labels_list
        self.augment = augment
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        feat = self.features[idx].copy()
        
        # SpecAugment
        if self.augment:
            # Frequency masking
            if np.random.rand() < 0.5:
                f0 = np.random.randint(0, feat.shape[0] - 5)
                f_mask = np.random.randint(1, 6)
                feat[f0:f0+f_mask, :] = 0
            
            # Time masking
            if np.random.rand() < 0.5:
                t0 = np.random.randint(0, feat.shape[1] - 15)
                t_mask = np.random.randint(1, 16)
                feat[:, t0:t0+t_mask] = 0
                
        return torch.tensor(feat, dtype=torch.float32), self.labels[idx]


def prepare_dataset(mp4_dir, labels_dict, wav_cache_dir, label_encoder=None):
    """Extract audio from MP4s, compute MFCCs, return features and encoded labels."""
    os.makedirs(wav_cache_dir, exist_ok=True)
    
    features = []
    emotions = []
    skipped = 0
    processed = 0
    
    mp4_files = glob.glob(os.path.join(mp4_dir, "*.mp4"))
    print(f"\nProcessing {len(mp4_files)} MP4 files from {mp4_dir}...")
    
    for i, mp4_path in enumerate(mp4_files):
        filename = os.path.basename(mp4_path)
        
        if filename not in labels_dict:
            skipped += 1
            continue
        
        emotion = labels_dict[filename]
        wav_path = os.path.join(wav_cache_dir, filename.replace('.mp4', '.wav'))
        
        # Extract audio
        if not extract_audio_from_mp4(mp4_path, wav_path):
            skipped += 1
            continue
        
        # Extract features
        try:
            padded_mfcc, _ = extract_features_from_audio(
                *load_audio(wav_path, target_sr=SR),
                n_mfcc=N_MFCC, max_len=MAX_LEN
            )
            features.append(padded_mfcc)
            emotions.append(emotion)
            processed += 1
        except Exception as e:
            skipped += 1
            continue
        
        if (i + 1) % 100 == 0:
            print(f"  Processed {i+1}/{len(mp4_files)} ({processed} valid, {skipped} skipped)")
    
    print(f"  Final: {processed} valid samples, {skipped} skipped")
    
    if not features:
        return np.array([]), np.array([]), label_encoder
    
    X = np.array(features)
    
    if label_encoder is None:
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(emotions)
    else:
        y = label_encoder.transform(emotions)
    
    return X, y, label_encoder


def train_model(train_loader, val_loader, model, num_classes, label_encoder, class_weights=None):
    """Train the CNN model."""
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(DEVICE) if class_weights is not None else None)
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
    
    best_val_acc = 0.0
    best_model_path = os.path.join(MODEL_DIR, "speech_emotion_hybrid_model.pth")
    
    
    # Clear old history
    try:
        if os.path.exists(os.path.join(MODEL_DIR, "training_history.json")):
            os.remove(os.path.join(MODEL_DIR, "training_history.json"))
    except: pass
    
    for epoch in range(EPOCHS):
        # Train
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * X_batch.size(0)
            _, predicted = outputs.max(1)
            train_total += y_batch.size(0)
            train_correct += predicted.eq(y_batch).sum().item()
        
        train_acc = train_correct / train_total if train_total > 0 else 0
        train_loss = train_loss / train_total if train_total > 0 else 0
        
        # Validate
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(DEVICE)
                y_batch = y_batch.to(DEVICE)
                
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                
                val_loss += loss.item() * X_batch.size(0)
                _, predicted = outputs.max(1)
                val_total += y_batch.size(0)
                val_correct += predicted.eq(y_batch).sum().item()
        
        val_acc = val_correct / val_total if val_total > 0 else 0
        val_loss = val_loss / val_total if val_total > 0 else 0
        
        scheduler.step(val_loss)
        
        
        print(f"  Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")
        
        # Save to history
        try:
            hist_path = os.path.join(MODEL_DIR, "training_history.json")
            if os.path.exists(hist_path):
                with open(hist_path, "r") as hf:
                    hdata = json.load(hf)
            else:
                hdata = {"history": {"epochs":[], "train_loss":[], "val_loss":[], "val_acc":[]}}
            
            hdata["epoch"] = epoch + 1
            hdata["total_epochs"] = EPOCHS
            hdata["val_accuracy"] = round(val_acc * 100, 2)
            hdata["val_loss"] = round(val_loss, 3)
            hdata["history"]["epochs"].append(epoch + 1)
            hdata["history"]["train_loss"].append(train_loss)
            hdata["history"]["val_loss"].append(val_loss)
            hdata["history"]["val_acc"].append(val_acc * 100)
            
            with open(hist_path, "w") as hf:
                json.dump(hdata, hf)
        except: pass

        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            print(f"    -> Saved best model (val_acc={val_acc:.4f})")
    
    print(f"\nTraining complete! Best validation accuracy: {best_val_acc:.4f}")
    return best_val_acc


def main():
    print("=" * 60)
    print("MELD Speech Emotion Recognition Training Pipeline")
    print(f"Device: {DEVICE}")
    print("=" * 60)
    
    # 1. Find MELD structure
    csv_files, mp4_dirs = find_meld_structure(os.path.join(BASE_DIR, "data"))
    
    if not csv_files:
        print("ERROR: No CSV files found. Check MELD extraction.")
        return
    
    if not mp4_dirs:
        print("ERROR: No MP4 directories found. Check MELD extraction.")
        return
    
    # 2. Load labels from all CSVs
    all_labels = {}
    for csv_path in csv_files:
        if 'train' in csv_path.lower() or 'dev' in csv_path.lower() or 'test' in csv_path.lower():
            labels = load_meld_labels(csv_path)
            all_labels.update(labels)
    
    if not all_labels:
        # Try all CSVs
        for csv_path in csv_files:
            labels = load_meld_labels(csv_path)
            all_labels.update(labels)
    
    print(f"\nTotal labels loaded: {len(all_labels)}")
    emotion_counts = Counter(all_labels.values())
    print(f"Emotion distribution: {dict(emotion_counts)}")
    

    # 3. Prepare train and val datasets
    cache_train_x = os.path.join(MODEL_DIR, "train_X.npy")
    cache_train_y = os.path.join(MODEL_DIR, "train_y.npy")
    cache_val_x = os.path.join(MODEL_DIR, "val_X.npy")
    cache_val_y = os.path.join(MODEL_DIR, "val_y.npy")
    cache_le = os.path.join(MODEL_DIR, "label_encoder.joblib")
    
    if os.path.exists(cache_train_x) and os.path.exists(cache_le):
        print("Loading completely cached datasets...")
        label_encoder = joblib.load(cache_le)
        
        active_ds = ["meld"]
        try:
            with open(os.path.join(MODEL_DIR, "active_datasets.json"), "r") as f:
                active_ds = json.load(f)
        except: pass

        Xs, ys, val_Xs, val_ys = [], [], [], []
        
        if "meld" in active_ds:
            Xs.append(np.load(cache_train_x))
            ys.append(np.load(cache_train_y))
            val_Xs.append(np.load(cache_val_x))
            val_ys.append(np.load(cache_val_y))
            
        if "ravdess" in active_ds:
            try:
                r_X = np.load(os.path.join(MODEL_DIR, "ravdess_X.npy"))
                r_y = np.load(os.path.join(MODEL_DIR, "ravdess_y.npy"))
                r_y_enc = label_encoder.transform(r_y)
                split = int(0.8 * len(r_X))
                Xs.append(r_X[:split])
                ys.append(r_y_enc[:split])
                val_Xs.append(r_X[split:])
                val_ys.append(r_y_enc[split:])
            except Exception as e: print("RAVDESS not found:", e)
            
        if "tess" in active_ds:
            try:
                t_X = np.load(os.path.join(MODEL_DIR, "tess_X.npy"))
                t_y = np.load(os.path.join(MODEL_DIR, "tess_y.npy"))
                t_y_enc = label_encoder.transform(t_y)
                split = int(0.8 * len(t_X))
                Xs.append(t_X[:split])
                ys.append(t_y_enc[:split])
                val_Xs.append(t_X[split:])
                val_ys.append(t_y_enc[split:])
            except: pass
            
        if "emodb" in active_ds:
            try:
                e_X = np.load(os.path.join(MODEL_DIR, "emodb_X.npy"))
                e_y = np.load(os.path.join(MODEL_DIR, "emodb_y.npy"))
                e_y_enc = label_encoder.transform(e_y)
                split = int(0.8 * len(e_X))
                Xs.append(e_X[:split])
                ys.append(e_y_enc[:split])
                val_Xs.append(e_X[split:])
                val_ys.append(e_y_enc[split:])
            except: pass
            
        train_X = np.concatenate(Xs, axis=0)
        train_y = np.concatenate(ys, axis=0)
        val_X = np.concatenate(val_Xs, axis=0)
        val_y = np.concatenate(val_ys, axis=0)
        
        # 4. Create DataLoaders
        train_dataset = MELDDataset(train_X, train_y, augment=True)
        val_dataset = MELDDataset(val_X, val_y, augment=False)
        
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
        
        num_classes = len(label_encoder.classes_)
        model = SpeechCNNLSTM(in_channels=N_MFCC, num_classes=num_classes).to(DEVICE)
        
        # 6. Train
        from sklearn.utils.class_weight import compute_class_weight
        classes = np.unique(train_y)
        weights = compute_class_weight('balanced', classes=classes, y=train_y)
        class_weights_tensor = torch.tensor(weights, dtype=torch.float32)

        best_acc = train_model(train_loader, val_loader, model, num_classes, label_encoder, class_weights=class_weights_tensor)
        return best_acc

    wav_cache = os.path.join(BASE_DIR, "data", "wav_cache")

    
    # Try to find train/dev/test splits
    train_labels = {}
    val_labels = {}
    
    for csv_path in csv_files:
        bn = os.path.basename(csv_path).lower()
        if 'train' in bn:
            train_labels.update(load_meld_labels(csv_path))
        elif 'dev' in bn:
            val_labels.update(load_meld_labels(csv_path))
        elif 'test' in bn:
            val_labels.update(load_meld_labels(csv_path))
    
    if not train_labels:
        # Fall back: use all labels, split 80/20
        all_items = list(all_labels.items())
        np.random.seed(42)
        np.random.shuffle(all_items)
        split = int(0.8 * len(all_items))
        train_labels = dict(all_items[:split])
        val_labels = dict(all_items[split:])
    
    print(f"\nTrain samples: {len(train_labels)}, Val samples: {len(val_labels)}")
    
    # Find MP4 directories
    # Process from all mp4 dirs
    train_X_all = []
    train_y_all = []
    val_X_all = []
    val_y_all = []
    label_encoder = None
    
    # First pass: collect all emotions to fit label encoder
    all_emotions = list(set(list(train_labels.values()) + list(val_labels.values())))
    label_encoder = LabelEncoder()
    label_encoder.fit(all_emotions)
    print(f"Classes: {label_encoder.classes_}")
    
    for mp4_dir, count in mp4_dirs:
        print(f"\n--- Processing directory: {mp4_dir} ({count} files) ---")
        
        # Try train labels
        X, y, _ = prepare_dataset(mp4_dir, train_labels, wav_cache, label_encoder)
        if len(X) > 0:
            train_X_all.append(X)
            train_y_all.append(y)
        
        # Try val labels
        X, y, _ = prepare_dataset(mp4_dir, val_labels, wav_cache, label_encoder)
        if len(X) > 0:
            val_X_all.append(X)
            val_y_all.append(y)
    
    if not train_X_all:
        print("ERROR: No training data was extracted.")
        return
    
    train_X = np.concatenate(train_X_all)
    train_y = np.concatenate(train_y_all)
    
    if val_X_all:
        val_X = np.concatenate(val_X_all)
        val_y = np.concatenate(val_y_all)
    else:
        # Split from training
        split = int(0.8 * len(train_X))
        val_X = train_X[split:]
        val_y = train_y[split:]
        train_X = train_X[:split]
        train_y = train_y[:split]
    

    print(f"\nFinal dataset sizes:")
    print(f"  Train: {train_X.shape}")
    print(f"  Val:   {val_X.shape}")
    print(f"  Classes: {label_encoder.classes_}")

    # SAVE CACHE
    np.save(os.path.join(MODEL_DIR, "train_X.npy"), train_X)
    np.save(os.path.join(MODEL_DIR, "train_y.npy"), train_y)
    np.save(os.path.join(MODEL_DIR, "val_X.npy"), val_X)
    np.save(os.path.join(MODEL_DIR, "val_y.npy"), val_y)

    
    # 4. Create DataLoaders
    train_dataset = MELDDataset(train_X, train_y)
    val_dataset = MELDDataset(val_X, val_y)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # 5. Create model
    num_classes = len(label_encoder.classes_)
    model = SpeechCNNLSTM(in_channels=N_MFCC, num_classes=num_classes).to(DEVICE)
    print(f"\nModel: SpeechCNNLSTM (in_channels={N_MFCC}, num_classes={num_classes})")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
        # 6. Train
    from sklearn.utils.class_weight import compute_class_weight
    classes = np.unique(train_y)
    weights = compute_class_weight('balanced', classes=classes, y=train_y)
    class_weights_tensor = torch.tensor(weights, dtype=torch.float32)

    os.makedirs(MODEL_DIR, exist_ok=True)
    best_acc = train_model(train_loader, val_loader, model, num_classes, label_encoder, class_weights=class_weights_tensor)
    
    # 7. Save label encoder
    le_path = os.path.join(MODEL_DIR, "label_encoder.joblib")
    joblib.dump(label_encoder, le_path)
    print(f"\nSaved label encoder to {le_path}")
    print(f"Final classes: {label_encoder.classes_}")
    print("Done!")


if __name__ == "__main__":
    main()
