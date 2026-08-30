import os
import sys
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

base_dir = os.path.dirname(os.path.abspath(__file__))
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

from data.dataset_loader import load_or_create_dataset
from src.models import CRNN_CTC

def decode_predictions(preds, classes):
    """
    Decodes CTC output into text string.
    preds: (Batch, SeqLen) of class indices
    Returns: list of decoded strings
    """
    decoded = []
    blank_idx = len(classes)
    for b in range(preds.shape[0]):
        seq = preds[b]
        chars = []
        prev_idx = -1
        for idx in seq:
            idx = idx.item()
            if idx != blank_idx and idx != prev_idx:
                chars.append(classes[idx])
            prev_idx = idx
        decoded.append("".join(chars))
    return decoded

def train_character_recognition(epochs: int = 12, batch_size: int = 16):
    models_dir = os.path.join(base_dir, "models")
    results_dir = os.path.join(base_dir, "results")
    
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    train_set, test_set, classes = load_or_create_dataset(num_train=100, num_test=20)
    
    # Custom collate_fn for variable length targets
    def collate_fn(batch):
        images, targets = zip(*batch)
        images = torch.stack(images, 0)
        target_lengths = torch.tensor([len(t) for t in targets], dtype=torch.long)
        targets = torch.cat(targets, 0)
        return images, targets, target_lengths

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    
    with open(os.path.join(models_dir, "class_mapping.json"), "w") as f:
        json.dump({"classes": classes}, f, indent=4)
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Task 3] Training CRNN on device: {device} | Classes: {len(classes)}")
    
    model = CRNN_CTC(num_classes=len(classes)).to(device)
    
    # blank_idx is len(classes)
    criterion = nn.CTCLoss(blank=len(classes), zero_infinity=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    history = {"train_loss": [], "test_loss": [], "test_acc": []}
    best_test_acc = 0.0
    best_model_path = os.path.join(models_dir, "handwritten_crnn_best.pth")
    
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        
        for i, (inputs, targets, target_lengths) in enumerate(train_loader):
            if i % 10 == 0:
                print(f"Epoch {epoch} - Batch {i}/{len(train_loader)}", end="\r")
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            logits = model(inputs) # (B, T, C)
            
            # log_softmax is required by CTCLoss
            log_probs = F.log_softmax(logits, dim=2)
            
            # PyTorch CTCLoss expects (T, B, C)
            log_probs = log_probs.permute(1, 0, 2)
            
            # Input lengths is T for all elements in the batch (64)
            batch_size_curr = inputs.size(0)
            input_lengths = torch.full(size=(batch_size_curr,), fill_value=64, dtype=torch.long)
            
            loss = criterion(log_probs, targets, input_lengths, target_lengths)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * batch_size_curr
            
        scheduler.step()
        
        # Validation
        model.eval()
        test_loss = 0.0
        correct_sequences = 0
        total_sequences = 0
        
        with torch.no_grad():
            for inputs, targets, target_lengths in test_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                
                logits = model(inputs)
                log_probs = F.log_softmax(logits, dim=2)
                log_probs_t = log_probs.permute(1, 0, 2)
                
                batch_size_curr = inputs.size(0)
                input_lengths = torch.full(size=(batch_size_curr,), fill_value=64, dtype=torch.long)
                
                loss = criterion(log_probs_t, targets, input_lengths, target_lengths)
                test_loss += loss.item() * batch_size_curr
                
                # Decode predictions
                _, preds = torch.max(logits, 2) # (B, T)
                decoded_preds = decode_predictions(preds, classes)
                
                # Reconstruct ground truth strings
                gt_strings = []
                idx = 0
                for tl in target_lengths:
                    gt_str = "".join([classes[c.item()] for c in targets[idx:idx+tl]])
                    gt_strings.append(gt_str)
                    idx += tl
                    
                for pred_str, gt_str in zip(decoded_preds, gt_strings):
                    if pred_str == gt_str:
                        correct_sequences += 1
                total_sequences += batch_size_curr
                
        tr_l = train_loss / len(train_set)
        te_l = test_loss / len(test_set)
        te_acc = correct_sequences / total_sequences
        
        history["train_loss"].append(tr_l)
        history["test_loss"].append(te_l)
        history["test_acc"].append(te_acc)
        
        print(f"Epoch [{epoch:02d}/{epochs:02d}] | Train Loss: {tr_l:.4f} | Test Loss: {te_l:.4f} | Sequence Acc: {te_acc*100:.2f}%")
        
        if te_acc >= best_test_acc:
            best_test_acc = te_acc
            torch.save(model.state_dict(), best_model_path)
            
    print(f"[Task 3] Training Complete! Best Sequence Accuracy: {best_test_acc*100:.2f}%")
    print(f"[Task 3] Saved model weights to: {best_model_path}")
    
    with open(os.path.join(results_dir, "training_history.json"), "w") as f:
        json.dump(history, f, indent=4)
        
    return best_test_acc

if __name__ == "__main__":
    train_character_recognition(epochs=1)
