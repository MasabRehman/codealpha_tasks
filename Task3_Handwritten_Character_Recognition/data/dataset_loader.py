import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image, ImageDraw, ImageFont
import urllib.request
import zipfile

# Classes mapping
CLASSES_36 = [str(i) for i in range(10)] + [chr(ord('A') + i) for i in range(26)]
# For IAM we need space, lowercase, punctuation, etc. 
# But for simplicity, we can upper-case everything and filter to alphanumeric.
CHAR2IDX = {char: idx for idx, char in enumerate(CLASSES_36)}

class IAMDataset(Dataset):
    def __init__(self, data_dir, split='train', max_len=32, target_height=64, target_width=256):
        self.data_dir = data_dir
        self.max_len = max_len
        self.target_height = target_height
        self.target_width = target_width
        self.samples = []
        
        words_txt = os.path.join(data_dir, 'iam_words', 'words.txt')
        words_img_dir = os.path.join(data_dir, 'iam_words', 'words')
        
        if not os.path.exists(words_txt) or not os.path.exists(words_img_dir):
            raise FileNotFoundError(f"IAM dataset not found at {data_dir}. Please download words.tgz and ascii.tgz")
            
        with open(words_txt, 'r') as f:
            lines = f.readlines()
            
        # Parse words.txt
        valid_samples = []
        for line in lines:
            if line.startswith('#'): continue
            parts = line.strip().split(' ')
            if len(parts) >= 9 and parts[1] == 'ok':
                word_id = parts[0]
                text = parts[-1].upper()
                
                # Filter to only alphanumeric
                text = "".join([c for c in text if c in CHAR2IDX])
                if len(text) == 0: continue
                
                # Path format: a01/a01-000u/a01-000u-00.png
                parts_id = word_id.split('-')
                dir1 = parts_id[0]
                dir2 = f"{parts_id[0]}-{parts_id[1]}"
                img_path = os.path.join(words_img_dir, dir1, dir2, f"{word_id}.png")
                
                if os.path.exists(img_path):
                    valid_samples.append((img_path, text))
                    
        # Split train/test (90/10)
        random.seed(42)
        random.shuffle(valid_samples)
        split_idx = int(len(valid_samples) * 0.9)
        if split == 'train':
            self.samples = valid_samples[:split_idx]
        else:
            self.samples = valid_samples[split_idx:]
            
        print(f"[IAM Dataset] Loaded {len(self.samples)} samples for {split}")

    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        img_path, text = self.samples[idx]
        img = Image.open(img_path).convert("L")
        arr = np.array(img, dtype=np.uint8)
        
        # Invert if necessary (IAM is usually dark on light, but let's be safe)
        border = np.concatenate([arr[0, :], arr[-1, :], arr[:, 0], arr[:, -1]])
        if np.mean(border) > 127:
            arr = 255 - arr
            
        # Simple resize/pad inline for dataset
        import cv2
        _, thresh = cv2.threshold(arr, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        h, w = thresh.shape
        scale = self.target_height / float(max(1, h))
        new_w = int(w * scale)
        resized = cv2.resize(thresh, (new_w, self.target_height), interpolation=cv2.INTER_AREA)
        
        padded = np.zeros((self.target_height, self.target_width), dtype=np.uint8)
        if new_w > self.target_width:
            padded[:, :] = resized[:, :self.target_width]
        else:
            padded[:, :new_w] = resized
            
        # Morphological Augmentation: Simulate thick markers or thin pens
        if hasattr(self, 'split') and self.split == 'train' and random.random() > 0.5:
            kernel = np.ones((3, 3), np.uint8)
            if random.random() > 0.5:
                padded = cv2.dilate(padded, kernel, iterations=1) # Thicker stroke
            else:
                padded = cv2.erode(padded, kernel, iterations=1)  # Thinner stroke
            
        img_tensor = torch.tensor(padded.astype(np.float32) / 255.0).unsqueeze(0)
        
        targets = [CHAR2IDX[c] for c in text if c in CHAR2IDX]
        target_tensor = torch.tensor(targets, dtype=torch.long)
        
        return img_tensor, target_tensor

# --- Synthetic cursive generator fallback ---
def render_synthetic_sequence(text: str, height: int = 64, width: int = 256) -> np.ndarray:
    img = Image.new("L", (width, height), color=0)
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()
    
    # Render and scale
    temp_img = Image.new("L", (width, height), color=0)
    temp_draw = ImageDraw.Draw(temp_img)
    temp_draw.text((10, 20), text, fill=255, font=font)
    temp_img = temp_img.resize((width, height), Image.Resampling.BICUBIC)
    
    arr = np.array(temp_img, dtype=np.float32)
    arr += np.random.normal(0, 15, arr.shape)
    arr = np.clip(arr, 0, 255)
    return arr.astype(np.uint8)

class SyntheticSequenceDataset(Dataset):
    def __init__(self, num_samples=1000, max_len=6):
        self.num_samples = num_samples
        self.samples = []
        for _ in range(num_samples):
            length = random.randint(3, max_len)
            text = "".join(random.choices(CLASSES_36, k=length))
            self.samples.append(text)
            
    def __len__(self):
        return self.num_samples
        
    def __getitem__(self, idx):
        text = self.samples[idx]
        img_arr = render_synthetic_sequence(text, height=64, width=256)
        img_tensor = torch.tensor(img_arr.astype(np.float32) / 255.0).unsqueeze(0)
        targets = [CHAR2IDX[c] for c in text]
        return img_tensor, torch.tensor(targets, dtype=torch.long)

def load_or_create_dataset(num_train=2000, num_test=400):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    iam_dir = os.path.join(base_dir, "data", "raw", "IAM")
    
    if os.path.exists(os.path.join(iam_dir, "iam_words", "words")):
        print("[Task 3] IAM Dataset found! Loading real handwritten data...")
        train_dataset = IAMDataset(iam_dir, split='train')
        test_dataset = IAMDataset(iam_dir, split='test')
    else:
        print("[Task 3] IAM Dataset not found. Falling back to synthetic sequences.")
        train_dataset = SyntheticSequenceDataset(num_samples=num_train)
        test_dataset = SyntheticSequenceDataset(num_samples=num_test)
        
    return train_dataset, test_dataset, CLASSES_36

def save_sample_test_images(output_dir: str = None):
    if output_dir is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        output_dir = os.path.join(base_dir, "data", "sample_images")
        
    os.makedirs(output_dir, exist_ok=True)
    sample_texts = ["HELLO", "WORLD", "CRNN", "123AB", "TEST9"]
    for text in sample_texts:
        img_arr = render_synthetic_sequence(text)
        img = Image.fromarray(img_arr)
        img.save(os.path.join(output_dir, f"sample_{text}.png"))

if __name__ == '__main__':
    save_sample_test_images()
