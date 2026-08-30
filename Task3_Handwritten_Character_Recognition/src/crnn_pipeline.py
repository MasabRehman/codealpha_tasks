import torch
import torch.nn as nn
import numpy as np
import cv2
from PIL import Image

# Step 1: Image Preprocessing & Formatting
def preprocess_for_crnn(image_input, target_height=64, max_width=256):
    """
    Standardizing inputs into uniform NumPy arrays.
    Grayscale -> Binarize -> Scale to fixed height (64px) -> Pad to max_width (256px) -> Normalize.
    """
    if isinstance(image_input, str):
        img = cv2.imread(image_input, cv2.IMREAD_GRAYSCALE)
    elif isinstance(image_input, Image.Image):
        img = np.array(image_input.convert('L'))
    else:
        img = image_input

    # Binarization threshold (Otsu's method)
    _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Scale to fixed height while maintaining aspect ratio
    h, w = binary.shape
    aspect_ratio = w / h
    new_w = int(target_height * aspect_ratio)
    
    # Cap the width at max_width to prevent overflow
    new_w = min(new_w, max_width)
    
    resized = cv2.resize(binary, (new_w, target_height), interpolation=cv2.INTER_AREA)
    
    # Pad the right side with background (0) until they match maximum sequence width
    padded = np.zeros((target_height, max_width), dtype=np.float32)
    padded[:, :new_w] = resized / 255.0 # Normalize pixel values to fall between 0 and 1
    
    # Shape: (1, 64, 256)
    return padded[np.newaxis, ...]

# Step 2: Line Segmentation
def segment_into_lines(page_image_path):
    """
    Use horizontal projection profiles to detect gaps between lines of text.
    """
    img = cv2.imread(page_image_path, cv2.IMREAD_GRAYSCALE)
    _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Horizontal projection profile
    h_proj = np.sum(binary, axis=1)
    
    lines = []
    in_line = False
    start_y = 0
    
    for y, val in enumerate(h_proj):
        if val > 0 and not in_line:
            in_line = True
            start_y = y
        elif val == 0 and in_line:
            in_line = False
            if y - start_y > 10: # Minimum line height
                lines.append(img[start_y:y, :])
                
    if in_line and h_proj.shape[0] - start_y > 10:
        lines.append(img[start_y:, :])
        
    return lines

# Step 5: Decoding via CTC Loss
def ctc_decode(predictions, classes):
    """
    Take the most probable character at each step, merge adjacent repeating characters,
    and remove the CTC blank tokens (index 0).
    """
    # predictions: (Sequence Length, Num Classes)
    max_probs = torch.argmax(predictions, dim=-1)
    
    decoded = []
    prev_idx = -1
    
    for idx in max_probs:
        idx = idx.item()
        # 0 is the CTC blank token
        if idx != 0 and idx != prev_idx:
            # -1 because blank token shifts all classes by 1
            decoded.append(classes[idx - 1])
        prev_idx = idx
        
    return "".join(decoded)

# Example CTC Loss Initialization for Training
def get_ctc_loss():
    # blank=0 means the 0th index is the blank token
    return nn.CTCLoss(blank=0, zero_infinity=True)

