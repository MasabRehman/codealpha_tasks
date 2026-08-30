import cv2
import numpy as np
from PIL import Image

def preprocess_sequence_image(image_input, target_height=64, target_width=256) -> np.ndarray:
    if isinstance(image_input, str):
        img = Image.open(image_input).convert("L")
    elif isinstance(image_input, np.ndarray):
        if image_input.ndim == 3:
            img = Image.fromarray(image_input).convert("L")
        else:
            img = Image.fromarray(image_input)
    elif isinstance(image_input, Image.Image):
        img = image_input.convert("L")
    else:
        raise ValueError("Unsupported image input type.")
        
    arr = np.array(img, dtype=np.uint8)
    
    # Auto-inversion
    border_pixels = np.concatenate([arr[0, :], arr[-1, :], arr[:, 0], arr[:, -1]])
    if np.mean(border_pixels) > 127:
        arr = 255 - arr
        
    # Otsu binarization
    _, thresh = cv2.threshold(arr, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Scale to fixed height maintaining aspect ratio
    h, w = thresh.shape
    scale = target_height / float(h)
    new_w = int(w * scale)
    
    # Resize
    resized = cv2.resize(thresh, (new_w, target_height), interpolation=cv2.INTER_AREA)
    
    # Pad right side to reach target_width
    padded = np.zeros((target_height, target_width), dtype=np.uint8)
    if new_w > target_width:
        # If too wide, truncate
        padded[:, :] = resized[:, :target_width]
    else:
        padded[:, :new_w] = resized
        
    # Normalize 0 to 1
    normalized = padded.astype(np.float32) / 255.0
    return normalized

def segment_lines(image_input) -> list:
    """
    Uses horizontal projection profiles to detect gaps between lines of text.
    Returns a list of image strips (numpy arrays), one for each line.
    """
    if isinstance(image_input, str):
        img = Image.open(image_input).convert("L")
    elif isinstance(image_input, Image.Image):
        img = image_input.convert("L")
    else:
        img = Image.fromarray(image_input).convert("L")
        
    arr = np.array(img, dtype=np.uint8)
    border = np.concatenate([arr[0, :], arr[-1, :], arr[:, 0], arr[:, -1]])
    if np.mean(border) > 127:
        arr = 255 - arr
        
    _, binary = cv2.threshold(arr, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Horizontal projection profile
    proj = np.sum(binary, axis=1)
    
    # Find line segments based on projection sum > threshold
    threshold = np.max(proj) * 0.05
    in_line = False
    lines = []
    start_y = 0
    
    for y, val in enumerate(proj):
        if val > threshold and not in_line:
            in_line = True
            start_y = max(0, y - 5)
        elif val <= threshold and in_line:
            in_line = False
            end_y = min(binary.shape[0], y + 5)
            # Filter out very thin noise lines
            if end_y - start_y > 10:
                lines.append(arr[start_y:end_y, :])
                
    # If it ended while in a line
    if in_line:
        end_y = binary.shape[0]
        if end_y - start_y > 10:
            lines.append(arr[start_y:end_y, :])
            
    # Fallback if no lines found (e.g. single small line)
    if not lines:
        lines.append(arr)
        
    return lines
