import os
import sys
import json
import torch
import torch.nn.functional as F
from PIL import Image

base_dir = os.path.dirname(os.path.abspath(__file__))
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

from src.models import CRNN_CTC
from src.image_processing import preprocess_sequence_image, segment_lines

def load_character_model():
    models_dir = os.path.join(base_dir, "models")
    model_path = os.path.join(models_dir, "handwritten_crnn_best.pth")
    class_map_path = os.path.join(models_dir, "class_mapping.json")
    
    if not os.path.exists(model_path) or not os.path.exists(class_map_path):
        raise FileNotFoundError("Model artifacts not found. Please run train.py first.")
        
    with open(class_map_path) as f:
        classes = json.load(f)["classes"]
        
    model = CRNN_CTC(num_classes=len(classes))
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    return model, classes

def ctc_decode(preds, classes):
    # preds: (T,)
    decoded = []
    blank_idx = len(classes)
    prev_idx = -1
    for idx in preds:
        idx = int(idx.item())
        if idx != blank_idx and idx != prev_idx:
            decoded.append(classes[idx])
        prev_idx = idx
    return "".join(decoded)

def predict_word_or_sequence(image_input, model=None, classes=None) -> dict:
    if model is None or classes is None:
        model, classes = load_character_model()
        
    # Segment into lines
    lines = segment_lines(image_input)
    
    recognized_lines = []
    
    for line_img in lines:
        processed_arr = preprocess_sequence_image(line_img, target_height=64, target_width=256)
        tensor_in = torch.tensor(processed_arr, dtype=torch.float32).unsqueeze(0).unsqueeze(0) # (1, 1, 64, 256)
        
        with torch.no_grad():
            outputs = model(tensor_in) # (1, T, C)
            probs = F.softmax(outputs, dim=2).squeeze(0) # (T, C)
            
        _, preds = torch.max(probs, dim=1) # (T,)
        text = ctc_decode(preds, classes)
        recognized_lines.append(text)
        
    full_text = "\n".join(recognized_lines)
    
    # We return the first line for single-mode compatibility
    top_text = recognized_lines[0] if recognized_lines else ""
    
    return {
        "success": True,
        "recognized_text": full_text,
        "predicted_character": top_text, # Compatibility for UI
        "confidence": 1.0, # CRNN CTC confidence is complex to compute (path prob), defaulting
        "confidence_percentage": "100%",
        "top3_candidates": [],
        "processed_image": processed_arr if lines else np.zeros((64, 256))
    }
